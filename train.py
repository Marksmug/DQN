import argparse

import gym
from gym.wrappers import AtariPreprocessing
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

import config
from utils import preprocess
from evaluate import evaluate_policy
from dqn import DQN, ReplayMemory, optimize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser()
parser.add_argument('--env', choices=['CartPole-v0', 'Pong-v0'])
parser.add_argument('--evaluate_freq', type=int, default=25, help='How often to run evaluation.', nargs='?')
parser.add_argument('--evaluation_episodes', type=int, default=5, help='Number of evaluation episodes.', nargs='?')

# Hyperparameter configurations for different environments. See config.py.
ENV_CONFIGS = {
    'CartPole-v0': config.CartPole,
    'Pong-v0': config.Pong
}

if __name__ == '__main__':
    args = parser.parse_args()
    print(device)
  
    # Initialize environment and config.
    env = gym.make(args.env)
    if args.env == 'Pong-v0':
      # convert the frame to 84 * 84 grey scale
      env = AtariPreprocessing(env, screen_size=84, grayscale_obs=True, frame_skip=1, noop_max=30)

    env_config = ENV_CONFIGS[args.env]

    # Initialize deep Q-networks.
    dqn = DQN(env_config=env_config).to(device)
    # TODO: Create and initialize target Q-network.

    target_dqn = DQN(env_config=env_config).to(device)
    # Initianize scores/rewards every episode/mean loss every episode records
    scores = []
    reward_epi = []
    loss_epi = []
    # Initialize observation stack
    obs_stack = []
    # Create replay memory.
    memory = ReplayMemory(env_config['memory_size'])

    # Initialize optimizer used for training the DQN. We use Adam rather than RMSProp.
    optimizer = torch.optim.Adam(dqn.parameters(), lr=env_config['lr'])

    # Keep track of best evaluation mean return achieved so far.
    best_mean_return = -float("Inf")
    
    # keep track the steps 
    global step_counter 
    step_counter = 0

    for episode in range(env_config['n_episodes']):
        done = False

        
        obs = preprocess(env.reset(), env=args.env).unsqueeze(0)
        # Initialize total loss of current episode
        total_loss_per_epi = 0
        
        # Initialize total reward of current episode
        total_rewards_per_epi = 0
        
        # Initianize obs_stack for pong
        if args.env == 'Pong-v0':
          obs_stack = torch.cat(env_config['stack_size'] * [obs]).unsqueeze(0).to(device)
        #Initialize inner episode step counter 
        t = 0
        while not done:
            step_counter += 1
            t += 1
            
            # TODO: Get action from DQN.
            if args.env == 'Pong-v0':
              action = dqn.act(obs_stack).item()
            else:
              action = dqn.act(obs).item()
            #print(action)
          


            # Act in the true environment.
            next_obs, reward, done, info = env.step(action)

            # Adding reward of current step
            total_rewards_per_epi += reward

            # Preprocess incoming observation.
            if not done:
                next_obs = preprocess(next_obs, env=args.env).unsqueeze(0).to(device)
                if args.env == 'Pong-v0':
                  next_obs_stack = torch.cat((obs_stack[:, 1:, ...], next_obs.unsqueeze(1)), dim=1).to(device)
                  next_obs_stack = preprocess(next_obs_stack, env = args.env).to(device)
            else:
                next_obs = None
                if args.env == 'Pong-v0':
                  next_obs_stack = None
            
            # TODO: Add the transition to the replay memory. Remember to convert
            #       everything to PyTorch tensors!
            action = torch.tensor(action, device=device)
            reward = torch.tensor(reward, device=device)
            if args.env == 'CartPole-v0':
              memory.push(obs, action, next_obs, reward)
            else:
              memory.push(obs_stack, action, next_obs_stack, reward)
            obs = next_obs
            if args.env == 'Pong-v0':
              obs_stack = next_obs_stack

            # TODO: Run DQN.optimize() every env_config["train_frequency"] steps
            if step_counter % env_config["train_frequency"] == 0:
              loss = optimize(dqn, target_dqn, memory, optimizer)
              if loss != None:
                total_loss_per_epi += loss
                

            # TODO: Update the target network every env_config["target_update_frequency"] steps.
            if step_counter % env_config["target_update_frequency"] == 0:
              target_dqn.load_state_dict(dqn.state_dict())

            

              

        # Evaluate the current agent.
        mean_loss = total_loss_per_epi/t
        loss_epi.append(mean_loss)
        reward_epi.append(total_rewards_per_epi)
        print("Mean loss of current episode is:" , mean_loss)
        print("Total reward of cuurent episode is:", total_rewards_per_epi)
        if episode % args.evaluate_freq == 0:
            mean_return = evaluate_policy(dqn, env, env_config, args, n_episodes=args.evaluation_episodes)
            scores.append(mean_return)
            
            print(f'Episode {episode}/{env_config["n_episodes"]}: {mean_return}')

            # Save current agent if it has the best performance so far.
            if mean_return >= best_mean_return:
                best_mean_return = mean_return

                print('Best performance so far! Saving model.')
                torch.save(dqn, f'models/{args.env}_best.pt')
              
        
    # Close environment after training is completed.
    env.close()
    plt.figure(1)
    plt.plot(range(1, env_config['n_episodes'], args.evaluate_freq), scores)
    plt.xlabel("Number of Episode")
    plt.ylabel("Mean Score");
    plt.title("Evaluation Scores Every 25 Episodes")
    plt.savefig(f'figures/Evaluation_reward.jpg')

    plt.figure(2)
    plt.plot(range(0, env_config['n_episodes'], 1), reward_epi)
    plt.xlabel("Number of Episode")
    plt.ylabel("Score");
    plt.title("Scores of Every Episode")
    plt.savefig(f'figures/Every_reward.jpg')

    plt.figure(3)
    plt.plot(range(0, env_config['n_episodes'], 1), loss_epi)
    plt.xlabel("Number of Episode")
    plt.ylabel("Mean loss");
    plt.title("Mean Loss of Every Episode")
    plt.savefig(f'figures/Every_Loss.jpg')
