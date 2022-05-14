import random
import numpy as np
import gym
import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ReplayMemory:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def __len__(self):
        return len(self.memory)

    # store the experience into the memory
    def push(self, obs, action, next_obs, reward):
        if len(self.memory) < self.capacity:
            self.memory.append(None)

        self.memory[self.position] = (obs, action, next_obs, reward)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """
        Samples batch_size transitions from the replay memory and returns a tuple
            (obs, action, next_obs, reward)
        """
        sample = random.sample(self.memory, batch_size)
        return tuple(zip(*sample))


class DQN(nn.Module):
    def __init__(self, env_config):
        super(DQN, self).__init__()

        # Save hyperparameters needed in the DQN class.
        self.env_name = env_config["env_name"]
        self.batch_size = env_config["batch_size"]
        self.gamma = env_config["gamma"]
        self.eps_start = env_config["eps_start"]
        self.eps_end = env_config["eps_end"]
        self.eps = env_config["eps_start"]
        self.anneal_length = env_config["anneal_length"]
        self.n_actions = env_config["n_actions"]
        if self.env_name == 'CartPole-v0':
          self.fc1 = nn.Linear(4, 256)
          self.fc2 = nn.Linear(256, self.n_actions)

        elif self.env_name == 'Pong-v0':
          self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4, padding=0)
          self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0)
          self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0)
          self.fc1 = nn.Linear(3136, 512)
          self.fc2 = nn.Linear(512, self.n_actions)
        
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()

    def forward(self, x):
        """Runs the forward pass of the NN depending on architecture."""
        if self.env_name == 'CartPole-v0':
          x = self.relu(self.fc1(x))
          x = self.fc2(x)
        if self.env_name == 'Pong-v0':
          x = self.relu(self.conv1(x))
          x = self.relu(self.conv2(x))
          x = self.relu(self.conv3(x))
          x = self.flatten(x)
          x = self.relu(self.fc1(x))
          x = self.fc2(x)


        return x

    def act(self, observation, exploit=False):
        """Selects an action with an epsilon-greedy exploration strategy."""
        # TODO: Implement action selection using the Deep Q-network. This function
        #       takes an observation tensor and should return a tensor of actions.
        #       For example, if the state dimension is 4 and the batch size is 32,
        #       the input would be a [32, 4] tensor and the output a [32, 1] tensor.
        

        
        # TODO: Implement epsilon-greedy exploration.
        if random.random() > self.eps:
          with torch.no_grad():
            action_values = self.forward(observation)
            action_chose = action_values.max(1)[1]
            if self.env_name == 'Pong-v0':
              action_chose = action_chose + 2
            
    

        else:
          if self.env_name == 'CartPole-v0':
            action_chose = torch.tensor(np.random.choice(self.n_actions))
          elif self.env_name == 'Pong-v0':
            # We only take 3 valid actions in pong: 1-stay, 2-right, 3-left
            action_chose = torch.tensor(np.random.choice([2, 3]))
          action_chose = action_chose.unsqueeze(0)
          
        
        if self.eps > self.eps_end:
          self.eps -= (self.eps_start - self.eps_end)/self.anneal_length
        return action_chose
        raise NotImplmentedError

def optimize(dqn, target_dqn, memory, optimizer):
    """This function samples a batch from the replay buffer and optimizes the Q-network."""
    # If we don't have enough transitions stored yet, we don't train.
    if len(memory) < dqn.batch_size:
        return

    # TODO: Sample a batch from the replay memory and concatenate so that there are
    #       four tensors in total: observations, actions, next observations and rewards.
    #       Remember to move them to GPU if it is available, e.g., by using Tensor.to(device).
    #       Note that special care is needed for terminal transitions!
    #Initialzing the q_value_targets and the q_values
    q_values = []
    q_value_targets = []
    experience = memory.sample(dqn.batch_size)
   
    non_terminal_next_states = [s for s in experience[2] if s is not None]
    non_terminal_next_states = torch.cat(non_terminal_next_states).to(device)
    states = torch.cat(experience[0]).to(device) #size = 32 * 4
    actions = torch.stack(experience[1]).to(device) #size = 32
    actions = torch.unsqueeze(actions, 1)  #size = 32 * 1
    rewards = torch.stack(experience[3]).to(device) #size = 32
  
    # TODO: Compute the current estimates of the Q-values for each state-action
    #       pair (s,a). Here, torch.gather() is useful for selecting the Q-values
    #       corresponding to the chosen actions.

    q_values = dqn.forward(states)
    if dqn.env_name == 'CartPole-v0':
      q_values = q_values.gather(1, actions)
    elif dqn.env_name == 'Pong-v0':
      q_values = q_values.gather(1, actions - 2)
    
    
    # TODO: Compute the Q-value targets. Only do this for non-terminal transitions!
    mask = []         #find the indexes of non-terminal-next-states
    index = 0
    for s in experience[2]:
      if s is not None:
          mask.append(index)
      index += 1 
    mask = torch.from_numpy(np.array(mask)).to(device)

    next_action_values = torch.zeros(dqn.batch_size).to(device)
    next_action_values[mask] = target_dqn.forward(non_terminal_next_states).max(1)[0] #find the maximum action value amone multiple action-value paris
    q_value_targets = (next_action_values * target_dqn.gamma) + rewards

    # Compute loss.
    loss = F.mse_loss(q_values.squeeze(), q_value_targets)


    # Perform gradient descent.
    optimizer.zero_grad()

    loss.backward()
    optimizer.step()

    return loss.item()
