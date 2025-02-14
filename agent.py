import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from system import gauss_jackson
from constants import *
from dynamic import *

# Define the Deep Q-Network (DQN)
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
    
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)

# Define the agent
class DQNAgent:
    def __init__(self, dynamic, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):

        self.dynamic: Dynamic = dynamic

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net = DQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = []
        self.batch_size = 64
        self.max_memory = 10000
    
        self.state_json = {}

        self.total_reward = 0
        self.active = True

    def store_transition(self, transition):
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def select_action(self):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(self.dynamic.state, dtype=torch.float32).to(self.device).unsqueeze(0)
        with torch.no_grad():
            return torch.argmax(self.policy_net(state)).item()
    
    def train(self):
        if len(self.memory) < self.batch_size:
            return
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.tensor(states, dtype=torch.float32).to(self.device)
        actions = torch.tensor(actions, dtype=torch.int64).to(self.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        next_states = torch.tensor(next_states, dtype=torch.float32).to(self.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(self.device)
        
        q_values = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        next_q_values = self.target_net(next_states).max(1)[0].detach()
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values
        
        loss = nn.MSELoss()(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
    
    def update_target_network(self, new_dinamic):
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.dynamic.update_to(new_dinamic)
        self.total_reward = 0
        self.active = True

    def update_state(self, action):
        x, y, z, vx, vy, vz, ax, ay, az = self.dynamic.state

        thrust_x = 0
        thrust_y = 0
        thrust_z = 0

        if action == 0:
            thrust_x = 150

        elif action == 1:
            thrust_y = 150
            
        elif action == 2:
            thrust_z = 150
        
        ax = thrust_x
        ay = thrust_y
        az = thrust_z


        new_dyn = gauss_jackson(np.array([[x, y, z, vx, vy, vz, ax, ay, az],[0,0,0,0,0,0,0,0,0]]), 10, [self.dynamic.mass, EARTH_MASS])

        d = np.linalg.norm([x,y,z])
        v = np.linalg.norm([vx,vy,vz])

        reward = -abs(d-6371e3) -abs(v) # Reward staying centered & stable speed
        self.active = not (d > 15371e3 or d < 6371e3) # Reset if altitude exceeds initial height

        return new_dyn, reward
    
    def train_me(self):
        action = self.select_action()
        next_dynamic, reward = self.update_state(action)
        self.store_transition((self.dynamic.state, action, reward, next_dynamic.state, self.active))
        self.dynamic.update_to(next_dynamic)
        self.total_reward += reward
        self.train()