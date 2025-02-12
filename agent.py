import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# Colors
WHITE = (255, 255, 255)
RED = (255, 0, 0)

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
    def __init__(self, state_dim, action_dim, lr=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995, epsilon_min=0.01):
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
    
    def store_transition(self, transition):
        if len(self.memory) > self.max_memory:
            self.memory.pop(0)
        self.memory.append(transition)
    
    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)
        state = torch.tensor(state, dtype=torch.float32).to(self.device).unsqueeze(0)
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
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

# Rocket Simulation
def get_rocket_state():
    #return np.array([WIDTH // 2, HEIGHT - 100, 0, 0, 0, 0])  # x, y, vx, vy, ax, ay
    return np.array([-6545, -3490, 2500, -3.457, 6.618, 2.533, 0, 0, 0])

def update_rocket(state, action):
    x, y, z, vx, vy, vz, ax, ay, az= state

    thrust_x = 0
    thrust_y = 0
    thrust_z = 0

    if action == 0:
        thrust_x = 1.5

    elif action == 1:
        thrust_y = 1.5
        
    elif action == 2:
        thrust_z = 1.5
    
    ax = thrust_x
    ay = thrust_y - 0.1  # Gravity
    az = thrust_z

    vx += ax
    vy += ay
    vz += az
    x += vx
    y += vy
    z += vz

    d = np.linalg.norm([x,y,z])

    reward = -abs((d - 6371)/1000)  # Reward staying centered & stable speed
    done = d > 6371*2 or d < 6371 # Reset if altitude exceeds initial height
    
    return np.array([x, y, z, vx, vy, vz, ax, ay, az]), reward, done

# Main loop
def run_simulation(episodes=500):
    agent = DQNAgent(state_dim=6, action_dim=9)
    
    for episode in range(episodes):
        state = get_rocket_state()
        total_reward = 0
        done = False
        
        while not done:
            action = agent.select_action(state)
            next_state, reward, done = update_rocket(state, action)
            agent.store_transition((state, action, reward, next_state, done))
            state = next_state
            total_reward += reward
            agent.train()
            
            #pygame.draw.rect(screen, RED, (state[0] - 10, state[1] - 20, 20, 40))
            #socketio.emit('800', state)
            #pygame.display.flip()
            #clock.tick(30)
        
        agent.update_target_network()
        print(f"Episode {episode+1}: Total Reward: {total_reward}")
    
    return agent

if __name__ == "__main__":
    trained_agent = run_simulation()
