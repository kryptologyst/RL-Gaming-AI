"""
Project 630: RL for Gaming AI - Legacy Implementation

This file contains the original implementation that has been modernized and refactored
into the comprehensive RL Gaming AI project. The modern implementation can be found in:

- src/algorithms/dqn.py: Advanced DQN with Double DQN, Prioritized Replay, Noisy Networks
- src/algorithms/ppo.py: PPO implementation with GAE
- src/train/trainer.py: Modern training and evaluation framework
- scripts/train.py: Training script with configuration management
- scripts/evaluate.py: Evaluation script with statistical analysis
- demo/app.py: Interactive Streamlit demo

For the modern implementation, please use:
    python scripts/train.py --config configs/dqn_config.yaml
    python scripts/evaluate.py --config configs/dqn_config.yaml --model_path checkpoints/dqn_final.pth
    streamlit run demo/app.py

⚠️ DISCLAIMER: This project is for research and educational purposes only.
Not for production control of real systems.
"""

# Original implementation preserved for reference
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# 1. Define the Q-network (Deep Q-Network) for gaming AI
class QNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)  # Output: Q-values for each action

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Output: Q-values for each action

# 2. Define the DQN agent for gaming AI
class DQNAgent:
    def __init__(self, model, learning_rate=0.001, gamma=0.99, epsilon=1.0, epsilon_decay=0.995):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.criterion = nn.MSELoss()

    def select_action(self, state):
        # Epsilon-greedy action selection
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.model.fc3.out_features)  # Random action (exploration)
        else:
            q_values = self.model(torch.tensor(state, dtype=torch.float32))
            return torch.argmax(q_values).item()  # Select action with the highest Q-value

    def update(self, state, action, reward, next_state, done):
        # Q-learning update rule
        q_values = self.model(torch.tensor(state, dtype=torch.float32))
        next_q_values = self.model(torch.tensor(next_state, dtype=torch.float32))
        target = reward + self.gamma * torch.max(next_q_values) * (1 - done)
        loss = self.criterion(q_values[action], target)  # Compute loss (MSE)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay epsilon (exploration rate)
        if done:
            self.epsilon *= self.epsilon_decay

        return loss.item()

# 3. Initialize the environment and DQN agent
env = gym.make('CartPole-v1')  # Example gaming environment
model = QNetwork(input_size=env.observation_space.shape[0], output_size=env.action_space.n)
agent = DQNAgent(model)

# 4. Train the agent using DQN for gaming AI
num_episodes = 1000
for episode in range(num_episodes):
    state, _ = env.reset()
    done = False
    total_reward = 0
    while not done:
        action = agent.select_action(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated

        # Update the agent using DQN
        loss = agent.update(state, action, reward, next_state, done)
        total_reward += reward
        state = next_state

    if episode % 100 == 0:
        print(f"Episode {episode}, Total Reward: {total_reward}, Loss: {loss:.4f}")

# 5. Evaluate the agent after training (no exploration, only exploitation)
state, _ = env.reset()
done = False
total_reward = 0
while not done:
    action = agent.select_action(state)
    next_state, reward, terminated, truncated, _ = env.step(action)
    done = terminated or truncated
    total_reward += reward
    state = next_state

print(f"Total reward after DQN training for Gaming AI: {total_reward}")
env.close()
