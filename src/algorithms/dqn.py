"""Modern DQN implementations for gaming AI."""

import random
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from ..utils import get_device, set_seed


class DQNNetwork(nn.Module):
    """Deep Q-Network architecture for gaming AI.
    
    Args:
        input_size: Size of input state.
        output_size: Number of actions.
        hidden_sizes: List of hidden layer sizes.
        dropout: Dropout probability.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_sizes: List[int] = [64, 64],
        dropout: float = 0.0,
    ):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network.
        
        Args:
            x: Input tensor.
            
        Returns:
            Q-values for each action.
        """
        return self.network(x)


class NoisyLinear(nn.Module):
    """Noisy linear layer for exploration in DQN."""
    
    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init
        
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        
        self.register_buffer('weight_epsilon', torch.empty(out_features, in_features))
        self.register_buffer('bias_epsilon', torch.empty(out_features))
        
        self.reset_parameters()
        self.reset_noise()
    
    def reset_parameters(self):
        """Reset parameters."""
        mu_range = 1 / np.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(self.std_init / np.sqrt(self.in_features))
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(self.std_init / np.sqrt(self.out_features))
    
    def reset_noise(self):
        """Reset noise."""
        epsilon_in = self._scale_noise(self.in_features)
        epsilon_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)
    
    def _scale_noise(self, size: int) -> torch.Tensor:
        """Scale noise."""
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """Forward pass with noise."""
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        
        return F.linear(input, weight, bias)


class NoisyDQNNetwork(nn.Module):
    """Noisy DQN network for exploration without epsilon-greedy."""
    
    def __init__(self, input_size: int, output_size: int, hidden_size: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = NoisyLinear(hidden_size, hidden_size)
        self.fc3 = NoisyLinear(hidden_size, output_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
    def reset_noise(self):
        """Reset noise in noisy layers."""
        self.fc2.reset_noise()
        self.fc3.reset_noise()


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay Buffer for DQN."""
    
    def __init__(
        self,
        capacity: int,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
    ):
        self.capacity = capacity
        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.position = 0
        self.size = 0
        
    def add(self, experience: Tuple) -> None:
        """Add experience to buffer."""
        max_priority = self.priorities.max() if self.size > 0 else 1.0
        
        if self.size < self.capacity:
            self.buffer.append(experience)
        else:
            self.buffer[self.position] = experience
            
        self.priorities[self.position] = max_priority
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size: int) -> Tuple[List, np.ndarray, np.ndarray]:
        """Sample batch from buffer."""
        if self.size < batch_size:
            batch_size = self.size
            
        priorities = self.priorities[:self.size]
        probabilities = priorities ** self.alpha
        probabilities /= probabilities.sum()
        
        indices = np.random.choice(self.size, batch_size, p=probabilities)
        
        experiences = [self.buffer[i] for i in indices]
        weights = (self.size * probabilities[indices]) ** (-self.beta)
        weights /= weights.max()
        
        return experiences, indices, weights
    
    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        """Update priorities for sampled experiences."""
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
        
        self.beta = min(1.0, self.beta + self.beta_increment)


class DQNAgent:
    """Modern DQN Agent with advanced features.
    
    Args:
        state_size: Size of state space.
        action_size: Number of actions.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor.
        epsilon: Initial exploration rate.
        epsilon_decay: Epsilon decay rate.
        epsilon_min: Minimum epsilon value.
        buffer_size: Replay buffer size.
        batch_size: Training batch size.
        target_update: Target network update frequency.
        device: PyTorch device.
        use_double_dqn: Whether to use Double DQN.
        use_prioritized_replay: Whether to use prioritized replay.
        use_noisy_dqn: Whether to use Noisy DQN.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01,
        buffer_size: int = 10000,
        batch_size: int = 32,
        target_update: int = 100,
        device: Optional[torch.device] = None,
        use_double_dqn: bool = True,
        use_prioritized_replay: bool = False,
        use_noisy_dqn: bool = False,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.target_update = target_update
        self.device = device or get_device()
        
        self.use_double_dqn = use_double_dqn
        self.use_prioritized_replay = use_prioritized_replay
        self.use_noisy_dqn = use_noisy_dqn
        
        # Initialize networks
        if use_noisy_dqn:
            self.q_network = NoisyDQNNetwork(state_size, action_size).to(self.device)
            self.target_network = NoisyDQNNetwork(state_size, action_size).to(self.device)
        else:
            self.q_network = DQNNetwork(state_size, action_size).to(self.device)
            self.target_network = DQNNetwork(state_size, action_size).to(self.device)
        
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Initialize replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(buffer_size)
        else:
            self.replay_buffer = []
            self.buffer_size = buffer_size
        
        self.step_count = 0
        
    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """Select action using epsilon-greedy or noisy network.
        
        Args:
            state: Current state.
            training: Whether in training mode.
            
        Returns:
            Selected action.
        """
        if self.use_noisy_dqn and training:
            # Noisy DQN doesn't need epsilon-greedy
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.q_network(state_tensor)
            return q_values.argmax().item()
        else:
            # Epsilon-greedy action selection
            if training and np.random.random() < self.epsilon:
                return np.random.choice(self.action_size)
            else:
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.q_network(state_tensor)
                return q_values.argmax().item()
    
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add experience to replay buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            done: Whether episode is done.
        """
        experience = (state, action, reward, next_state, done)
        
        if self.use_prioritized_replay:
            self.replay_buffer.add(experience)
        else:
            if len(self.replay_buffer) < self.buffer_size:
                self.replay_buffer.append(experience)
            else:
                self.replay_buffer[self.step_count % self.buffer_size] = experience
        
        self.step_count += 1
    
    def learn(self) -> Optional[float]:
        """Learn from replay buffer.
        
        Returns:
            Loss value if learning occurred, None otherwise.
        """
        if self.use_prioritized_replay:
            if self.replay_buffer.size < self.batch_size:
                return None
            experiences, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            if len(self.replay_buffer) < self.batch_size:
                return None
            experiences = random.sample(self.replay_buffer, self.batch_size)
            weights = np.ones(self.batch_size)
            indices = None
        
        # Convert experiences to tensors
        states = torch.FloatTensor([e[0] for e in experiences]).to(self.device)
        actions = torch.LongTensor([e[1] for e in experiences]).to(self.device)
        rewards = torch.FloatTensor([e[2] for e in experiences]).to(self.device)
        next_states = torch.FloatTensor([e[3] for e in experiences]).to(self.device)
        dones = torch.BoolTensor([e[4] for e in experiences]).to(self.device)
        weights = torch.FloatTensor(weights).to(self.device)
        
        # Compute current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Compute target Q values
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: use main network to select action, target network to evaluate
                next_actions = self.q_network(next_states).argmax(1)
                next_q_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1))
            else:
                # Standard DQN: use target network for both selection and evaluation
                next_q_values = self.target_network(next_states).max(1)[0].unsqueeze(1)
            
            target_q_values = rewards.unsqueeze(1) + (self.gamma * next_q_values * ~dones.unsqueeze(1))
        
        # Compute loss
        loss = F.mse_loss(current_q_values, target_q_values, reduction='none')
        weighted_loss = (loss * weights.unsqueeze(1)).mean()
        
        # Update network
        self.optimizer.zero_grad()
        weighted_loss.backward()
        self.optimizer.step()
        
        # Update priorities if using prioritized replay
        if self.use_prioritized_replay and indices is not None:
            priorities = loss.detach().cpu().numpy().flatten() + 1e-6
            self.replay_buffer.update_priorities(indices, priorities)
        
        # Update target network
        if self.step_count % self.target_update == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        # Decay epsilon
        if not self.use_noisy_dqn:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        
        # Reset noise for noisy DQN
        if self.use_noisy_dqn:
            self.q_network.reset_noise()
            self.target_network.reset_noise()
        
        return weighted_loss.item()
    
    def save(self, filepath: str) -> None:
        """Save agent state.
        
        Args:
            filepath: Path to save file.
        """
        torch.save({
            'q_network_state_dict': self.q_network.state_dict(),
            'target_network_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state.
        
        Args:
            filepath: Path to load file from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.q_network.load_state_dict(checkpoint['q_network_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
