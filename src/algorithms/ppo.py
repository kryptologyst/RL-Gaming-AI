"""PPO implementation for gaming AI."""

import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..utils import get_device


class PPONetwork(nn.Module):
    """PPO Actor-Critic Network for gaming AI.
    
    Args:
        state_size: Size of state space.
        action_size: Number of actions.
        hidden_sizes: List of hidden layer sizes.
        activation: Activation function.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        hidden_sizes: List[int] = [64, 64],
        activation: str = "tanh",
    ):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size
        
        # Choose activation function
        if activation == "tanh":
            self.activation = torch.tanh
        elif activation == "relu":
            self.activation = F.relu
        else:
            raise ValueError(f"Unknown activation: {activation}")
        
        # Shared layers
        self.shared_layers = nn.ModuleList()
        prev_size = state_size
        
        for hidden_size in hidden_sizes:
            self.shared_layers.append(nn.Linear(prev_size, hidden_size))
            prev_size = hidden_size
        
        # Actor head (policy)
        self.actor_head = nn.Linear(prev_size, action_size)
        
        # Critic head (value function)
        self.critic_head = nn.Linear(prev_size, 1)
        
    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass through the network.
        
        Args:
            state: Input state tensor.
            
        Returns:
            Tuple of (action_logits, value).
        """
        x = state
        
        # Shared layers
        for layer in self.shared_layers:
            x = self.activation(layer(x))
        
        # Actor and critic heads
        action_logits = self.actor_head(x)
        value = self.critic_head(x)
        
        return action_logits, value
    
    def get_action_and_value(
        self, 
        state: torch.Tensor, 
        action: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get action, log probability, value, and entropy.
        
        Args:
            state: Input state tensor.
            action: Action tensor (optional).
            
        Returns:
            Tuple of (action, log_prob, value, entropy).
        """
        action_logits, value = self.forward(state)
        
        # Create categorical distribution
        dist = torch.distributions.Categorical(logits=action_logits)
        
        if action is None:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, value.squeeze(-1), entropy


class PPOBuffer:
    """PPO Experience Buffer.
    
    Args:
        buffer_size: Size of the buffer.
        state_size: Size of state space.
        action_size: Number of actions.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        device: PyTorch device.
    """
    
    def __init__(
        self,
        buffer_size: int,
        state_size: int,
        action_size: int,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        device: Optional[torch.device] = None,
    ):
        self.buffer_size = buffer_size
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.device = device or get_device()
        
        # Initialize buffers
        self.states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions = np.zeros((buffer_size,), dtype=np.int32)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.values = np.zeros((buffer_size,), dtype=np.float32)
        self.log_probs = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.bool_)
        
        self.ptr = 0
        self.size = 0
        
    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add experience to buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            value: Value estimate.
            log_prob: Log probability of action.
            done: Whether episode is done.
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.values[self.ptr] = value
        self.log_probs[self.ptr] = log_prob
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def compute_gae(self, next_value: float) -> Tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.
        
        Args:
            next_value: Value of next state.
            
        Returns:
            Tuple of (advantages, returns).
        """
        advantages = np.zeros(self.size, dtype=np.float32)
        returns = np.zeros(self.size, dtype=np.float32)
        
        last_advantage = 0
        
        for t in reversed(range(self.size)):
            if t == self.size - 1:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = next_value
            else:
                next_non_terminal = 1.0 - self.dones[t]
                next_value_t = self.values[t + 1]
            
            delta = self.rewards[t] + self.gamma * next_value_t * next_non_terminal - self.values[t]
            advantages[t] = last_advantage = delta + self.gamma * self.gae_lambda * next_non_terminal * last_advantage
        
        returns = advantages + self.values[:self.size]
        
        return advantages, returns
    
    def get(self) -> Dict[str, torch.Tensor]:
        """Get all data from buffer.
        
        Returns:
            Dictionary of tensors.
        """
        advantages, returns = self.compute_gae(0)  # Assuming episode ended
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return {
            'states': torch.FloatTensor(self.states[:self.size]).to(self.device),
            'actions': torch.LongTensor(self.actions[:self.size]).to(self.device),
            'old_log_probs': torch.FloatTensor(self.log_probs[:self.size]).to(self.device),
            'advantages': torch.FloatTensor(advantages).to(self.device),
            'returns': torch.FloatTensor(returns).to(self.device),
        }
    
    def clear(self) -> None:
        """Clear the buffer."""
        self.ptr = 0
        self.size = 0


class PPOAgent:
    """PPO Agent for gaming AI.
    
    Args:
        state_size: Size of state space.
        action_size: Number of actions.
        learning_rate: Learning rate for optimizer.
        gamma: Discount factor.
        gae_lambda: GAE lambda parameter.
        clip_ratio: PPO clip ratio.
        value_loss_coef: Value loss coefficient.
        entropy_coef: Entropy coefficient.
        max_grad_norm: Maximum gradient norm for clipping.
        buffer_size: Experience buffer size.
        batch_size: Training batch size.
        n_epochs: Number of training epochs per update.
        device: PyTorch device.
    """
    
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_ratio: float = 0.2,
        value_loss_coef: float = 0.5,
        entropy_coef: float = 0.01,
        max_grad_norm: float = 0.5,
        buffer_size: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_ratio = clip_ratio
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = device or get_device()
        
        # Initialize network
        self.network = PPONetwork(state_size, action_size).to(self.device)
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        
        # Initialize buffer
        self.buffer = PPOBuffer(buffer_size, state_size, action_size, gamma, gae_lambda, self.device)
        
    def select_action(self, state: np.ndarray) -> Tuple[int, float, float]:
        """Select action using current policy.
        
        Args:
            state: Current state.
            
        Returns:
            Tuple of (action, log_prob, value).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            action, log_prob, value, _ = self.network.get_action_and_value(state_tensor)
        
        return action.item(), log_prob.item(), value.item()
    
    def add_experience(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        value: float,
        log_prob: float,
        done: bool,
    ) -> None:
        """Add experience to buffer.
        
        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            value: Value estimate.
            log_prob: Log probability of action.
            done: Whether episode is done.
        """
        self.buffer.add(state, action, reward, value, log_prob, done)
    
    def learn(self) -> Optional[Dict[str, float]]:
        """Learn from buffer.
        
        Returns:
            Dictionary of loss values if learning occurred, None otherwise.
        """
        if self.buffer.size < self.batch_size:
            return None
        
        data = self.buffer.get()
        
        # Training loop
        for _ in range(self.n_epochs):
            # Create batches
            indices = torch.randperm(self.buffer.size)
            
            for start_idx in range(0, self.buffer.size, self.batch_size):
                end_idx = min(start_idx + self.batch_size, self.buffer.size)
                batch_indices = indices[start_idx:end_idx]
                
                # Get batch data
                batch_states = data['states'][batch_indices]
                batch_actions = data['actions'][batch_indices]
                batch_old_log_probs = data['old_log_probs'][batch_indices]
                batch_advantages = data['advantages'][batch_indices]
                batch_returns = data['returns'][batch_indices]
                
                # Forward pass
                _, new_log_probs, new_values, entropy = self.network.get_action_and_value(
                    batch_states, batch_actions
                )
                
                # Calculate ratios
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # Calculate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_ratio, 1 + self.clip_ratio) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                value_loss = F.mse_loss(new_values, batch_returns)
                
                entropy_loss = -entropy.mean()
                
                total_loss = actor_loss + self.value_loss_coef * value_loss + self.entropy_coef * entropy_loss
                
                # Backward pass
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()
        
        # Clear buffer
        self.buffer.clear()
        
        return {
            'actor_loss': actor_loss.item(),
            'value_loss': value_loss.item(),
            'entropy_loss': entropy_loss.item(),
            'total_loss': total_loss.item(),
        }
    
    def save(self, filepath: str) -> None:
        """Save agent state.
        
        Args:
            filepath: Path to save file.
        """
        torch.save({
            'network_state_dict': self.network.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, filepath)
    
    def load(self, filepath: str) -> None:
        """Load agent state.
        
        Args:
            filepath: Path to load file from.
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.network.load_state_dict(checkpoint['network_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
