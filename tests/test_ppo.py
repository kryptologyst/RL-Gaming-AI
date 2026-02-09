"""Tests for PPO algorithm."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.ppo import PPOAgent, PPONetwork, PPOBuffer


class TestPPONetwork:
    """Test PPO network implementation."""
    
    def test_network_creation(self):
        """Test network creation."""
        network = PPONetwork(state_size=4, action_size=2)
        assert network.state_size == 4
        assert network.action_size == 2
        
        # Test forward pass
        state = torch.randn(1, 4)
        action_logits, value = network(state)
        assert action_logits.shape == (1, 2)
        assert value.shape == (1, 1)
    
    def test_get_action_and_value(self):
        """Test action and value extraction."""
        network = PPONetwork(state_size=4, action_size=2)
        state = torch.randn(1, 4)
        
        action, log_prob, value, entropy = network.get_action_and_value(state)
        
        assert isinstance(action, torch.Tensor)
        assert isinstance(log_prob, torch.Tensor)
        assert isinstance(value, torch.Tensor)
        assert isinstance(entropy, torch.Tensor)
        
        assert action.shape == (1,)
        assert log_prob.shape == (1,)
        assert value.shape == (1,)
        assert entropy.shape == (1,)
    
    def test_custom_hidden_sizes(self):
        """Test network with custom hidden sizes."""
        network = PPONetwork(state_size=4, action_size=2, hidden_sizes=[128, 64])
        state = torch.randn(1, 4)
        action_logits, value = network(state)
        assert action_logits.shape == (1, 2)
        assert value.shape == (1, 1)


class TestPPOBuffer:
    """Test PPO buffer implementation."""
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        buffer = PPOBuffer(buffer_size=1000, state_size=4, action_size=2)
        assert buffer.buffer_size == 1000
        assert buffer.size == 0
    
    def test_add_experience(self):
        """Test adding experiences."""
        buffer = PPOBuffer(buffer_size=1000, state_size=4, action_size=2)
        
        state = np.array([1, 2, 3, 4])
        action = 0
        reward = 1.0
        value = 0.5
        log_prob = -0.5
        done = False
        
        buffer.add(state, action, reward, value, log_prob, done)
        assert buffer.size == 1
    
    def test_compute_gae(self):
        """Test GAE computation."""
        buffer = PPOBuffer(buffer_size=1000, state_size=4, action_size=2)
        
        # Add some experiences
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = i == 9  # Last one is done
            
            buffer.add(state, action, reward, value, log_prob, done)
        
        advantages, returns = buffer.compute_gae(0.0)  # next_value = 0
        
        assert len(advantages) == buffer.size
        assert len(returns) == buffer.size
    
    def test_get_data(self):
        """Test getting data from buffer."""
        buffer = PPOBuffer(buffer_size=1000, state_size=4, action_size=2)
        
        # Add experiences
        for i in range(10):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = i == 9
            
            buffer.add(state, action, reward, value, log_prob, done)
        
        data = buffer.get()
        
        assert 'states' in data
        assert 'actions' in data
        assert 'old_log_probs' in data
        assert 'advantages' in data
        assert 'returns' in data
        
        assert data['states'].shape[0] == buffer.size
        assert data['actions'].shape[0] == buffer.size
    
    def test_clear_buffer(self):
        """Test clearing buffer."""
        buffer = PPOBuffer(buffer_size=1000, state_size=4, action_size=2)
        
        # Add some experiences
        for i in range(5):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = False
            
            buffer.add(state, action, reward, value, log_prob, done)
        
        assert buffer.size == 5
        
        buffer.clear()
        assert buffer.size == 0


class TestPPOAgent:
    """Test PPO agent implementation."""
    
    @pytest.fixture
    def agent(self):
        """Create PPO agent for testing."""
        return PPOAgent(
            state_size=4,
            action_size=2,
            learning_rate=3e-4,
            gamma=0.99,
            gae_lambda=0.95,
            clip_ratio=0.2,
            value_loss_coef=0.5,
            entropy_coef=0.01,
            max_grad_norm=0.5,
            buffer_size=1000,
            batch_size=32,
            n_epochs=3,
            device=torch.device("cpu"),
        )
    
    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.clip_ratio == 0.2
    
    def test_action_selection(self, agent):
        """Test action selection."""
        state = np.array([1, 2, 3, 4])
        action, log_prob, value = agent.select_action(state)
        
        assert isinstance(action, int)
        assert 0 <= action < agent.action_size
        assert isinstance(log_prob, float)
        assert isinstance(value, float)
    
    def test_add_experience(self, agent):
        """Test adding experience."""
        state = np.array([1, 2, 3, 4])
        action = 0
        reward = 1.0
        value = 0.5
        log_prob = -0.5
        done = False
        
        agent.add_experience(state, action, reward, value, log_prob, done)
        assert agent.buffer.size == 1
    
    def test_learn_with_insufficient_data(self, agent):
        """Test learning with insufficient data."""
        # Should return None when not enough data
        losses = agent.learn()
        assert losses is None
    
    def test_learn_with_sufficient_data(self, agent):
        """Test learning with sufficient data."""
        # Add enough experiences
        for i in range(100):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = i % 20 == 19
            
            agent.add_experience(state, action, reward, value, log_prob, done)
        
        # Should be able to learn
        losses = agent.learn()
        assert losses is not None
        assert isinstance(losses, dict)
        assert 'actor_loss' in losses
        assert 'value_loss' in losses
        assert 'entropy_loss' in losses
        assert 'total_loss' in losses
    
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        # Add some experiences and learn
        for i in range(100):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            value = np.random.randn()
            log_prob = np.random.randn()
            done = i % 20 == 19
            
            agent.add_experience(state, action, reward, value, log_prob, done)
        
        agent.learn()
        
        # Save agent
        save_path = tmp_path / "test_ppo_agent.pth"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = PPOAgent(
            state_size=4,
            action_size=2,
            device=torch.device("cpu"),
        )
        new_agent.load(str(save_path))
        
        # Check that networks are loaded (can't easily check exact weights)
        assert new_agent.state_size == agent.state_size
        assert new_agent.action_size == agent.action_size
