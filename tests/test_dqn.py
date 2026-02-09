"""Tests for DQN algorithm."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.dqn import DQNAgent, DQNNetwork, NoisyDQNNetwork, PrioritizedReplayBuffer


class TestDQNNetwork:
    """Test DQN network implementation."""
    
    def test_network_creation(self):
        """Test network creation with different configurations."""
        network = DQNNetwork(input_size=4, output_size=2)
        assert network.input_size == 4
        assert network.output_size == 2
        
        # Test forward pass
        x = torch.randn(1, 4)
        output = network(x)
        assert output.shape == (1, 2)
    
    def test_network_with_dropout(self):
        """Test network with dropout."""
        network = DQNNetwork(input_size=4, output_size=2, dropout=0.5)
        x = torch.randn(1, 4)
        output = network(x)
        assert output.shape == (1, 2)
    
    def test_custom_hidden_sizes(self):
        """Test network with custom hidden sizes."""
        network = DQNNetwork(input_size=4, output_size=2, hidden_sizes=[128, 64, 32])
        x = torch.randn(1, 4)
        output = network(x)
        assert output.shape == (1, 2)


class TestNoisyDQNNetwork:
    """Test Noisy DQN network implementation."""
    
    def test_noisy_network_creation(self):
        """Test noisy network creation."""
        network = NoisyDQNNetwork(input_size=4, output_size=2)
        x = torch.randn(1, 4)
        output = network(x)
        assert output.shape == (1, 2)
    
    def test_noise_reset(self):
        """Test noise reset functionality."""
        network = NoisyDQNNetwork(input_size=4, output_size=2)
        network.reset_noise()
        # Should not raise any errors
        assert True


class TestPrioritizedReplayBuffer:
    """Test prioritized replay buffer implementation."""
    
    def test_buffer_creation(self):
        """Test buffer creation."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        assert buffer.capacity == 1000
        assert buffer.size == 0
    
    def test_add_experience(self):
        """Test adding experiences to buffer."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        experience = (np.array([1, 2, 3, 4]), 0, 1.0, np.array([1, 2, 3, 4]), False)
        buffer.add(experience)
        assert buffer.size == 1
    
    def test_sample_experience(self):
        """Test sampling experiences from buffer."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        
        # Add some experiences
        for i in range(10):
            experience = (np.array([i, i+1, i+2, i+3]), i % 2, float(i), np.array([i+1, i+2, i+3, i+4]), False)
            buffer.add(experience)
        
        experiences, indices, weights = buffer.sample(5)
        assert len(experiences) == 5
        assert len(indices) == 5
        assert len(weights) == 5
    
    def test_update_priorities(self):
        """Test priority updates."""
        buffer = PrioritizedReplayBuffer(capacity=1000)
        
        # Add experience
        experience = (np.array([1, 2, 3, 4]), 0, 1.0, np.array([1, 2, 3, 4]), False)
        buffer.add(experience)
        
        # Sample and update priorities
        experiences, indices, weights = buffer.sample(1)
        buffer.update_priorities(indices, np.array([0.5]))
        
        # Should not raise errors
        assert True


class TestDQNAgent:
    """Test DQN agent implementation."""
    
    @pytest.fixture
    def agent(self):
        """Create DQN agent for testing."""
        return DQNAgent(
            state_size=4,
            action_size=2,
            learning_rate=0.001,
            gamma=0.99,
            epsilon=0.1,
            epsilon_decay=0.995,
            epsilon_min=0.01,
            buffer_size=1000,
            batch_size=32,
            target_update=100,
            device=torch.device("cpu"),
        )
    
    def test_agent_creation(self, agent):
        """Test agent creation."""
        assert agent.state_size == 4
        assert agent.action_size == 2
        assert agent.epsilon == 0.1
    
    def test_action_selection(self, agent):
        """Test action selection."""
        state = np.array([1, 2, 3, 4])
        action = agent.select_action(state, training=True)
        assert isinstance(action, int)
        assert 0 <= action < agent.action_size
    
    def test_add_experience(self, agent):
        """Test adding experience."""
        state = np.array([1, 2, 3, 4])
        action = 0
        reward = 1.0
        next_state = np.array([2, 3, 4, 5])
        done = False
        
        agent.add_experience(state, action, reward, next_state, done)
        assert agent.step_count == 1
    
    def test_learn_with_insufficient_data(self, agent):
        """Test learning with insufficient data."""
        # Should return None when not enough data
        loss = agent.learn()
        assert loss is None
    
    def test_learn_with_sufficient_data(self, agent):
        """Test learning with sufficient data."""
        # Add enough experiences
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 9
            
            agent.add_experience(state, action, reward, next_state, done)
        
        # Should be able to learn
        loss = agent.learn()
        assert loss is not None
        assert isinstance(loss, float)
    
    def test_epsilon_decay(self, agent):
        """Test epsilon decay."""
        initial_epsilon = agent.epsilon
        
        # Add experience and learn
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 9
            
            agent.add_experience(state, action, reward, next_state, done)
        
        agent.learn()
        
        # Epsilon should have decayed
        assert agent.epsilon <= initial_epsilon
    
    def test_save_load(self, agent, tmp_path):
        """Test saving and loading agent."""
        # Train agent a bit
        for i in range(50):
            state = np.random.randn(4)
            action = np.random.randint(0, 2)
            reward = np.random.randn()
            next_state = np.random.randn(4)
            done = i % 10 == 9
            
            agent.add_experience(state, action, reward, next_state, done)
        
        agent.learn()
        
        # Save agent
        save_path = tmp_path / "test_agent.pth"
        agent.save(str(save_path))
        
        # Create new agent and load
        new_agent = DQNAgent(
            state_size=4,
            action_size=2,
            device=torch.device("cpu"),
        )
        new_agent.load(str(save_path))
        
        # Check that loaded agent has same epsilon
        assert new_agent.epsilon == agent.epsilon
        assert new_agent.step_count == agent.step_count
