"""Tests for utility functions."""

import pytest
import torch
import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from utils import (
    set_seed,
    get_device,
    calculate_confidence_interval,
    format_time,
    EarlyStopping,
    validate_config,
)


class TestSeeding:
    """Test seeding functionality."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Test numpy
        np.random.seed(42)
        val1 = np.random.rand()
        np.random.seed(42)
        val2 = np.random.rand()
        assert val1 == val2
        
        # Test torch
        torch.manual_seed(42)
        val1 = torch.rand(1).item()
        torch.manual_seed(42)
        val2 = torch.rand(1).item()
        assert val1 == val2


class TestDeviceHandling:
    """Test device handling."""
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        assert isinstance(device, torch.device)
        
        # Should be one of the supported devices
        assert device.type in ['cpu', 'cuda', 'mps']


class TestConfidenceInterval:
    """Test confidence interval calculation."""
    
    def test_confidence_interval_normal_data(self):
        """Test confidence interval with normal data."""
        data = np.random.normal(0, 1, 1000)
        mean, lower, upper = calculate_confidence_interval(data)
        
        assert isinstance(mean, float)
        assert isinstance(lower, float)
        assert isinstance(upper, float)
        assert lower <= mean <= upper
    
    def test_confidence_interval_different_levels(self):
        """Test confidence interval with different confidence levels."""
        data = np.random.normal(0, 1, 1000)
        
        mean_95, lower_95, upper_95 = calculate_confidence_interval(data, confidence=0.95)
        mean_99, lower_99, upper_99 = calculate_confidence_interval(data, confidence=0.99)
        
        # 99% CI should be wider than 95% CI
        assert (upper_99 - lower_99) > (upper_95 - lower_95)
    
    def test_confidence_interval_single_value(self):
        """Test confidence interval with single value."""
        data = np.array([5.0])
        mean, lower, upper = calculate_confidence_interval(data)
        
        assert mean == 5.0
        assert lower == 5.0
        assert upper == 5.0


class TestTimeFormatting:
    """Test time formatting."""
    
    def test_format_time_seconds(self):
        """Test formatting seconds."""
        assert format_time(30.5) == "30.5s"
        assert format_time(59.9) == "59.9s"
    
    def test_format_time_minutes(self):
        """Test formatting minutes."""
        assert format_time(60) == "1.0m"
        assert format_time(120) == "2.0m"
        assert format_time(3599) == "60.0m"
    
    def test_format_time_hours(self):
        """Test formatting hours."""
        assert format_time(3600) == "1.0h"
        assert format_time(7200) == "2.0h"


class TestEarlyStopping:
    """Test early stopping functionality."""
    
    def test_early_stopping_improvement(self):
        """Test early stopping with improvement."""
        model = torch.nn.Linear(1, 1)
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        
        # Simulate improving scores
        scores = [0.5, 0.6, 0.7, 0.8, 0.9]
        
        for score in scores:
            should_stop = early_stopping(score, model)
            assert not should_stop
        
        assert early_stopping.best_score == 0.9
    
    def test_early_stopping_no_improvement(self):
        """Test early stopping with no improvement."""
        model = torch.nn.Linear(1, 1)
        early_stopping = EarlyStopping(patience=3, min_delta=0.1)
        
        # Simulate no improvement
        scores = [0.5, 0.5, 0.5, 0.5, 0.5]
        
        for i, score in enumerate(scores):
            should_stop = early_stopping(score, model)
            if i < 3:
                assert not should_stop
            else:
                assert should_stop
    
    def test_early_stopping_min_delta(self):
        """Test early stopping with min_delta."""
        model = torch.nn.Linear(1, 1)
        early_stopping = EarlyStopping(patience=2, min_delta=0.1)
        
        # Small improvement should not count
        scores = [0.5, 0.55, 0.55, 0.55]
        
        for i, score in enumerate(scores):
            should_stop = early_stopping(score, model)
            if i < 3:
                assert not should_stop
            else:
                assert should_stop


class TestConfigValidation:
    """Test configuration validation."""
    
    def test_valid_config(self):
        """Test valid configuration."""
        config = {
            "env_name": "CartPole-v1",
            "algorithm": "dqn",
            "total_timesteps": 100000,
            "learning_rate": 0.001,
            "gamma": 0.99,
        }
        
        # Should not raise any errors
        validate_config(config)
    
    def test_missing_required_key(self):
        """Test missing required key."""
        config = {
            "env_name": "CartPole-v1",
            "algorithm": "dqn",
            # Missing total_timesteps
        }
        
        with pytest.raises(ValueError, match="Missing required config key"):
            validate_config(config)
    
    def test_invalid_total_timesteps(self):
        """Test invalid total_timesteps."""
        config = {
            "env_name": "CartPole-v1",
            "algorithm": "dqn",
            "total_timesteps": -1000,
        }
        
        with pytest.raises(ValueError, match="total_timesteps must be positive"):
            validate_config(config)
    
    def test_invalid_learning_rate(self):
        """Test invalid learning_rate."""
        config = {
            "env_name": "CartPole-v1",
            "algorithm": "dqn",
            "total_timesteps": 100000,
            "learning_rate": -0.001,
        }
        
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            validate_config(config)
    
    def test_invalid_gamma(self):
        """Test invalid gamma."""
        config = {
            "env_name": "CartPole-v1",
            "algorithm": "dqn",
            "total_timesteps": 100000,
            "gamma": 1.5,  # Should be <= 1
        }
        
        with pytest.raises(ValueError, match="gamma must be between 0 and 1"):
            validate_config(config)
