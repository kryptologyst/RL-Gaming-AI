"""Utility functions for RL Gaming AI project."""

import random
import numpy as np
import torch
from typing import Any, Dict, Optional, Tuple, Union
import gymnasium as gym


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device() -> torch.device:
    """Get the best available device (CUDA -> MPS -> CPU).
    
    Returns:
        PyTorch device.
    """
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


def set_env_seed(env: gym.Env, seed: int) -> None:
    """Set environment seed for reproducibility.
    
    Args:
        env: Gymnasium environment.
        seed: Random seed value.
    """
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env.observation_space.seed(seed)


def calculate_confidence_interval(
    data: np.ndarray, confidence: float = 0.95
) -> Tuple[float, float, float]:
    """Calculate confidence interval for data.
    
    Args:
        data: Array of values.
        confidence: Confidence level (default: 0.95).
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound).
    """
    mean = np.mean(data)
    std = np.std(data)
    n = len(data)
    
    # Calculate standard error
    se = std / np.sqrt(n)
    
    # Calculate critical value (approximate for normal distribution)
    alpha = 1 - confidence
    z_score = 1.96 if confidence == 0.95 else 2.576 if confidence == 0.99 else 1.645
    
    margin_error = z_score * se
    
    return mean, mean - margin_error, mean + margin_error


def format_time(seconds: float) -> str:
    """Format time in seconds to human readable string.
    
    Args:
        seconds: Time in seconds.
        
    Returns:
        Formatted time string.
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


class EarlyStopping:
    """Early stopping utility to prevent overfitting.
    
    Args:
        patience: Number of epochs to wait before stopping.
        min_delta: Minimum change to qualify as improvement.
        restore_best_weights: Whether to restore best weights.
    """
    
    def __init__(
        self,
        patience: int = 10,
        min_delta: float = 0.0,
        restore_best_weights: bool = True,
    ):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        
    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Check if training should stop early.
        
        Args:
            score: Current validation score.
            model: PyTorch model.
            
        Returns:
            True if training should stop.
        """
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                if self.restore_best_weights:
                    model.load_state_dict(self.best_weights)
                return True
        else:
            self.best_score = score
            self.counter = 0
            self.save_checkpoint(model)
            
        return False
    
    def save_checkpoint(self, model: torch.nn.Module) -> None:
        """Save model checkpoint.
        
        Args:
            model: PyTorch model.
        """
        if self.restore_best_weights:
            self.best_weights = model.state_dict().copy()


def validate_config(config: Dict[str, Any]) -> None:
    """Validate configuration parameters.
    
    Args:
        config: Configuration dictionary.
        
    Raises:
        ValueError: If configuration is invalid.
    """
    required_keys = ["env_name", "algorithm", "total_timesteps"]
    
    for key in required_keys:
        if key not in config:
            raise ValueError(f"Missing required config key: {key}")
    
    if config["total_timesteps"] <= 0:
        raise ValueError("total_timesteps must be positive")
    
    if config.get("learning_rate", 0) <= 0:
        raise ValueError("learning_rate must be positive")
    
    if not 0 <= config.get("gamma", 0) <= 1:
        raise ValueError("gamma must be between 0 and 1")
