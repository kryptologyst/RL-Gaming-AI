"""Main training script for RL Gaming AI."""

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml
from omegaconf import OmegaConf

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from train.trainer import Trainer, plot_training_curves
from utils import get_device, set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to configuration file.
        
    Returns:
        Configuration dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL Gaming AI Agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/dqn_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default="auto",
        help="Device to use (auto, cpu, cuda, mps)"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--total_timesteps", 
        type=int, 
        default=None,
        help="Total training timesteps (overrides config)"
    )
    parser.add_argument(
        "--eval_freq", 
        type=int, 
        default=None,
        help="Evaluation frequency (overrides config)"
    )
    parser.add_argument(
        "--save_freq", 
        type=int, 
        default=None,
        help="Save frequency (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.device != "auto":
        config["device"] = args.device
    if args.seed is not None:
        config["seed"] = args.seed
    if args.total_timesteps is not None:
        config["total_timesteps"] = args.total_timesteps
    if args.eval_freq is not None:
        config["eval_freq"] = args.eval_freq
    if args.save_freq is not None:
        config["save_freq"] = args.save_freq
    
    # Set device
    if config["device"] == "auto":
        device = get_device()
    else:
        device = torch.device(config["device"])
    
    print(f"Using device: {device}")
    print(f"Configuration: {config}")
    
    # Create trainer
    trainer = Trainer(
        env_name=config["env_name"],
        algorithm=config["algorithm"],
        config=config,
        device=device,
        seed=config["seed"],
    )
    
    # Create output directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Train agent
    metrics = trainer.train(
        total_timesteps=config["total_timesteps"],
        eval_freq=config.get("eval_freq", 5000),
        save_freq=config.get("save_freq", 10000),
    )
    
    # Plot training curves
    plot_training_curves(
        metrics, 
        save_path=f"assets/{config['algorithm']}_training_curves.png",
        show=False
    )
    
    # Final evaluation
    print("\nFinal Evaluation:")
    eval_rewards = trainer.evaluate(n_episodes=100)
    mean_reward = sum(eval_rewards) / len(eval_rewards)
    print(f"Average reward over 100 episodes: {mean_reward:.2f}")
    
    # Save final model
    final_model_path = f"checkpoints/{config['algorithm']}_final.pth"
    trainer.save_checkpoint(final_model_path)
    print(f"Final model saved to: {final_model_path}")


if __name__ == "__main__":
    main()
