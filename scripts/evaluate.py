"""Evaluation script for RL Gaming AI."""

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from algorithms.dqn import DQNAgent
from algorithms.ppo import PPOAgent
from train.trainer import Evaluator
from utils import get_device, set_seed


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def create_agent(config: dict, device: torch.device):
    """Create agent from configuration."""
    env_name = config["env_name"]
    algorithm = config["algorithm"]
    
    # Create dummy environment to get dimensions
    import gymnasium as gym
    env = gym.make(env_name)
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n
    env.close()
    
    if algorithm.lower() == 'dqn':
        agent = DQNAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config.get('learning_rate', 0.001),
            gamma=config.get('gamma', 0.99),
            epsilon=0.0,  # No exploration during evaluation
            epsilon_decay=1.0,
            epsilon_min=0.0,
            buffer_size=config.get('buffer_size', 10000),
            batch_size=config.get('batch_size', 32),
            target_update=config.get('target_update', 100),
            device=device,
            use_double_dqn=config.get('use_double_dqn', True),
            use_prioritized_replay=config.get('use_prioritized_replay', False),
            use_noisy_dqn=config.get('use_noisy_dqn', False),
        )
    elif algorithm.lower() == 'ppo':
        agent = PPOAgent(
            state_size=state_size,
            action_size=action_size,
            learning_rate=config.get('learning_rate', 3e-4),
            gamma=config.get('gamma', 0.99),
            gae_lambda=config.get('gae_lambda', 0.95),
            clip_ratio=config.get('clip_ratio', 0.2),
            value_loss_coef=config.get('value_loss_coef', 0.5),
            entropy_coef=config.get('entropy_coef', 0.01),
            max_grad_norm=config.get('max_grad_norm', 0.5),
            buffer_size=config.get('buffer_size', 2048),
            batch_size=config.get('batch_size', 64),
            n_epochs=config.get('n_epochs', 10),
            device=device,
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm}")
    
    return agent


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate RL Gaming AI Agent")
    parser.add_argument(
        "--config", 
        type=str, 
        default="configs/dqn_config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model_path", 
        type=str, 
        required=True,
        help="Path to trained model"
    )
    parser.add_argument(
        "--n_episodes", 
        type=int, 
        default=100,
        help="Number of episodes to evaluate"
    )
    parser.add_argument(
        "--render", 
        action="store_true",
        help="Render environment during evaluation"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42,
        help="Random seed"
    )
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set device
    device = get_device()
    print(f"Using device: {device}")
    
    # Set seed
    set_seed(args.seed)
    
    # Create agent
    agent = create_agent(config, device)
    
    # Load trained model
    agent.load(args.model_path)
    print(f"Loaded model from: {args.model_path}")
    
    # Create evaluator
    evaluator = Evaluator(config["env_name"], device)
    
    # Evaluate agent
    print(f"Evaluating agent over {args.n_episodes} episodes...")
    results = evaluator.evaluate_agent(
        agent, 
        config["algorithm"], 
        n_episodes=args.n_episodes
    )
    
    # Print results
    print("\nEvaluation Results:")
    print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
    print(f"95% Confidence Interval: [{results['reward_ci'][0]:.2f}, {results['reward_ci'][1]:.2f}]")
    print(f"Mean Episode Length: {results['mean_length']:.2f} ± {results['std_length']:.2f}")
    print(f"Success Rate: {results['success_rate']:.2%}")
    
    # Save results
    results_file = f"assets/{config['algorithm']}_evaluation_results.npz"
    os.makedirs("assets", exist_ok=True)
    np.savez(results_file, **results)
    print(f"Results saved to: {results_file}")
    
    # Render episodes if requested
    if args.render:
        print("\nRendering episodes...")
        import gymnasium as gym
        env = gym.make(config["env_name"], render_mode="human")
        
        for episode in range(5):  # Render 5 episodes
            state, _ = env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if config["algorithm"].lower() == 'dqn':
                    action = agent.select_action(state, training=False)
                elif config["algorithm"].lower() == 'ppo':
                    action, _, _ = agent.select_action(state)
                
                next_state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
            
            print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")
        
        env.close()


if __name__ == "__main__":
    main()
