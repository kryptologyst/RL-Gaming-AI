"""Training and evaluation modules for RL Gaming AI."""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm

from ..algorithms.dqn import DQNAgent
from ..algorithms.ppo import PPOAgent
from ..utils import calculate_confidence_interval, format_time, set_env_seed, set_seed


class Trainer:
    """Base trainer class for RL agents.
    
    Args:
        env_name: Name of the environment.
        algorithm: Algorithm to use ('dqn' or 'ppo').
        config: Training configuration.
        device: PyTorch device.
        seed: Random seed.
    """
    
    def __init__(
        self,
        env_name: str,
        algorithm: str,
        config: Dict[str, Any],
        device: Optional[torch.device] = None,
        seed: int = 42,
    ):
        self.env_name = env_name
        self.algorithm = algorithm
        self.config = config
        self.device = device or torch.device("cpu")
        self.seed = seed
        
        # Set seeds
        set_seed(seed)
        
        # Create environment
        self.env = gym.make(env_name)
        set_env_seed(self.env, seed)
        
        # Get state and action sizes
        self.state_size = self.env.observation_space.shape[0]
        self.action_size = self.env.action_space.n
        
        # Initialize agent
        self.agent = self._create_agent()
        
        # Training metrics
        self.episode_rewards = []
        self.episode_lengths = []
        self.losses = []
        self.training_time = 0
        
    def _create_agent(self):
        """Create agent based on algorithm."""
        if self.algorithm.lower() == 'dqn':
            return DQNAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.config.get('learning_rate', 0.001),
                gamma=self.config.get('gamma', 0.99),
                epsilon=self.config.get('epsilon', 1.0),
                epsilon_decay=self.config.get('epsilon_decay', 0.995),
                epsilon_min=self.config.get('epsilon_min', 0.01),
                buffer_size=self.config.get('buffer_size', 10000),
                batch_size=self.config.get('batch_size', 32),
                target_update=self.config.get('target_update', 100),
                device=self.device,
                use_double_dqn=self.config.get('use_double_dqn', True),
                use_prioritized_replay=self.config.get('use_prioritized_replay', False),
                use_noisy_dqn=self.config.get('use_noisy_dqn', False),
            )
        elif self.algorithm.lower() == 'ppo':
            return PPOAgent(
                state_size=self.state_size,
                action_size=self.action_size,
                learning_rate=self.config.get('learning_rate', 3e-4),
                gamma=self.config.get('gamma', 0.99),
                gae_lambda=self.config.get('gae_lambda', 0.95),
                clip_ratio=self.config.get('clip_ratio', 0.2),
                value_loss_coef=self.config.get('value_loss_coef', 0.5),
                entropy_coef=self.config.get('entropy_coef', 0.01),
                max_grad_norm=self.config.get('max_grad_norm', 0.5),
                buffer_size=self.config.get('buffer_size', 2048),
                batch_size=self.config.get('batch_size', 64),
                n_epochs=self.config.get('n_epochs', 10),
                device=self.device,
            )
        else:
            raise ValueError(f"Unknown algorithm: {self.algorithm}")
    
    def train(self, total_timesteps: int, eval_freq: int = 1000, save_freq: int = 10000) -> Dict[str, List]:
        """Train the agent.
        
        Args:
            total_timesteps: Total number of training timesteps.
            eval_freq: Frequency of evaluation.
            save_freq: Frequency of saving checkpoints.
            
        Returns:
            Dictionary of training metrics.
        """
        print(f"Starting training with {self.algorithm.upper()} on {self.env_name}")
        print(f"Device: {self.device}")
        print(f"Total timesteps: {total_timesteps}")
        
        start_time = time.time()
        timestep = 0
        episode = 0
        
        # Training loop
        with tqdm(total=total_timesteps, desc="Training") as pbar:
            while timestep < total_timesteps:
                episode += 1
                episode_reward = 0
                episode_length = 0
                
                # Reset environment
                state, _ = self.env.reset()
                done = False
                
                while not done and timestep < total_timesteps:
                    # Select action
                    if self.algorithm.lower() == 'dqn':
                        action = self.agent.select_action(state, training=True)
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        
                        # Add experience to buffer
                        self.agent.add_experience(state, action, reward, next_state, done)
                        
                        # Learn
                        loss = self.agent.learn()
                        if loss is not None:
                            self.losses.append(loss)
                        
                        state = next_state
                        
                    elif self.algorithm.lower() == 'ppo':
                        action, log_prob, value = self.agent.select_action(state)
                        next_state, reward, terminated, truncated, _ = self.env.step(action)
                        done = terminated or truncated
                        
                        # Add experience to buffer
                        self.agent.add_experience(state, action, reward, value, log_prob, done)
                        
                        state = next_state
                    
                    episode_reward += reward
                    episode_length += 1
                    timestep += 1
                    pbar.update(1)
                
                # Learn for PPO at end of episode
                if self.algorithm.lower() == 'ppo':
                    losses = self.agent.learn()
                    if losses is not None:
                        self.losses.append(losses['total_loss'])
                
                # Record episode metrics
                self.episode_rewards.append(episode_reward)
                self.episode_lengths.append(episode_length)
                
                # Update progress bar
                if episode % 10 == 0:
                    avg_reward = np.mean(self.episode_rewards[-10:])
                    pbar.set_postfix({
                        'Episode': episode,
                        'Avg Reward': f"{avg_reward:.2f}",
                        'Epsilon': f"{self.agent.epsilon:.3f}" if hasattr(self.agent, 'epsilon') else "N/A"
                    })
                
                # Evaluation
                if timestep % eval_freq == 0:
                    eval_rewards = self.evaluate(n_episodes=5)
                    avg_eval_reward = np.mean(eval_rewards)
                    print(f"\nTimestep {timestep}: Average eval reward = {avg_eval_reward:.2f}")
                
                # Save checkpoint
                if timestep % save_freq == 0:
                    self.save_checkpoint(f"checkpoints/{self.algorithm}_{timestep}.pth")
        
        self.training_time = time.time() - start_time
        
        print(f"\nTraining completed in {format_time(self.training_time)}")
        print(f"Total episodes: {episode}")
        print(f"Average reward (last 100 episodes): {np.mean(self.episode_rewards[-100:]):.2f}")
        
        return {
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths,
            'losses': self.losses,
            'training_time': self.training_time,
        }
    
    def evaluate(self, n_episodes: int = 10, render: bool = False) -> List[float]:
        """Evaluate the agent.
        
        Args:
            n_episodes: Number of episodes to evaluate.
            render: Whether to render the environment.
            
        Returns:
            List of episode rewards.
        """
        eval_rewards = []
        
        for episode in range(n_episodes):
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                if self.algorithm.lower() == 'dqn':
                    action = self.agent.select_action(state, training=False)
                elif self.algorithm.lower() == 'ppo':
                    action, _, _ = self.agent.select_action(state)
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                state = next_state
                
                if render:
                    self.env.render()
            
            eval_rewards.append(episode_reward)
        
        return eval_rewards
    
    def save_checkpoint(self, filepath: str) -> None:
        """Save training checkpoint.
        
        Args:
            filepath: Path to save checkpoint.
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        self.agent.save(filepath)
        
        # Save training metrics
        metrics_file = filepath.replace('.pth', '_metrics.npz')
        np.savez(metrics_file,
                 episode_rewards=self.episode_rewards,
                 episode_lengths=self.episode_lengths,
                 losses=self.losses,
                 training_time=self.training_time)
    
    def load_checkpoint(self, filepath: str) -> None:
        """Load training checkpoint.
        
        Args:
            filepath: Path to load checkpoint from.
        """
        self.agent.load(filepath)
        
        # Load training metrics
        metrics_file = filepath.replace('.pth', '_metrics.npz')
        if os.path.exists(metrics_file):
            metrics = np.load(metrics_file)
            self.episode_rewards = metrics['episode_rewards'].tolist()
            self.episode_lengths = metrics['episode_lengths'].tolist()
            self.losses = metrics['losses'].tolist()
            self.training_time = float(metrics['training_time'])


class Evaluator:
    """Evaluation utilities for RL agents."""
    
    def __init__(self, env_name: str, device: Optional[torch.device] = None):
        self.env_name = env_name
        self.device = device or torch.device("cpu")
        self.env = gym.make(env_name)
    
    def evaluate_agent(
        self,
        agent,
        algorithm: str,
        n_episodes: int = 100,
        seeds: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """Evaluate agent with statistical analysis.
        
        Args:
            agent: Trained agent.
            algorithm: Algorithm name.
            n_episodes: Number of episodes to evaluate.
            seeds: List of seeds for evaluation.
            
        Returns:
            Dictionary of evaluation metrics.
        """
        if seeds is None:
            seeds = [42 + i for i in range(n_episodes)]
        
        episode_rewards = []
        episode_lengths = []
        
        for i, seed in enumerate(seeds):
            set_env_seed(self.env, seed)
            state, _ = self.env.reset()
            done = False
            episode_reward = 0
            episode_length = 0
            
            while not done:
                if algorithm.lower() == 'dqn':
                    action = agent.select_action(state, training=False)
                elif algorithm.lower() == 'ppo':
                    action, _, _ = agent.select_action(state)
                else:
                    raise ValueError(f"Unknown algorithm: {algorithm}")
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                
                episode_reward += reward
                episode_length += 1
                state = next_state
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
        
        # Calculate statistics
        mean_reward, lower_ci, upper_ci = calculate_confidence_interval(np.array(episode_rewards))
        mean_length, _, _ = calculate_confidence_interval(np.array(episode_lengths))
        
        return {
            'episode_rewards': episode_rewards,
            'episode_lengths': episode_lengths,
            'mean_reward': mean_reward,
            'reward_ci': (lower_ci, upper_ci),
            'mean_length': mean_length,
            'success_rate': np.mean([r > 0 for r in episode_rewards]),
            'std_reward': np.std(episode_rewards),
            'std_length': np.std(episode_lengths),
        }
    
    def compare_algorithms(
        self,
        agents: Dict[str, Any],
        algorithms: List[str],
        n_episodes: int = 100,
    ) -> Dict[str, Any]:
        """Compare multiple algorithms.
        
        Args:
            agents: Dictionary of trained agents.
            algorithms: List of algorithm names.
            n_episodes: Number of episodes per algorithm.
            
        Returns:
            Dictionary of comparison results.
        """
        results = {}
        
        for algorithm in algorithms:
            if algorithm in agents:
                results[algorithm] = self.evaluate_agent(
                    agents[algorithm], algorithm, n_episodes
                )
        
        return results


def plot_training_curves(
    metrics: Dict[str, List],
    save_path: Optional[str] = None,
    show: bool = True,
) -> None:
    """Plot training curves.
    
    Args:
        metrics: Dictionary of training metrics.
        save_path: Path to save plot.
        show: Whether to show plot.
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Episode rewards
    axes[0, 0].plot(metrics['episode_rewards'])
    axes[0, 0].set_title('Episode Rewards')
    axes[0, 0].set_xlabel('Episode')
    axes[0, 0].set_ylabel('Reward')
    axes[0, 0].grid(True)
    
    # Episode lengths
    axes[0, 1].plot(metrics['episode_lengths'])
    axes[0, 1].set_title('Episode Lengths')
    axes[0, 1].set_xlabel('Episode')
    axes[0, 1].set_ylabel('Length')
    axes[0, 1].grid(True)
    
    # Losses
    if metrics['losses']:
        axes[1, 0].plot(metrics['losses'])
        axes[1, 0].set_title('Training Loss')
        axes[1, 0].set_xlabel('Update')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True)
    
    # Moving average rewards
    window = 100
    if len(metrics['episode_rewards']) >= window:
        moving_avg = np.convolve(metrics['episode_rewards'], np.ones(window)/window, mode='valid')
        axes[1, 1].plot(moving_avg)
        axes[1, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Average Reward')
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    if show:
        plt.show()
    
    plt.close()
