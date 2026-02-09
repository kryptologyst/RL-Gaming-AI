"""Streamlit demo for RL Gaming AI."""

import os
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import yaml
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

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
            epsilon=0.0,  # No exploration during demo
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


def plot_training_metrics(metrics_file: str):
    """Plot training metrics from saved file."""
    if not os.path.exists(metrics_file):
        return None
    
    data = np.load(metrics_file)
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Episode Rewards', 'Episode Lengths', 'Training Loss', 'Moving Average Rewards'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Episode rewards
    fig.add_trace(
        go.Scatter(y=data['episode_rewards'], mode='lines', name='Rewards'),
        row=1, col=1
    )
    
    # Episode lengths
    fig.add_trace(
        go.Scatter(y=data['episode_lengths'], mode='lines', name='Lengths'),
        row=1, col=2
    )
    
    # Training loss
    if len(data['losses']) > 0:
        fig.add_trace(
            go.Scatter(y=data['losses'], mode='lines', name='Loss'),
            row=2, col=1
        )
    
    # Moving average rewards
    window = 100
    if len(data['episode_rewards']) >= window:
        moving_avg = np.convolve(data['episode_rewards'], np.ones(window)/window, mode='valid')
        fig.add_trace(
            go.Scatter(y=moving_avg, mode='lines', name=f'Moving Avg (window={window})'),
            row=2, col=2
        )
    
    fig.update_layout(height=600, showlegend=False, title_text="Training Metrics")
    return fig


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="RL Gaming AI Demo",
        page_icon="🎮",
        layout="wide"
    )
    
    st.title("🎮 Reinforcement Learning Gaming AI Demo")
    st.markdown("""
    This demo showcases modern reinforcement learning algorithms for gaming AI, including 
    advanced DQN variants and PPO. The agents are trained to play classic games like CartPole.
    
    **⚠️ DISCLAIMER: This is a research/educational project. Not for production control of real systems.**
    """)
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    
    # Algorithm selection
    algorithm = st.sidebar.selectbox(
        "Algorithm",
        ["dqn", "ppo"],
        help="Choose the RL algorithm to use"
    )
    
    # Model selection
    model_dir = "checkpoints"
    if os.path.exists(model_dir):
        model_files = [f for f in os.listdir(model_dir) if f.endswith('.pth') and algorithm in f]
        if model_files:
            model_file = st.sidebar.selectbox(
                "Trained Model",
                model_files,
                help="Select a trained model to load"
            )
            model_path = os.path.join(model_dir, model_file)
        else:
            st.sidebar.error(f"No trained {algorithm.upper()} models found in {model_dir}")
            model_path = None
    else:
        st.sidebar.error(f"Model directory {model_dir} not found")
        model_path = None
    
    # Environment selection
    env_name = st.sidebar.selectbox(
        "Environment",
        ["CartPole-v1", "Acrobot-v1", "MountainCar-v0"],
        help="Choose the game environment"
    )
    
    # Evaluation parameters
    st.sidebar.header("Evaluation Parameters")
    n_episodes = st.sidebar.slider(
        "Number of Episodes",
        min_value=1,
        max_value=100,
        value=10,
        help="Number of episodes to evaluate"
    )
    
    render_mode = st.sidebar.selectbox(
        "Render Mode",
        ["rgb_array", "human"],
        help="How to render the environment"
    )
    
    # Main content
    if model_path and os.path.exists(model_path):
        # Load configuration
        config_path = f"configs/{algorithm}_config.yaml"
        if os.path.exists(config_path):
            config = load_config(config_path)
            config["env_name"] = env_name
            
            # Set device
            device = get_device()
            
            # Create agent
            try:
                agent = create_agent(config, device)
                agent.load(model_path)
                
                st.success(f"✅ Loaded {algorithm.upper()} model from {model_file}")
                
                # Evaluation section
                st.header("Agent Evaluation")
                
                if st.button("Run Evaluation", type="primary"):
                    with st.spinner("Evaluating agent..."):
                        evaluator = Evaluator(env_name, device)
                        results = evaluator.evaluate_agent(agent, algorithm, n_episodes)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            st.metric(
                                "Mean Reward",
                                f"{results['mean_reward']:.2f}",
                                f"±{results['std_reward']:.2f}"
                            )
                        
                        with col2:
                            st.metric(
                                "Success Rate",
                                f"{results['success_rate']:.1%}"
                            )
                        
                        with col3:
                            st.metric(
                                "Mean Length",
                                f"{results['mean_length']:.1f}",
                                f"±{results['std_length']:.1f}"
                            )
                        
                        with col4:
                            ci_lower, ci_upper = results['reward_ci']
                            st.metric(
                                "95% CI",
                                f"[{ci_lower:.1f}, {ci_upper:.1f}]"
                            )
                        
                        # Plot reward distribution
                        fig = px.histogram(
                            x=results['episode_rewards'],
                            nbins=20,
                            title="Episode Reward Distribution",
                            labels={'x': 'Episode Reward', 'y': 'Frequency'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                
                # Training metrics section
                st.header("Training Metrics")
                
                # Look for training metrics
                metrics_file = model_path.replace('.pth', '_metrics.npz')
                if os.path.exists(metrics_file):
                    fig = plot_training_metrics(metrics_file)
                    if fig:
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No training metrics found for this model.")
                
                # Live gameplay section
                st.header("Live Gameplay")
                
                if st.button("Start Live Demo", type="secondary"):
                    import gymnasium as gym
                    
                    # Create environment
                    env = gym.make(env_name, render_mode=render_mode)
                    
                    # Run episodes
                    episode_rewards = []
                    
                    for episode in range(5):  # Run 5 episodes for demo
                        state, _ = env.reset()
                        done = False
                        episode_reward = 0
                        step = 0
                        
                        # Create placeholder for episode info
                        episode_placeholder = st.empty()
                        
                        while not done and step < 1000:  # Max steps per episode
                            if algorithm.lower() == 'dqn':
                                action = agent.select_action(state, training=False)
                            elif algorithm.lower() == 'ppo':
                                action, _, _ = agent.select_action(state)
                            
                            next_state, reward, terminated, truncated, _ = env.step(action)
                            done = terminated or truncated
                            
                            episode_reward += reward
                            state = next_state
                            step += 1
                            
                            # Update episode info
                            episode_placeholder.text(f"Episode {episode + 1}: Step {step}, Reward: {episode_reward:.2f}")
                        
                        episode_rewards.append(episode_reward)
                        episode_placeholder.text(f"Episode {episode + 1} Complete: Total Reward = {episode_reward:.2f}")
                    
                    env.close()
                    
                    # Display results
                    st.success(f"Demo completed! Average reward: {np.mean(episode_rewards):.2f}")
                
            except Exception as e:
                st.error(f"Error loading agent: {str(e)}")
    
    else:
        st.warning("⚠️ Please train a model first using the training script.")
        st.markdown("""
        To train a model, run:
        ```bash
        python scripts/train.py --config configs/dqn_config.yaml
        ```
        or
        ```bash
        python scripts/train.py --config configs/ppo_config.yaml
        ```
        """)
    
    # Information section
    st.header("About This Demo")
    
    st.markdown("""
    ### Features
    
    - **Advanced DQN**: Double DQN, Prioritized Experience Replay, Noisy Networks
    - **PPO**: Proximal Policy Optimization with GAE
    - **Modern Stack**: PyTorch 2.x, Gymnasium, proper device handling
    - **Comprehensive Evaluation**: Confidence intervals, statistical analysis
    - **Interactive Demo**: Live gameplay and visualization
    
    ### Algorithms
    
    **DQN (Deep Q-Network)**:
    - Value-based method for discrete action spaces
    - Experience replay and target networks for stability
    - Advanced variants: Double DQN, Prioritized Replay, Noisy Networks
    
    **PPO (Proximal Policy Optimization)**:
    - Policy-gradient method for both discrete and continuous actions
    - Clipped objective for stable learning
    - Generalized Advantage Estimation (GAE)
    
    ### Safety Notice
    
    This project is designed for research and educational purposes only. 
    The algorithms and models are not intended for production control of real-world systems.
    """)
    
    # Footer
    st.markdown("---")
    st.markdown("Built with ❤️ using Streamlit, PyTorch, and Gymnasium")


if __name__ == "__main__":
    main()
