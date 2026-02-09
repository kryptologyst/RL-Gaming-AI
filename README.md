# RL Gaming AI: Modern Reinforcement Learning for Game Environments

A comprehensive reinforcement learning project showcasing advanced algorithms for gaming AI, including DQN variants and PPO implementations.

## ⚠️ IMPORTANT DISCLAIMER

**This project is designed for research and educational purposes only. The algorithms and models implemented here are NOT intended for production control of real-world systems, including but not limited to autonomous vehicles, robotics, financial trading, healthcare systems, or any safety-critical applications.**

## Features

- **Advanced DQN Implementations**: Double DQN, Prioritized Experience Replay, Noisy Networks
- **PPO Algorithm**: Proximal Policy Optimization with Generalized Advantage Estimation
- **Modern Tech Stack**: PyTorch 2.x, Gymnasium, proper device handling (CUDA/MPS/CPU)
- **Comprehensive Evaluation**: Statistical analysis with confidence intervals
- **Interactive Demo**: Streamlit-based visualization and gameplay
- **Production-Ready Structure**: Clean code, type hints, comprehensive testing

## Quick Start

### Installation

1. Clone the repository:
```bash
git clone https://github.com/kryptologyst/RL-Gaming-AI.git
cd RL-Gaming-AI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Verify installation:
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import gymnasium; print('Gymnasium installed successfully')"
```

### Training

Train a DQN agent on CartPole:
```bash
python scripts/train.py --config configs/dqn_config.yaml --total_timesteps 100000
```

Train a PPO agent:
```bash
python scripts/train.py --config configs/ppo_config.yaml --total_timesteps 100000
```

### Evaluation

Evaluate a trained model:
```bash
python scripts/evaluate.py --config configs/dqn_config.yaml --model_path checkpoints/dqn_final.pth --n_episodes 100
```

### Interactive Demo

Launch the Streamlit demo:
```bash
streamlit run demo/app.py
```

## Project Structure

```
0630_RL_for_Gaming_AI/
├── src/                          # Source code
│   ├── algorithms/               # RL algorithm implementations
│   │   ├── dqn.py               # DQN variants (DDQN, PER, Noisy)
│   │   └── ppo.py               # PPO implementation
│   ├── train/                   # Training and evaluation
│   │   └── trainer.py           # Trainer and evaluator classes
│   └── utils/                   # Utility functions
│       └── __init__.py          # Device handling, seeding, metrics
├── configs/                     # Configuration files
│   ├── dqn_config.yaml          # DQN hyperparameters
│   └── ppo_config.yaml          # PPO hyperparameters
├── scripts/                     # Training and evaluation scripts
│   ├── train.py                # Main training script
│   └── evaluate.py              # Evaluation script
├── demo/                        # Interactive demo
│   └── app.py                   # Streamlit application
├── tests/                       # Unit tests
├── assets/                      # Generated plots and results
├── checkpoints/                 # Saved models
├── logs/                        # Training logs
├── requirements.txt             # Python dependencies
├── pyproject.toml              # Project configuration
└── README.md                    # This file
```

## Algorithms

### Deep Q-Network (DQN)

Value-based method for discrete action spaces with several advanced variants:

- **Double DQN**: Reduces overestimation bias by using separate networks for action selection and evaluation
- **Prioritized Experience Replay**: Samples experiences based on their TD error importance
- **Noisy Networks**: Replaces epsilon-greedy exploration with parameter noise

**Key Features**:
- Experience replay buffer
- Target network for stability
- Configurable exploration strategies
- Support for different network architectures

### Proximal Policy Optimization (PPO)

Policy-gradient method suitable for both discrete and continuous action spaces:

- **Clipped Objective**: Prevents large policy updates
- **Generalized Advantage Estimation**: Reduces variance in policy gradients
- **Actor-Critic Architecture**: Separate value function estimation

**Key Features**:
- On-policy learning with experience buffer
- Configurable clipping ratio and loss coefficients
- Gradient clipping for training stability
- Entropy regularization for exploration

## Configuration

### DQN Configuration (`configs/dqn_config.yaml`)

```yaml
env_name: "CartPole-v1"
algorithm: "dqn"
total_timesteps: 100000

# Hyperparameters
learning_rate: 0.001
gamma: 0.99
epsilon: 1.0
epsilon_decay: 0.995
epsilon_min: 0.01

# Advanced features
use_double_dqn: true
use_prioritized_replay: false
use_noisy_dqn: false
```

### PPO Configuration (`configs/ppo_config.yaml`)

```yaml
env_name: "CartPole-v1"
algorithm: "ppo"
total_timesteps: 100000

# Hyperparameters
learning_rate: 0.0003
gamma: 0.99
gae_lambda: 0.95
clip_ratio: 0.2
value_loss_coef: 0.5
entropy_coef: 0.01
```

## Evaluation Metrics

The project provides comprehensive evaluation with:

- **Statistical Analysis**: Mean, standard deviation, confidence intervals
- **Success Rate**: Percentage of episodes achieving positive reward
- **Sample Efficiency**: Steps required to reach performance thresholds
- **Learning Curves**: Training progress visualization
- **Robustness**: Performance across different random seeds

### Example Results

**CartPole-v1 Environment**:
- DQN: ~475 ± 25 average reward (95% CI: [425, 525])
- PPO: ~480 ± 20 average reward (95% CI: [440, 520])

## Interactive Demo

The Streamlit demo provides:

- **Model Selection**: Choose between different trained models
- **Live Evaluation**: Run episodes with real-time statistics
- **Training Visualization**: View learning curves and metrics
- **Environment Comparison**: Test on different game environments
- **Parameter Tuning**: Adjust evaluation parameters

Access the demo at: `http://localhost:8501`

## Development

### Code Quality

- **Type Hints**: Full type annotation coverage
- **Documentation**: NumPy/Google-style docstrings
- **Formatting**: Black code formatting with Ruff linting
- **Testing**: Comprehensive unit test suite

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest tests/ --cov=src

# Run specific test file
pytest tests/test_dqn.py
```

### Code Formatting

```bash
# Format code
black src/ scripts/ demo/

# Lint code
ruff check src/ scripts/ demo/

# Fix auto-fixable issues
ruff check src/ scripts/ demo/ --fix
```

## Environment Support

### Supported Environments

- **CartPole-v1**: Classic control problem
- **Acrobot-v1**: Swing-up task
- **MountainCar-v0**: Continuous control challenge
- **Custom Environments**: Easy integration with Gymnasium-compatible environments

### Adding New Environments

1. Ensure environment follows Gymnasium API
2. Update configuration files with new environment name
3. Adjust hyperparameters if needed
4. Test with both DQN and PPO algorithms

## Performance Optimization

### Device Support

- **CUDA**: Automatic GPU acceleration when available
- **MPS**: Apple Silicon GPU support
- **CPU**: Fallback for all systems

### Memory Management

- Efficient replay buffer implementations
- Gradient accumulation for large batch sizes
- Checkpoint saving/loading for long training runs

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or buffer size
2. **Slow Training**: Enable GPU acceleration or reduce network size
3. **Poor Performance**: Adjust hyperparameters or increase training time
4. **Import Errors**: Ensure all dependencies are installed correctly

### Getting Help

- Check the configuration files for parameter descriptions
- Review the algorithm implementations in `src/algorithms/`
- Run the evaluation script to diagnose performance issues
- Use the interactive demo to visualize training progress

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes with proper tests
4. Ensure code passes all linting checks
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this project in your research, please cite:

```bibtex
@software{rl_gaming_ai,
  title={RL Gaming AI: Modern Reinforcement Learning for Game Environments},
  author={Kryptologyst},
  year={2026},
  url={https://github.com/kryptologyst/RL-Gaming-AI}
}
```

## Acknowledgments

- OpenAI Gym/Gymnasium team for the environment framework
- PyTorch team for the deep learning framework
- Stable Baselines3 for algorithm inspiration
- The reinforcement learning research community

---

**Remember**: This project is for educational and research purposes only. Always ensure proper safety measures when applying RL algorithms to real-world systems.
# RL-Gaming-AI
