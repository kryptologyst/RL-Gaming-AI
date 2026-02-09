#!/usr/bin/env python3
"""Quick start script for RL Gaming AI project."""

import os
import sys
import subprocess
from pathlib import Path

def main():
    """Main quick start function."""
    print("🎮 RL Gaming AI - Quick Start")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not Path("src").exists() or not Path("configs").exists():
        print("❌ Error: Please run this script from the project root directory")
        print("   The project root should contain 'src/' and 'configs/' directories")
        return
    
    print("✅ Project structure detected")
    
    # Check Python version
    if sys.version_info < (3, 10):
        print("❌ Error: Python 3.10+ is required")
        print(f"   Current version: {sys.version}")
        return
    
    print(f"✅ Python version: {sys.version.split()[0]}")
    
    # Check if requirements are installed
    try:
        import torch
        import gymnasium
        import streamlit
        print("✅ Core dependencies installed")
    except ImportError as e:
        print(f"❌ Missing dependency: {e}")
        print("   Please run: pip install -r requirements.txt")
        return
    
    print(f"✅ PyTorch version: {torch.__version__}")
    
    # Create necessary directories
    os.makedirs("checkpoints", exist_ok=True)
    os.makedirs("assets", exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    print("✅ Directories created")
    
    # Show available commands
    print("\n🚀 Available Commands:")
    print("1. Train DQN agent:")
    print("   python scripts/train.py --config configs/dqn_config.yaml --total_timesteps 10000")
    print()
    print("2. Train PPO agent:")
    print("   python scripts/train.py --config configs/ppo_config.yaml --total_timesteps 10000")
    print()
    print("3. Evaluate trained model:")
    print("   python scripts/evaluate.py --config configs/dqn_config.yaml --model_path checkpoints/dqn_final.pth")
    print()
    print("4. Launch interactive demo:")
    print("   streamlit run demo/app.py")
    print()
    print("5. Run tests:")
    print("   pytest tests/")
    print()
    print("6. Format code:")
    print("   black src/ scripts/ demo/")
    print("   ruff check src/ scripts/ demo/")
    
    # Ask if user wants to run a quick demo
    print("\n" + "=" * 50)
    response = input("Would you like to run a quick training demo? (y/n): ").lower().strip()
    
    if response in ['y', 'yes']:
        print("\n🏃 Running quick DQN training demo...")
        try:
            # Run a short training session
            cmd = [
                sys.executable, "scripts/train.py",
                "--config", "configs/dqn_config.yaml",
                "--total_timesteps", "5000",
                "--eval_freq", "2500",
                "--save_freq", "5000"
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ Training demo completed successfully!")
                print("\n📊 Training output:")
                print(result.stdout)
            else:
                print("❌ Training demo failed:")
                print(result.stderr)
                
        except subprocess.TimeoutExpired:
            print("⏰ Training demo timed out (this is normal for longer training)")
        except Exception as e:
            print(f"❌ Error running demo: {e}")
    
    print("\n🎯 Next Steps:")
    print("1. Explore the code in src/ directory")
    print("2. Modify configs/ for different hyperparameters")
    print("3. Try different environments in the configs")
    print("4. Run the interactive demo: streamlit run demo/app.py")
    print("5. Check out the Jupyter notebook: notebooks/quick_start_tutorial.ipynb")
    
    print("\n⚠️  Remember: This project is for research and educational purposes only.")
    print("   Not for production control of real systems.")
    
    print("\n🎉 Happy learning!")

if __name__ == "__main__":
    main()
