"""
Quick Start Example for RL Research Framework
===========================================

This example demonstrates the core functionality of the RL research framework,
including:
1. Environment setup and configuration
2. Algorithm configuration and training
3. Experiment tracking with Weights & Biases (WandB)
4. Hyperparameter management with Hydra
5. Visualization and analysis

The framework integrates several key libraries:
- Gymnasium: Standard interface for RL environments
- Stable-Baselines3: Implementation of RL algorithms
- WandB: Experiment tracking and visualization
- Hydra: Configuration management
- PyTorch: Deep learning backend
"""

import os
from typing import Dict, Any
import gymnasium as gym
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common.monitor import Monitor
from rl_research.utils.logger import setup_logging

# =============================================
# Configuration Setup
# =============================================
@hydra.main(version_base=None, config_path="../rl_research/experiments/configs/examples", config_name="quickstart")
def main(cfg: DictConfig) -> None:
    """Main training loop with configuration management.
    
    Hydra automatically creates a new working directory for each run,
    managing experiments and their outputs.
    """
    # Print current configuration
    print("\nConfiguration:")
    print("=" * 50)
    print(OmegaConf.to_yaml(cfg))
    
    try:
        # =============================================
        # Environment Setup
        # =============================================
        """
        Gymnasium provides:
        1. Standard RL environment interface
        2. Various pre-built environments
        3. Environment wrappers for customization
        """
        # Create environment with config parameters
        env_kwargs = dict(cfg.env.params)
        env = gym.make(cfg.env.id, **env_kwargs)
        print("\nEnvironment Info:")
        print("=" * 50)
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        # =============================================
        # Logging Setup
        # =============================================
        """
        The framework provides:
        1. Experiment tracking with WandB
        2. Video recording and logging
        3. Training metrics and visualization
        4. Model checkpointing
        """
        logger = setup_logging(cfg, env)
        
        # =============================================
        # Algorithm Setup
        # =============================================
        """
        Stable-Baselines3 provides:
        1. Implementations of popular RL algorithms
        2. Pre-built neural network policies
        3. Training and evaluation utilities
        """
        # Determine device based on policy type
        # Use CPU for MLP policies (better performance) and GPU for CNN policies
        policy_type = cfg.algorithm.policy_type
        device = "cpu"  # MLP policies run faster on CPU
        if torch.cuda.is_available() and "CnnPolicy" in policy_type:
            device = "cuda"
            print("\nUsing GPU for CNN policy")
        else:
            print("\nUsing CPU for MLP policy (recommended)")
        
        # Create model with config parameters
        model = PPO(
            policy=policy_type,
            env=env,
            learning_rate=float(cfg.algorithm.params.learning_rate),
            n_steps=int(cfg.algorithm.params.n_steps),
            batch_size=int(cfg.algorithm.params.batch_size),
            n_epochs=int(cfg.algorithm.params.n_epochs),
            gamma=float(cfg.algorithm.params.gamma),
            device=device,
            verbose=1
        )
        
        # =============================================
        # Training Loop
        # =============================================
        print("\nStarting Training:")
        print("=" * 50)
        
        # Train the agent with proper callbacks
        model.learn(
            total_timesteps=cfg.experiment.total_timesteps,
            callback=logger.get_callbacks(),
            progress_bar=True
        )
        
        # Save the final model
        logger.save_model(model, "final_model")
        
        # =============================================
        # Final Evaluation
        # =============================================
        print("\nFinal Score:", logger.get_final_score())
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Always cleanup
        env.close()
        logger.finish()

if __name__ == "__main__":
    """
    Run this script with:
    ```bash
    # Basic run
    python examples/quickstart.py
    
    # Disable WandB logging
    WANDB_MODE=disabled python examples/quickstart.py
    
    # Run on CPU only
    CUDA_VISIBLE_DEVICES="" python examples/quickstart.py
    ```
    """
    main() 