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
from typing import Dict, Any, List
import gymnasium as gym
import torch
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra
from stable_baselines3 import PPO
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy
from rl_research.callbacks import QuickStartCallback

def record_video_episodes(
    model: PPO,
    env: gym.Env,
    num_episodes: int = 3,
    max_steps: int = 1000,
) -> List[np.ndarray]:
    """Record video of evaluation episodes.
    
    Args:
        model: Trained model to evaluate
        env: Environment to evaluate in
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        
    Returns:
        List of episode frames as numpy arrays
    """
    # Create a new environment with rgb_array rendering
    env_id = env.unwrapped.spec.id if env.unwrapped.spec is not None else "CartPole-v1"
    eval_env = gym.make(env_id, render_mode="rgb_array")
    
    all_episode_frames = []
    for episode in range(num_episodes):
        frames = []
        obs, info = eval_env.reset()
        done = False
        truncated = False
        step = 0
        episode_reward = 0.0  # Initialize as float
        
        while not (done or truncated) and step < max_steps:
            # Get action from model
            action, _ = model.predict(obs, deterministic=True)
            
            # Execute action
            obs, reward, done, truncated, info = eval_env.step(action)
            episode_reward = float(episode_reward) + float(reward)  # Explicit float conversion
            
            # Capture frame
            frame = eval_env.render()
            # Ensure frame is in correct format (H, W, C) and uint8
            if isinstance(frame, np.ndarray):
                if frame.dtype != np.uint8:
                    if frame.max() <= 1.0:
                        frame = np.array(frame * 255, dtype=np.uint8)
                    else:
                        frame = np.array(frame, dtype=np.uint8)
            elif isinstance(frame, list):
                frame = np.array(frame, dtype=np.uint8)
            frames.append(frame)
            step += 1
        
        if frames:
            # Stack frames for this episode [T, H, W, C]
            episode_frames = np.stack(frames)
            print(f"\nEpisode {episode + 1}:")
            print(f"- Total reward: {episode_reward}")
            print(f"- Number of frames: {len(frames)}")
            print(f"- Frame shape: {frames[0].shape}")
            print(f"- Array shape: {episode_frames.shape}")
            print(f"- Data type: {episode_frames.dtype}")
            
            all_episode_frames.append(episode_frames)
    
    eval_env.close()
    return all_episode_frames

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
    
    # =============================================
    # WandB Setup
    # =============================================
    """
    WandB provides:
    1. Experiment tracking
    2. Metric visualization
    3. Hyperparameter logging
    4. Model artifact storage
    
    You can view your results at: https://wandb.ai/your-username
    """
    run = wandb.init(
        project="rl-quickstart",
        config={
            "algorithm": "PPO",
            "environment": "CartPole-v1",
            "total_timesteps": 50000,
        },
        # Set to True if you haven't set up WandB yet
        mode="disabled" if os.getenv("WANDB_DISABLED") else "online"
    )
    
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
        env = gym.make("CartPole-v1")
        print("\nEnvironment Info:")
        print("=" * 50)
        print(f"Observation Space: {env.observation_space}")
        print(f"Action Space: {env.action_space}")
        
        # =============================================
        # Algorithm Setup
        # =============================================
        """
        Stable-Baselines3 provides:
        1. Implementations of popular RL algorithms
        2. Pre-built neural network policies
        3. Training and evaluation utilities
        """
        model = PPO(
            policy="MlpPolicy",
            env=env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            device="cuda" if torch.cuda.is_available() else "cpu",
            verbose=1
        )
        
        # =============================================
        # Training Loop
        # =============================================
        print("\nStarting Training:")
        print("=" * 50)
        
        # Create callback for tracking
        callback = QuickStartCallback()
        
        # Train the agent
        model.learn(
            total_timesteps=50000,
            callback=callback,
            progress_bar=True
        )
        
        # Save the trained model
        model.save("quickstart_model")
        wandb.save("quickstart_model.zip")
        
        # =============================================
        # Evaluation and Visualization
        # =============================================
        """
        The framework provides visualization utilities for:
        1. Learning curves
        2. Policy evaluation
        3. Environment renders
        """
        # Plot learning curve
        plt.figure(figsize=(10, 6))
        sns.lineplot(
            x=range(len(callback.episode_rewards)),
            y=callback.episode_rewards,
            label="Episode Reward"
        )
        plt.title("Training Progress")
        plt.xlabel("Episode")
        plt.ylabel("Reward")
        plt.savefig("learning_curve.png")
        wandb.log({"learning_curve": wandb.Image("learning_curve.png")})
        
        # Record and log evaluation episodes
        print("\nRecording evaluation episodes...")
        eval_episodes = record_video_episodes(model, env, num_episodes=3)
        
        # Log videos to WandB
        print("\nLogging evaluation videos...")
        for i, frames in enumerate(eval_episodes):
            # Ensure frames are in uint8 format
            if frames.dtype != np.uint8:
                if frames.max() <= 1.0:
                    frames = (frames * 255).astype(np.uint8)
                else:
                    frames = frames.astype(np.uint8)
            
            # Log video with metadata
            wandb.log({
                f"eval_episode_{i+1}": wandb.Video(
                    frames,
                    fps=30,
                    format="mp4",
                    caption=f"Evaluation Episode {i+1} - {len(frames)} frames"
                ),
                f"eval_episode_{i+1}_length": len(frames)
            })
        
        # Also log some evaluation metrics
        eval_env = gym.make("CartPole-v1")
        mean_reward, std_reward = evaluate_policy(
            model,
            eval_env,
            n_eval_episodes=10,
            deterministic=True
        )
        wandb.log({
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward
        })
        eval_env.close()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Always cleanup
        env.close()
        if run is not None:
            run.finish()

if __name__ == "__main__":
    """
    Run this script with:
    ```bash
    # Basic run
    python examples/quickstart.py
    
    # Disable WandB logging
    WANDB_DISABLED=true python examples/quickstart.py
    
    # Run on CPU only
    CUDA_VISIBLE_DEVICES="" python examples/quickstart.py
    ```
    """
    main() 