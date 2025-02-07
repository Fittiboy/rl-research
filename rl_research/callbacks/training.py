"""Training callbacks for reinforcement learning experiments."""

from typing import List
import numpy as np
import wandb
from stable_baselines3.common.callbacks import BaseCallback


class QuickStartCallback(BaseCallback):
    """Custom callback for tracking training progress.
    
    This demonstrates how to:
    1. Track custom metrics
    2. Log to WandB
    3. Save checkpoints
    """
    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self._current_episode_reward: float = 0.0
    
    def _on_step(self) -> bool:
        """Called after each step in the environment."""
        # Accumulate reward for the current episode
        rewards = self.locals.get("rewards")
        reward = rewards[0] if rewards and len(rewards) > 0 else 0
        self._current_episode_reward += reward
        
        # When episode ends, log the total reward
        dones = self.locals.get("dones") 
        if dones and len(dones) > 0 and dones[0]:
            self.episode_rewards.append(self._current_episode_reward)
            self.episode_lengths.append(self.n_calls - (len(self.episode_lengths) * self.n_calls))
            
            # Log to WandB
            if len(self.episode_rewards) % 10 == 0:  # Log every 10 episodes
                wandb.log({
                    "episode_reward": np.mean(self.episode_rewards[-10:]),
                    "episode_length": np.mean(self.episode_lengths[-10:]),
                    "total_timesteps": self.num_timesteps,
                })
            
            # Reset episode reward accumulator
            self._current_episode_reward = 0.0
        
        return True 