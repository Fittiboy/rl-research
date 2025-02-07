"""Logging utilities for experiment tracking."""
import os
from typing import List, Optional
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
from omegaconf import DictConfig

class ExperimentLogger:
    """Logger for tracking experiments with WandB integration."""
    
    def __init__(self, config: DictConfig, env: gym.Env):
        """Initialize logger with configuration."""
        self.config = config
        self.env = env
        self._setup_wandb()
        
    def _setup_wandb(self):
        """Initialize WandB run."""
        self.run = wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            config=self.config,
            tags=self.config.wandb.tags,
            monitor_gym=True,
        )
    
    def get_callbacks(self) -> List[BaseCallback]:
        """Get list of callbacks for training."""
        callbacks = []
        
        # WandB callback for logging
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"models/{wandb.run.id}",
            verbose=2,
        )
        callbacks.append(wandb_callback)
        
        # Evaluation callback
        if hasattr(self.config.experiment, "eval_frequency"):
            eval_env = Monitor(gym.make(self.config.env.id))
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=f"models/{wandb.run.id}/best_model",
                log_path=f"logs/{wandb.run.id}",
                eval_freq=self.config.experiment.eval_frequency,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        return callbacks
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to WandB."""
        wandb.log(metrics, step=step)
    
    def save_model(self, model, name: str = "final_model"):
        """Save model with WandB logging."""
        path = f"models/{wandb.run.id}/{name}"
        model.save(path)
        wandb.save(path)
    
    def finish(self):
        """Clean up logging."""
        if self.run is not None:
            self.run.finish()

def setup_logging(config: DictConfig, env: gym.Env) -> ExperimentLogger:
    """Initialize logging for experiment."""
    return ExperimentLogger(config, env) 