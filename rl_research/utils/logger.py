"""Logging utilities for experiment tracking."""
import os
from typing import List, Optional
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf

class ExperimentLogger:
    """Logger for tracking experiments with WandB integration."""
    
    def __init__(self, config: DictConfig, env: gym.Env):
        """Initialize logger with configuration."""
        self.config = config
        self.env = env
        self.run = None
        self._setup_wandb()
    
    def _setup_wandb(self):
        """Initialize WandB run."""
        # Convert OmegaConf to dict for WandB
        wandb_config = OmegaConf.to_container(self.config, resolve=True)
        
        # Get the save directory from config or use default
        save_dir = self.config.wandb.get("dir", ".")
        
        self.run = wandb.init(
            project=self.config.wandb.project,
            group=self.config.wandb.group,
            config=wandb_config,
            tags=self.config.wandb.tags,
            dir=save_dir,
            mode=self.config.wandb.get("mode", "online"),
            monitor_gym=True,
        )
    
    def get_callbacks(self) -> List[BaseCallback]:
        """Get list of callbacks for training."""
        callbacks = []
        
        # Get the save directory
        save_dir = self.config.wandb.get("dir", ".")
        
        # WandB callback for logging
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=os.path.join(save_dir, "models"),
            verbose=2,
        )
        callbacks.append(wandb_callback)
        
        # Evaluation callback
        if hasattr(self.config.experiment, "eval_frequency"):
            eval_env = Monitor(gym.make(self.config.env.id))
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "models", "best"),
                log_path=os.path.join(save_dir, "logs"),
                eval_freq=self.config.experiment.eval_frequency,
                deterministic=True,
                render=False,
            )
            callbacks.append(eval_callback)
        
        return callbacks
    
    def log_metrics(self, metrics: dict, step: Optional[int] = None):
        """Log metrics to WandB."""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def save_model(self, model, name: str = "final_model"):
        """Save model with WandB logging."""
        if self.run is not None:
            # Get the save directory
            save_dir = self.config.wandb.get("dir", ".")
            
            # Create the models directory if it doesn't exist
            model_dir = os.path.join(save_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            # Save the model
            path = os.path.join(model_dir, name)
            model.save(path)
            
            # Log to WandB if not in disabled mode
            if self.config.wandb.get("mode", "online") != "disabled":
                wandb.save(path + ".zip")
    
    def finish(self):
        """Clean up logging."""
        if self.run is not None:
            self.run.finish()
            self.run = None

def setup_logging(config: DictConfig, env: gym.Env) -> ExperimentLogger:
    """Initialize logging for experiment."""
    return ExperimentLogger(config, env) 