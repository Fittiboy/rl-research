"""Logging utilities for experiment tracking."""
import os
from typing import List, Optional, Dict, Any, Union
import wandb
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from wandb.integration.sb3 import WandbCallback
import gymnasium as gym
from omegaconf import DictConfig, OmegaConf
import numpy as np
from .viz import record_video_episodes

class VideoEvalCallback(EvalCallback):
    """Evaluation callback with video recording."""
    
    def __init__(
        self,
        eval_env: gym.Env,
        callback_on_new_best: Optional[BaseCallback] = None,
        callback_after_eval: Optional[BaseCallback] = None,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[str] = None,
        best_model_save_path: Optional[str] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        warn: bool = True,
        # Video recording params
        video_enabled: bool = True,  # Whether to record videos at all
        record_video_freq: int = 1,  # Record every N evaluations
        num_video_episodes: int = 3,
        video_fps: int = 30,
        save_local: bool = False,  # Whether to save videos locally
        video_dir: str = "videos",  # Directory for local videos
        video_format: str = "mp4",  # Format for local videos
        video_prefix: str = "",  # Prefix for video filenames
        wandb_enabled: bool = True,  # Whether to log videos to WandB
    ):
        """Initialize callback.
        
        Args:
            video_enabled: Whether to record videos at all
            record_video_freq: Record video every N evaluations (1 = every time)
            num_video_episodes: Number of episodes to record
            video_fps: FPS for video recording
            save_local: Whether to save videos locally
            video_dir: Directory for local videos
            video_format: Format for local videos (mp4, avi)
            video_prefix: Prefix for video filenames
            wandb_enabled: Whether to log videos to WandB
            
            Other args: Same as EvalCallback
        """
        super().__init__(
            eval_env=eval_env,
            callback_on_new_best=callback_on_new_best,
            callback_after_eval=callback_after_eval,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            log_path=log_path,
            best_model_save_path=best_model_save_path,
            deterministic=deterministic,
            render=render,
            verbose=verbose,
            warn=warn,
        )
        self.video_enabled = video_enabled
        self.record_video_freq = record_video_freq
        self.num_video_episodes = num_video_episodes
        self.video_fps = video_fps
        self.save_local = save_local
        self.video_dir = video_dir
        self.video_format = video_format
        self.video_prefix = video_prefix
        self.wandb_enabled = wandb_enabled
        self.eval_idx = 0
    
    def _on_step(self) -> bool:
        """Called after each step in training."""
        continue_training = super()._on_step()
        
        # Check if we just did an evaluation
        if self.num_timesteps % self.eval_freq == 0:
            print(f"\nEvaluation at timestep {self.num_timesteps}")
            # Check if we should record video
            if self.video_enabled and self.eval_idx % self.record_video_freq == 0:
                print(f"Recording evaluation videos...")
                # Get the actual environment ID
                if hasattr(self.eval_env, "envs"):
                    # Vectorized environment
                    env = self.eval_env.envs[0]
                else:
                    # Regular environment
                    env = self.eval_env
                env_id = env.unwrapped.spec.id if hasattr(env.unwrapped, "spec") else "CartPole-v1"
                
                # Record and log video episodes
                episode_frames, episode_rewards = record_video_episodes(
                    model=self.model,
                    env_id=env_id,
                    num_episodes=self.num_video_episodes,
                    deterministic=self.deterministic,
                    render_fps=self.video_fps,
                    save_local=self.save_local,
                    output_dir=self.video_dir,
                    video_format=self.video_format,
                    prefix=self.video_prefix,
                    timestep=self.num_timesteps,
                )
                
                print(f"Recorded {len(episode_frames)} episodes")
                for i, (frames, reward) in enumerate(zip(episode_frames, episode_rewards)):
                    print(f"Episode {i+1}: {len(frames)} frames, reward: {reward:.2f}")
                    
                    # Log to WandB if enabled
                    if self.wandb_enabled:
                        try:
                            # Ensure frames are in the correct format for WandB
                            if not isinstance(frames, np.ndarray):
                                frames = np.array(frames)
                            
                            # Ensure frames are uint8 and in range [0, 255]
                            if frames.dtype != np.uint8:
                                if frames.max() <= 1.0:
                                    frames = (frames * 255).astype(np.uint8)
                                else:
                                    frames = frames.astype(np.uint8)
                            
                            # Ensure shape is (T, H, W, C)
                            if frames.ndim != 4:
                                print(f"Warning: Unexpected frame dimensions: {frames.shape}")
                                continue
                            
                            # Transpose from (T, H, W, C) to (T, C, H, W) for WandB
                            frames = np.transpose(frames, (0, 3, 1, 2))
                            print(f"Transposed frames for WandB - Shape: {frames.shape}, dtype: {frames.dtype}")
                            
                            # Create video with explicit format
                            video = wandb.Video(
                                frames,
                                fps=self.video_fps,
                                format="mp4",
                                caption=f"Eval {self.eval_idx} - Episode {i+1}"
                            )
                            
                            # Log video and metrics
                            wandb.log({
                                f"videos/episode_{i+1}": video,
                                f"videos/episode_{i+1}_reward": reward,
                                f"videos/episode_{i+1}_length": len(frames)
                            }, step=self.num_timesteps)
                            print(f"Successfully logged video {i+1} to WandB")
                        except Exception as e:
                            print(f"Error logging video to WandB: {str(e)}")
                            print(f"Frame info - Shape: {frames.shape}, dtype: {frames.dtype}, "
                                  f"min: {frames.min()}, max: {frames.max()}")
            
            self.eval_idx += 1
        
        return continue_training

class ExperimentLogger:
    """Logger for tracking experiments with WandB integration."""
    
    def __init__(self, config: DictConfig, env: gym.Env):
        """Initialize logger with configuration."""
        self.config = config
        # Wrap environment with Monitor
        self.env = Monitor(env, filename=None)  # filename=None means don't save to disk
        self.run = None
        self._setup_wandb()
    
    def _setup_wandb(self) -> None:
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
        
        # Evaluation callback with video recording
        if hasattr(self.config.experiment, "eval_frequency"):
            # Create a separate environment for evaluation
            eval_env = Monitor(gym.make(self.config.env.id))
            
            # Get video recording config
            video_config = self.config.get("video", {}) if hasattr(self.config, "get") else {}
            record_video_freq = video_config.get("record_freq", 1)  # Record every evaluation by default
            num_video_episodes = video_config.get("num_episodes", 3)
            video_fps = video_config.get("fps", 30)
            
            eval_callback = VideoEvalCallback(
                eval_env,
                best_model_save_path=os.path.join(save_dir, "models", "best"),
                log_path=os.path.join(save_dir, "logs"),
                eval_freq=self.config.experiment.eval_frequency,
                deterministic=True,
                render=False,
                record_video_freq=record_video_freq,
                num_video_episodes=num_video_episodes,
                video_fps=video_fps,
                save_local=video_config.get("save_local", False),
                video_dir=video_config.get("dir", "videos"),
                video_enabled=video_config.get("enabled", True),
                video_format=video_config.get("format", "mp4"),
                video_prefix=video_config.get("prefix", ""),
                wandb_enabled=video_config.get("wandb_enabled", True),
            )
            callbacks.append(eval_callback)
        
        # Add custom callback for episode logging
        episode_callback = EpisodeLoggingCallback(self)
        callbacks.append(episode_callback)
        
        return callbacks
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB."""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def save_model(self, model: Any, name: str = "final_model") -> None:
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
    
    def finish(self) -> None:
        """Clean up logging."""
        if self.run is not None:
            self.run.finish()
            self.run = None

class EpisodeLoggingCallback(BaseCallback):
    """Callback for logging episode metrics."""
    
    def __init__(self, logger: ExperimentLogger, verbose: int = 0):
        """Initialize callback with logger."""
        super().__init__(verbose)
        self._logger = logger
        self._episode_count = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
    
    def _on_step(self) -> bool:
        """Called after each step in the environment."""
        # Get episode info from monitor
        for info in self.locals["infos"]:
            if "episode" in info:
                self._episode_count += 1
                episode_info = info["episode"]
                self._episode_rewards.append(episode_info["r"])
                self._episode_lengths.append(episode_info["l"])
                
                # Calculate statistics over last 100 episodes
                window = 100
                recent_rewards = self._episode_rewards[-window:]
                recent_lengths = self._episode_lengths[-window:]
                
                # Log metrics
                self._logger.log_metrics({
                    "train/episode_reward": episode_info["r"],
                    "train/episode_length": episode_info["l"],
                    "train/episode_reward_mean": sum(recent_rewards) / len(recent_rewards),
                    "train/episode_reward_std": float(np.std(recent_rewards)),
                    "train/episode_length_mean": sum(recent_lengths) / len(recent_lengths),
                    "train/episode_count": self._episode_count,
                }, step=self.num_timesteps)
        
        return True

def setup_logging(config: DictConfig, env: gym.Env) -> ExperimentLogger:
    """Initialize logging for experiment."""
    return ExperimentLogger(config, env) 