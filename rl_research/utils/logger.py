"""Logging utilities for experiment tracking."""
import os
from typing import List, Dict, Any, Optional, Union, Tuple, cast, TypeVar
import numpy as np
from numpy.typing import NDArray
import wandb
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from omegaconf import DictConfig, OmegaConf

ObsType = TypeVar("ObsType", np.ndarray, Dict[str, np.ndarray])

class VideoEvalCallback(EvalCallback):
    """Evaluation callback with video recording."""
    
    def __init__(
        self,
        eval_env: Union[gym.Env, VecEnv],
        eval_freq: int,
        video_fps: int = 30,
        n_eval_episodes: int = 2,
        deterministic: bool = True,
    ):
        """Initialize callback.
        
        Args:
            eval_env: Environment for evaluation
            eval_freq: Evaluate every N steps
            video_fps: FPS for video recording
            n_eval_episodes: Number of episodes to record
            deterministic: Whether to use deterministic actions
        """
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
        )
        self.video_fps = video_fps
        self.eval_idx = 0
    
    def _on_step(self) -> bool:
        """Record videos during evaluation."""
        continue_training = super()._on_step()
        
        if self.num_timesteps % self.eval_freq == 0:
            print(f"\nEvaluation at timestep {self.num_timesteps}")
            
            # Record episodes
            episode_frames: List[NDArray] = []
            episode_rewards: List[float] = []
            
            for episode in range(self.n_eval_episodes):
                frames: List[NDArray] = []
                total_reward = 0.0
                
                # Handle both vectorized and non-vectorized environments
                if isinstance(self.eval_env, VecEnv):
                    obs = self.eval_env.reset()
                else:
                    obs = cast(ObsType, self.eval_env.reset()[0])
                
                done = False
                
                while not done:
                    # Get action from model
                    action, _ = self.model.predict(cast(ObsType, obs), deterministic=self.deterministic)
                    
                    # Execute action
                    if isinstance(self.eval_env, VecEnv):
                        obs, reward, dones, info = self.eval_env.step(action)
                        done = dones[0]  # Only care about first env
                        reward = reward[0]
                    else:
                        step_result = self.eval_env.step(action)
                        obs, reward, terminated, truncated = step_result[:4]
                        done = terminated or truncated
                    
                    total_reward += float(reward)
                    
                    # Render and capture frame
                    frame = self.eval_env.render()
                    if frame is not None and isinstance(frame, np.ndarray):
                        frames.append(frame)
                
                if frames:
                    frames_array = np.stack(frames)
                    episode_frames.append(frames_array)
                    episode_rewards.append(total_reward)
                    print(f"Episode {episode + 1}: {len(frames)} frames, reward: {total_reward:.2f}")
            
            # Log videos to WandB
            for i, (frames, reward) in enumerate(zip(episode_frames, episode_rewards)):
                try:
                    # Ensure frames are uint8 and in range [0, 255]
                    if frames.dtype != np.uint8:
                        frames = (frames * 255).astype(np.uint8)
                    
                    # Transpose for WandB (T, H, W, C) -> (T, C, H, W)
                    frames = np.transpose(frames, (0, 3, 1, 2))
                    
                    # Create and log video
                    video = wandb.Video(
                        frames,
                        fps=self.video_fps,
                        format="mp4",
                        caption=f"Eval {self.eval_idx} - Episode {i+1}"
                    )
                    
                    wandb.log({
                        f"videos/episode_{i+1}": video,
                        f"videos/episode_{i+1}_reward": reward,
                        f"videos/episode_{i+1}_length": len(frames)
                    }, step=self.num_timesteps)
                except Exception as e:
                    print(f"Error logging video to WandB: {str(e)}")
            
            self.eval_idx += 1
        
        return continue_training

class ExperimentLogger:
    """Logger for tracking experiments with WandB integration."""
    
    def __init__(self, config: DictConfig, env: Union[gym.Env, VecEnv]):
        """Initialize logger with configuration."""
        self.config = config
        # Only wrap with Monitor if not already a VecEnv
        self.env = env if isinstance(env, VecEnv) else Monitor(env)
        self.run = None
        self._setup_wandb()
    
    def _setup_wandb(self) -> None:
        """Initialize WandB run."""
        wandb_config = OmegaConf.to_container(self.config, resolve=True)
        save_dir = self.config.wandb.get("dir", ".")
        
        # Convert config to dict and ensure it's serializable
        config_dict = dict(wandb_config) if wandb_config else None
        if isinstance(config_dict, dict):
            config_dict = {str(k): v for k, v in config_dict.items()}
        
        self.run = wandb.init(
            project=str(self.config.wandb.project),
            group=str(self.config.wandb.group),
            config=config_dict,
            tags=list(self.config.wandb.tags),
            dir=str(save_dir),
            mode=str(self.config.wandb.get("mode", "online")),
        )
    
    def get_callbacks(self) -> List[BaseCallback]:
        """Get list of callbacks for training."""
        callbacks = []
        
        # Evaluation callback with video recording
        if hasattr(self.config.experiment, "eval_frequency"):
            # Create evaluation environment with same settings as training
            env_kwargs = dict(self.config.env.params)
            env_kwargs["render_mode"] = "rgb_array"  # Override render_mode for evaluation
            
            eval_env = Monitor(gym.make(self.config.env.id, **env_kwargs))
            
            # If using multiple environments, wrap in DummyVecEnv
            if hasattr(self.config.algorithm, "n_envs") and self.config.algorithm.n_envs > 1:
                eval_env = DummyVecEnv([lambda: eval_env])
            
            eval_callback = VideoEvalCallback(
                eval_env=eval_env,
                eval_freq=self.config.experiment.eval_frequency,
                video_fps=self.config.video.fps,
                n_eval_episodes=self.config.video.num_episodes,
            )
            callbacks.append(eval_callback)
        
        # Episode logging callback
        callbacks.append(EpisodeLoggingCallback(self))
        
        return callbacks
    
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None) -> None:
        """Log metrics to WandB."""
        if self.run is not None:
            self.run.log(metrics, step=step)
    
    def save_model(self, model: Any, name: str = "final_model") -> None:
        """Save model with WandB logging."""
        if self.run is not None:
            save_dir = self.config.wandb.get("dir", ".")
            model_dir = os.path.join(save_dir, "models")
            os.makedirs(model_dir, exist_ok=True)
            
            path = os.path.join(model_dir, name)
            model.save(path)
            
            if self.config.wandb.get("mode", "online") != "disabled":
                wandb.save(path + ".zip", base_path=save_dir, policy="now")
    
    def finish(self) -> None:
        """Clean up logging."""
        if self.run is not None:
            if self.config.wandb.get("mode", "online") != "disabled":
                save_dir = self.config.wandb.get("dir", ".")
                wandb.save(
                    os.path.join(save_dir, "**", "*"),
                    base_path=save_dir,
                    policy="now"
                )
            self.run.finish()
            self.run = None

class EpisodeLoggingCallback(BaseCallback):
    """Callback for logging episode metrics."""
    
    def __init__(self, logger: ExperimentLogger):
        """Initialize callback with logger."""
        super().__init__(verbose=0)
        self._logger = logger
        self._episode_count = 0
        self._episode_rewards: List[float] = []
        self._episode_lengths: List[int] = []
    
    def _on_step(self) -> bool:
        """Log episode metrics."""
        for info in self.locals["infos"]:
            if "episode" in info:
                self._episode_count += 1
                episode_info = info["episode"]
                self._episode_rewards.append(episode_info["r"])
                self._episode_lengths.append(episode_info["l"])
                
                # Calculate statistics over last 100 episodes
                window = min(100, len(self._episode_rewards))
                recent_rewards = self._episode_rewards[-window:]
                recent_lengths = self._episode_lengths[-window:]
                
                self._logger.log_metrics({
                    "train/episode_reward": episode_info["r"],
                    "train/episode_length": episode_info["l"],
                    "train/episode_reward_mean": sum(recent_rewards) / window,
                    "train/episode_reward_std": float(np.std(recent_rewards)),
                    "train/episode_length_mean": sum(recent_lengths) / window,
                    "train/episode_count": self._episode_count,
                }, step=self.num_timesteps)
        
        return True

def setup_logging(config: DictConfig, env: Union[gym.Env, VecEnv]) -> ExperimentLogger:
    """Initialize logging for experiment."""
    return ExperimentLogger(config, env) 