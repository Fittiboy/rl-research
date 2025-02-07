"""Logging utilities for experiment tracking."""
import os
import warnings
from typing import List, Dict, Any, Optional, Union, Tuple, cast, TypeVar, Literal, Protocol, Sequence
import numpy as np
from numpy.typing import NDArray, ArrayLike
import wandb
import gymnasium as gym
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import VecEnv, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.type_aliases import GymObs, GymStepReturn
from stable_baselines3.common.preprocessing import is_image_space
from omegaconf import DictConfig, OmegaConf
from gymnasium import spaces

# Suppress warning about environment type mismatch
warnings.filterwarnings('ignore', message='Training and eval env are not of the same type')

# Type definitions
GymEnv = Union[gym.Env, VecEnv]
ObsType = TypeVar('ObsType', np.ndarray, Dict[str, np.ndarray])
VecResetReturn = Tuple[GymObs, Dict[str, Any]]
GymResetReturn = Tuple[GymObs, Dict[str, Any]]
VecStepReturn = Tuple[GymObs, np.ndarray, np.ndarray, List[Dict[str, Any]]]

class SupportsRender(Protocol):
    """Protocol for environments that support rendering."""
    def render(self) -> Optional[NDArray[np.uint8]]: ...

Frame = NDArray[np.uint8]  # Shape: (H, W, C)
FrameStack = NDArray[np.uint8]  # Shape: (T, H, W, C)
Frames = List[Frame]  # For collections of frames

class VideoEvalCallback(EvalCallback):
    """Evaluation callback with video recording."""
    
    def __init__(
        self,
        eval_env: GymEnv,
        eval_freq: int,
        video_fps: int = 30,
        n_eval_episodes: int = 2,
        deterministic: bool = True,
    ):
        """Initialize callback."""
        super().__init__(
            eval_env=eval_env,
            n_eval_episodes=n_eval_episodes,
            eval_freq=eval_freq,
            deterministic=deterministic,
        )
        self.video_fps = video_fps
        self.eval_idx = 0
        self.last_mean_reward = 0.0
        self._last_frames: Optional[NDArray[np.uint8]] = None  # Store last recorded frames for testing
    
    def _on_step(self) -> bool:
        """Record videos during evaluation."""
        continue_training = super()._on_step()
        
        if self.num_timesteps % self.eval_freq == 0:
            print(f"\nEvaluation at timestep {self.num_timesteps}")
            
            # Record episodes
            episode_frames: List[FrameStack] = []
            episode_rewards: List[float] = []
            
            for episode in range(self.n_eval_episodes):
                frames: List[Frame] = []
                total_reward = 0.0
                
                # Handle both vectorized and non-vectorized environments
                if isinstance(self.eval_env, VecEnv):
                    obs = self.eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = cast(VecResetReturn, obs)[0]  # New Gym API returns (obs, info)
                else:
                    obs = self.eval_env.reset()
                    if isinstance(obs, tuple):
                        obs = cast(GymResetReturn, obs)[0]  # New Gym API returns (obs, info)
                
                done = False
                
                while not done:
                    # Get action from model
                    # Cast observation to the correct type for predict
                    if isinstance(obs, np.ndarray):
                        obs_for_predict = obs
                    elif isinstance(obs, dict):
                        obs_for_predict = {k: v for k, v in obs.items()}
                    else:
                        raise ValueError(f"Unexpected observation type: {type(obs)}")
                    
                    action, _ = self.model.predict(obs_for_predict, deterministic=self.deterministic)
                    
                    # Execute action
                    if isinstance(self.eval_env, VecEnv):
                        next_obs, rewards, dones, infos = cast(VecStepReturn, self.eval_env.step(action))
                        obs = next_obs
                        done = bool(dones[0])  # Only care about first env
                        reward = float(rewards[0])
                    else:
                        next_obs, reward, terminated, truncated, info = cast(GymStepReturn, self.eval_env.step(action))
                        obs = next_obs
                        done = terminated or truncated
                        reward = float(reward)
                    
                    total_reward += reward
                    
                    # Render and capture frame
                    frame = self.eval_env.render()
                    if frame is not None and isinstance(frame, np.ndarray):
                        frame_array = frame.astype(np.uint8)
                        frames.append(frame_array)
                
                if frames:
                    frames_array = np.stack(frames)
                    episode_frames.append(frames_array)
                    episode_rewards.append(total_reward)
                    print(f"Episode {episode + 1}: {len(frames)} frames, reward: {total_reward:.2f}")
            
            # Store mean reward
            if episode_rewards:
                self.last_mean_reward = float(np.mean(episode_rewards))
            
            # Log videos to WandB
            for i, (frames_stack, reward) in enumerate(zip(episode_frames, episode_rewards)):
                try:
                    # Ensure frames are uint8 and in range [0, 255]
                    frames_array: NDArray[np.uint8] = frames_stack
                    if np.max(frames_array) <= 1.0:
                        frames_array = (frames_array * 255).astype(np.uint8)
                    
                    # Convert from (T, H, W, C) to (T, C, H, W) for WandB
                    # WandB expects PyTorch-style TCHW format
                    frames_array = np.transpose(frames_array, (0, 3, 1, 2))
                    
                    # Store frames for testing (in original HWC format)
                    self._last_frames = np.transpose(frames_array, (0, 2, 3, 1))
                    
                    # Create and log video
                    video = wandb.Video(
                        frames_array,  # type: ignore
                        fps=self.video_fps,
                        format="mp4",
                        caption=f"Eval {self.eval_idx} - Episode {i+1}"
                    )
                    
                    wandb.log({
                        f"videos/episode_{i+1}": video,
                        f"videos/episode_{i+1}_reward": reward,
                        f"videos/episode_{i+1}_length": len(frames_stack),
                        "eval/mean_reward": self.last_mean_reward
                    }, step=self.num_timesteps)
                except Exception as e:
                    print(f"Error logging video to WandB: {str(e)}")
            
            self.eval_idx += 1
        
        return continue_training

class ExperimentLogger:
    """Logger for tracking experiments with WandB integration."""
    
    def __init__(self, config: DictConfig, env: GymEnv):
        """Initialize logger with configuration."""
        self.config = config
        # Only wrap with Monitor if not already a VecEnv and not already monitored
        self.env = env if isinstance(env, (VecEnv, Monitor)) else Monitor(env)
        self.run = None
        self._callbacks: Optional[List[BaseCallback]] = None
        self._setup_wandb()
    
    def _setup_wandb(self) -> None:
        """Initialize WandB run."""
        wandb_config = OmegaConf.to_container(self.config, resolve=True)
        save_dir = self.config.wandb.get("dir", ".")
        
        # Convert config to dict and ensure it's serializable
        config_dict: Dict[str, Any] = {}
        if isinstance(wandb_config, dict):
            config_dict = {str(k): v for k, v in wandb_config.items()}
        
        mode = str(self.config.wandb.get("mode", "online"))
        if mode not in ("online", "offline", "disabled"):
            mode = "online"
        
        self.run = wandb.init(
            project=str(self.config.wandb.project),
            group=str(self.config.wandb.group),
            config=config_dict,
            tags=list(self.config.wandb.tags),
            dir=str(save_dir),
            mode=cast(Literal["online", "offline", "disabled"], mode),
        )
    
    def get_callbacks(self):
        """Get list of callbacks for training."""
        # Create evaluation environment
        eval_env = DummyVecEnv([lambda: Monitor(gym.make(self.config.env.id, **self.config.env.params))])
        
        # Only apply VecTransposeImage if observation space is an image
        if is_image_space(eval_env.observation_space) or isinstance(eval_env.observation_space, spaces.Dict):
            eval_env = VecTransposeImage(eval_env)
        
        # Create callbacks
        callbacks = []
        
        # Add episode logging callback
        callbacks.append(EpisodeLoggingCallback(self))
        
        # Add video evaluation callback
        callbacks.append(
            VideoEvalCallback(
                eval_env,  # Uses original HWC format
                eval_freq=self.config.experiment.eval_frequency,
                n_eval_episodes=self.config.video.num_episodes,
                video_fps=self.config.video.fps,
            )
        )
        
        return callbacks
    
    def log_metrics(self, metrics: Dict[str, Any]) -> None:
        """Log metrics to WandB."""
        wandb.log(metrics)
    
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
    
    def get_final_score(self) -> float:
        """Get the final evaluation score from the last evaluation."""
        if self._callbacks is None:
            return 0.0
        
        eval_callbacks = [cb for cb in self._callbacks if isinstance(cb, VideoEvalCallback)]
        if eval_callbacks and hasattr(eval_callbacks[0], "last_mean_reward"):
            return float(eval_callbacks[0].last_mean_reward)
        return 0.0

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
            
        # Clean up callbacks
        if self._callbacks is not None:
            for callback in self._callbacks:
                if hasattr(callback, "eval_env"):
                    callback.eval_env.close()  # type: ignore
            self._callbacks = None

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
        """Log episode metrics when an episode ends."""
        for info in self.locals.get("infos", []):
            if "episode" in info:
                episode_info = info["episode"]
                self._episode_rewards.append(episode_info["r"])
                self._episode_lengths.append(episode_info["l"])
                self._episode_count += 1
                
                # Calculate rolling statistics
                window = min(100, len(self._episode_rewards))
                recent_rewards = self._episode_rewards[-window:]
                recent_lengths = self._episode_lengths[-window:]
                
                self._logger.log_metrics({
                    "train/episode_reward": episode_info["r"],
                    "train/episode_length": episode_info["l"],
                    "train/episode_count": self._episode_count,
                    "train/mean_reward": np.mean(recent_rewards),
                    "train/mean_length": np.mean(recent_lengths),
                })
        
        return True

def setup_logging(config: DictConfig, env: GymEnv) -> ExperimentLogger:
    """Initialize logging for experiment."""
    return ExperimentLogger(config, env) 