"""Visualization utilities for experiment analysis."""
from typing import List, Optional, Union, Any, Tuple, cast, TypeVar, Protocol
import numpy as np
import numpy.typing as npt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy
import os

class SupportsShape(Protocol):
    """Protocol for objects that have shape, dtype, min, and max."""
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def dtype(self) -> np.dtype: ...
    def min(self) -> Any: ...
    def max(self) -> Any: ...

# Type definitions
RawFrame = TypeVar('RawFrame', bound=SupportsShape)
Frame = npt.NDArray[np.uint8]  # For processed frames
FrameList = List[Frame]
FrameStack = npt.NDArray[np.uint8]  # For stacked frames (T, H, W, C)

# OpenCV types
try:
    import cv2
    from typing_extensions import TypeAlias
    
    VideoWriter: TypeAlias = 'cv2.VideoWriter'
    FourCC = int  # Type for fourcc codes
except ImportError:
    cv2 = None

def process_frame(raw_frame: RawFrame) -> Frame:
    """Process a raw frame into a standardized format.
    
    Args:
        raw_frame: Raw frame data that supports shape, dtype, min, and max
    
    Returns:
        Processed frame in uint8 RGB format with shape (H, W, 3)
    """
    frame_array = np.asarray(raw_frame)
    
    # Transpose if channels are first
    if frame_array.shape[0] == 3:
        frame_array = np.transpose(frame_array, (1, 2, 0))
    
    # Convert to uint8
    if frame_array.dtype != np.uint8:
        if frame_array.max() <= 1.0:
            frame_array = (frame_array * 255).astype(np.uint8)
        else:
            frame_array = frame_array.astype(np.uint8)
    
    # Ensure 3 channels (RGB)
    if len(frame_array.shape) == 2:
        frame_array = np.stack([frame_array] * 3, axis=-1)
    elif frame_array.shape[-1] == 1:
        frame_array = np.repeat(frame_array, 3, axis=-1)
    elif frame_array.shape[-1] == 4:  # RGBA
        frame_array = frame_array[..., :3]  # Keep only RGB
    
    return cast(Frame, frame_array)

def get_video_writer(
    path: str,
    fps: int,
    size: tuple[int, int],
    fourcc: Optional[FourCC] = None
) -> Optional[VideoWriter]:
    """Create a video writer with proper type handling.
    
    Args:
        path: Output video file path
        fps: Frames per second
        size: Video dimensions (width, height)
        fourcc: FourCC code for video codec, defaults to mp4v
    
    Returns:
        VideoWriter object if cv2 is available, None otherwise
    """
    if cv2 is None:
        return None
    
    if fourcc is None:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # type: ignore
    
    return cv2.VideoWriter(path, fourcc, fps, size)  # type: ignore

def set_style():
    """Set the style for all plots."""
    sns.set_theme()  # This is the modern way to set seaborn style
    
def plot_learning_curves(
    runs: List[Any],  # wandb.sdk.wandb_run.Run
    metric: str = "rollout/ep_rew_mean",
    window: int = 100,
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot learning curves from multiple runs."""
    plt.figure(figsize=figsize)
    
    has_plots = False
    for run in runs:
        # Get history and smooth
        history = pd.DataFrame(run.history())
        if metric in history.columns:
            values = history[metric].rolling(window=window).mean()
            steps = history.get('global_step', range(len(values)))
            plt.plot(steps, values, label=f"{run.config['algorithm']['name']}")
            has_plots = True
    
    plt.title(title or f"Learning Curves - {metric}")
    plt.xlabel("Steps")
    plt.ylabel(metric)
    if has_plots:  # Only show legend if we have plots
        plt.legend()
    plt.grid(True)
    
def plot_evaluation_results(
    results: Union[List[float], np.ndarray],
    title: str = "Evaluation Results",
    figsize: tuple = (10, 6)
):
    """Plot evaluation results as a box plot."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=results)
    plt.title(title)
    plt.ylabel("Return")
    plt.grid(True)

def plot_environment_renders(
    frames: FrameList,
    rows: int = 2,
    cols: int = 3,
    figsize: tuple = (15, 10)
):
    """Plot a grid of environment renders."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, frame in enumerate(frames[:rows*cols]):
        if len(frame.shape) == 3:
            axes[idx].imshow(frame)
        else:
            axes[idx].imshow(frame, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f"Frame {idx}")
    
    plt.tight_layout()
    
def save_figure(path: str, dpi: int = 300):
    """Save the current figure."""
    plt.savefig(path, dpi=dpi, bbox_inches='tight')

def record_video_episodes(
    model: Any,
    env_id: str,
    num_episodes: int = 3,
    max_steps: int = 1000,
    deterministic: bool = True,
    render_fps: int = 30,
    save_local: bool = False,
    output_dir: str = "videos",
    video_format: str = "mp4",
    prefix: str = "",
    timestep: Optional[int] = None,
) -> Tuple[List[FrameStack], List[float]]:
    """Record video of evaluation episodes.
    
    Args:
        model: The trained model to evaluate
        env_id: Environment ID to create
        num_episodes: Number of episodes to record
        max_steps: Maximum steps per episode
        deterministic: Whether to use deterministic actions
        render_fps: FPS for video recording
        save_local: Whether to save videos locally
        output_dir: Directory to save local videos
        video_format: Format for local videos (mp4, avi)
        prefix: Prefix for video filenames
        timestep: Current timestep for unique filenames
    
    Returns:
        Tuple of (episode frames, episode rewards)
    """
    if save_local and cv2 is None:
        print("Warning: cv2 not found. Please install opencv-python to save videos locally.")
        save_local = False
    
    if save_local:
        os.makedirs(output_dir, exist_ok=True)
    
    print(f"\nCreating environment {env_id} for video recording...")
    env = gym.make(env_id, render_mode="rgb_array")
    print(f"Environment created with render_mode: {env.render_mode}")
    
    # Test render to check frame format
    obs, _ = env.reset()
    test_frame = env.render()
    if test_frame is not None:
        test_frame_array = process_frame(cast(RawFrame, test_frame))
        print(f"Initial frame shape: {test_frame_array.shape}, dtype: {test_frame_array.dtype}, "
              f"min: {test_frame_array.min()}, max: {test_frame_array.max()}")
    
    episode_frames: List[FrameStack] = []
    episode_rewards: List[float] = []
    
    try:
        for episode in range(num_episodes):
            print(f"\nRecording episode {episode + 1}/{num_episodes}")
            frames: FrameList = []
            total_reward = 0.0
            obs, _ = env.reset()
            done = False
            truncated = False
            step = 0
            
            while not (done or truncated) and step < max_steps:
                # Get action from model
                action, _ = model.predict(obs, deterministic=deterministic)
                
                # Execute action
                obs, reward, done, truncated, _ = env.step(action)
                total_reward += float(reward)
                
                # Render and capture frame
                frame = env.render()
                if frame is None:
                    print("Warning: Environment returned None for render!")
                    continue
                
                # Process frame into standardized format
                frame_array = process_frame(cast(RawFrame, frame))
                
                # Debug first frame of each episode
                if step == 0:
                    print(f"First frame shape: {frame_array.shape}, dtype: {frame_array.dtype}, "
                          f"min: {frame_array.min()}, max: {frame_array.max()}")
                
                # Convert BGR to RGB for OpenCV if saving locally
                if save_local:
                    frame_array = cv2.cvtColor(frame_array, cv2.COLOR_RGB2BGR)  # type: ignore
                
                frames.append(frame_array)
                step += 1
            
            if frames:
                print(f"Episode completed with {len(frames)} frames and reward {total_reward:.2f}")
                frames_array = np.stack(frames)
                print(f"Stacked frames shape: {frames_array.shape}, "
                      f"dtype: {frames_array.dtype}, min: {frames_array.min()}, max: {frames_array.max()}")
                
                # Save video locally if requested
                if save_local:
                    # Create descriptive filename
                    timestamp = f"step_{timestep}_" if timestep is not None else ""
                    base_name = f"{prefix}_" if prefix else ""
                    filename = f"{base_name}{timestamp}episode_{episode+1}.{video_format}"
                    video_path = os.path.join(output_dir, filename)
                    
                    # Create video writer
                    height, width = frames[0].shape[:2]
                    writer = get_video_writer(video_path, render_fps, (width, height))
                    
                    if writer is not None:
                        for frame in frames:
                            writer.write(frame)  # type: ignore
                        writer.release()  # type: ignore
                        print(f"Saved video to {video_path}")
                    else:
                        print("Warning: Could not create video writer")
                
                # Convert back to RGB for WandB if we converted to BGR
                if save_local:
                    frames_array = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames])  # type: ignore
                
                episode_frames.append(cast(FrameStack, frames_array))
                episode_rewards.append(total_reward)
            else:
                print("Warning: No frames captured in this episode!")
    
    except Exception as e:
        print(f"Error during video recording: {str(e)}")
        raise
    finally:
        env.close()
    
    return episode_frames, episode_rewards

def log_video_to_wandb(
    model: Any,
    env_id: str,
    num_episodes: int = 3,
    max_steps: int = 1000,
    name_prefix: str = "eval_episode",
    deterministic: bool = True,
    render_fps: int = 30,
) -> None:
    """Record and log evaluation episodes to WandB."""
    episode_frames, episode_rewards = record_video_episodes(
        model=model,
        env_id=env_id,
        num_episodes=num_episodes,
        max_steps=max_steps,
        deterministic=deterministic,
        render_fps=render_fps,
    )
    
    # Log each episode
    for i, (frames, reward) in enumerate(zip(episode_frames, episode_rewards)):
        try:
            # Ensure frames are in format (T, H, W, C)
            if frames.ndim != 4:
                print(f"Warning: Unexpected frame dimensions: {frames.shape}")
                continue
            
            print(f"Logging video {i+1} with shape {frames.shape}, "
                  f"dtype: {frames.dtype}, min: {frames.min()}, max: {frames.max()}")
            
            video = wandb.Video(
                frames,
                fps=render_fps,
                format="mp4",
                caption=f"Episode {i+1} - Reward: {reward:.2f}"
            )
            
            wandb.log({
                f"videos/episode_{i+1}": video,
                f"videos/episode_{i+1}_reward": reward,
                f"videos/episode_{i+1}_length": len(frames)
            })
            print(f"Successfully logged video {i+1} to WandB")
        except Exception as e:
            print(f"Error logging video {i+1} to WandB: {str(e)}")
    
    # Log summary statistics
    wandb.log({
        f"{name_prefix}_mean_reward": np.mean(episode_rewards),
        f"{name_prefix}_std_reward": np.std(episode_rewards),
        f"{name_prefix}_min_reward": np.min(episode_rewards),
        f"{name_prefix}_max_reward": np.max(episode_rewards),
    }) 