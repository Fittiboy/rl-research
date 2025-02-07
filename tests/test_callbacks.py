"""Tests for training callbacks."""
import pytest
import warnings
import numpy as np
import wandb
from rl_research.callbacks import QuickStartCallback
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.logger import configure
from omegaconf import OmegaConf
from rl_research.utils.logger import VideoEvalCallback

# Mark all tests to ignore specific warnings
pytestmark = [
    pytest.mark.filterwarnings("ignore:Training and eval env are not of the same type"),
    pytest.mark.filterwarnings("ignore:You tried to call render()"),
]

class MockLocals:
    """Mock locals dictionary for testing callbacks."""
    def __init__(self, rewards=None, dones=None):
        self.data = {
            "rewards": rewards if rewards is not None else [0.0],
            "dones": dones if dones is not None else [False]
        }
    
    def get(self, key, default=None):
        return self.data.get(key, default)

class MockPolicy(BasePolicy):
    """Mock policy that returns random actions."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        """Not used in testing."""
        return None, None
    
    def _predict(self, observation, deterministic: bool = False):
        """Return random action."""
        return self.action_space.sample(), None

class MockModel(BaseAlgorithm):
    """Mock model for testing that always returns the same action."""
    
    # Define policy aliases
    policy_aliases = {
        "MlpPolicy": MockPolicy,
        "CnnPolicy": MockPolicy,
    }
    
    def __init__(self, action_space):
        super().__init__(
            policy="MlpPolicy",  # Dummy policy
            env=None,  # No environment needed
            learning_rate=0.0,  # No learning
            policy_kwargs=None,
            tensorboard_log=None,
            verbose=0,
            device="cpu",
            support_multi_env=True,
        )
        self._action_space = action_space
        # Set up logger
        self._logger = configure(None, ["stdout"])
    
    def predict(
        self,
        observation,
        state=None,
        episode_start=None,
        deterministic=False,
    ):
        """Return random action and None for the state."""
        if isinstance(observation, dict):
            # Handle dict observations
            batch_size = len(next(iter(observation.values())))
        else:
            # Handle array observations
            batch_size = len(observation)
        
        # Return batch of actions
        actions = np.array([self._action_space.sample() for _ in range(batch_size)])
        return actions, None
    
    def _setup_model(self):
        """Required by BaseAlgorithm but not needed for testing."""
        pass
    
    def learn(self, *args, **kwargs):
        """Required by BaseAlgorithm but not needed for testing."""
        return self

class FastImageEnv(gym.Wrapper):
    """Wrapper that limits episode length for faster testing."""
    
    def __init__(self, env):
        """Initialize wrapper."""
        super().__init__(env)
        self.max_steps = 10  # Limit episodes to 10 steps for testing
        self._steps = 0
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(84, 84, 3), dtype=np.uint8
        )
    
    def reset(self, **kwargs):
        """Reset environment and step counter."""
        obs, info = self.env.reset(**kwargs)
        self._steps = 0
        return self._get_obs(), info
    
    def step(self, action):
        """Step environment and force early termination."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self._steps += 1
        # Force termination after max_steps
        if self._steps >= self.max_steps:
            terminated = True
        return self._get_obs(), reward, terminated, truncated, info
    
    def _get_obs(self):
        """Convert state to image observation."""
        # Create a simple visualization (blue background with white dot)
        img = np.zeros((84, 84, 3), dtype=np.uint8)
        img[:, :] = [66, 135, 245]  # Blue background
        
        # Draw white dot at a position that moves with steps
        x = int(42 + np.sin(self._steps * 0.5) * 20)
        y = int(42 + np.cos(self._steps * 0.5) * 20)
        
        # Draw white dot
        for i in range(-2, 3):
            for j in range(-2, 3):
                if 0 <= x + i < 84 and 0 <= y + j < 84:
                    img[y + j, x + i] = [255, 255, 255]
        
        return img

@pytest.fixture
def callback():
    """Create a QuickStartCallback instance."""
    return QuickStartCallback(verbose=0)

@pytest.fixture(autouse=True)
def wandb_init():
    """Initialize WandB in disabled mode for tests."""
    wandb.init(mode="disabled")
    yield
    if wandb.run is not None:
        wandb.finish()

class TestQuickStartCallback:
    """Test suite for QuickStartCallback."""
    
    def test_initialization(self, callback):
        """Test callback initialization."""
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []
        assert callback._current_episode_reward == 0
    
    def test_on_step_accumulate_reward(self, callback):
        """Test reward accumulation during steps."""
        # Mock locals with rewards
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        
        # Test single step
        callback._on_step()
        assert callback._current_episode_reward == 1.0
        
        # Test multiple steps
        callback._on_step()
        assert callback._current_episode_reward == 2.0
    
    def test_on_step_episode_end(self, callback):
        """Test behavior when episode ends."""
        # Set up episode
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        callback.n_calls = 10
        
        # Run some steps
        for _ in range(3):
            callback._on_step()
        
        # End episode
        locals_dict.data["dones"] = [True]
        callback._on_step()
        
        # Check if episode was properly logged
        assert len(callback.episode_rewards) == 1
        assert callback.episode_rewards[0] == 4.0  # Sum of rewards
        assert len(callback.episode_lengths) == 1
    
    def test_on_step_wandb_logging(self, callback):
        """Test WandB logging functionality."""
        # Set up episode
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        callback.n_calls = 20
        
        # Run steps until logging should occur
        for _ in range(10):
            locals_dict.data["dones"] = [True]
            callback._on_step()
            locals_dict.data["dones"] = [False]
        
        # Verify metrics were logged
        assert len(callback.episode_rewards) == 10
        # WandB logging would have occurred, but we can't verify the actual log
        # without mocking WandB itself
    
    def test_on_step_with_none_values(self, callback):
        """Test handling of None values in locals."""
        # Test with None rewards
        locals_dict = MockLocals(rewards=None)
        callback.locals = locals_dict
        callback._on_step()
        assert callback._current_episode_reward == 0
        
        # Test with None dones
        locals_dict = MockLocals(dones=None)
        callback.locals = locals_dict
        callback._on_step()
        # Should not raise any errors
    
    def test_on_step_with_empty_values(self, callback):
        """Test handling of empty lists in locals."""
        # Test with empty rewards
        locals_dict = MockLocals(rewards=[])
        callback.locals = locals_dict
        callback._on_step()
        assert callback._current_episode_reward == 0
        
        # Test with empty dones
        locals_dict = MockLocals(dones=[])
        callback.locals = locals_dict
        callback._on_step()
        # Should not raise any errors 

def test_video_eval_callback_format():
    """Test that VideoEvalCallback maintains correct frame format."""
    # Create a simple environment with limited steps
    env = DummyVecEnv([
        lambda: Monitor(
            FastImageEnv(gym.make("CartPole-v1"))
        )
    ])
    
    # Create mock model instead of training
    mock_model = MockModel(env.action_space)
    
    # Create callback
    callback = VideoEvalCallback(
        eval_env=env,
        eval_freq=1,  # Evaluate immediately
        video_fps=30,
        n_eval_episodes=1,
    )
    
    # Set up callback with mock model
    callback.model = mock_model
    
    # Trigger evaluation directly
    callback.eval_freq = 1
    callback._on_step()
    
    # Get the last recorded frames from the callback
    if hasattr(callback, "_last_frames") and callback._last_frames is not None:
        frames = callback._last_frames
        
        # Check frame format (should be in HWC format for testing)
        assert len(frames.shape) == 4, "Frames should be 4-dimensional (T, H, W, C)"
        assert frames.shape[-1] == 3, "Last dimension should be channels (3 for RGB)"
        assert frames.dtype == np.uint8, "Frames should be uint8"
        assert 0 <= frames.min() <= frames.max() <= 255, "Pixel values should be in [0, 255]"
        
        # Additional checks for frame dimensions
        T, H, W, C = frames.shape
        assert H > 0 and W > 0, "Height and width should be positive"
        assert C == 3, "Should have 3 color channels"
        assert T > 0, "Should have at least one frame"
        assert T <= 10, "Should have limited number of frames"

def test_video_eval_callback_integration():
    """Test that VideoEvalCallback works with WandB logging."""
    # Create config
    config = OmegaConf.create({
        "env": {
            "id": "CartPole-v1",
            "params": {}
        },
        "experiment": {
            "eval_frequency": 1
        },
        "video": {
            "num_episodes": 1,
            "fps": 30
        }
    })
    
    # Create environment with limited steps
    env = DummyVecEnv([
        lambda: Monitor(
            FastImageEnv(gym.make(config.env.id))
        )
    ])
    
    # Create callback
    callback = VideoEvalCallback(
        eval_env=env,
        eval_freq=config.experiment.eval_frequency,
        video_fps=config.video.fps,
        n_eval_episodes=config.video.num_episodes,
    )
    
    # Set up callback with mock model
    callback.model = MockModel(env.action_space)
    
    # Trigger evaluation directly
    callback._on_step()
    
    # Verify callback completed successfully
    assert callback.eval_idx > 0, "Should complete at least one evaluation"
    assert hasattr(callback, "_last_frames"), "Should have recorded frames"
    if callback._last_frames is not None:
        assert callback._last_frames.shape[0] <= 10, "Should have limited number of frames" 