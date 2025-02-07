"""Tests for experiment logging utilities."""
import os
import pytest
import gymnasium as gym
from omegaconf import OmegaConf
import wandb
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from rl_research.utils.logger import ExperimentLogger

@pytest.fixture
def mock_config():
    """Create a mock configuration for testing."""
    return OmegaConf.create({
        "wandb": {
            "project": "test_project",
            "group": "test_group",
            "tags": ["test"],
            "mode": "disabled"  # Important: Use disabled mode for tests
        },
        "experiment": {
            "eval_frequency": 1000,
            "name": "test_experiment"
        },
        "env": {
            "id": "CartPole-v1",
            "type": "gym",
            "params": {}  # Add empty params to avoid KeyError
        },
        "algorithm": {
            "type": "stable_baselines3",
            "name": "ppo",
            "params": {
                "learning_rate": 0.0003,
                "n_steps": 128
            }
        },
        "video": {
            "enabled": True,
            "fps": 30,
            "num_episodes": 2
        }
    })

@pytest.fixture
def mock_env():
    """Create a mock environment for testing."""
    return gym.make("CartPole-v1")

class TestExperimentLogger:
    """Test suite for ExperimentLogger."""
    
    def test_initialization(self, mock_config, mock_env):
        """Test logger initialization."""
        logger = ExperimentLogger(mock_config, mock_env)
        assert logger.config == mock_config
        # Check that env is wrapped in Monitor
        assert isinstance(logger.env, Monitor)
        # Check that the base environment is a Gym environment
        assert isinstance(logger.env.unwrapped, gym.Env)
    
    def test_get_callbacks(self, mock_config, mock_env):
        """Test callback creation."""
        logger = ExperimentLogger(mock_config, mock_env)
        callbacks = logger.get_callbacks()
        
        # Should return a list of callbacks
        assert isinstance(callbacks, list)
        assert len(callbacks) > 0
        
        # Check if evaluation callback is included when eval_frequency is set
        assert any(callback.__class__.__name__ == "VideoEvalCallback" 
                  for callback in callbacks)
        
        # Check if episode logging callback is included
        assert any(callback.__class__.__name__ == "EpisodeLoggingCallback" 
                  for callback in callbacks)
        
        logger.finish()
    
    def test_log_metrics(self, mock_config, mock_env):
        """Test metric logging."""
        logger = ExperimentLogger(mock_config, mock_env)
        
        # Test logging different types of metrics
        metrics = {
            "test_scalar": 1.0,
            "test_list": [1, 2, 3],
            "test_dict": {"a": 1, "b": 2}
        }
        
        logger.log_metrics(metrics)
        logger.log_metrics(metrics, step=10)
        
        logger.finish()
    
    def test_save_model(self, mock_config, mock_env, tmp_path):
        """Test model saving functionality."""
        # Update config to use tmp_path for model saving
        mock_config.wandb.dir = str(tmp_path)
        logger = ExperimentLogger(mock_config, mock_env)
        
        # Create a test model
        model = PPO("MlpPolicy", mock_env, device="cpu")  # Force CPU to avoid warnings
        
        # Test saving
        logger.save_model(model, name="test_model")
        
        # Verify model was saved (in disabled mode, files are saved locally)
        model_path = os.path.join(tmp_path, "models", "test_model.zip")
        assert os.path.exists(model_path), f"Model file not found at {model_path}"
        
        logger.finish()
    
    def test_finish(self, mock_config, mock_env):
        """Test logger cleanup."""
        logger = ExperimentLogger(mock_config, mock_env)
        
        # Should not raise any errors
        logger.finish()
        
        # Second call should not raise errors
        logger.finish()
        
        # Verify run is properly closed
        assert logger.run is None

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after tests."""
    yield
    if wandb.run is not None:
        wandb.finish() 