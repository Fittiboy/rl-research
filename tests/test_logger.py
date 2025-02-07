"""Tests for logging functionality."""
import os
from typing import Dict, Any
import pytest
import numpy as np
import gymnasium as gym
from omegaconf import OmegaConf
import wandb
from stable_baselines3.common.monitor import Monitor

from rl_research.utils.logger import (
    ExperimentLogger,
    VideoEvalCallback,
    EpisodeLoggingCallback,
    setup_logging,
)

class DummyModel:
    """Dummy model for testing."""
    def save(self, path: str) -> None:
        """Mock save method."""
        with open(path, "w") as f:
            f.write("dummy model")

@pytest.fixture
def mock_config() -> Dict[str, Any]:
    """Create a mock configuration for testing."""
    return {
        "wandb": {
            "project": "test_project",
            "group": "test_group",
            "tags": ["test"],
            "mode": "disabled",
            "dir": "test_output"
        },
        "experiment": {
            "eval_frequency": 100,
            "name": "test_experiment"
        },
        "env": {
            "id": "CartPole-v1",
            "params": {}
        },
        "algorithm": {
            "name": "ppo",
            "n_envs": 1,
            "params": {}
        },
        "video": {
            "fps": 30,
            "num_episodes": 2
        }
    }

@pytest.fixture
def config(mock_config):
    """Create OmegaConf configuration."""
    return OmegaConf.create(mock_config)

@pytest.fixture
def env():
    """Create test environment."""
    return gym.make("CartPole-v1")

def test_experiment_logger_initialization(config, env):
    """Test ExperimentLogger initialization."""
    logger = ExperimentLogger(config, env)
    assert logger.config == config
    assert logger.run is not None
    assert isinstance(logger.env, Monitor)

def test_experiment_logger_callbacks(config, env):
    """Test callback creation."""
    logger = ExperimentLogger(config, env)
    callbacks = logger.get_callbacks()
    
    # Should have both VideoEvalCallback and EpisodeLoggingCallback
    assert len(callbacks) == 2
    assert any(isinstance(cb, VideoEvalCallback) for cb in callbacks)
    assert any(isinstance(cb, EpisodeLoggingCallback) for cb in callbacks)

def test_experiment_logger_metrics(config, env):
    """Test metric logging."""
    logger = ExperimentLogger(config, env)
    metrics = {"test_metric": 1.0}
    logger.log_metrics(metrics)
    # Since we're in disabled mode, this just tests that the call doesn't error

def test_experiment_logger_model_saving(config, env, tmp_path):
    """Test model saving functionality."""
    config.wandb.dir = str(tmp_path)
    logger = ExperimentLogger(config, env)
    
    # Create a dummy model
    model = DummyModel()
    
    # Test saving
    logger.save_model(model, "test_model")
    assert os.path.exists(os.path.join(tmp_path, "models", "test_model"))

def test_video_eval_callback(config, env):
    """Test VideoEvalCallback functionality."""
    eval_env = gym.make("CartPole-v1", render_mode="rgb_array")
    callback = VideoEvalCallback(
        eval_env=eval_env,
        eval_freq=10,
        video_fps=30,
        n_eval_episodes=1
    )
    
    assert callback.video_fps == 30
    assert callback.eval_idx == 0
    assert callback.last_mean_reward == 0.0

def test_episode_logging_callback(config, env):
    """Test EpisodeLoggingCallback functionality."""
    logger = ExperimentLogger(config, env)
    callback = EpisodeLoggingCallback(logger)
    
    # Test episode logging
    callback.locals = {"infos": [{"episode": {"r": 1.0, "l": 10}}]}
    callback.num_timesteps = 100
    
    assert callback._on_step()
    assert len(callback._episode_rewards) == 1
    assert len(callback._episode_lengths) == 1
    assert callback._episode_count == 1

def test_get_final_score(config, env):
    """Test get_final_score functionality."""
    logger = ExperimentLogger(config, env)
    score = logger.get_final_score()
    assert score == 0.0  # Should be 0.0 when no evaluation has been done

def test_setup_logging(config, env):
    """Test setup_logging utility function."""
    logger = setup_logging(config, env)
    assert isinstance(logger, ExperimentLogger)

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after tests."""
    yield
    if wandb.run is not None:
        wandb.finish() 