"""Tests for utility functions."""
import pytest
import os
import gymnasium as gym
from omegaconf import OmegaConf
import wandb
from rl_research.utils import get_environment, get_algorithm, setup_logging

@pytest.fixture
def test_config():
    """Create test configuration."""
    return OmegaConf.create({
        "env": {
            "type": "gym",
            "id": "CartPole-v1",
            "params": {}
        },
        "algorithm": {
            "type": "stable_baselines3",
            "name": "ppo",
            "params": {
                "learning_rate": 0.0003,
                "n_steps": 128,
                "device": "cpu"
            }
        },
        "experiment": {
            "name": "test_experiment",
            "eval_frequency": 1000
        },
        "video": {
            "enabled": True,
            "fps": 30,
            "num_episodes": 2
        },
        "wandb": {
            "project": "test_project",
            "group": "test_group",
            "tags": ["test"],
            "mode": "disabled"
        }
    })

def test_environment_creation(test_config):
    """Test environment creation."""
    env = get_environment(test_config.env)
    assert env is not None
    env.close()

def test_algorithm_creation(test_config):
    """Test algorithm creation."""
    env = get_environment(test_config.env)
    algo = get_algorithm(test_config.algorithm, env)
    assert algo is not None
    env.close()

def test_logger_setup(test_config):
    """Test logger setup with WandB integration."""
    env = get_environment(test_config.env)
    logger = setup_logging(test_config, env)
    
    assert logger is not None
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
    env.close()

def test_logger_model_saving(test_config, tmp_path):
    """Test model saving functionality."""
    # Update config to use tmp_path
    test_config.wandb.dir = str(tmp_path)
    
    env = get_environment(test_config.env)
    algo = get_algorithm(test_config.algorithm, env)
    logger = setup_logging(test_config, env)
    
    # Test saving
    logger.save_model(algo, name="test_model")
    
    # Verify model was saved
    model_path = os.path.join(tmp_path, "models", "test_model.zip")
    assert os.path.exists(model_path)
    
    logger.finish()
    env.close()

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after tests."""
    yield
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 