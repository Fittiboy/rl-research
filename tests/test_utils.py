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
            "id": "CartPole-v1"
        },
        "algorithm": {
            "type": "stable_baselines3",
            "name": "ppo",
            "params": {
                "learning_rate": 3e-4,
                "n_steps": 128,
                "batch_size": 32,
                "device": "cpu"  # Force CPU to avoid warnings
            }
        },
        "experiment": {
            "name": "test_experiment",
            "eval_frequency": 1000
        },
        "wandb": {
            "project": "test_project",
            "group": "test_group",
            "tags": ["test"],
            "mode": "disabled"  # Important: Use disabled mode for tests
        }
    })

def test_environment_creation(test_config):
    """Test environment creation utility."""
    env = get_environment(test_config.env)
    assert isinstance(env, gym.Env)
    assert env.unwrapped.spec.id == "CartPole-v1"
    env.close()

def test_algorithm_creation(test_config):
    """Test algorithm creation utility."""
    env = get_environment(test_config.env)
    algo = get_algorithm(test_config.algorithm, env)
    
    # Check if algorithm was created with correct parameters
    assert algo.learning_rate == test_config.algorithm.params.learning_rate
    assert algo.n_steps == test_config.algorithm.params.n_steps
    assert algo.batch_size == test_config.algorithm.params.batch_size
    
    env.close()

def test_logger_setup(test_config):
    """Test logger setup with WandB integration."""
    env = get_environment(test_config.env)
    logger = setup_logging(test_config, env)
    
    assert logger is not None
    callbacks = logger.get_callbacks()
    
    # Check if we got the expected callbacks
    assert len(callbacks) > 0
    assert any(callback.__class__.__name__ == "WandbCallback" 
              for callback in callbacks)
    
    # Test metric logging
    test_metrics = {
        "test_metric": 1.0,
        "test_list": [1, 2, 3],
        "test_dict": {"a": 1, "b": 2}
    }
    logger.log_metrics(test_metrics)
    logger.log_metrics(test_metrics, step=10)
    
    # Cleanup
    env.close()
    logger.finish()

def test_logger_model_saving(test_config, tmp_path):
    """Test model saving with WandB integration."""
    # Update config to use tmp_path for model saving
    test_config.wandb.dir = str(tmp_path)
    
    env = get_environment(test_config.env)
    logger = setup_logging(test_config, env)
    
    # Create and save a test model
    algo = get_algorithm(test_config.algorithm, env)
    logger.save_model(algo, name="test_model")
    
    # Verify model was saved (in disabled mode, files are saved locally)
    model_path = os.path.join(tmp_path, "models", "test_model.zip")
    assert os.path.exists(model_path), f"Model file not found at {model_path}"
    
    # Cleanup
    env.close()
    logger.finish()

@pytest.fixture(autouse=True)
def cleanup():
    """Clean up after tests."""
    yield
    if wandb.run is not None:
        wandb.finish()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 