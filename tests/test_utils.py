"""Tests for utility functions."""
import pytest
from omegaconf import OmegaConf
import gymnasium as gym
from rl_research.utils import (
    get_algorithm,
    get_environment,
    setup_logging,
)

def test_environment_creation():
    """Test environment creation from config."""
    config = OmegaConf.create({
        "type": "gym",
        "id": "CartPole-v1",
        "params": {
            "max_episode_steps": 500
        }
    })
    
    env = get_environment(config)
    assert isinstance(env, gym.Env)
    assert env.spec.id == "CartPole-v1"
    env.close()

def test_algorithm_creation():
    """Test algorithm creation from config."""
    env_config = OmegaConf.create({
        "type": "gym",
        "id": "CartPole-v1"
    })
    env = get_environment(env_config)
    
    algo_config = OmegaConf.create({
        "type": "stable_baselines3",
        "name": "ppo",
        "params": {
            "learning_rate": 0.0003,
            "n_steps": 2048,
            "device": "cpu"  # Force CPU to avoid warnings
        }
    })
    
    algo = get_algorithm(algo_config, env)
    assert algo is not None
    env.close()

@pytest.mark.skip(reason="Requires WandB authentication")
def test_logger_setup():
    """Test logger setup with config."""
    config = OmegaConf.create({
        "experiment": {
            "name": "test_experiment",
            "eval_frequency": 1000
        },
        "wandb": {
            "project": "test_project",
            "group": "test_group",
            "tags": ["test"]
        },
        "env": {
            "type": "gym",
            "id": "CartPole-v1"
        }
    })
    
    env = get_environment(config.env)
    logger = setup_logging(config, env)
    
    assert logger is not None
    callbacks = logger.get_callbacks()
    assert len(callbacks) > 0
    
    env.close()
    logger.finish()

if __name__ == "__main__":
    pytest.main([__file__, "-v"]) 