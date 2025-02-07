"""Tests for project setup and configuration."""
import os
from typing import Dict, Any, List, Union
import pytest
import torch
import yaml
from omegaconf import OmegaConf

def test_imports():
    """Test if all required packages are installed."""
    import gymnasium as gym
    import stable_baselines3
    import wandb
    import hydra
    import matplotlib
    import seaborn
    
    assert all([gym, stable_baselines3, wandb, hydra, matplotlib, seaborn])

def test_gpu_availability():
    """Test if CUDA is available."""
    # Just check if torch can detect GPU, don't require it
    _ = torch.cuda.is_available()

def test_environment():
    """Test if we can create a basic environment."""
    import gymnasium as gym
    env = gym.make("CartPole-v1")
    assert env is not None
    env.close()

def test_config_loading():
    """Test if we can load our configuration files."""
    with open("rl_research/experiments/configs/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # Check base config structure
    assert isinstance(config, dict), "Config should be a dictionary"
    assert "defaults" in config, "Config should have 'defaults' key"
    defaults = config["defaults"]
    assert isinstance(defaults, list), "Defaults should be a list"
    
    # Check for algorithm and env in defaults
    has_algorithm = False
    has_env = False
    for item in defaults:
        if isinstance(item, dict):
            if "algorithm" in item:
                has_algorithm = True
            if "env" in item:
                has_env = True
    assert has_algorithm, "Config should include algorithm in defaults"
    assert has_env, "Config should include env in defaults"
    
    # Check if we can load algorithm configs
    algo_path = "rl_research/experiments/configs/algorithm/ppo.yaml"
    assert os.path.exists(algo_path), "PPO config file should exist"
    with open(algo_path, "r") as f:
        algo_config = yaml.safe_load(f)
    
    assert isinstance(algo_config, dict), "Algorithm config should be a dictionary"
    assert "name" in algo_config, "Algorithm config should have 'name' key"
    assert "type" in algo_config, "Algorithm config should have 'type' key"
    assert "params" in algo_config, "Algorithm config should have 'params' key"
    
    # Check if we can load environment configs
    env_path = "rl_research/experiments/configs/env/cartpole.yaml"
    assert os.path.exists(env_path), "CartPole config file should exist"
    with open(env_path, "r") as f:
        env_config = yaml.safe_load(f)
    
    assert isinstance(env_config, dict), "Environment config should be a dictionary"
    assert "id" in env_config, "Environment config should have 'id' key"
    assert "type" in env_config, "Environment config should have 'type' key"

if __name__ == "__main__":
    print("Running setup tests...")
    test_imports()
    test_gpu_availability()
    test_environment()
    test_config_loading()
    print("All tests passed successfully!") 