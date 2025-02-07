import pytest
import gymnasium as gym
import torch
import stable_baselines3
import wandb
import hydra
from omegaconf import OmegaConf
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

def test_imports():
    """Test if all required packages are properly installed."""
    assert gym.__version__ >= "0.29.1"
    assert torch.__version__ >= "2.1.0"
    assert wandb.__version__ >= "0.16.0"

def test_gpu_availability():
    """Test if CUDA is available (not required but good to know)."""
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")

def test_environment():
    """Test if we can create and run a basic environment."""
    env = gym.make("CartPole-v1")
    obs, _ = env.reset()
    assert obs.shape == (4,)
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    assert isinstance(reward, float)
    env.close()

def test_config_loading():
    """Test if we can load our configuration files."""
    with open("rl_research/experiments/configs/experiment.yaml", "r") as f:
        config = yaml.safe_load(f)
    assert "experiment" in config
    assert "wandb" in config

if __name__ == "__main__":
    print("Running setup tests...")
    test_imports()
    test_gpu_availability()
    test_environment()
    test_config_loading()
    print("All tests passed successfully!") 