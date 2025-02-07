"""Registry for algorithms and environments."""
from typing import Dict, Type, Any, Optional
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from omegaconf import DictConfig

ALGORITHM_REGISTRY = {
    "stable_baselines3": {
        "ppo": PPO,
        "sac": SAC,
        "dqn": DQN,
    },
    "custom": {
        # Custom algorithms will be registered here
    }
}

ENVIRONMENT_REGISTRY = {
    "gym": {
        # Standard Gym environments are handled directly
    },
    "custom": {
        # Custom environments will be registered here
    }
}

def register_algorithm(name: str, algorithm_class: Type, algorithm_type: str = "custom"):
    """Register a new algorithm."""
    if algorithm_type not in ALGORITHM_REGISTRY:
        ALGORITHM_REGISTRY[algorithm_type] = {}
    ALGORITHM_REGISTRY[algorithm_type][name] = algorithm_class

def register_environment(name: str, environment_class: Type, environment_type: str = "custom"):
    """Register a new environment."""
    if environment_type not in ENVIRONMENT_REGISTRY:
        ENVIRONMENT_REGISTRY[environment_type] = {}
    ENVIRONMENT_REGISTRY[environment_type][name] = environment_class

def get_algorithm(config: DictConfig, env: gym.Env):
    """Get algorithm instance from config."""
    algo_type = config.type
    algo_name = config.name
    
    if algo_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    if algo_name not in ALGORITHM_REGISTRY[algo_type]:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    algo_class = ALGORITHM_REGISTRY[algo_type][algo_name]
    return algo_class(
        policy="MlpPolicy",
        env=env,
        **config.params
    )

def get_environment(config: DictConfig) -> gym.Env:
    """Get environment instance from config."""
    env_type = config.type
    
    if env_type == "gym":
        env = gym.make(config.id)
        # Apply environment parameters if specified
        if hasattr(config, "params"):
            for key, value in config.params.items():
                if hasattr(env, key):
                    setattr(env, key, value)
        return env
    
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    if config.id not in ENVIRONMENT_REGISTRY[env_type]:
        raise ValueError(f"Unknown environment: {config.id}")
    
    env_class = ENVIRONMENT_REGISTRY[env_type][config.id]
    return env_class(**config.get("params", {})) 