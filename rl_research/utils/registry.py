"""Registry for algorithms and environments."""
from typing import Dict, Type, Any, Optional, Union
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv
from stable_baselines3.common.env_util import make_vec_env
from omegaconf import DictConfig

# Import and register ALE environments
try:
    import ale_py
    gym.register_envs(ale_py)
except ImportError:
    ale_py = None

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

def get_algorithm(config: DictConfig, env: Union[gym.Env, VecEnv]):
    """Get algorithm instance from config."""
    algo_type = config.type
    algo_name = config.name
    
    if algo_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    if algo_name not in ALGORITHM_REGISTRY[algo_type]:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    algo_class = ALGORITHM_REGISTRY[algo_type][algo_name]
    
    # Get all parameters from config
    params = dict(config.params)
    
    # Extract policy if specified, otherwise use default
    policy = params.pop("policy", "MlpPolicy")
    
    return algo_class(
        policy=policy,
        env=env,
        **params
    )

def get_environment(config: DictConfig) -> Union[gym.Env, VecEnv]:
    """Get environment instance from config."""
    env_type = config.type
    
    if env_type == "gym":
        # For Atari environments, ensure ale_py is imported and environments are registered
        if config.id.startswith("ALE/"):
            if ale_py is None:
                raise ImportError(
                    "ale_py is required for Atari environments. "
                    "Please install it with: pip install ale-py"
                )
        
        # Get environment parameters
        env_kwargs = dict(config.params) if hasattr(config, "params") else {}
        
        # Create vectorized environment if n_envs is specified
        n_envs = getattr(config, "n_envs", 1)
        if n_envs > 1:
            env = make_vec_env(
                env_id=config.id,
                n_envs=n_envs,
                env_kwargs=env_kwargs,
                vec_env_cls=SubprocVecEnv if n_envs > 1 else DummyVecEnv
            )
        else:
            env = gym.make(config.id, **env_kwargs)
        
        return env
    
    if env_type not in ENVIRONMENT_REGISTRY:
        raise ValueError(f"Unknown environment type: {env_type}")
    
    if config.id not in ENVIRONMENT_REGISTRY[env_type]:
        raise ValueError(f"Unknown environment: {config.id}")
    
    env_class = ENVIRONMENT_REGISTRY[env_type][config.id]
    return env_class(**config.get("params", {})) 