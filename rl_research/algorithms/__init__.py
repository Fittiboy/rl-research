"""Algorithm implementations and factory."""
from typing import Any, Dict, Optional
import gymnasium as gym
from stable_baselines3 import PPO
from .rllib_wrapper import RLlibWrapper


def create_algorithm(
    env: gym.Env,
    config: Dict[str, Any],
    seed: Optional[int] = None
) -> Any:
    """Create an RL algorithm based on configuration.
    
    Args:
        env: The environment to train on
        config: Algorithm configuration
        seed: Random seed
        
    Returns:
        The algorithm instance
    """
    if config["type"] == "stable_baselines3":
        if config["name"] == "ppo":
            return PPO(
                "MlpPolicy",
                env,
                seed=seed,
                **config["params"]
            )
    elif config["type"] == "rllib":
        return RLlibWrapper(
            env=env,
            config=config,
            seed=seed
        )
    else:
        raise ValueError(f"Unknown algorithm type: {config['type']}")
