"""Algorithm implementations and factory."""
from typing import Any, Dict, Optional
import gymnasium as gym
from stable_baselines3 import PPO, SAC, DQN

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
        elif config["name"] == "sac":
            return SAC(
                "MlpPolicy",
                env,
                seed=seed,
                **config["params"]
            )
        elif config["name"] == "dqn":
            return DQN(
                "MlpPolicy",
                env,
                seed=seed,
                **config["params"]
            )
        else:
            raise ValueError(f"Unknown algorithm: {config['name']}")
    else:
        raise ValueError(f"Unknown algorithm type: {config['type']}")

__all__ = ["create_algorithm"]
