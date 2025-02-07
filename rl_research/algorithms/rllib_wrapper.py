"""RLlib algorithm wrapper for consistent interface."""
from typing import Any, Dict, Optional, Union
import os
import ray
from ray import tune
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
import gymnasium as gym


class RLlibWrapper:
    """Wrapper for RLlib algorithms to match our framework's interface."""
    
    def __init__(
        self,
        env: gym.Env,
        config: Dict[str, Any],
        seed: Optional[int] = None
    ):
        """Initialize RLlib algorithm.
        
        Args:
            env: The environment to train on
            config: Algorithm configuration
            seed: Random seed
        """
        self.env = env
        self.config = config
        self.seed = seed
        
        # Initialize Ray if not already
        if not ray.is_initialized():
            ray.init()
        
        # Create algorithm config
        algo_config = (
            AlgorithmConfig()
            .environment(env=type(env))
            .framework(config["params"]["framework"])
            .training(
                train_batch_size=config["params"]["train_batch_size"],
                sgd_minibatch_size=config["params"]["sgd_minibatch_size"],
                num_sgd_iter=config["params"]["num_sgd_iter"],
                lr=config["params"]["lr"],
                gamma=config["params"]["gamma"],
                lambda_=config["params"]["lambda_"],
                clip_param=config["params"]["clip_param"],
                vf_clip_param=config["params"]["vf_clip_param"],
                entropy_coeff=config["params"]["entropy_coeff"],
                vf_loss_coeff=config["params"]["vf_loss_coeff"],
            )
            .resources(
                num_gpus=config["params"]["num_gpus"],
                num_workers=config["params"]["num_workers"],
            )
            .rollouts(num_rollout_workers=config["params"]["num_workers"])
            .evaluation(
                evaluation_config=config["evaluation_config"]
            )
        )
        
        if seed is not None:
            algo_config.seed(seed)
        
        # Set model config
        algo_config.training(model=config["params"]["model"])
        
        # Create algorithm
        self.algo = algo_config.build()
    
    def learn(
        self,
        total_timesteps: int,
        callback: Optional[Any] = None,
        **kwargs
    ) -> "RLlibWrapper":
        """Train the algorithm.
        
        Args:
            total_timesteps: Number of environment steps to train for
            callback: Callback for logging (not used, RLlib has its own logging)
            **kwargs: Additional arguments passed to tune.run()
            
        Returns:
            self
        """
        # Convert timesteps to training iterations
        train_batch_size = self.config["params"]["train_batch_size"]
        num_workers = self.config["params"]["num_workers"]
        iters = total_timesteps // (train_batch_size * num_workers)
        
        for _ in range(iters):
            result = self.algo.train()
            
            # Log metrics if callback is provided
            if callback is not None:
                metrics = {
                    "train/episode_reward_mean": result["episode_reward_mean"],
                    "train/episode_length_mean": result["episode_len_mean"],
                    "train/total_timesteps": result["timesteps_total"],
                }
                callback.on_step(metrics)
        
        return self
    
    def predict(
        self,
        observation: Any,
        deterministic: bool = False
    ) -> tuple:
        """Get action from policy.
        
        Args:
            observation: Environment observation
            deterministic: Whether to use deterministic action selection
            
        Returns:
            action, state
        """
        action = self.algo.compute_single_action(
            observation,
            explore=not deterministic
        )
        return action, None  # RLlib doesn't use state
    
    def save(self, path: str):
        """Save the model.
        
        Args:
            path: Path to save directory
        """
        checkpoint_dir = os.path.join(path, "checkpoint")
        self.algo.save(checkpoint_dir)
    
    def close(self):
        """Clean up resources."""
        if self.algo is not None:
            self.algo.stop()
        
        # Shutdown Ray if initialized
        if ray.is_initialized():
            ray.shutdown() 