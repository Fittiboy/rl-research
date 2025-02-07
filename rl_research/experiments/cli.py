"""Command-line interface for running experiments."""
import os
import logging
from typing import Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import wandb

from rl_research.utils import get_algorithm, get_environment, setup_logging

# Configure logging
logger = logging.getLogger(__name__)

@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def run_experiment(cfg: DictConfig) -> Optional[float]:
    """Main entry point for running experiments.
    
    Args:
        cfg: Hydra configuration object containing experiment settings
    
    Returns:
        Optional[float]: Final evaluation score if available, None otherwise
    
    The configuration is automatically loaded from the configs directory:
    - Base config: experiment.yaml
    - Algorithm config: algorithm/{algo_name}.yaml
    - Environment config: env/{env_name}.yaml
    
    Example usage:
        # Basic run with default config
        python -m rl_research.experiments.cli
        
        # Override parameters
        python -m rl_research.experiments.cli algorithm=ppo env=cartpole
        
        # Multiple seeds
        python -m rl_research.experiments.cli experiment.seed=1,2,3
        
        # Parameter sweep
        python -m rl_research.experiments.cli -m algorithm.params.learning_rate=0.0001,0.0003,0.001
    """
    logger.info("Starting experiment with configuration:")
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    final_score = None
    env = None
    algo = None
    experiment_logger = None
    
    try:
        # Create environment
        logger.info(f"Creating environment: {cfg.env.id}")
        env = get_environment(cfg.env)
        
        # Setup logging and experiment tracking
        logger.info("Setting up experiment tracking")
        experiment_logger = setup_logging(cfg, env)
        
        # Initialize algorithm
        logger.info(f"Initializing algorithm: {cfg.algorithm.name}")
        algo = get_algorithm(cfg.algorithm, env)
        
        # Training loop
        logger.info(f"Starting training for {cfg.experiment.total_timesteps} timesteps")
        algo.learn(
            total_timesteps=cfg.experiment.total_timesteps,
            callback=experiment_logger.get_callbacks(),
        )
        
        # Save final model and configs
        logger.info("Saving final model and configurations")
        experiment_logger.save_model(algo)
        
        # Get final evaluation score if available
        if hasattr(experiment_logger, "get_final_score"):
            final_score = experiment_logger.get_final_score()
            logger.info(f"Final evaluation score: {final_score}")
        
        return final_score
        
    except KeyboardInterrupt:
        logger.warning("\nTraining interrupted by user")
    except Exception as e:
        logger.error(f"\nError during training: {str(e)}", exc_info=True)
        raise
    finally:
        # Cleanup
        logger.info("Cleaning up resources")
        if env is not None:
            env.close()
        if experiment_logger is not None:
            experiment_logger.finish()
        if wandb.run is not None:
            wandb.finish()
    
    return final_score

if __name__ == "__main__":
    run_experiment() 