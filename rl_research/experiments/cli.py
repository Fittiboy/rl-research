"""Command-line interface for running experiments."""
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import wandb
from rl_research.utils import get_algorithm, get_environment, setup_logging

@hydra.main(config_path="configs", config_name="experiment", version_base=None)
def run_experiment(cfg: DictConfig) -> None:
    """Main entry point for running experiments.
    
    Args:
        cfg: Hydra configuration object containing experiment settings
    
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
    print("Configuration:")
    print(OmegaConf.to_yaml(cfg))
    
    try:
        # Create environment
        env = get_environment(cfg.env)
        
        # Setup logging and experiment tracking
        logger = setup_logging(cfg, env)
        
        # Initialize algorithm
        algo = get_algorithm(cfg.algorithm, env)
        
        # Training loop
        algo.learn(
            total_timesteps=cfg.experiment.total_timesteps,
            callback=logger.get_callbacks(),
        )
        
        # Save final model and configs
        logger.save_model(algo)
        
        # Cleanup
        env.close()
        logger.finish()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    except Exception as e:
        print(f"\nError during training: {str(e)}")
        raise
    finally:
        # Ensure wandb run is properly closed
        if wandb.run is not None:
            wandb.finish()

if __name__ == "__main__":
    run_experiment() 