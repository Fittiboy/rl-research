Quickstart Guide
==============

This guide will help you get started with RL Research quickly. We'll cover the basics of running experiments and training agents.

Basic Usage
----------

Here's a minimal example to get you started with training a PPO agent on the CartPole environment:

.. code-block:: python

   from rl_research.experiments.cli import run_experiment
   
   # Run with default configuration
   run_experiment()

This will:
1. Create a CartPole environment
2. Initialize a PPO agent
3. Start training with default parameters
4. Log metrics to Weights & Biases

Configuration
------------

RL Research uses Hydra for configuration management. You can override default settings via command line:

.. code-block:: bash

   # Change environment
   python -m rl_research.experiments.cli env.id=LunarLander-v2

   # Modify algorithm parameters
   python -m rl_research.experiments.cli algorithm.params.learning_rate=0.0003

   # Run multiple seeds
   python -m rl_research.experiments.cli experiment.seed=1,2,3,4,5

Example Script
-------------

Here's a complete example showing common usage patterns:

.. code-block:: python

   import gymnasium as gym
   from stable_baselines3 import PPO
   from rl_research.utils.logger import ExperimentLogger
   from omegaconf import OmegaConf

   # Create configuration
   config = OmegaConf.create({
       "wandb": {
           "project": "my_project",
           "group": "my_experiment",
           "tags": ["training"]
       },
       "experiment": {
           "eval_frequency": 1000,
           "name": "ppo_cartpole"
       },
       "env": {
           "id": "CartPole-v1"
       }
   })

   # Initialize environment and logger
   env = gym.make(config.env.id)
   logger = ExperimentLogger(config, env)

   # Create agent
   model = PPO("MlpPolicy", env)

   # Get callbacks for logging
   callbacks = logger.get_callbacks()

   # Train the agent
   model.learn(
       total_timesteps=50000,
       callback=callbacks
   )

   # Save the trained model
   logger.save_model(model)

   # Clean up
   logger.finish()
   env.close()

Visualization
------------

To visualize your agent's performance:

.. code-block:: python

   from rl_research.utils.viz import plot_learning_curve

   # Plot training curves
   plot_learning_curve("runs/my_experiment")

Next Steps
---------

- Check out the :doc:`configuration` guide for detailed configuration options
- Learn about :doc:`experiments` for advanced experiment management
- See :doc:`custom_environments` to create your own environments
- Explore :doc:`custom_algorithms` to implement new algorithms

For more examples, visit the `examples directory <https://github.com/yourusername/rl-research/tree/main/examples>`_ in the repository. 