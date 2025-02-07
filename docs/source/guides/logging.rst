Logging System
=============

RL Research provides a comprehensive logging system for tracking experiments, metrics, and artifacts using Weights & Biases (WandB).

ExperimentLogger
--------------

The main logging interface is the ``ExperimentLogger`` class:

.. code-block:: python

   from rl_research.utils.logger import setup_logging

   # Initialize logger with config and environment
   logger = setup_logging(config, env)

Configuration
-----------

The logger is configured through the ``wandb`` section in your configuration:

.. code-block:: yaml

   wandb:
     project: my_project
     group: experiment_group
     tags: ["training"]
     mode: online  # or "disabled" for offline use
     dir: ${hydra:runtime.output_dir}

Logging Metrics
-------------

Basic Metrics
~~~~~~~~~~~

Log scalar values:

.. code-block:: python

   logger.log_metrics({
       "reward": 100.0,
       "loss": 0.5
   })

Nested Metrics
~~~~~~~~~~~~

Group related metrics using slashes:

.. code-block:: python

   logger.log_metrics({
       "training/reward": 100.0,
       "training/loss": 0.5,
       "eval/reward": 150.0
   })

Training Callbacks
---------------

Get Callbacks
~~~~~~~~~~~

The logger provides callbacks for training:

.. code-block:: python

   # Get training callbacks
   callbacks = logger.get_callbacks()

   # Use in training
   model.learn(
       total_timesteps=50000,
       callback=callbacks
   )

Available Callbacks
~~~~~~~~~~~~~~~~

The logger automatically sets up:

1. ``EpisodeLoggingCallback``:
   - Logs training metrics
   - Tracks episode rewards and lengths
   - Calculates rolling statistics

2. ``VideoEvalCallback``:
   - Records evaluation videos
   - Saves frames in correct format
   - Logs video metrics to WandB

Model Management
-------------

Saving Models
~~~~~~~~~~~

Save trained models with automatic WandB logging:

.. code-block:: python

   # Save final model
   logger.save_model(model)

   # Save with custom name
   logger.save_model(model, name="checkpoint_1000")

   # Get final evaluation score
   final_score = logger.get_final_score()

Cleanup
------

Always clean up the logger when done:

.. code-block:: python

   try:
       # Training code here
       logger.save_model(model)
   finally:
       logger.finish()

Example Usage
-----------

Complete example of logger usage:

.. code-block:: python

   from rl_research.utils.logger import setup_logging
   import gymnasium as gym
   from stable_baselines3 import PPO
   from omegaconf import OmegaConf

   # Create configuration
   config = OmegaConf.create({
       "wandb": {
           "project": "my_project",
           "group": "experiment_1",
           "tags": ["training"],
           "mode": "online"
       },
       "experiment": {
           "eval_frequency": 1000
       },
       "env": {
           "id": "CartPole-v1",
           "params": {}
       },
       "video": {
           "fps": 30,
           "num_episodes": 2
       }
   })

   # Setup environment and logger
   env = gym.make(config.env.id)
   logger = setup_logging(config, env)

   try:
       # Create and train model
       model = PPO("MlpPolicy", env)
       model.learn(
           total_timesteps=50000,
           callback=logger.get_callbacks()
       )

       # Save final model and get score
       logger.save_model(model)
       final_score = logger.get_final_score()
       print(f"Final evaluation score: {final_score}")

   finally:
       # Cleanup
       env.close()
       logger.finish()

Best Practices
------------

1. **Configuration**
   - Use meaningful project and group names
   - Add relevant tags
   - Set appropriate save directories

2. **Metric Logging**
   - Use consistent naming conventions
   - Group related metrics
   - Keep metrics organized by category

3. **Resource Management**
   - Clean up logger with finish()
   - Close environments
   - Monitor storage usage

4. **Error Handling**
   - Use try/finally blocks
   - Handle interruptions gracefully
   - Verify logging state

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Connection Problems**
   - Check internet connection
   - Verify WandB API key
   - Try offline mode

2. **Missing Data**
   - Check metric names
   - Verify callback setup
   - Ensure proper initialization

3. **Storage Issues**
   - Monitor disk usage
   - Clean up old runs
   - Use appropriate logging frequency

For more information, see the `Weights & Biases documentation <https://docs.wandb.ai/>`_. 