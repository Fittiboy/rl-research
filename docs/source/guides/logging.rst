Logging System
=============

RL Research provides a comprehensive logging system for tracking experiments, metrics, and artifacts.

ExperimentLogger
--------------

The main logging interface is the ``ExperimentLogger`` class:

.. code-block:: python

   from rl_research.utils.logger import ExperimentLogger

   # Initialize logger
   logger = ExperimentLogger(config, env)

Configuration
-----------

The logger is configured through the ``wandb`` section in your configuration:

.. code-block:: yaml

   wandb:
     project: my_project
     group: experiment_group
     tags: ["training", "ppo"]
     mode: online
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

Custom Metrics
~~~~~~~~~~~~

Log any custom metrics:

.. code-block:: python

   logger.log_metrics({
       "custom/metric": value,
       "custom/nested/metric": value
   })

Logging Artifacts
---------------

Models
~~~~~

Save trained models:

.. code-block:: python

   # Save model
   logger.save_model(model, name="best_model")

   # Save model with custom metadata
   logger.save_model(
       model,
       name="checkpoint",
       metadata={
           "step": 1000,
           "reward": 100.0
       }
   )

Configurations
~~~~~~~~~~~~

Configurations are automatically logged, but you can also log custom configs:

.. code-block:: python

   logger.log_config({
       "custom_config": {
           "param1": value1,
           "param2": value2
       }
   })

Media
~~~~

Log images, videos, and other media:

.. code-block:: python

   # Log image
   logger.log_image("state", state_image)

   # Log video
   logger.log_video("episode", episode_video)

Callbacks
--------

The logger provides callbacks for integration with training loops:

.. code-block:: python

   # Get callbacks
   callbacks = logger.get_callbacks()

   # Use in training
   model.learn(
       total_timesteps=50000,
       callback=callbacks
   )

Available callbacks:

* ``WandbCallback``: Logs metrics to W&B
* ``EvalCallback``: Performs periodic evaluation
* ``CheckpointCallback``: Saves model checkpoints
* ``VideoRecorderCallback``: Records videos of episodes

Custom Callbacks
~~~~~~~~~~~~~

Create custom callbacks:

.. code-block:: python

   from stable_baselines3.common.callbacks import BaseCallback

   class CustomCallback(BaseCallback):
       def __init__(self, logger):
           super().__init__()
           self.logger = logger

       def _on_step(self):
           # Log custom metrics
           self.logger.log_metrics({
               "custom": self.n_calls
           })
           return True

Offline Logging
-------------

For environments without internet access:

.. code-block:: yaml

   wandb:
     mode: offline
     dir: logs

Later sync the logs:

.. code-block:: bash

   wandb sync logs/wandb/offline-run-*

Best Practices
------------

1. **Metric Names**
   
   * Use clear, descriptive names
   * Group related metrics
   * Be consistent with naming

2. **Logging Frequency**
   
   * Don't log too frequently
   * Use appropriate step counts
   * Consider storage limitations

3. **Organization**
   
   * Use meaningful run names
   * Add relevant tags
   * Group related experiments

4. **Resource Management**
   
   * Clean up old logs
   * Monitor storage usage
   * Use appropriate logging modes

5. **Documentation**
   
   * Document custom metrics
   * Add metric descriptions
   * Note important events

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Connection Problems**
   
   * Check internet connection
   * Verify API key
   * Try offline mode

2. **Missing Data**
   
   * Check logging frequency
   * Verify metric names
   * Inspect callback setup

3. **Storage Issues**
   
   * Clean up old runs
   * Use appropriate logging frequency
   * Monitor disk usage

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check W&B status
2. Review logger configuration
3. Inspect error messages
4. Contact support team

For more information, see the `Weights & Biases documentation <https://docs.wandb.ai/>`_. 