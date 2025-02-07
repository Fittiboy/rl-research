Experiment Management
===================

This guide covers how to run, manage, and track experiments using RL Research.

Basic Concepts
-------------

An experiment in RL Research consists of:

1. **Environment**: The training environment (e.g., CartPole, LunarLander)
2. **Algorithm**: The RL algorithm (e.g., PPO, DQN)
3. **Configuration**: Parameters for the environment and algorithm
4. **Logging**: Metrics tracking and visualization
5. **Artifacts**: Saved models and data

Running Experiments
-----------------

Command Line Interface
~~~~~~~~~~~~~~~~~~~~

The simplest way to run an experiment is through the CLI:

.. code-block:: bash

   # Basic run
   python -m rl_research.experiments.cli

   # With custom configuration
   python -m rl_research.experiments.cli \
     experiment.name=my_experiment \
     env.id=LunarLander-v2

Python API
~~~~~~~~~

For more control, use the Python API:

.. code-block:: python

   from rl_research.experiments.cli import run_experiment
   from omegaconf import OmegaConf

   # Create configuration
   config = OmegaConf.create({
       "experiment": {
           "name": "my_experiment",
           "seed": 42
       },
       "env": {
           "id": "CartPole-v1"
       }
   })

   # Run experiment
   run_experiment(config)

Experiment Organization
---------------------

Directory Structure
~~~~~~~~~~~~~~~~~

Experiments are organized as follows:

.. code-block:: bash

   runs/
   ├── experiment_name/
   │   ├── models/
   │   │   └── best_model.zip
   │   ├── configs/
   │   │   └── config.yaml
   │   ├── logs/
   │   │   └── metrics.csv
   │   └── videos/
   │       └── evaluation.mp4
   └── ...

Experiment Tracking
-----------------

Weights & Biases Integration
~~~~~~~~~~~~~~~~~~~~~~~~~~

RL Research uses W&B for experiment tracking:

.. code-block:: python

   from rl_research.utils.logger import ExperimentLogger

   # Initialize logger
   logger = ExperimentLogger(config, env)

   # Log metrics
   logger.log_metrics({
       "reward": 100,
       "loss": 0.5
   })

   # Save model
   logger.save_model(model, name="best_model")

   # Finish logging
   logger.finish()

Tracked Metrics
~~~~~~~~~~~~~

Default metrics include:

* Episode rewards
* Episode lengths
* Learning rate
* Policy loss
* Value loss
* Explained variance

Custom metrics can be added:

.. code-block:: python

   logger.log_metrics({
       "custom_metric": value,
       "nested/metric": value
   })

Experiment Evaluation
-------------------

Evaluating Models
~~~~~~~~~~~~~~~

Evaluate trained models:

.. code-block:: python

   from rl_research.utils.evaluation import evaluate_policy

   # Evaluate policy
   mean_reward, std_reward = evaluate_policy(
       model,
       env,
       n_eval_episodes=10
   )

Recording Videos
~~~~~~~~~~~~~~

Record agent behavior:

.. code-block:: python

   from rl_research.utils.evaluation import record_video

   # Record video
   record_video(
       model,
       env,
       video_path="videos/agent.mp4",
       n_episodes=1
   )

Parameter Studies
---------------

Grid Search
~~~~~~~~~~

Run grid search over parameters:

.. code-block:: bash

   python -m rl_research.experiments.cli --multirun \
     algorithm.params.learning_rate=1e-4,3e-4,1e-3 \
     algorithm.params.batch_size=32,64,128

Random Search
~~~~~~~~~~~

Perform random search:

.. code-block:: python

   import numpy as np
   from rl_research.utils.search import random_search

   # Define parameter space
   param_space = {
       "learning_rate": lambda: np.random.loguniform(1e-5, 1e-2),
       "batch_size": lambda: np.random.choice([32, 64, 128])
   }

   # Run random search
   best_params = random_search(
       param_space,
       n_trials=10,
       evaluation_fn=run_experiment
   )

Best Practices
------------

1. **Reproducibility**
   
   * Set random seeds
   * Version control configurations
   * Document environment details

2. **Organization**
   
   * Use meaningful experiment names
   * Group related experiments
   * Clean up old experiments

3. **Resource Management**
   
   * Monitor GPU usage
   * Clean up environments
   * Use appropriate batch sizes

4. **Documentation**
   
   * Document experiment purpose
   * Note important findings
   * Track failed experiments

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Out of Memory**
   
   * Reduce batch size
   * Clean up old models
   * Monitor memory usage

2. **Poor Performance**
   
   * Check hyperparameters
   * Verify environment setup
   * Inspect learning curves

3. **Crashes**
   
   * Check error messages
   * Verify dependencies
   * Monitor system resources

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check the logs
2. Review configurations
3. Search GitHub issues
4. Create detailed bug reports 