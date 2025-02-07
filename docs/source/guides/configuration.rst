Configuration System
==================

RL Research uses Hydra for configuration management, providing a flexible and powerful way to configure experiments.

Configuration Structure
---------------------

The configuration system is organized hierarchically:

.. code-block:: bash

   rl_research/experiments/configs/
   ├── algorithm/
   │   ├── ppo.yaml
   │   ├── dqn.yaml
   │   └── sac.yaml
   ├── env/
   │   ├── cartpole.yaml
   │   ├── lunarlander.yaml
   │   └── custom_envs.yaml
   ├── experiment.yaml
   └── examples/
       └── quickstart.yaml

Base Configuration
----------------

The base configuration is defined in ``experiment.yaml``:

.. code-block:: yaml

   # Default experiment configuration
   wandb:
     project: rl_research
     group: ${experiment.name}
     tags: []
     mode: online
     dir: ${hydra:runtime.output_dir}

   experiment:
     name: default
     seed: 42
     eval_frequency: 1000
     save_frequency: 10000

   env:
     id: CartPole-v1
     max_episode_steps: 500

   algorithm:
     name: PPO
     policy: MlpPolicy
     params:
       learning_rate: 3.0e-4
       n_steps: 2048
       batch_size: 64
       n_epochs: 10
       gamma: 0.99
       gae_lambda: 0.95
       clip_range: 0.2
       normalize_advantage: true

Command-line Configuration
------------------------

Override configuration values via command line:

.. code-block:: bash

   # Single value
   python -m rl_research.experiments.cli algorithm.params.learning_rate=0.0001

   # Multiple values (sweep)
   python -m rl_research.experiments.cli algorithm.params.learning_rate=0.0001,0.0003,0.001

   # Multiple parameters
   python -m rl_research.experiments.cli \
     algorithm.params.learning_rate=0.0001 \
     algorithm.params.batch_size=128

Environment Configuration
-----------------------

Configure environments in ``env/`` directory:

.. code-block:: yaml

   # env/lunarlander.yaml
   id: LunarLander-v2
   max_episode_steps: 1000
   reward_threshold: 200

Algorithm Configuration
---------------------

Algorithm configurations in ``algorithm/`` directory:

.. code-block:: yaml

   # algorithm/ppo.yaml
   name: PPO
   policy: MlpPolicy
   params:
     learning_rate: 3.0e-4
     n_steps: 2048
     batch_size: 64
     n_epochs: 10
     gamma: 0.99
     gae_lambda: 0.95
     clip_range: 0.2
     normalize_advantage: true

Custom Configurations
-------------------

Create custom configurations by combining existing ones:

.. code-block:: yaml

   # configs/my_experiment.yaml
   defaults:
     - experiment
     - algorithm: ppo
     - env: lunarlander
     - _self_

   experiment:
     name: ppo_lunarlander
     seed: 42

   algorithm:
     params:
       learning_rate: 0.0001
       batch_size: 128

Using Configuration in Code
-------------------------

Access configuration in your code:

.. code-block:: python

   from omegaconf import DictConfig, OmegaConf

   def run_experiment(cfg: DictConfig):
       # Access configuration values
       env_id = cfg.env.id
       lr = cfg.algorithm.params.learning_rate
       
       # Print configuration
       print(OmegaConf.to_yaml(cfg))

Configuration Best Practices
--------------------------

1. **Version Control**
   
   * Keep all configurations in version control
   * Document changes in configuration files

2. **Naming Conventions**
   
   * Use descriptive names for configuration files
   * Follow consistent naming patterns

3. **Documentation**
   
   * Comment complex configuration options
   * Keep a changelog for major configuration changes

4. **Organization**
   
   * Group related parameters together
   * Use hierarchical structure for clarity

5. **Validation**
   
   * Add parameter validation where possible
   * Use type hints in configuration classes

Advanced Features
---------------

1. **Configuration Groups**

   Create configuration groups for different experiment types:

   .. code-block:: bash

      configs/
      ├── experiment/
      │   ├── training.yaml
      │   └── evaluation.yaml
      └── config.yaml

2. **Interpolation**

   Use value interpolation in configurations:

   .. code-block:: yaml

      wandb:
         group: ${experiment.name}_${env.id}
         dir: ${hydra:runtime.output_dir}

3. **Multirun**

   Run multiple configurations:

   .. code-block:: bash

      python -m rl_research.experiments.cli --multirun \
         algorithm=ppo,dqn \
         env=cartpole,lunarlander

For more details, see the `Hydra documentation <https://hydra.cc/docs/intro/>`_. 