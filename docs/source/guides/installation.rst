Installation Guide
=================

This guide will help you set up the RL Research framework on your system.

Prerequisites
------------

Before installing RL Research, ensure you have the following prerequisites:

* Python 3.8 or higher
* pip (Python package installer)
* git (for version control)
* conda (recommended for environment management)

Basic Installation
----------------

1. Create and activate a conda environment:

   .. code-block:: bash

      conda create -n rl-research python=3.8
      conda activate rl-research

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/rl-research.git
      cd rl-research

3. Install the package:

   .. code-block:: bash

      pip install -e .

Core Dependencies
---------------

The framework includes the following major dependencies (automatically installed):

* **Reinforcement Learning**:
   * ``gymnasium[all]>=0.29.1``: Core RL environments
   * ``stable-baselines3[extra]>=2.2.1``: RL algorithms
   * ``ale-py>=0.8.0``: Atari environments
   * ``box2d-py>=2.3.5``: Box2D environments
   * ``autorom[accept-rom-license]>=0.6.1``: ROM management for Atari

* **Deep Learning**:
   * ``torch>=2.1.0``: Deep learning backend

* **Experiment Management**:
   * ``wandb>=0.16.0``: Experiment tracking
   * ``hydra-core>=1.3.2``: Configuration management
   * ``omegaconf>=2.3.0``: Configuration system
   * ``PyYAML>=6.0.1``: YAML file support
   * ``tensorboard>=2.15.0``: Training visualization

* **Visualization**:
   * ``matplotlib>=3.8.0``: Plotting utilities
   * ``seaborn>=0.13.0``: Statistical visualization
   * ``opencv-python>=4.8.0``: Video processing

* **Testing**:
   * ``pytest>=7.0.0``: Testing framework

Package Data
-----------

The package includes:

* All YAML configuration files
* Excludes runtime directories:
   * ``wandb/``
   * ``outputs/``
   * ``models/``
   * ``logs/``

Weights & Biases Setup
--------------------

RL Research uses Weights & Biases for experiment tracking. To set it up:

1. Create a free account at `wandb.ai <https://wandb.ai>`_ if you haven't already

2. Log in to your wandb account:

   .. code-block:: bash

      wandb login

3. Set your API key:

   .. code-block:: bash

      export WANDB_API_KEY=your_key_here

Optional Dependencies
-------------------

Depending on your needs, you might want to install additional packages:

* For PyTorch with CUDA support:

  .. code-block:: bash

     conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia

Troubleshooting
--------------

Common Issues
~~~~~~~~~~~~

1. CUDA not found
   
   Make sure you have NVIDIA drivers installed and CUDA toolkit is properly set up.

2. Package conflicts

   Try creating a fresh conda environment and installing dependencies one by one.

3. ImportError

   Ensure you're in the correct conda environment:

   .. code-block:: bash

      conda activate rl-research

4. Atari ROM issues

   If you encounter issues with Atari environments:

   .. code-block:: bash

      # Install AutoROM
      pip install autorom[accept-rom-license]
      # Download Atari ROMs
      AutoROM --accept-license

Getting Help
~~~~~~~~~~~

If you encounter any issues:

1. Check the `GitHub Issues <https://github.com/yourusername/rl-research/issues>`_
2. Create a new issue with:
   * Your system information
   * Error message
   * Steps to reproduce
   * What you've tried 