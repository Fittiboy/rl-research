Installation Guide
=================

This guide will help you set up the RL Research framework on your system.

Prerequisites
------------

Before installing RL Research, ensure you have the following prerequisites:

* Python 3.10 or higher
* pip (Python package installer)
* git (for version control)
* conda (recommended for environment management)

Basic Installation
----------------

1. Create and activate a conda environment:

   .. code-block:: bash

      conda create -n rl-research python=3.10
      conda activate rl-research

2. Clone the repository:

   .. code-block:: bash

      git clone https://github.com/yourusername/rl-research.git
      cd rl-research

3. Install the package:

   .. code-block:: bash

      # For basic installation
      pip install -e .

      # For development installation (includes testing tools)
      pip install -e ".[dev]"

      # For visualization tools only
      pip install -e ".[viz]"

Development Installation
----------------------

For development, you'll want to install additional dependencies:

.. code-block:: bash

   pip install -e ".[dev]"

This will install:

* pytest (for running tests)
* black (for code formatting)
* isort (for import sorting)
* flake8 (for linting)
* mypy (for type checking)

Weights & Biases Setup
--------------------

RL Research uses Weights & Biases for experiment tracking. To set it up:

1. Install wandb:

   .. code-block:: bash

      pip install wandb

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

* For visualization tools:

  .. code-block:: bash

     pip install -e ".[viz]"

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

Getting Help
~~~~~~~~~~~

If you encounter any issues:

1. Check the `GitHub Issues <https://github.com/yourusername/rl-research/issues>`_
2. Create a new issue with:
   * Your system information
   * Error message
   * Steps to reproduce
   * What you've tried 