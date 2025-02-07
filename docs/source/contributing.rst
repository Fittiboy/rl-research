Contributing
===========

Thank you for your interest in contributing to RL Research! This guide will help you get started.

Development Setup
---------------

1. Fork the repository on GitHub
2. Clone your fork locally:

   .. code-block:: bash

      git clone https://github.com/yourusername/rl-research.git
      cd rl-research

3. Install development dependencies:

   .. code-block:: bash

      pip install -e ".[dev]"

4. Set up pre-commit hooks:

   .. code-block:: bash

      pre-commit install

Code Style
---------

We follow these coding standards:

* PEP 8 style guide
* Google docstring format
* Type hints for function arguments and return values
* Maximum line length of 88 characters (Black default)

Use our automated tools to ensure compliance:

.. code-block:: bash

   # Format code
   black .
   
   # Sort imports
   isort .
   
   # Check types
   mypy .
   
   # Run linter
   flake8

Testing
------

Write tests for new features:

.. code-block:: bash

   # Run tests
   pytest
   
   # Run tests with coverage
   pytest --cov=rl_research

Pull Request Process
-----------------

1. Create a new branch for your feature:

   .. code-block:: bash

      git checkout -b feature-name

2. Make your changes and commit:

   .. code-block:: bash

      git add .
      git commit -m "Description of changes"

3. Push to your fork:

   .. code-block:: bash

      git push origin feature-name

4. Open a Pull Request on GitHub

Documentation
-----------

Update documentation for new features:

1. Add docstrings to new code
2. Update API documentation if needed
3. Add examples to user guides
4. Build and test documentation:

   .. code-block:: bash

      cd docs
      make html

Code Review
---------

All submissions require review:

1. Address review comments
2. Update your branch:

   .. code-block:: bash

      git fetch origin
      git rebase origin/main
      git push -f origin feature-name

Release Process
------------

1. Update version in ``pyproject.toml``
2. Update changelog
3. Create release notes
4. Tag the release:

   .. code-block:: bash

      git tag -a v0.1.0 -m "Release v0.1.0"
      git push origin v0.1.0

Project Structure
--------------

.. code-block:: bash

   rl_research/
   ├── algorithms/        # RL algorithms
   ├── environments/      # Custom environments
   ├── experiments/       # Experiment management
   ├── utils/            # Utility functions
   └── tests/            # Test suite

   docs/                 # Documentation
   examples/             # Example scripts
   notebooks/           # Jupyter notebooks

Issue Labels
----------

* ``bug``: Bug reports
* ``enhancement``: New features
* ``documentation``: Documentation updates
* ``good first issue``: Good for newcomers
* ``help wanted``: Extra attention needed

Getting Help
----------

* Check existing issues
* Join our community chat
* Contact maintainers
* Read the documentation

Code of Conduct
-------------

Please read our `Code of Conduct <https://github.com/yourusername/rl-research/blob/main/CODE_OF_CONDUCT.md>`_
before contributing. 