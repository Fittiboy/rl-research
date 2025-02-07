# RL Research Framework

A modular framework for reinforcement learning research with integrated experiment tracking and configuration management.

## Features

- 🚀 Easy-to-use experiment management
- 📊 Integrated logging with Weights & Biases
- ⚙️ Flexible configuration system using Hydra
- 🔧 Modular architecture for custom environments and algorithms
- 📈 Built-in visualization utilities
- 🧪 Comprehensive test suite

## Project Structure
```
rl_research/
├── algorithms/          # RL algorithm implementations
│   └── custom/         # Custom algorithm implementations
├── environments/       # Environment definitions
│   ├── wrappers/      # Custom gym wrappers
│   └── custom_envs/   # Custom environment implementations
├── experiments/       # Experiment management
│   ├── configs/       # Configuration files
│   │   ├── algorithm/ # Algorithm-specific configs
│   │   ├── env/      # Environment-specific configs
│   │   └── experiment.yaml
│   ├── runs/         # Experiment run data
│   └── cli.py        # Command-line interface
├── utils/            # Utility functions
└── tests/           # Test suite
```

## Installation

1. Create and activate conda environment:
```bash
conda create -n rl-research python=3.10
conda activate rl-research
```

2. Install the package:
```bash
# For basic installation
pip install -e .

# For development installation (includes testing tools)
pip install -e ".[dev]"

# For visualization tools only
pip install -e ".[viz]"
```

3. Set up Weights & Biases:
```bash
# Set your WANDB API key
export WANDB_API_KEY=your_key_here
```

## Quick Start

1. Run a basic experiment:
```bash
python -m rl_research.experiments.cli
```

2. Run a parameter sweep:
```bash
python -m rl_research.experiments.cli algorithm.params.learning_rate=0.0001,0.0003,0.001
```

3. Run with multiple seeds:
```bash
python -m rl_research.experiments.cli experiment.seed=1,2,3,4,5
```

## Documentation

- [API Documentation](docs/api/README.md)
- [User Guides](docs/guides/README.md)
- [Contributing Guidelines](CONTRIBUTING.md)

## Development

See our [Contributing Guidelines](CONTRIBUTING.md) for information on how to contribute to the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 