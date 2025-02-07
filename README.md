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
conda create -n rl-research python=3.8
conda activate rl-research
```

2. Install the package:
```bash
# Install in development mode
pip install -e .
```

3. Set up Weights & Biases:
```bash
# Set your WANDB API key
export WANDB_API_KEY=your_key_here
```

### Dependencies

The framework requires the following major dependencies (automatically installed):

#### Reinforcement Learning
- `gymnasium[all]>=0.29.1`: Core RL environments
- `stable-baselines3[extra]>=2.2.1`: RL algorithms
- `ale-py>=0.8.0`: Atari environments
- `box2d-py>=2.3.5`: Box2D environments
- `autorom[accept-rom-license]>=0.6.1`: ROM management for Atari

#### Deep Learning
- `torch>=2.1.0`: Deep learning backend

#### Experiment Management
- `wandb>=0.16.0`: Experiment tracking
- `hydra-core>=1.3.2`: Configuration management
- `omegaconf>=2.3.0`: Configuration system
- `PyYAML>=6.0.1`: YAML file support
- `tensorboard>=2.15.0`: Training visualization

#### Visualization
- `matplotlib>=3.8.0`: Plotting utilities
- `seaborn>=0.13.0`: Statistical visualization
- `opencv-python>=4.8.0`: Video processing

#### Testing
- `pytest>=7.0.0`: Testing framework

### Package Data

The package includes:
- All YAML configuration files
- Excludes runtime directories: `wandb/`, `outputs/`, `models/`, `logs/`

## Quick Start

1. Run a basic experiment:
```bash
python -m rl_research.experiments.cli
```

2. Run a parameter sweep:
```bash
python -m rl_research.experiments.cli -m algorithm.params.learning_rate=0.0001,0.0003,0.001
```

3. Run with multiple seeds:
```bash
python -m rl_research.experiments.cli experiment.seed=1,2,3,4,5
```

4. Run with specific algorithm and environment:
```bash
python -m rl_research.experiments.cli algorithm=ppo env=cartpole
```

## Documentation

The documentation is built using Sphinx and can be found in the `docs/` directory:

- [Installation Guide](docs/source/guides/installation.rst)
- [Quick Start Guide](docs/source/guides/quickstart.rst)
- [Configuration Guide](docs/source/guides/configuration.rst)
- [Experiments Guide](docs/source/guides/experiments.rst)
- [Custom Environments Guide](docs/source/guides/custom_environments.rst)
- [Custom Algorithms Guide](docs/source/guides/custom_algorithms.rst)
- [Logging Guide](docs/source/guides/logging.rst)
- [Visualization Guide](docs/source/guides/visualization.rst)

To build the documentation locally:

```bash
cd docs
make html
```

Then open `docs/build/html/index.html` in your browser.

## Testing

Run the test suite to ensure everything is working correctly:

```bash
pytest tests/
```

The test suite covers:
- Core functionality
- Environment wrappers
- Algorithm implementations
- Visualization utilities
- Logging and experiment tracking

## Development

See our [Contributing Guidelines](docs/source/contributing.rst) for information on how to contribute to the project.

## License

This project is licensed under the MIT License - see the LICENSE file for details. 