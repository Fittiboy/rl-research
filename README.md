# RL Research Framework

A modular framework for reinforcement learning research with integrated experiment tracking and configuration management.

## Project Structure
```
rl_research/
├── algorithms/
│   └── custom/
├── environments/
│   ├── wrappers/
│   └── custom_envs/
├── experiments/
│   ├── configs/
│   │   ├── algorithm/
│   │   ├── env/
│   │   └── experiment.yaml
│   ├── runs/
│   └── cli.py
├── utils/
└── tests/
```

## Setup

1. Create and activate conda environment:
```bash
conda create -n rl-research python=3.10
conda activate rl-research
```

2. Install the package in development mode:
```bash
pip install -e .
```

3. Set up Weights & Biases:
```bash
# Set your WANDB API key
export WANDB_API_KEY=your_key_here
```

## Running Experiments

Basic experiment:
```bash
python -m experiments.cli
```

Parameter sweep:
```bash
python -m experiments.cli algorithm.params.learning_rate=0.0001,0.0003,0.001
```

Multiple seeds:
```bash
python -m experiments.cli experiment.seed=1,2,3,4,5
``` 