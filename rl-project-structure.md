# RL Research Repository - Experiment Workflow

## Refined Project Structure
```
rl_research/
├── algorithms/
│   └── custom/
│       └── your_algorithm/
│           ├── models.py
│           ├── policy.py
│           └── algorithm.py
├── environments/
│   ├── wrappers/
│   └── custom_envs/
├── experiments/
│   ├── configs/
│   │   ├── algorithm/           # Algorithm-specific configs
│   │   │   ├── ppo.yaml
│   │   │   └── custom_algo.yaml
│   │   ├── env/                 # Environment configs
│   │   │   ├── cartpole.yaml
│   │   │   └── custom_env.yaml
│   │   └── experiment.yaml      # Base config
│   ├── runs/                    # Automatically organized by wandb
│   └── cli.py                   # Command-line interface
├── utils/
│   ├── registry.py              # Algorithm/env registration
│   ├── logger.py                # Custom logging
│   └── viz.py                   # Visualization tools
├── tests/
├── pyproject.toml               # Project dependencies
└── README.md

```

## Complete Experiment Workflow

### 1. Configuration Setup

```yaml
# configs/experiment.yaml
defaults:
  - algorithm: ppo         # Choose algorithm config
  - env: cartpole         # Choose environment config
  - _self_

experiment:
  name: ${algorithm.name}_${env.name}
  seed: 42
  total_timesteps: 1_000_000
  eval_frequency: 10_000
  
wandb:
  project: "rl_research"
  group: "${experiment.name}"
  tags: []
```

```yaml
# configs/algorithm/ppo.yaml
name: ppo
type: "stable_baselines3"  # or "custom"
params:
  learning_rate: 0.0003
  n_steps: 2048
  batch_size: 64
  n_epochs: 10
  gamma: 0.99
  gae_lambda: 0.95
```

### 2. Command-Line Interface (experiments/cli.py)
```python
import hydra
from omegaconf import DictConfig
import wandb
from utils.registry import get_algorithm, get_environment
from utils.logger import setup_logging

@hydra.main(config_path="configs", config_name="experiment")
def run(cfg: DictConfig):
    """Main entry point for experiments."""
    # Setup logging and experiment tracking
    logger = setup_logging(cfg)
    run = wandb.init(
        project=cfg.wandb.project,
        group=cfg.wandb.group,
        config=dict(cfg),
        tags=cfg.wandb.tags,
    )
    
    # Create environment
    env = get_environment(cfg.env)
    
    # Initialize algorithm
    algo = get_algorithm(cfg.algorithm, env)
    
    # Training loop with automatic logging
    algo.learn(
        total_timesteps=cfg.experiment.total_timesteps,
        callback=logger.get_callback(),
    )
    
    # Save final model and configs
    algo.save(f"models/{wandb.run.id}")
    
if __name__ == "__main__":
    run()
```

### 3. Algorithm Registry (utils/registry.py)
```python
from stable_baselines3 import PPO, SAC, DQN
from typing import Dict, Type, Any
import gymnasium as gym

ALGORITHM_REGISTRY = {
    "stable_baselines3": {
        "ppo": PPO,
        "sac": SAC,
        "dqn": DQN,
    },
    "custom": {
        # Your custom algorithms here
    }
}

def get_algorithm(config: Dict[str, Any], env: gym.Env):
    """Get algorithm instance from config."""
    algo_type = config.type
    algo_name = config.name
    
    if algo_type not in ALGORITHM_REGISTRY:
        raise ValueError(f"Unknown algorithm type: {algo_type}")
    
    if algo_name not in ALGORITHM_REGISTRY[algo_type]:
        raise ValueError(f"Unknown algorithm: {algo_name}")
    
    algo_class = ALGORITHM_REGISTRY[algo_type][algo_name]
    return algo_class(
        policy="MlpPolicy",
        env=env,
        **config.params
    )
```

### 4. Custom Logger (utils/logger.py)
```python
from stable_baselines3.common.callbacks import EvalCallback
from wandb.integration.sb3 import WandbCallback

class Logger:
    def __init__(self, config):
        self.config = config
        
    def get_callback(self):
        """Get composite callback for training."""
        callbacks = [
            WandbCallback(
                gradient_save_freq=100,
                model_save_path=f"models/{wandb.run.id}"
            ),
            EvalCallback(
                eval_env=self.env,
                eval_freq=self.config.experiment.eval_frequency,
                log_path=f"eval/{wandb.run.id}",
            ),
        ]
        return callbacks

def setup_logging(config):
    """Initialize logging for experiment."""
    return Logger(config)
```

## Running Experiments

1. **Single Experiment**
```bash
# From project root
python -m experiments.cli
```

2. **Parameter Sweep**
```bash
# Sweep over learning rates
python -m experiments.cli algorithm.params.learning_rate=0.0001,0.0003,0.001
```

3. **Multiple Seeds**
```bash
# Run with different seeds
python -m experiments.cli experiment.seed=1,2,3,4,5
```

## Best Practices for Experiments

1. **Experiment Organization**
   - Each experiment gets unique wandb run ID
   - All artifacts automatically saved under run ID
   - Configs saved with experiments for reproducibility

2. **Results Analysis**
```python
# Example analysis script
import wandb
api = wandb.Api()

# Get experiment runs
runs = api.runs(
    "your-project/rl_research",
    filters={
        "group": "ppo_cartpole",
    }
)

# Analyze results
for run in runs:
    # Access metrics
    metrics = run.history()
    # Access configs
    config = run.config
    
    # Your analysis here
```

3. **Visualization**
```python
# utils/viz.py
import matplotlib.pyplot as plt
import seaborn as sns

def plot_learning_curves(runs, metric="episode_reward"):
    plt.figure(figsize=(10, 6))
    for run in runs:
        sns.lineplot(
            data=run.history()[metric],
            label=f"{run.config['algorithm']['name']}"
        )
    plt.title("Learning Curves")
    plt.xlabel("Timesteps")
    plt.ylabel(metric)
    plt.show()
```

## Setup

1. **Install Package**
```bash
# From project root
pip install -e .
```

2. **Environment Variables**
```bash
# .env
WANDB_API_KEY=your_key_here
```

This structure provides:
- Clear experiment entry points
- Automatic logging and organization
- Easy parameter sweeps
- Reproducible configs
- Simple results analysis

The key improvements from the previous version:
1. Clearer separation of configs
2. Streamlined CLI interface
3. Better algorithm registry
4. Improved logging setup
5. More organized experiment tracking
