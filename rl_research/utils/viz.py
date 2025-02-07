"""Visualization utilities for experiment analysis."""
from typing import List, Optional, Union, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

def set_style():
    """Set the style for all plots."""
    plt.style.use('seaborn')
    sns.set_palette("husl")
    
def plot_learning_curves(
    runs: List[Any],  # wandb.sdk.wandb_run.Run
    metric: str = "rollout/ep_rew_mean",
    window: int = 100,
    title: Optional[str] = None,
    figsize: tuple = (10, 6)
):
    """Plot learning curves from multiple runs."""
    plt.figure(figsize=figsize)
    
    for run in runs:
        # Get history and smooth
        history = pd.DataFrame(run.history())
        if metric in history.columns:
            values = history[metric].rolling(window=window).mean()
            steps = history.get('global_step', range(len(values)))
            plt.plot(steps, values, label=f"{run.config['algorithm']['name']}")
    
    plt.title(title or f"Learning Curves - {metric}")
    plt.xlabel("Steps")
    plt.ylabel(metric)
    plt.legend()
    plt.grid(True)
    
def plot_evaluation_results(
    results: Union[List[float], np.ndarray],
    title: str = "Evaluation Results",
    figsize: tuple = (10, 6)
):
    """Plot evaluation results as a box plot."""
    plt.figure(figsize=figsize)
    sns.boxplot(data=results)
    plt.title(title)
    plt.ylabel("Return")
    plt.grid(True)

def plot_environment_renders(
    frames: List[np.ndarray],
    rows: int = 2,
    cols: int = 3,
    figsize: tuple = (15, 10)
):
    """Plot a grid of environment renders."""
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten()
    
    for idx, frame in enumerate(frames[:rows*cols]):
        if len(frame.shape) == 3:
            axes[idx].imshow(frame)
        else:
            axes[idx].imshow(frame, cmap='gray')
        axes[idx].axis('off')
        axes[idx].set_title(f"Frame {idx}")
    
    plt.tight_layout()
    
def save_figure(path: str, dpi: int = 300):
    """Save the current figure."""
    plt.savefig(path, dpi=dpi, bbox_inches='tight') 