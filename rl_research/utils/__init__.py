"""Utilities for the RL research framework."""
from .registry import (
    register_algorithm,
    register_environment,
    get_algorithm,
    get_environment,
)
from .logger import setup_logging, ExperimentLogger
from .viz import (
    set_style,
    plot_learning_curves,
    plot_evaluation_results,
    plot_environment_renders,
    save_figure,
)

__all__ = [
    # Registry
    "register_algorithm",
    "register_environment",
    "get_algorithm",
    "get_environment",
    # Logging
    "setup_logging",
    "ExperimentLogger",
    # Visualization
    "set_style",
    "plot_learning_curves",
    "plot_evaluation_results",
    "plot_environment_renders",
    "save_figure",
]
