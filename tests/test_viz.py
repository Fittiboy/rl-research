"""Tests for visualization utilities."""
import pytest
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from rl_research.utils.viz import set_style, plot_learning_curves

class MockWandbRun:
    """Mock WandB run for testing."""
    def __init__(self, history_data, config):
        self.history_data = history_data
        self.config = config
    
    def history(self):
        return self.history_data

@pytest.fixture
def mock_runs():
    """Create mock runs for testing."""
    # Create sample data
    n_steps = 100
    runs = []
    
    for algo in ["PPO", "DQN"]:
        history_data = pd.DataFrame({
            "rollout/ep_rew_mean": np.random.normal(0, 1, n_steps).cumsum(),
            "global_step": range(n_steps)
        })
        
        config = {
            "algorithm": {
                "name": algo
            }
        }
        
        runs.append(MockWandbRun(history_data, config))
    
    return runs

def test_set_style():
    """Test if style setting works without errors."""
    set_style()
    assert plt.style.available is not None

def test_plot_learning_curves(mock_runs):
    """Test if learning curves can be plotted."""
    # Test with default parameters
    plot_learning_curves(mock_runs)
    fig = plt.gcf()
    assert fig is not None
    plt.close()
    
    # Test with custom parameters
    plot_learning_curves(
        runs=mock_runs,
        metric="rollout/ep_rew_mean",
        window=10,
        title="Test Plot",
        figsize=(8, 4)
    )
    fig = plt.gcf()
    assert fig is not None
    assert fig.get_size_inches().tolist() == [8, 4]
    plt.close()

def test_plot_learning_curves_missing_metric(mock_runs):
    """Test plotting with missing metrics."""
    # Create a run with missing metric
    history_data = pd.DataFrame({
        "other_metric": np.random.normal(0, 1, 100)
    })
    config = {"algorithm": {"name": "TestAlgo"}}
    mock_runs.append(MockWandbRun(history_data, config))
    
    # Should not raise an error
    plot_learning_curves(mock_runs, metric="rollout/ep_rew_mean")
    fig = plt.gcf()
    assert fig is not None
    plt.close()

def test_plot_learning_curves_empty_runs():
    """Test plotting with empty runs list."""
    plot_learning_curves([])
    fig = plt.gcf()
    assert fig is not None
    plt.close() 