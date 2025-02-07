"""Tests for training callbacks."""
import pytest
import numpy as np
import wandb
from examples.quickstart import QuickStartCallback

class MockLocals:
    """Mock locals dictionary for testing callbacks."""
    def __init__(self, rewards=None, dones=None):
        self.data = {
            "rewards": rewards if rewards is not None else [0.0],
            "dones": dones if dones is not None else [False]
        }
    
    def get(self, key, default=None):
        return self.data.get(key, default)

@pytest.fixture
def callback():
    """Create a QuickStartCallback instance."""
    return QuickStartCallback(verbose=0)

@pytest.fixture(autouse=True)
def wandb_init():
    """Initialize WandB in disabled mode for tests."""
    wandb.init(mode="disabled")
    yield
    if wandb.run is not None:
        wandb.finish()

class TestQuickStartCallback:
    """Test suite for QuickStartCallback."""
    
    def test_initialization(self, callback):
        """Test callback initialization."""
        assert callback.episode_rewards == []
        assert callback.episode_lengths == []
        assert callback._current_episode_reward == 0
    
    def test_on_step_accumulate_reward(self, callback):
        """Test reward accumulation during steps."""
        # Mock locals with rewards
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        
        # Test single step
        callback._on_step()
        assert callback._current_episode_reward == 1.0
        
        # Test multiple steps
        callback._on_step()
        assert callback._current_episode_reward == 2.0
    
    def test_on_step_episode_end(self, callback):
        """Test behavior when episode ends."""
        # Set up episode
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        callback.n_calls = 10
        
        # Run some steps
        for _ in range(3):
            callback._on_step()
        
        # End episode
        locals_dict.data["dones"] = [True]
        callback._on_step()
        
        # Check if episode was properly logged
        assert len(callback.episode_rewards) == 1
        assert callback.episode_rewards[0] == 4.0  # Sum of rewards
        assert len(callback.episode_lengths) == 1
    
    def test_on_step_wandb_logging(self, callback):
        """Test WandB logging functionality."""
        # Set up episode
        locals_dict = MockLocals(rewards=[1.0])
        callback.locals = locals_dict
        callback.n_calls = 20
        
        # Run steps until logging should occur
        for _ in range(10):
            locals_dict.data["dones"] = [True]
            callback._on_step()
            locals_dict.data["dones"] = [False]
        
        # Verify metrics were logged
        assert len(callback.episode_rewards) == 10
        # WandB logging would have occurred, but we can't verify the actual log
        # without mocking WandB itself
    
    def test_on_step_with_none_values(self, callback):
        """Test handling of None values in locals."""
        # Test with None rewards
        locals_dict = MockLocals(rewards=None)
        callback.locals = locals_dict
        callback._on_step()
        assert callback._current_episode_reward == 0
        
        # Test with None dones
        locals_dict = MockLocals(dones=None)
        callback.locals = locals_dict
        callback._on_step()
        # Should not raise any errors
    
    def test_on_step_with_empty_values(self, callback):
        """Test handling of empty lists in locals."""
        # Test with empty rewards
        locals_dict = MockLocals(rewards=[])
        callback.locals = locals_dict
        callback._on_step()
        assert callback._current_episode_reward == 0
        
        # Test with empty dones
        locals_dict = MockLocals(dones=[])
        callback.locals = locals_dict
        callback._on_step()
        # Should not raise any errors 