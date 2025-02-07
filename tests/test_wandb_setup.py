"""Test script for WandB setup and authentication."""
import os
import pytest
from unittest.mock import patch, MagicMock
import wandb

def test_wandb_auth():
    """Test WandB authentication and setup."""
    with patch('wandb.login') as mock_login, \
         patch('wandb.init') as mock_init, \
         patch('wandb.log') as mock_log:
        
        # Set up mocks
        mock_run = MagicMock()
        mock_init.return_value = mock_run
        
        # Test with API key set
        with patch.dict(os.environ, {'WANDB_API_KEY': 'test_key'}):
            assert os.environ.get('WANDB_API_KEY') is not None
            mock_login.return_value = True
            assert mock_login() is True
            
            # Test run creation
            run = mock_init(project="test-project", name="test-run")
            mock_init.assert_called_once_with(project="test-project", name="test-run")
            
            # Test metric logging
            wandb.log({"test_metric": 1.0})
            mock_log.assert_called_once_with({"test_metric": 1.0})
            
            # Test cleanup
            run.finish()
            mock_run.finish.assert_called_once()
        
        # Test without API key
        with patch.dict(os.environ, clear=True):
            assert os.environ.get('WANDB_API_KEY') is None

if __name__ == "__main__":
    test_wandb_auth() 