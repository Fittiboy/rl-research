"""Test script for WandB setup and authentication."""
import os
import wandb

def test_wandb_auth():
    """Test WandB authentication and setup."""
    print("\nTesting WandB Setup")
    print("-" * 50)
    
    # Check if API key is set
    api_key = os.environ.get("WANDB_API_KEY")
    if api_key:
        print("✓ WANDB_API_KEY environment variable is set")
    else:
        print("✗ WANDB_API_KEY environment variable is not set")
        print("  Please set it using: export WANDB_API_KEY=your_key_here")
    
    # Try to authenticate
    try:
        wandb.login()
        print("✓ Successfully authenticated with WandB")
        
        # Test creating a run
        run = wandb.init(project="test-project", name="test-run")
        print("✓ Successfully created a test run")
        
        # Log some test metrics
        wandb.log({"test_metric": 1.0})
        print("✓ Successfully logged test metrics")
        
        # Cleanup
        run.finish()
        print("✓ Successfully finished the test run")
        
    except wandb.errors.UsageError:
        print("✗ Authentication failed")
        print("  Please check your API key and internet connection")
    except Exception as e:
        print(f"✗ Error during testing: {str(e)}")
    
    print("\nSetup Guide:")
    print("1. Create an account at https://wandb.ai/")
    print("2. Get your API key from https://wandb.ai/settings")
    print("3. Set up your key using one of these methods:")
    print("   a. Export environment variable:")
    print("      export WANDB_API_KEY=your_key_here")
    print("   b. Create config file:")
    print("      mkdir -p ~/.config/wandb")
    print("      echo 'api_key: your_key_here' > ~/.config/wandb/settings")
    print("   c. Login via CLI:")
    print("      wandb login")

if __name__ == "__main__":
    test_wandb_auth() 