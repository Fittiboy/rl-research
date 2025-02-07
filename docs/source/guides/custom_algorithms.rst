Custom Algorithms
================

This guide explains how to implement custom reinforcement learning algorithms in RL Research.

Basic Structure
-------------

Custom algorithms should follow the Stable-Baselines3 interface:

.. code-block:: python

   from stable_baselines3.common.base_class import BaseAlgorithm
   import torch as th
   import numpy as np

   class CustomAlgorithm(BaseAlgorithm):
       """Custom RL Algorithm."""
       
       def __init__(
           self,
           policy,
           env,
           learning_rate=3e-4,
           batch_size=64,
           device="auto",
           **kwargs
       ):
           super().__init__(
               policy=policy,
               env=env,
               learning_rate=learning_rate,
               device=device,
               **kwargs
           )
           
           self.batch_size = batch_size
           self._setup_model()
       
       def _setup_model(self):
           """Initialize policy and other components."""
           self.policy = self.policy_class(
               observation_space=self.observation_space,
               action_space=self.action_space,
               lr_schedule=self.lr_schedule,
               **self.policy_kwargs
           )
       
       def train(self):
           """Update policy using collected data."""
           # Implement training logic
           pass
       
       def learn(
           self,
           total_timesteps,
           callback=None,
           log_interval=1,
           tb_log_name="run",
           reset_num_timesteps=True,
           progress_bar=False,
       ):
           """Train the agent."""
           return super().learn(
               total_timesteps=total_timesteps,
               callback=callback,
               log_interval=log_interval,
               tb_log_name=tb_log_name,
               reset_num_timesteps=reset_num_timesteps,
               progress_bar=progress_bar,
           )

Example Implementation
-------------------

Here's an example implementation of a simple policy gradient algorithm:

.. code-block:: python

   import torch as th
   import torch.nn as nn
   import torch.optim as optim
   from stable_baselines3.common.policies import ActorCriticPolicy
   from stable_baselines3.common.base_class import BaseAlgorithm

   class SimplePG(BaseAlgorithm):
       """Simple Policy Gradient algorithm."""
       
       def __init__(
           self,
           policy,
           env,
           learning_rate=3e-4,
           n_steps=2048,
           gamma=0.99,
           device="auto",
           **kwargs
       ):
           super().__init__(
               policy=policy,
               env=env,
               learning_rate=learning_rate,
               device=device,
               **kwargs
           )
           
           self.n_steps = n_steps
           self.gamma = gamma
           self._setup_model()
       
       def _setup_model(self):
           """Initialize policy and optimizer."""
           self.policy = self.policy_class(
               observation_space=self.observation_space,
               action_space=self.action_space,
               lr_schedule=self.lr_schedule,
           )
           
           self.optimizer = optim.Adam(
               self.policy.parameters(),
               lr=self.learning_rate
           )
       
       def collect_rollouts(self):
           """Collect experience."""
           observations = []
           actions = []
           rewards = []
           
           obs, _ = self.env.reset()
           
           for _ in range(self.n_steps):
               # Get action
               action, _ = self.policy.forward(obs)
               
               # Execute action
               next_obs, reward, terminated, truncated, _ = self.env.step(action)
               
               # Store experience
               observations.append(obs)
               actions.append(action)
               rewards.append(reward)
               
               obs = next_obs
               
               if terminated or truncated:
                   obs, _ = self.env.reset()
           
           return observations, actions, rewards
       
       def compute_returns(self, rewards):
           """Compute discounted returns."""
           returns = []
           G = 0
           
           for r in reversed(rewards):
               G = r + self.gamma * G
               returns.insert(0, G)
           
           returns = th.tensor(returns, device=self.device)
           returns = (returns - returns.mean()) / (returns.std() + 1e-8)
           
           return returns
       
       def train(self):
           """Update policy."""
           # Collect experience
           obs, actions, rewards = self.collect_rollouts()
           
           # Convert to tensors
           obs = th.tensor(np.array(obs), device=self.device)
           actions = th.tensor(np.array(actions), device=self.device)
           returns = self.compute_returns(rewards)
           
           # Get action log probabilities
           action_logprobs = self.policy.evaluate_actions(
               obs,
               actions
           )[0]
           
           # Compute loss
           loss = -(action_logprobs * returns).mean()
           
           # Update policy
           self.optimizer.zero_grad()
           loss.backward()
           self.optimizer.step()
           
           return {
               "policy_loss": loss.item(),
               "mean_return": np.mean(returns.cpu().numpy())
           }

Custom Policies
-------------

Implement custom neural network policies:

.. code-block:: python

   import torch.nn as nn
   from stable_baselines3.common.policies import BasePolicy

   class CustomPolicy(BasePolicy):
       """Custom neural network policy."""
       
       def __init__(
           self,
           observation_space,
           action_space,
           lr_schedule,
           net_arch=[64, 64],
           activation_fn=nn.Tanh,
       ):
           super().__init__(
               observation_space,
               action_space,
               features_extractor_class=None,
               normalize_images=True,
           )
           
           self.net_arch = net_arch
           self.activation_fn = activation_fn
           
           self._build()
       
       def _build(self):
           """Build neural network."""
           # Policy network
           policy_net = []
           
           # Input layer
           policy_net.append(nn.Linear(
               self.observation_space.shape[0],
               self.net_arch[0]
           ))
           policy_net.append(self.activation_fn())
           
           # Hidden layers
           for i in range(len(self.net_arch)-1):
               policy_net.append(nn.Linear(
                   self.net_arch[i],
                   self.net_arch[i+1]
               ))
               policy_net.append(self.activation_fn())
           
           # Output layer
           policy_net.append(nn.Linear(
               self.net_arch[-1],
               self.action_space.n
           ))
           
           self.policy_net = nn.Sequential(*policy_net)
       
       def forward(self, obs):
           """Forward pass."""
           # Convert observation to tensor
           obs = th.as_tensor(obs).float()
           
           # Get action logits
           logits = self.policy_net(obs)
           
           # Sample action
           action = th.argmax(logits).item()
           
           return action, None
       
       def evaluate_actions(self, obs, actions):
           """Compute action log probabilities."""
           logits = self.policy_net(obs)
           log_probs = nn.functional.log_softmax(logits, dim=-1)
           
           return (
               th.gather(log_probs, 1, actions.unsqueeze(1)).squeeze(),
               None,
               None
           )

Using Custom Algorithms
--------------------

Use your custom algorithm:

.. code-block:: python

   # Create environment
   env = gym.make("CartPole-v1")

   # Initialize algorithm
   model = SimplePG(
       policy=CustomPolicy,
       env=env,
       learning_rate=3e-4,
       n_steps=2048,
       gamma=0.99
   )

   # Train
   model.learn(total_timesteps=50000)

Advanced Features
--------------

Vectorized Training
~~~~~~~~~~~~~~~~

Support parallel training:

.. code-block:: python

   from stable_baselines3.common.vec_env import SubprocVecEnv

   def make_env():
       return lambda: gym.make("CartPole-v1")

   # Create vectorized environment
   env = SubprocVecEnv([make_env() for _ in range(4)])

   # Train with vectorized environment
   model = SimplePG(policy=CustomPolicy, env=env)
   model.learn(total_timesteps=50000)

Custom Exploration
~~~~~~~~~~~~~~~

Implement custom exploration strategies:

.. code-block:: python

   class EpsilonGreedyPolicy(BasePolicy):
       def __init__(self, *args, epsilon=0.1, **kwargs):
           super().__init__(*args, **kwargs)
           self.epsilon = epsilon
       
       def forward(self, obs):
           if np.random.random() < self.epsilon:
               action = self.action_space.sample()
           else:
               action = super().forward(obs)[0]
           
           return action, None

Best Practices
------------

1. **Code Organization**
   
   * Use clear class hierarchy
   * Separate policy and algorithm
   * Follow existing patterns

2. **Performance**
   
   * Vectorize operations
   * Use GPU when available
   * Profile critical sections

3. **Debugging**
   
   * Add informative logging
   * Monitor gradients
   * Track key metrics

4. **Testing**
   
   * Unit test components
   * Integration test training
   * Benchmark performance

Testing Algorithms
---------------

Basic Tests
~~~~~~~~~

Test algorithm functionality:

.. code-block:: python

   def test_simple_pg():
       # Create environment
       env = gym.make("CartPole-v1")
       
       # Initialize algorithm
       model = SimplePG(
           policy=CustomPolicy,
           env=env
       )
       
       # Test training step
       model.train()
       
       # Test prediction
       obs, _ = env.reset()
       action, _ = model.predict(obs)
       
       assert env.action_space.contains(action)

Learning Tests
~~~~~~~~~~~~

Test learning performance:

.. code-block:: python

   def test_learning():
       env = gym.make("CartPole-v1")
       model = SimplePG(policy=CustomPolicy, env=env)
       
       # Train for few steps
       model.learn(total_timesteps=1000)
       
       # Evaluate
       mean_reward = evaluate_policy(model, env)
       assert mean_reward > 0

Common Issues
-----------

1. **Numerical Stability**
   
   * Use log probabilities
   * Normalize advantages
   * Clip gradients

2. **Training Stability**
   
   * Monitor value estimates
   * Check gradient norms
   * Use proper initialization

3. **Memory Usage**
   
   * Clear unused tensors
   * Use appropriate batch sizes
   * Monitor memory consumption

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check algorithm implementation
2. Review mathematical derivation
3. Debug training process
4. Seek community help

For more details, see the `Stable-Baselines3 documentation <https://stable-baselines3.readthedocs.io/>`_. 