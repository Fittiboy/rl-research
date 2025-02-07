Custom Environments
==================

This guide explains how to create custom environments in RL Research using the Gymnasium interface.

Basic Structure
-------------

A custom environment must implement the Gymnasium interface:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from gymnasium import spaces

   class CustomEnv(gym.Env):
       """Custom Environment that follows gym interface."""
       
       def __init__(self):
           super().__init__()
           
           # Define action and observation space
           self.action_space = spaces.Discrete(2)
           self.observation_space = spaces.Box(
               low=-np.inf,
               high=np.inf,
               shape=(4,),
               dtype=np.float32
           )
       
       def step(self, action):
           # Execute action and return next state
           next_state = self._get_next_state(action)
           
           # Calculate reward
           reward = self._calculate_reward(next_state)
           
           # Check if episode is done
           terminated = self._is_terminated(next_state)
           truncated = self._is_truncated()
           
           # Optional info dictionary
           info = {}
           
           return next_state, reward, terminated, truncated, info
       
       def reset(self, seed=None, options=None):
           # Reset environment state
           super().reset(seed=seed)
           
           # Initialize state
           self.state = self._init_state()
           
           # Optional info dictionary
           info = {}
           
           return self.state, info
       
       def render(self):
           # Implement visualization
           pass
       
       def close(self):
           # Clean up resources
           pass

Registration
-----------

Register your environment with Gymnasium:

.. code-block:: python

   from gymnasium.envs.registration import register

   register(
       id="CustomEnv-v0",
       entry_point="rl_research.environments.custom_envs:CustomEnv",
       max_episode_steps=500,
   )

Then use it like any other environment:

.. code-block:: python

   import gymnasium as gym
   env = gym.make("CustomEnv-v0")

Example Environment
-----------------

Here's a complete example of a simple 2D navigation environment:

.. code-block:: python

   import gymnasium as gym
   import numpy as np
   from gymnasium import spaces

   class NavigationEnv(gym.Env):
       """2D navigation environment."""
       
       def __init__(self):
           super().__init__()
           
           # Environment parameters
           self.size = 10
           self.target = np.array([8, 8])
           
           # Action space: up, down, left, right
           self.action_space = spaces.Discrete(4)
           
           # Observation space: agent position (x, y)
           self.observation_space = spaces.Box(
               low=0,
               high=self.size,
               shape=(2,),
               dtype=np.float32
           )
           
           # Initialize state
           self.state = None
       
       def step(self, action):
           # Current position
           x, y = self.state
           
           # Update position based on action
           if action == 0:  # up
               y = min(y + 1, self.size)
           elif action == 1:  # down
               y = max(y - 1, 0)
           elif action == 2:  # left
               x = max(x - 1, 0)
           elif action == 3:  # right
               x = min(x + 1, self.size)
           
           # Update state
           self.state = np.array([x, y])
           
           # Calculate distance to target
           distance = np.linalg.norm(self.state - self.target)
           
           # Reward: negative distance to target
           reward = -distance
           
           # Done if agent reaches target
           terminated = distance < 0.5
           truncated = False
           
           return self.state, reward, terminated, truncated, {}
       
       def reset(self, seed=None, options=None):
           super().reset(seed=seed)
           
           # Random initial position
           self.state = self.np_random.uniform(
               low=0,
               high=self.size,
               size=2
           )
           
           return self.state, {}
       
       def render(self):
           print(f"Agent position: {self.state}")
           print(f"Target position: {self.target}")

Environment Wrappers
------------------

Create custom wrappers to modify environment behavior:

.. code-block:: python

   import gymnasium as gym

   class NormalizeObservation(gym.ObservationWrapper):
       """Normalize observations to [-1, 1]."""
       
       def __init__(self, env):
           super().__init__(env)
           self.low = self.observation_space.low
           self.high = self.observation_space.high
       
       def observation(self, obs):
           return 2.0 * (obs - self.low) / (self.high - self.low) - 1.0

Use wrappers:

.. code-block:: python

   # Create and wrap environment
   env = gym.make("CustomEnv-v0")
   env = NormalizeObservation(env)

Advanced Features
--------------

Vectorized Environments
~~~~~~~~~~~~~~~~~~~~

Support parallel environments:

.. code-block:: python

   from gymnasium.vector import VectorEnv

   class VectorizedCustomEnv(VectorEnv):
       def __init__(self, num_envs):
           super().__init__(
               num_envs=num_envs,
               observation_space=spaces.Box(...),
               action_space=spaces.Discrete(...)
           )
       
       def step_async(self, actions):
           self.actions = actions
       
       def step_wait(self):
           observations = []
           rewards = []
           dones = []
           infos = []
           
           for i, action in enumerate(self.actions):
               obs, rew, done, info = self.envs[i].step(action)
               observations.append(obs)
               rewards.append(rew)
               dones.append(done)
               infos.append(info)
           
           return (
               np.array(observations),
               np.array(rewards),
               np.array(dones),
               infos
           )

Custom Rewards
~~~~~~~~~~~~

Implement complex reward functions:

.. code-block:: python

   def calculate_reward(self, state, action, next_state):
       # Base reward
       distance_reward = -np.linalg.norm(next_state - self.target)
       
       # Action penalty
       action_penalty = -0.1 * np.sum(np.abs(action))
       
       # Progress reward
       old_distance = np.linalg.norm(state - self.target)
       new_distance = np.linalg.norm(next_state - self.target)
       progress_reward = old_distance - new_distance
       
       return distance_reward + action_penalty + progress_reward

Best Practices
------------

1. **State Space**
   
   * Use appropriate data types
   * Normalize observations
   * Document space meanings

2. **Action Space**
   
   * Keep actions simple
   * Use reasonable bounds
   * Handle invalid actions

3. **Rewards**
   
   * Make rewards informative
   * Avoid sparse rewards
   * Scale appropriately

4. **Performance**
   
   * Optimize computations
   * Use vectorized operations
   * Cache when possible

5. **Testing**
   
   * Test edge cases
   * Verify reward function
   * Check termination conditions

Testing Environments
-----------------

Basic Tests
~~~~~~~~~

Test environment functionality:

.. code-block:: python

   import pytest

   def test_custom_env():
       env = gym.make("CustomEnv-v0")
       
       # Test initialization
       obs, info = env.reset()
       assert env.observation_space.contains(obs)
       
       # Test step
       action = env.action_space.sample()
       obs, reward, done, truncated, info = env.step(action)
       assert env.observation_space.contains(obs)
       assert isinstance(reward, float)
       assert isinstance(done, bool)

Random Agent Test
~~~~~~~~~~~~~~

Test with random actions:

.. code-block:: python

   def test_random_rollout():
       env = gym.make("CustomEnv-v0")
       obs, info = env.reset()
       
       for _ in range(100):
           action = env.action_space.sample()
           obs, reward, done, truncated, info = env.step(action)
           
           if done or truncated:
               obs, info = env.reset()

Common Issues
-----------

1. **Space Mismatch**
   
   * Verify observation shapes
   * Check data types
   * Handle edge cases

2. **Reward Design**
   
   * Test reward bounds
   * Ensure meaningful gradients
   * Avoid reward hacking

3. **Performance**
   
   * Profile slow operations
   * Optimize bottlenecks
   * Use appropriate data structures

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check Gymnasium documentation
2. Review example environments
3. Test systematically
4. Ask for community help

For more details, see the `Gymnasium documentation <https://gymnasium.farama.org/>`_. 