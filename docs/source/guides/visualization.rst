Visualization Utilities
=====================

RL Research provides tools for visualizing experiment results and agent behavior.

Basic Setup
----------

Set the plotting style:

.. code-block:: python

   from rl_research.utils.viz import set_style

   # Set the default style (uses seaborn)
   set_style()

Learning Curves
-------------

Plot Training Progress
~~~~~~~~~~~~~~~~~~~

Visualize training metrics from multiple runs:

.. code-block:: python

   from rl_research.utils.viz import plot_learning_curves

   # Plot learning curves
   plot_learning_curves(
       runs=wandb_runs,  # List of WandB runs
       metric="rollout/ep_rew_mean",
       window=100,
       title="Training Progress",
       figsize=(10, 6)
   )

Evaluation Results
----------------

Plot evaluation metrics:

.. code-block:: python

   from rl_research.utils.viz import plot_evaluation_results

   # Plot evaluation results
   plot_evaluation_results(
       results=[100, 150, 200, 180, 190],  # List of evaluation returns
       title="Evaluation Results",
       figsize=(10, 6)
   )

Environment Visualization
----------------------

Record Episodes
~~~~~~~~~~~~~

Record environment episodes:

.. code-block:: python

   from rl_research.utils.viz import record_video_episodes

   # Record episodes
   frames, rewards = record_video_episodes(
       model=trained_model,
       env_id="CartPole-v1",
       num_episodes=3,
       render_fps=30,
       save_local=True,
       output_dir="videos",
       video_format="mp4",
       prefix="eval"
   )

Saving Visualizations
------------------

Save plots to disk:

.. code-block:: python

   from rl_research.utils.viz import save_figure

   # Save the current figure
   save_figure(
       path="learning_curve.png",
       dpi=300
   )

Example Usage
-----------

Complete example combining multiple visualizations:

.. code-block:: python

   from rl_research.utils.viz import (
       set_style,
       plot_learning_curves,
       plot_evaluation_results,
       record_video_episodes,
       save_figure
   )

   # Set style
   set_style()

   # Plot learning curves
   plot_learning_curves(wandb_runs)
   save_figure("learning_curves.png")

   # Plot evaluation results
   plot_evaluation_results(eval_returns)
   save_figure("eval_results.png")

   # Record episodes
   frames, rewards = record_video_episodes(model, "CartPole-v1")

Best Practices
------------

1. **Style Consistency**
   - Call ``set_style()`` at the start of your script
   - Use consistent figure sizes
   - Keep plots simple and readable

2. **Memory Management**
   - Close figures after saving to free memory
   - Use appropriate DPI for your needs
   - Be mindful of frame buffer sizes

3. **Data Processing**
   - Smooth learning curves for clarity
   - Use appropriate window sizes
   - Handle missing data gracefully

4. **Organization**
   - Use descriptive filenames
   - Group related visualizations
   - Add proper titles and labels

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Missing Data**
   - Check if metrics exist in WandB runs
   - Verify metric names match exactly
   - Ensure data is properly logged

2. **Plot Quality**
   - Adjust figure size and DPI
   - Check smoothing window size
   - Verify data ranges

3. **Memory Issues**
   - Close figures after saving
   - Reduce number of frames
   - Clear plot cache

For more examples, check the ``examples/`` directory in the repository.

For more details, see the matplotlib_ and seaborn_ documentation.

.. _matplotlib: https://matplotlib.org/
.. _seaborn: https://seaborn.pydata.org/ 