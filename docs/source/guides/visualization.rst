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

Render Episodes
~~~~~~~~~~~~~

Plot environment frames:

.. code-block:: python

   from rl_research.utils.viz import plot_environment_renders

   # Plot grid of frames
   plot_environment_renders(
       frames=episode_frames,  # List of environment renders
       rows=2,
       cols=3,
       figsize=(15, 10)
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
       plot_environment_renders,
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

   # Plot environment frames
   plot_environment_renders(episode_frames)
   save_figure("environment_frames.png")

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

State Visualization
~~~~~~~~~~~~~~~~

Visualize environment states:

.. code-block:: python

   from rl_research.utils.viz import plot_state

   # Plot single state
   plot_state(state)

   # Plot state sequence
   plot_state_sequence(
       states,
       n_cols=4,
       figsize=(12, 8)
   )

Policy Visualization
-----------------

Action Distributions
~~~~~~~~~~~~~~~~~

Visualize policy decisions:

.. code-block:: python

   from rl_research.utils.viz import plot_action_dist

   # Plot action distribution
   plot_action_dist(
       model,
       state,
       title="Action Distribution"
   )

Value Function
~~~~~~~~~~~~

Visualize value estimates:

.. code-block:: python

   from rl_research.utils.viz import plot_value_function

   # Plot value function
   plot_value_function(
       model,
       states,
       title="State Values"
   )

Attention Maps
~~~~~~~~~~~~

For transformer-based policies:

.. code-block:: python

   from rl_research.utils.viz import plot_attention

   # Plot attention weights
   plot_attention(
       model,
       state,
       layer=0,
       head=0
   )

Comparative Analysis
-----------------

Compare Experiments
~~~~~~~~~~~~~~~~

Compare multiple runs:

.. code-block:: python

   from rl_research.utils.viz import compare_experiments

   # Compare learning curves
   compare_experiments(
       run_paths=["runs/exp1", "runs/exp2"],
       metrics=["reward", "loss"],
       labels=["Baseline", "Improved"]
   )

Statistical Analysis
~~~~~~~~~~~~~~~~~

Analyze experiment results:

.. code-block:: python

   from rl_research.utils.viz import plot_statistics

   # Plot performance statistics
   plot_statistics(
       run_paths=["runs/exp1", "runs/exp2"],
       metric="reward",
       ci=95  # confidence interval
   )

Interactive Visualization
----------------------

Jupyter Widgets
~~~~~~~~~~~~~

Interactive plots for notebooks:

.. code-block:: python

   from rl_research.utils.viz import interactive_plot

   # Create interactive plot
   interactive_plot(
       run_path="runs/experiment_name",
       metrics=["reward", "loss"]
   )

Real-time Monitoring
~~~~~~~~~~~~~~~~~

Monitor training progress:

.. code-block:: python

   from rl_research.utils.viz import LivePlot

   # Create live plot
   live_plot = LivePlot(
       metrics=["reward", "loss"],
       update_interval=1.0
   )

   # Update in training loop
   live_plot.update(metrics)

Export and Sharing
----------------

Save Plots
~~~~~~~~

Export visualizations:

.. code-block:: python

   from rl_research.utils.viz import save_plot

   # Save single plot
   save_plot(
       fig,
       path="plots/learning_curve.png",
       dpi=300
   )

   # Save multiple plots
   save_plots(
       figs,
       base_path="plots",
       prefix="experiment_"
   )

Generate Reports
~~~~~~~~~~~~~

Create experiment reports:

.. code-block:: python

   from rl_research.utils.viz import generate_report

   # Generate HTML report
   generate_report(
       run_path="runs/experiment_name",
       output_path="reports/report.html"
   )

Getting Help
~~~~~~~~~~

If you encounter issues:

1. Check matplotlib documentation
2. Review example notebooks
3. Search common solutions
4. Report bugs with examples

For more details, see the matplotlib_ and seaborn_ documentation.

.. _matplotlib: https://matplotlib.org/
.. _seaborn: https://seaborn.pydata.org/ 