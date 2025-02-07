Visualization Utilities
=====================

RL Research provides various tools for visualizing experiment results and agent behavior.

Learning Curves
-------------

Plot Training Progress
~~~~~~~~~~~~~~~~~~~

Visualize training metrics:

.. code-block:: python

   from rl_research.utils.viz import plot_learning_curve

   # Plot single run
   plot_learning_curve(
       run_path="runs/experiment_name",
       metric="reward",
       window=100
   )

   # Plot multiple runs
   plot_learning_curve(
       run_paths=["runs/exp1", "runs/exp2"],
       metric="reward",
       window=100,
       labels=["Experiment 1", "Experiment 2"]
   )

Customization Options
~~~~~~~~~~~~~~~~~~

Customize plot appearance:

.. code-block:: python

   plot_learning_curve(
       run_path="runs/experiment_name",
       metric="reward",
       window=100,
       title="Training Progress",
       xlabel="Steps",
       ylabel="Average Reward",
       figsize=(10, 6),
       style="seaborn"
   )

Environment Visualization
----------------------

Render Episodes
~~~~~~~~~~~~~

Record agent behavior:

.. code-block:: python

   from rl_research.utils.viz import record_episode

   # Record single episode
   record_episode(
       model,
       env,
       video_path="videos/episode.mp4"
   )

   # Record multiple episodes
   record_episode(
       model,
       env,
       video_path="videos/episodes.mp4",
       n_episodes=5
   )

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

Best Practices
------------

1. **Plot Clarity**
   
   * Use clear titles and labels
   * Add legends when comparing
   * Choose appropriate scales

2. **Data Processing**
   
   * Smooth noisy data
   * Use appropriate window sizes
   * Handle missing data

3. **Comparisons**
   
   * Use consistent scales
   * Show confidence intervals
   * Include baselines

4. **Resource Usage**
   
   * Optimize video quality
   * Manage file sizes
   * Clean up old visualizations

5. **Documentation**
   
   * Label axes clearly
   * Add plot descriptions
   * Document custom visualizations

Troubleshooting
-------------

Common Issues
~~~~~~~~~~~

1. **Display Problems**
   
   * Check matplotlib backend
   * Verify display settings
   * Update dependencies

2. **Performance Issues**
   
   * Reduce data points
   * Optimize plot updates
   * Use appropriate formats

3. **Export Problems**
   
   * Check file permissions
   * Verify paths
   * Monitor disk space

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