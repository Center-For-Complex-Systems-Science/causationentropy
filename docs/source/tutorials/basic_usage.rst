Basic Usage
===========

This tutorial demonstrates common workflows for causal network discovery
with the Causation Entropy library.

Loading Data and Discovering a Network
--------------------------------------

Time-series data should be provided as a two-dimensional array or pandas
DataFrame, with variables in columns and time points in rows.

For example:

.. code-block:: python

   import pandas as pd
   from causationentropy import discover_network

   data = pd.read_csv("data.csv")

   network = discover_network(
       data,
       method="standard",
       max_lag=5,
       random_state=42,
   )

The returned object is a NetworkX ``MultiDiGraph``. Its edges contain
information such as the source variable, target variable, lag, conditional
mutual information (CMI), and permutation-test p-value.

The network can be converted to a pandas DataFrame:

.. code-block:: python

   from causationentropy.graph import network_to_dataframe

   df = network_to_dataframe(network)

   print(df)

Reproducible Discovery
----------------------

Permutation tests use randomness. Providing the same integer
``random_state`` makes repeated discovery runs reproducible:

.. code-block:: python

   import numpy as np
   from causationentropy import discover_network

   rng = np.random.default_rng(0)
   data = rng.normal(size=(300, 3))

   network_1 = discover_network(
       data,
       max_lag=2,
       n_shuffles=100,
       random_state=42,
   )

   network_2 = discover_network(
       data,
       max_lag=2,
       n_shuffles=100,
       random_state=42,
   )

   print(
       list(network_1.edges(data=True))
       == list(network_2.edges(data=True))
   )

Use a different integer seed for an independent replicate, or pass
``random_state=None`` when independent randomness is desired.

Multiple-Testing Correction
---------------------------

After causal discovery, multiple-testing corrections can be applied to the
p-values of the retained edges.

For example, Benjamini-Hochberg false-discovery-rate correction can be
applied across the discovered graph:

.. code-block:: python

   import numpy as np
   from causationentropy import discover_network
   from causationentropy.graph import (
       apply_test_correction,
       network_to_dataframe,
   )

   rng = np.random.default_rng(42)
   data = rng.normal(size=(300, 3))

   network = discover_network(
       data,
       max_lag=2,
       n_shuffles=500,
       random_state=42,
   )

   apply_test_correction(
       network,
       method="bh",
       alpha=0.05,
       family="graph",
       n_shuffles=500,
   )

   df = network_to_dataframe(network)

   print(
       df[
           [
               "Source",
               "Sink",
               "Lag",
               "P_Value",
               "P_Adjusted",
           ]
       ]
   )

The ``family="graph"`` option applies one correction across the graph.
The ``family="target"`` option applies a separate correction to incoming
edges for each target node.

This is a reporting adjustment over the retained edges. It does not
retroactively correct the full set of candidate links considered during
network discovery.

Permutation-Test Early Stopping
-------------------------------

The permutation test supports an opt-in ``early_stop`` mode. Clearly
insignificant tests can stop before the full shuffle budget has been used,
while tests that could still pass the significance threshold continue to
the requested number of shuffles.

.. code-block:: python

   import numpy as np

   from causationentropy.core.discovery import shuffle_test
   from causationentropy.core.information.conditional_mutual_information import (
       conditional_mutual_information,
   )

   rng = np.random.default_rng(42)

   X = rng.normal(size=(200, 1))
   Y = rng.normal(size=(200, 1))

   observed_cmi = conditional_mutual_information(
       X,
       Y,
       None,
       method="gaussian",
   )

   result = shuffle_test(
       X,
       Y,
       None,
       observed_cmi,
       alpha=0.05,
       n_shuffles=500,
       rng=42,
       information="gaussian",
       early_stop=True,
   )

   print(f"P-value: {result['P_value']:.4f}")
   print(f"Shuffles completed: {result['N_Completed']}")
   print(f"Early stopped: {result['Early_Stopped']}")

Early stopping is opt-in. With ``early_stop=False``, the full requested
number of shuffles is generated.

Bootstrap Confidence Intervals
------------------------------

The library can estimate a conditional mutual information value together
with a percentile bootstrap confidence interval.

For time-series data, block-based resampling can preserve local temporal
structure:

.. code-block:: python

   import numpy as np

   from causationentropy.core.stats import (
       bootstrap_cmi_confidence_interval,
   )

   rng = np.random.default_rng(42)

   X = rng.normal(size=(300, 1))
   Y = X + 0.5 * rng.normal(size=(300, 1))

   estimate, lower, upper, bootstrap_estimates = (
       bootstrap_cmi_confidence_interval(
           X,
           Y,
           n_bootstraps=100,
           alpha=0.05,
           block_length=20,
           seed=42,
       )
   )

   print(f"CMI estimate: {estimate:.4f}")
   print(f"95% CI: [{lower:.4f}, {upper:.4f}]")

For a stationary bootstrap, use ``use_stationary=True`` and specify an
appropriate ``mean_block_length``.

Inspecting Significant and Insignificant Candidates
----------------------------------------------------

By default, ``discover_network`` returns only statistically significant
links. For smaller networks, setting ``only_return_significant=False``
keeps all tested source-lag candidates and records whether each candidate
was significant.

.. code-block:: python

   import numpy as np

   from causationentropy import discover_network
   from causationentropy.graph import network_to_dataframe

   rng = np.random.default_rng(42)
   data = rng.normal(size=(200, 3))

   network = discover_network(
       data,
       max_lag=3,
       n_shuffles=100,
       random_state=42,
       only_return_significant=False,
   )

   df = network_to_dataframe(network)

   print(
       df[
           [
               "Source",
               "Sink",
               "Lag",
               "CMI",
               "P_Value",
               "Significant",
           ]
       ]
   )

This option is particularly useful when examining the tested lag structure.
It can produce substantially larger graphs because insignificant
candidates are also retained.

Delay Analysis
--------------

When all tested candidates are retained, ``plot_delay_analysis`` can be
used to visualize CMI as a function of time lag.

.. code-block:: python

   import numpy as np

   from causationentropy import discover_network
   from causationentropy.core.plotting import plot_delay_analysis

   rng = np.random.default_rng(42)
   data = rng.normal(size=(200, 3))

   network = discover_network(
       data,
       max_lag=5,
       n_shuffles=100,
       random_state=42,
       only_return_significant=False,
   )

   fig, ax = plot_delay_analysis(
       network,
       show_plot=False,
   )

   fig.show()

The x-axis shows the tested lag and the y-axis shows the corresponding
CMI. Significant and insignificant candidates are displayed differently.

This visualization is most useful for small networks because retaining
every tested source-lag combination can produce many points.

Generating Synthetic Data from a Ground-Truth Graph
----------------------------------------------------

The ``linear_gaussian_from_graph`` utility can generate a stable linear
Gaussian time series directly from a directed graph with lagged edges.
This is useful for evaluating causal discovery against known ground truth.

.. code-block:: python

   import networkx as nx

   from causationentropy import discover_network
   from causationentropy.datasets.synthetic import (
       linear_gaussian_from_graph,
   )

   truth = nx.MultiDiGraph()
   truth.add_edge("X0", "X1", lag=1, weight=0.7)
   truth.add_edge("X1", "X2", lag=2, weight=0.7)

   data, truth = linear_gaussian_from_graph(
       truth,
       T=500,
       seed=42,
   )

   discovered = discover_network(
       data,
       max_lag=2,
       n_shuffles=200,
       random_state=42,
   )

   print("Ground-truth edges:")
   for source, target, attrs in truth.edges(data=True):
       print(
           f"{source} -> {target} "
           f"(lag={attrs['lag']})"
       )

   print("\nDiscovered edges:")
   for source, target, attrs in discovered.edges(data=True):
       print(
           f"{source} -> {target} "
           f"(lag={attrs['lag']})"
       )

The simulator interprets an edge ``source -> target`` with lag ``tau`` as
a contribution from the source at time ``t - tau`` to the target at time
``t``. The returned ``truth`` graph is the graph used to generate the
data.