Tutorials
=========

This section provides practical examples and guides for using the
``causationentropy`` package for causal network discovery from time-series data.

Basic Usage
-----------

Below is a self-contained example demonstrating how to discover a causal
network from synthetic time-series data:

.. code-block:: python

    import numpy as np
    from causationentropy import discover_network

    # Generate synthetic data (X0 causes X1 at lag 1)
    rng = np.random.default_rng(42)
    n_samples = 200

    x0 = rng.normal(0, 1, n_samples)
    x1 = np.zeros(n_samples)

    for t in range(1, n_samples):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * rng.normal()

    data = np.column_stack([x0, x1])

    # Discover the causal network
    network = discover_network(
        data,
        max_lag=2,
        n_shuffles=50,
        random_state=42,
    )

    # Inspect discovered causal edges
    for source, target, attrs in network.edges(data=True):
        print(
            f"Discovered causal edge: {source} -> {target} "
            f"(lag: {attrs.get('lag')})"
        )

Using a Reproducible Random State
---------------------------------

The ``random_state`` argument controls the permutation-test random stream.
Using the same integer seed makes repeated discovery runs reproducible:

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

    print(list(network_1.edges(data=True)) == list(network_2.edges(data=True)))

For independent permutation-test replicates, use a different seed or pass
``random_state=None``.

Multiple-Testing Correction
---------------------------

After causal discovery, multiple-testing corrections can be applied to the
p-values of the retained edges. For example, the Benjamini-Hochberg procedure
can be used to control the false discovery rate:

.. code-block:: python

    import numpy as np
    from causationentropy import discover_network
    from causationentropy.graph import apply_test_correction, network_to_dataframe

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

    print(df[["Source", "Sink", "Lag", "P_Value", "P_Adjusted"]])

The ``family="graph"`` option applies one correction across the edges in the
graph. Use ``family="target"`` to correct incoming edges separately for each
target node.

This is a reporting adjustment over the edges present in the discovered graph;
it does not retroactively correct every candidate link considered during the
network-discovery procedure.

Permutation-Test Early Stopping
--------------------------------

The lower-level ``shuffle_test`` function provides an optional
``early_stop`` mode. When enabled, clearly insignificant tests can stop before
all requested shuffles have been generated. Tests that could still reach the
significance threshold continue to the full shuffle budget.

.. code-block:: python

    import numpy as np
    from causationentropy.core.discovery import shuffle_test
    from causationentropy.core.information.conditional_mutual_information import (
        conditional_mutual_information,
    )

    rng = np.random.default_rng(42)

    X = rng.normal(size=(200, 1))
    Y = rng.normal(size=(200, 1))
    Z = None

    observed_cmi = conditional_mutual_information(
        X,
        Y,
        Z,
        method="gaussian",
    )

    result = shuffle_test(
        X,
        Y,
        Z,
        observed_cmi,
        alpha=0.05,
        n_shuffles=500,
        rng=42,
        information="gaussian",
        early_stop=True,
    )

    print(f"Pass: {result['Pass']}")
    print(f"P-value: {result['P_value']:.4f}")
    print(f"Shuffles completed: {result['N_Completed']}")
    print(f"Early stopped: {result['Early_Stopped']}")

Early stopping is opt-in. With ``early_stop=False``, the full requested
number of shuffles is generated.

Bootstrap Confidence Intervals
------------------------------

Bootstrap resampling can be used to quantify uncertainty in conditional
mutual information estimates. For time-series data, the library supports
moving-block and stationary bootstrap procedures.

.. code-block:: python

    import numpy as np
    from causationentropy.core.stats import bootstrap_cmi_confidence_interval

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

For strongly autocorrelated time series, the stationary bootstrap can be
selected with ``use_stationary=True`` and an appropriate
``mean_block_length``.

Interactive Notebooks
---------------------

The repository includes the following interactive Jupyter notebooks
demonstrating different estimators and use cases:

* `Quickstart Notebook <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/Quickstart.ipynb>`_ -
  Introductory workflow and basic network discovery.
* `Optimal Causation Entropy Tutorial <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/Optimal_Causation_Entropy_Tutorial.ipynb>`_ -
  Detailed walk-through of the oCSE algorithm.
* `Gaussian Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/gaussian_causal_discovery_example.ipynb>`_ -
  Network discovery under Gaussian assumptions.
* `kNN Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/knn_causal_discovery_example.ipynb>`_ -
  Nonparametric causal discovery using k-Nearest Neighbors.
* `Geometric kNN Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/geometric_knn_causal_discovery_example.ipynb>`_ -
  Nonparametric estimation using geometric kNN entropy corrections.
* `KDE Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/kde_causal_discovery_example.ipynb>`_ -
  Nonparametric causal discovery using Kernel Density Estimation.
* `Poisson Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/poisson_causal_discovery_example.ipynb>`_ -
  Causal discovery for count and event data with Poisson dynamics.

  Additional Tutorial
-------------------

For additional workflows covering result inspection, delay analysis, and
evaluation against known ground truth, see :doc:`basic_usage`.