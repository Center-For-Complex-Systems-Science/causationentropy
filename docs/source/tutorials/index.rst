Tutorials
=========

This section provides practical examples and guides for using the ``causationentropy`` package for causal network discovery from time-series data.

Basic Usage
-----------

Below is a self-contained example demonstrating how to discover a causal network from synthetic time-series data:

.. code-block:: python

    import numpy as np
    from causationentropy import discover_network

    # 1. Generate synthetic data (X0 causes X1 at lag 1)
    np.random.seed(42)
    n_samples = 200
    x0 = np.random.normal(0, 1, n_samples)
    x1 = np.zeros(n_samples)

    for t in range(1, n_samples):
        x1[t] = 0.7 * x0[t - 1] + 0.3 * np.random.normal()

    # Combine into a 2D array of shape (n_samples, n_variables)
    data = np.column_stack([x0, x1])

    # 2. Run causal network discovery
    network = discover_network(data, max_lag=2, n_shuffles=50)

    # 3. Inspect discovered causal edges
    for u, v, attrs in network.edges(data=True):
        print(f"Discovered causal edge: {u} -> {v} (lag: {attrs.get('lag')})")

Interactive Notebooks
---------------------

The repository includes the following interactive Jupyter notebooks demonstrating different estimators and use cases:

* `Quickstart Notebook <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/Quickstart.ipynb>`_ - Introductory workflow and basic network discovery.
* `Optimal Causation Entropy Tutorial <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/Optimal_Causation_Entropy_Tutorial.ipynb>`_ - Detailed walk-through of the oCSE algorithm.
* `Gaussian Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/gaussian_causal_discovery_example.ipynb>`_ - Network discovery under Gaussian assumptions.
* `kNN Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/knn_causal_discovery_example.ipynb>`_ - Nonparametric causal discovery using k-Nearest Neighbors.
* `Geometric kNN Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/geometric_knn_causal_discovery_example.ipynb>`_ - Nonparametric estimation using geometric kNN entropy corrections.
* `KDE Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/kde_causal_discovery_example.ipynb>`_ - Nonparametric causal discovery using Kernel Density Estimation.
* `Poisson Causal Discovery Example <https://github.com/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/poisson_causal_discovery_example.ipynb>`_ - Causal discovery for count and event data with Poisson dynamics.