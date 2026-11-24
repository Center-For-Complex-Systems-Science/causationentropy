"""Fast end-to-end check of Poisson network discovery in the default test suite."""

import networkx as nx
import numpy as np

from causationentropy.core.discovery import discover_network
from causationentropy.datasets.synthetic import poisson_coupled_oscillators


def test_poisson_discovery_recovers_true_edges():
    """Standard oCSE with the Poisson estimator recovers the generator's edges.

    The Poisson integration tests in ``test_data_integration.py`` are deselected
    by default, so this small case keeps the Poisson path covered in regular CI.
    """
    n = 3
    data, A = poisson_coupled_oscillators(n=n, T=200, seed=2)
    truth = np.asarray(A) > 0  # truth[j, i] is True when j drives i
    off_diagonal = ~np.eye(n, dtype=bool)

    G = discover_network(
        data,
        method="standard",
        information="poisson",
        max_lag=1,
        n_shuffles=30,
        random_state=2,
    )
    found = nx.to_numpy_array(G, nodelist=[f"X{i}" for i in range(n)]) > 0

    assert truth[off_diagonal].sum() > 0
    assert np.all(found[truth & off_diagonal])
