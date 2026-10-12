"""Dataset generation and method adapters for issue #18."""

from __future__ import annotations

import time

import networkx as nx
import numpy as np

from causationentropy.datasets.synthetic import linear_stochastic_gaussian_process

CANONICAL_RHO = 0.7
CANONICAL_P = 0.2
CANONICAL_EPSILON = 0.1


def generate_case(
    n: int, T: int, seed: int, rho: float = CANONICAL_RHO, p: float = CANONICAL_P
):
    """Generate one benchmark case.

    Returns ``(data, truth, node_order)`` where ``truth`` is the exact-match
    ``(source, target, lag)`` edge set (all ground-truth edges are lag 1)
    and ``node_order`` maps column ``k`` to node label ``'X{k}'``.
    """
    G_true = nx.erdos_renyi_graph(n, p, seed=seed, directed=True)
    data, _ = linear_stochastic_gaussian_process(
        rho, n=n, T=T, p=p, epsilon=CANONICAL_EPSILON, seed=seed, G=G_true
    )
    truth = {(u, v, 1) for u, v in G_true.edges()}
    node_order = [f"X{k}" for k in range(n)]
    return data, truth, node_order


def run_causation_entropy(
    data,
    max_lag: int,
    seed: int,
    n_shuffles: int,
    alpha_forward: float = 0.05,
    alpha_backward: float = 0.05,
):
    """Run Gaussian oCSE serially; return ``(graph, runtime_s)``."""
    from causationentropy.core.discovery import discover_network

    start = time.perf_counter()
    graph = discover_network(
        data,
        method="standard",
        information="gaussian",
        max_lag=max_lag,
        alpha_forward=alpha_forward,
        alpha_backward=alpha_backward,
        n_shuffles=n_shuffles,
        n_jobs=1,
        random_state=seed,
    )
    return graph, time.perf_counter() - start


def run_pcmci_parcorr(data, tau_max: int, pc_alpha: float = 0.05):
    """Run PCMCI + ParCorr; return ``(results_dict, runtime_s)``.

    Raises
    ------
    ImportError
        If ``tigramite`` is not installed.
    """
    try:
        from tigramite import data_processing as pp
        from tigramite.independence_tests.parcorr import ParCorr
        from tigramite.pcmci import PCMCI
    except ImportError as exc:
        raise ImportError(
            "tigramite is required for the PCMCI baseline "
            "(pip install 'causationentropy[dev]')."
        ) from exc

    dataframe = pp.DataFrame(data, var_names=[f"X{i}" for i in range(data.shape[1])])
    start = time.perf_counter()
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=pc_alpha)
    return results, time.perf_counter() - start
