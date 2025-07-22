from typing import List
import numpy as np
import networkx as nx


def discover_network_hawkes(
        data: List[np.ndarray],
        method: str,
        alpha_forward: float,
        alpha_backward: float,
        n_shuffles: int,
        rng: np.random.Generator
) -> nx.DiGraph:
    if not isinstance(data, (list, tuple)) or not all(isinstance(d, np.ndarray) and d.ndim == 1 for d in data):
        raise ValueError("For 'hawkes' information, data must be a list of 1D numpy arrays of event timestamps.")

    n = len(data)
    var_names = [f"X{i}" for i in range(n)]
    G = nx.DiGraph()
    G.add_nodes_from(var_names)

    for i in range(n):
        print(f"Estimating edges for node {i} ({var_names[i]})")

        Y = data[i]
        predictors = [data[j] for j in range(n) if j != i]

        if method == 'standard':
            S = standard_optimal_causation_entropy(
                predictors, Y, Z_init=[], rng=rng,
                alpha1=alpha_forward,
                alpha2=alpha_backward,
                n_shuffles=n_shuffles,
                information='hawkes'
            )
        elif method == 'alternative':
            S = alternative_optimal_causation_entropy(
                predictors, Y, rng=rng,
                alpha1=alpha_forward,
                alpha2=alpha_backward,
                n_shuffles=n_shuffles,
                information='hawkes'
            )
        else:
            raise NotImplementedError(f"Method '{method}' is not supported for Hawkes estimation.")

        for s in S:
            j = s if s < i else s + 1  # skip self-index from predictors
            G.add_edge(var_names[j], var_names[i], lag=1)  # lag=1 is a placeholder for Hawkes

    return G
