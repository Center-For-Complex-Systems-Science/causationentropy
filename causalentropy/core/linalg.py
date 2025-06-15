import numpy as np
import networkx as nx
from typing import Tuple, List, Optional


def edge_tensor_from_graph(
        G: nx.DiGraph,
        var_names:   Optional[List[str]] = None,
        tau_max:     Optional[int]       = None,
        attr:        str                 = "cmi",
        binary:      bool                = False,
        default:     float               = 0.0,
) -> Tuple[np.ndarray, List[str]]:
    """
    Convert a lag-labelled causal graph into a 3-D edge tensor.

    Parameters
    ----------
    G          : nx.DiGraph
        Output of `discover_network`.  Every edge must carry
        'lag' (int) and, optionally, the weight attribute `attr`.
    var_names  : list[str] or None
        Ordering of variables.  If None, uses `sorted(G.nodes())`.
    tau_max    : int or None
        Largest lag index to allocate.  If None, uses max edge lag.
    attr       : str
        Edge attribute to store (e.g. 'cmi').  Ignored if `binary=True`.
    binary     : bool
        If True, tensor cells are 1 when the edge exists, else 0.
        If False, store `G[u][v][attr]`, falling back to `default`.
    default    : float
        Fill value when an edge is absent or the attribute is missing.

    Returns
    -------
    tensor : ndarray  shape = (n, n, tau_max+1)
        tensor[i, j, τ]  =  weight of  j→i  at delay τ
    var_names : list[str]
        The variable order used in the tensor.
    """
    # -- node order -------------------------------------------------- #
    if var_names is None:
        var_names = sorted(G.nodes())
    name2idx = {v: k for k, v in enumerate(var_names)}
    n = len(var_names)

    # -- lag dimension ---------------------------------------------- #
    if tau_max is None:
        if not G.edges:
            tau_max = 0
        else:
            tau_max = max(data["lag"] for _, _, data in G.edges(data=True))

    tensor = np.full((n, n, tau_max + 1), default, dtype=float)

    for u, v, data in G.edges(data=True):
        src, tgt, lag = name2idx[u], name2idx[v], data["lag"]
        if lag > tau_max:       # optional: warn or resize
            continue
        if binary:
            tensor[tgt, src, lag] = 1.0
        else:
            tensor[tgt, src, lag] = data.get(attr, default)

    return tensor, var_names