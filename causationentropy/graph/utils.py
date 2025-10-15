import networkx as nx
import numpy as np
import pandas as pd

LINK_TYPE_SEMANTICS = {
    "-->": "directed",
    "<--": "directed",
    "o-o": "undirected",
    "-?>": "possible_directed",
    "x-x": "conflicting",
}

SEMANTIC_TO_LINK_TYPE = {
    "directed": "-->",
    "undirected": "o-o",
    "possible_directed": "-?>",
    "conflicting": "x-x",
}


def pcmci_to_networkx(
    results: dict, binarize: bool = False, p_value: float = 0.05
) -> nx.MultiDiGraph:
    """Convert a PCMCI results dictionary to a networkx MultiDiGraph.

    Parameters
    ----------
    results : dict
        The results dictionary from a PCMCI run.
    binarize : bool, optional
        Whether to binarize the graph based on the p-value, by default False.
    p_value : float, optional
        The p-value to use for binarization, by default 0.05.

    Returns
    -------
    nx.MultiDiGraph
        The networkx MultiDiGraph.
    """
    graph = np.asarray(results["graph"])
    val_matrix = np.asarray(results["val_matrix"])
    p_matrix = np.asarray(results["p_matrix"])

    if graph.ndim == 2:
        # Insert lag axis for contemporaneous-only runs.
        graph = graph[:, :, np.newaxis]
    elif graph.ndim != 3:
        raise ValueError("Expected PCMCI graph with 2 or 3 dimensions")

    if val_matrix.ndim == 2:
        val_matrix = val_matrix[:, :, np.newaxis]
    elif val_matrix.ndim != 3:
        raise ValueError("Expected value matrix with 2 or 3 dimensions")

    if p_matrix.ndim == 2:
        p_matrix = p_matrix[:, :, np.newaxis]
    elif p_matrix.ndim != 3:
        raise ValueError("Expected p-value matrix with 2 or 3 dimensions")

    if val_matrix.shape != graph.shape or p_matrix.shape != graph.shape:
        raise ValueError("PCMCI graph, value, and p-value matrices must share shape")

    N, _, tau_max_plus_1 = graph.shape

    G = nx.MultiDiGraph()
    G.add_nodes_from(range(N))

    for i in range(N):
        for j in range(N):
            for lag in range(tau_max_plus_1):
                link_type_symbol = graph[i, j, lag]
                if link_type_symbol == "":
                    continue

                semantic_link_type = LINK_TYPE_SEMANTICS.get(link_type_symbol)
                if semantic_link_type is None:
                    raise ValueError(f"Unknown link type: {link_type_symbol}")

                val = float(np.asarray(val_matrix[i, j, lag]))
                p = float(np.asarray(p_matrix[i, j, lag]))

                if binarize:
                    if link_type_symbol == "-->":
                        # graph[i, j, lag] = '-->' means j is caused by i (i -> j)
                        G.add_edge(
                            i,
                            j,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                            significant=p < p_value,
                        )
                    elif link_type_symbol == "<--":
                        # graph[i, j, lag] = '<--' means i is caused by j (j -> i)
                        # This is equivalent to graph[j, i, lag] = '-->'
                        G.add_edge(
                            j,
                            i,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                            significant=p < p_value,
                        )
                    elif link_type_symbol in ["o-o", "x-x"]:
                        # For undirected/conflicting links, PCMCI sets the same symbol at both [i,j] and [j,i]
                        # Process only once when we first encounter it (when i < j)
                        if i < j:
                            G.add_edge(
                                i,
                                j,
                                lag=lag,
                                val=val,
                                p_value=p,
                                link_type=semantic_link_type,
                                significant=p < p_value,
                            )
                            G.add_edge(
                                j,
                                i,
                                lag=lag,
                                val=val,
                                p_value=p,
                                link_type=semantic_link_type,
                                significant=p < p_value,
                            )
                    elif link_type_symbol == "-?>":
                        # Possible directed link from i to j
                        G.add_edge(
                            i,
                            j,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                            significant=p < p_value,
                        )
                else:
                    if link_type_symbol == "-->":
                        # graph[i, j, lag] = '-->' means j is caused by i (i -> j)
                        G.add_edge(
                            i,
                            j,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                        )
                    elif link_type_symbol == "<--":
                        # graph[i, j, lag] = '<--' means i is caused by j (j -> i)
                        # This is equivalent to graph[j, i, lag] = '-->'
                        G.add_edge(
                            j,
                            i,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                        )
                    elif link_type_symbol in ["o-o", "x-x"]:
                        # For undirected/conflicting links, PCMCI sets the same symbol at both [i,j] and [j,i]
                        # Process only once when we first encounter it (when i < j)
                        if i < j:
                            G.add_edge(
                                i,
                                j,
                                lag=lag,
                                val=val,
                                p_value=p,
                                link_type=semantic_link_type,
                            )
                            G.add_edge(
                                j,
                                i,
                                lag=lag,
                                val=val,
                                p_value=p,
                                link_type=semantic_link_type,
                            )
                    elif link_type_symbol == "-?>":
                        # Possible directed link from i to j
                        G.add_edge(
                            i,
                            j,
                            lag=lag,
                            val=val,
                            p_value=p,
                            link_type=semantic_link_type,
                        )

    return G


def networkx_to_pcmci(G: nx.MultiDiGraph) -> dict:
    """Convert a networkx MultiDiGraph to a PCMCI results dictionary.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The networkx MultiDiGraph.

    Returns
    -------
    dict
        The PCMCI results dictionary.
    """
    nodes = list(G.nodes())
    N = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    max_lag = 0
    for _, _, data in G.edges(data=True):
        if "lag" in data and data["lag"] > max_lag:
            max_lag = data["lag"]

    tau_max = max_lag

    graph = np.full((N, N, tau_max + 1), "", dtype="<U3")
    val_matrix = np.zeros((N, N, tau_max + 1))
    p_matrix = np.ones((N, N, tau_max + 1))

    # Track processed undirected/conflicting edges to avoid duplication
    processed_undirected = set()

    for u_node, v_node, data in G.edges(data=True):
        u, v = node_map[u_node], node_map[v_node]
        lag = data.get("lag", 0)

        semantic_link_type = data.get("link_type", "directed")

        val = data.get("val", data.get("cmi", 0.0))
        p = data.get("p_value", 1.0)

        if semantic_link_type == "directed":
            # Directed edge from u to v
            graph[u, v, lag] = "-->"
            val_matrix[u, v, lag] = val
            p_matrix[u, v, lag] = p
        elif semantic_link_type in ["undirected", "conflicting"]:
            # For undirected/conflicting, we expect edges in both directions
            # Process only once using the canonical form (min, max)
            edge_key = (min(u, v), max(u, v), lag)
            if edge_key in processed_undirected:
                continue
            processed_undirected.add(edge_key)

            symbol = "o-o" if semantic_link_type == "undirected" else "x-x"
            # Set the same symbol at both [u,v] and [v,u] positions
            graph[u, v, lag] = symbol
            graph[v, u, lag] = symbol
            val_matrix[u, v, lag] = val
            val_matrix[v, u, lag] = val
            p_matrix[u, v, lag] = p
            p_matrix[v, u, lag] = p
        elif semantic_link_type == "possible_directed":
            # Possible directed edge from u to v
            graph[u, v, lag] = "-?>"
            val_matrix[u, v, lag] = val
            p_matrix[u, v, lag] = p
        else:
            raise ValueError(f"Unknown semantic link type: {semantic_link_type}")

    return {"graph": graph, "val_matrix": val_matrix, "p_matrix": p_matrix}


def network_to_dataframe(
    G: nx.MultiDiGraph,
    method: str = None,
    information: str = None,
    alpha_forward: float = None,
    alpha_backward: float = None,
    metric: str = None,
    bandwidth: str = None,
    k_means: int = None,
    n_shuffles: int = None,
    max_lag: int = None,
):
    """
    Convert a CausationEntropy NetworkX MultiDiGraph to a pandas DataFrame.

    This function flattens the edge data from a causal network graph (generated by
    discover_network) into a tabular format suitable for analysis and export.

    Parameters
    ----------
    G : nx.MultiDiGraph
        The causal network graph from discover_network. Expected to contain edge
        attributes: 'lag', 'cmi', and 'p_value'.
    method : str, optional
        The discovery method used ('standard', 'alternative', 'information_lasso', 'lasso').
    information : str, optional
        The information estimator type used ('gaussian', 'knn', 'kde', 'geometric_knn', 'poisson').
    alpha_forward : float, optional
        Significance level for forward selection used in network construction.
    alpha_backward : float, optional
        Significance level for backward elimination used in network construction.
    metric : str, optional
        Distance metric used for k-NN based estimators (e.g., 'euclidean').
    bandwidth : str, optional
        Bandwidth selection method for KDE (e.g., 'silverman').
    k_means : int, optional
        Number of neighbors for k-NN based estimators.
    n_shuffles : int, optional
        Number of permutation shuffles used for statistical testing.
    max_lag : int, optional
        Maximum time lag considered in network construction.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with columns representing edge attributes. Base columns are:

        - 'Source': Source node of the causal edge
        - 'Sink': Sink (target) node of the causal edge
        - 'Lag': Time lag of the causal relationship
        - 'CMI': Conditional mutual information value
        - 'P_Value': Statistical p-value from permutation test

        Additional columns are added based on the optional parameters provided:

        - 'Method': Discovery method
        - 'Information': Information estimator type
        - 'Alpha_Forward': Forward selection significance level
        - 'Alpha_Backward': Backward elimination significance level
        - 'Metric': Distance metric
        - 'Bandwidth': KDE bandwidth
        - 'K_Means': Number of neighbors
        - 'N_Shuffles': Number of permutations
        - 'Max_Lag': Maximum lag considered

    Examples
    --------
    >>> from causationentropy.core.discovery import discover_network
    >>> from causationentropy.graph.utils import network_to_dataframe
    >>> import numpy as np
    >>>
    >>> # Create a network
    >>> data = np.random.randn(100, 3)
    >>> G = discover_network(data, method='standard', information='gaussian', max_lag=3)
    >>>
    >>> # Convert to DataFrame with metadata
    >>> df = network_to_dataframe(
    ...     G,
    ...     method='standard',
    ...     information='gaussian',
    ...     alpha_forward=0.05,
    ...     alpha_backward=0.05,
    ...     metric='euclidean',
    ...     max_lag=3
    ... )
    """
    edges_data = []

    for u, v, data in G.edges(data=True):
        edge_dict = {
            "Source": u,
            "Sink": v,
            "Lag": data.get("lag", 0),
            "CMI": data.get("cmi", None),
            "P_Value": data.get("p_value", None),
        }

        # Add optional metadata columns
        if method is not None:
            edge_dict["Method"] = method
        if information is not None:
            edge_dict["Information"] = information
        if alpha_forward is not None:
            edge_dict["Alpha_Forward"] = alpha_forward
        if alpha_backward is not None:
            edge_dict["Alpha_Backward"] = alpha_backward
        if metric is not None:
            edge_dict["Metric"] = metric
        if bandwidth is not None:
            edge_dict["Bandwidth"] = bandwidth
        if k_means is not None:
            edge_dict["K_Means"] = k_means
        if n_shuffles is not None:
            edge_dict["N_Shuffles"] = n_shuffles
        if max_lag is not None:
            edge_dict["Max_Lag"] = max_lag

        edges_data.append(edge_dict)

    df = pd.DataFrame(edges_data)

    # Handle empty graph case
    if len(df) == 0:
        base_cols = ["Source", "Sink", "Lag", "CMI", "P_Value"]
        return pd.DataFrame(columns=base_cols)

    # Reorder columns to have base columns first, then optional metadata
    base_cols = ["Source", "Sink", "Lag", "CMI", "P_Value"]
    metadata_order = [
        "Method",
        "Information",
        "Alpha_Forward",
        "Alpha_Backward",
        "Metric",
        "Bandwidth",
        "K_Means",
        "N_Shuffles",
        "Max_Lag",
    ]

    # Build final column order
    final_col_order = base_cols + [col for col in metadata_order if col in df.columns]
    df = df[final_col_order]

    return df


def pcmci_network_to_dataframe(G: nx.MultiDiGraph) -> pd.DataFrame:
    """Convert a PCMCI-style graph into a tabular representation.

    Handles graphs produced by :func:`pcmci_to_networkx`, including optional
    binarized significance flags, undirected/conflicting edges, and empty graphs.
    """

    base_columns = ["Source", "Sink", "Lag", "Val", "P_Value", "Link_Type"]
    optional_columns = ["Significant"]

    rows = []
    processed_undirected = set()

    for u, v, data in G.edges(data=True):
        link_type = data.get("link_type", "directed")
        lag = data.get("lag", 0)

        if link_type in {"undirected", "conflicting"}:
            canonical_nodes = tuple(sorted((u, v), key=str))
            dedup_key = canonical_nodes + (lag, link_type)
            if dedup_key in processed_undirected:
                continue
            processed_undirected.add(dedup_key)
            source, sink = canonical_nodes
        else:
            source, sink = u, v

        val_attr = data.get("val", data.get("cmi"))
        if val_attr in (None, ""):
            val = None
        else:
            val = float(np.asarray(val_attr))

        p_attr = data.get("p_value")
        if p_attr in (None, ""):
            p_val = None
        else:
            p_val = float(np.asarray(p_attr))

        row = {
            "Source": source,
            "Sink": sink,
            "Lag": lag,
            "Val": val,
            "P_Value": p_val,
            "Link_Type": link_type,
        }

        if "significant" in data:
            row["Significant"] = bool(data["significant"])

        rows.append(row)

    if not rows:
        return pd.DataFrame(columns=base_columns + optional_columns)

    df = pd.DataFrame(rows)
    ordered_columns = base_columns + [
        col for col in optional_columns if col in df.columns
    ]
    return df[ordered_columns]
