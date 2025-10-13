import numpy as np
import networkx as nx

LINK_TYPE_SEMANTICS = {
    '-->': 'directed',
    '<--': 'directed',
    'o-o': 'undirected',
    '-?>': 'possible_directed',
    'x-x': 'conflicting',
}

SEMANTIC_TO_LINK_TYPE = {
    'directed': '-->',
    'undirected': 'o-o',
    'possible_directed': '-?>',
    'conflicting': 'x-x',
}

def pcmci_to_networkx(results, binarize=False, p_value=0.05):
    graph = np.asarray(results['graph'])
    val_matrix = np.asarray(results['val_matrix'])
    p_matrix = np.asarray(results['p_matrix'])

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

    all_strengths = []

    for i in range(N):
        for j in range(N):
            for lag in range(tau_max_plus_1):
                link_type_symbol = graph[i, j, lag]
                if link_type_symbol == '':
                    continue

                semantic_link_type = LINK_TYPE_SEMANTICS.get(link_type_symbol)
                if semantic_link_type is None:
                    raise ValueError(f"Unknown link type: {link_type_symbol}")

                val = float(np.asarray(val_matrix[i, j, lag]))
                p = float(np.asarray(p_matrix[i, j, lag]))

                edge_attrs = {
                    'lag': lag,
                    'val': val,
                    'p_value': p,
                    'link_type': semantic_link_type
                }
                if binarize:
                    edge_attrs['significant'] = p < p_value

                if link_type_symbol == '-->':
                    # graph[i, j, lag] = '-->' means j is caused by i (i -> j)
                    G.add_edge(i, j, **edge_attrs)
                    all_strengths.append(val)
                elif link_type_symbol == '<--':
                    # graph[i, j, lag] = '<--' means i is caused by j (j -> i)
                    # This is equivalent to graph[j, i, lag] = '-->'
                    G.add_edge(j, i, **edge_attrs)
                    all_strengths.append(val)
                elif link_type_symbol in ['o-o', 'x-x']:
                    # For undirected/conflicting links, PCMCI sets the same symbol at both [i,j] and [j,i]
                    # Process only once when we first encounter it (when i < j)
                    if i < j:
                        G.add_edge(i, j, **edge_attrs)
                        G.add_edge(j, i, **edge_attrs)
                        all_strengths.append(val)
                elif link_type_symbol == '-?>':
                    # Possible directed link from i to j
                    G.add_edge(i, j, **edge_attrs)
                    all_strengths.append(val)
                else:
                    raise ValueError(f"Unknown link type: {link_type_symbol}")

    return G, all_strengths

def networkx_to_pcmci(G: nx.MultiDiGraph):
    nodes = list(G.nodes())
    N = len(nodes)
    node_map = {node: i for i, node in enumerate(nodes)}

    max_lag = 0
    for _, _, data in G.edges(data=True):
        if 'lag' in data and data['lag'] > max_lag:
            max_lag = data['lag']
    
    tau_max = max_lag
    
    graph = np.full((N, N, tau_max + 1), '', dtype='<U3')
    val_matrix = np.zeros((N, N, tau_max + 1))
    p_matrix = np.ones((N, N, tau_max + 1))

    # Track processed undirected/conflicting edges to avoid duplication
    processed_undirected = set()

    for u_node, v_node, data in G.edges(data=True):
        u, v = node_map[u_node], node_map[v_node]
        lag = data.get('lag', 0)

        semantic_link_type = data.get('link_type', 'directed')

        val = data.get('val', data.get('cmi', 0.0))
        p = data.get('p_value', 1.0)

        if semantic_link_type == 'directed':
            # Directed edge from u to v
            graph[u, v, lag] = '-->'
            val_matrix[u, v, lag] = val
            p_matrix[u, v, lag] = p
        elif semantic_link_type in ['undirected', 'conflicting']:
            # For undirected/conflicting, we expect edges in both directions
            # Process only once using the canonical form (min, max)
            edge_key = (min(u, v), max(u, v), lag)
            if edge_key in processed_undirected:
                continue
            processed_undirected.add(edge_key)

            symbol = 'o-o' if semantic_link_type == 'undirected' else 'x-x'
            # Set the same symbol at both [u,v] and [v,u] positions
            graph[u, v, lag] = symbol
            graph[v, u, lag] = symbol
            val_matrix[u, v, lag] = val
            val_matrix[v, u, lag] = val
            p_matrix[u, v, lag] = p
            p_matrix[v, u, lag] = p
        elif semantic_link_type == 'possible_directed':
            # Possible directed edge from u to v
            graph[u, v, lag] = '-?>'
            val_matrix[u, v, lag] = val
            p_matrix[u, v, lag] = p
        else:
            raise ValueError(f"Unknown semantic link type: {semantic_link_type}")

    return {'graph': graph, 'val_matrix': val_matrix, 'p_matrix': p_matrix}
