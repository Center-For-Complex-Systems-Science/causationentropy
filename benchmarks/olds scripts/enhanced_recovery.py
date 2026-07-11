import numpy as np
import networkx as nx
from causationentropy.core.linalg import companion_matrix


def recover_node_adj_with_companion(
    graph_nx,
    basis_meta,
    tau_max=1,
    use_lag_consistency=True,
    consistency_weight=0.5,
    threshold=0.01,
):
    """
    Recover node-level adjacency from expanded-variable graph using companion matrix structure.

    Key insight: If the true system is lag-1 Kuramoto dynamics, then:
    - Direct lag-1 effects: s_{i←j}(t-1) → v_i(t) should be strongest
    - Lag-k effects should be consistent with lag-1^k (weaker, transitive)

    We can use this to:
    1. Extract primary evidence from lag-1 relationships
    2. Use higher lags to validate/strengthen conclusions
    3. Penalize inconsistencies that suggest spurious edges

    Parameters
    ----------
    graph_nx : networkx.DiGraph
        Discovered graph with nodes labeled as (variable_index, -lag)
    basis_meta : dict
        Metadata from build_kuramoto_basis containing:
        - n: number of original nodes
        - pred_offset: where coupling variables start (= n)
        - basis_map: list mapping coupling var index to (i,j) pair
    tau_max : int
        Maximum lag used in discovery
    use_lag_consistency : bool
        Whether to use higher-lag information for validation
    consistency_weight : float
        How much to weight consistency vs direct evidence (0-1)
    threshold : float
        Minimum evidence threshold for edge inclusion

    Returns
    -------
    A_node : (n, n) np.ndarray
        Recovered adjacency where A_node[j, i] indicates j → i strength
    """
    n = basis_meta["n"]
    pred_offset = basis_meta["pred_offset"]
    basis_map = basis_meta["basis_map"]

    # Build evidence matrices for each lag
    evidence_by_lag = {}
    for lag in range(1, tau_max + 1):
        evidence_by_lag[lag] = np.zeros((n, n), dtype=float)

    # Extract evidence from graph edges
    for edge in graph_nx.edges(data=True):
        source_node, target_node, edge_data = edge
        src_var, src_lag = source_node
        tgt_var, tgt_lag = target_node

        # We want: coupling term at lag -k → velocity at lag 0
        if tgt_lag != 0:
            continue
        if tgt_var >= n:
            continue  # Target must be velocity
        if src_var < pred_offset:
            continue  # Source must be coupling term

        lag = -src_lag
        if lag < 1 or lag > tau_max:
            continue

        coupling_idx = src_var - pred_offset
        if coupling_idx >= len(basis_map):
            continue

        i, j = basis_map[coupling_idx]  # s_{i←j}

        if tgt_var != i:
            continue  # Coupling target must match velocity variable

        weight = edge_data.get("weight", 1.0)
        evidence_by_lag[lag][j, i] = abs(weight)

    # Primary evidence from lag-1
    A_primary = evidence_by_lag[1]

    if not use_lag_consistency or tau_max == 1:
        # Simple case: just threshold lag-1 evidence
        A_node = (A_primary > threshold).astype(float)
        return A_node

    # Advanced case: use companion matrix structure for validation
    # Build expected higher-lag evidence from lag-1
    A_binary = (A_primary > threshold).astype(float)

    # Compute expected evidence at each lag assuming lag-1 dynamics
    expected_by_lag = {}
    expected_by_lag[1] = A_primary.copy()

    A_current = A_primary.copy()
    for lag in range(2, tau_max + 1):
        # Expected lag-k effect is approximately proportional to A^k
        # (not exact due to nonlinearities, but should correlate)
        A_current = A_current @ A_primary
        expected_by_lag[lag] = A_current

    # Compute consistency score for each potential edge
    consistency_scores = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(n):
            if A_primary[j, i] < threshold:
                continue  # No primary evidence

            # Check if higher lags support this edge
            consistencies = []
            for lag in range(2, tau_max + 1):
                observed = evidence_by_lag[lag][j, i]
                expected = expected_by_lag[lag][j, i]

                if expected > 1e-6:  # Avoid division by zero
                    # Consistency: observed should be somewhat proportional to expected
                    ratio = observed / expected if expected > 0 else 0
                    # Good consistency: ratio near 1
                    # Over-evidence: ratio > 1 (could be spurious or confounding)
                    # Under-evidence: ratio < 1 (could be real but weak transitively)
                    consistency = np.exp(-abs(np.log(ratio + 1e-6)))
                    consistencies.append(consistency)

            if consistencies:
                consistency_scores[j, i] = np.mean(consistencies)
            else:
                consistency_scores[j, i] = 1.0  # No higher lag info = assume consistent

    # Combine primary evidence with consistency
    final_evidence = (
        1 - consistency_weight
    ) * A_primary + consistency_weight * A_primary * consistency_scores

    A_node = (final_evidence > threshold).astype(float)
    return A_node


def recover_node_adj_simple(graph_nx, basis_meta, threshold=0.0):
    """
    Simple recovery: just look at lag-1 coupling → velocity edges.

    This is the baseline approach without companion matrix structure.
    """
    n = basis_meta["n"]
    pred_offset = basis_meta["pred_offset"]
    basis_map = basis_meta["basis_map"]

    A_node = np.zeros((n, n), dtype=float)

    for edge in graph_nx.edges(data=True):
        source_node, target_node, edge_data = edge
        src_var, src_lag = source_node
        tgt_var, tgt_lag = target_node

        # Lag-1 coupling → lag-0 velocity
        if tgt_lag != 0 or src_lag != -1:
            continue
        if tgt_var >= n or src_var < pred_offset:
            continue

        coupling_idx = src_var - pred_offset
        if coupling_idx >= len(basis_map):
            continue

        i, j = basis_map[coupling_idx]
        if tgt_var != i:
            continue

        weight = edge_data.get("weight", 1.0)
        if abs(weight) > threshold:
            A_node[j, i] = 1.0

    return A_node.astype(int)


def analyze_lag_structure(graph_nx, basis_meta, tau_max=1):
    """
    Diagnostic function to understand what the discovery algorithm found.

    Returns statistics about edges at each lag to help debug recovery.
    """
    n = basis_meta["n"]
    pred_offset = basis_meta["pred_offset"]
    basis_map = basis_meta["basis_map"]

    stats = {
        "velocity_to_velocity": {},  # v → v edges by lag
        "coupling_to_velocity": {},  # s → v edges by lag (what we want)
        "velocity_to_coupling": {},  # v → s edges by lag (confounding?)
        "coupling_to_coupling": {},  # s → s edges by lag (indirect?)
    }

    for lag in range(tau_max + 1):
        stats["velocity_to_velocity"][lag] = []
        stats["coupling_to_velocity"][lag] = []
        stats["velocity_to_coupling"][lag] = []
        stats["coupling_to_coupling"][lag] = []

    for edge in graph_nx.edges(data=True):
        source_node, target_node, edge_data = edge
        src_var, src_lag = source_node
        tgt_var, tgt_lag = target_node

        weight = edge_data.get("weight", 1.0)
        lag = tgt_lag - src_lag  # Positive lag means source is in past

        if lag < 0 or lag > tau_max:
            continue

        src_is_velocity = src_var < pred_offset
        tgt_is_velocity = tgt_var < pred_offset

        edge_info = {
            "src_var": src_var,
            "tgt_var": tgt_var,
            "weight": weight,
            "lag": lag,
        }

        if src_is_velocity and tgt_is_velocity:
            stats["velocity_to_velocity"][lag].append(edge_info)
        elif not src_is_velocity and tgt_is_velocity:
            stats["coupling_to_velocity"][lag].append(edge_info)
        elif src_is_velocity and not tgt_is_velocity:
            stats["velocity_to_coupling"][lag].append(edge_info)
        else:
            stats["coupling_to_coupling"][lag].append(edge_info)

    # Print summary
    print("\n" + "=" * 80)
    print("EDGE TYPE DISTRIBUTION BY LAG")
    print("=" * 80)
    for lag in range(tau_max + 1):
        print(f"\nLag {lag}:")
        print(f"  v → v: {len(stats['velocity_to_velocity'][lag])} edges")
        print(
            f"  s → v: {len(stats['coupling_to_velocity'][lag])} edges (PRIMARY SIGNAL)"
        )
        print(f"  v → s: {len(stats['velocity_to_coupling'][lag])} edges")
        print(f"  s → s: {len(stats['coupling_to_coupling'][lag])} edges")

    return stats


# Example usage comparison
if __name__ == "__main__":
    print("This module provides enhanced adjacency recovery functions.")
    print("\nKey functions:")
    print("1. recover_node_adj_simple() - Baseline lag-1 only recovery")
    print("2. recover_node_adj_with_companion() - Uses companion matrix structure")
    print("3. analyze_lag_structure() - Diagnostic tool for debugging")
    print("\nThe companion approach validates edges using consistency across lags,")
    print("following the mathematical structure where A_k ≈ A_1^k for true dynamics.")
