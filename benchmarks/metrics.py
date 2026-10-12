"""Edge-set normalization and graph-recovery metrics for issue #18.

Canonical rule (see ``benchmarks/README.md``): a true positive requires an
exact ``(source, target, lag)`` match. Cross-links are the main metric;
self-lags are reported separately because the standard oCSE method
conditions on the target's own lags and cannot discover them.
"""

from __future__ import annotations


def normalize_ce_edges(graph, node_order) -> set:
    """Normalize a ``discover_network`` graph to ``(source, target, lag)`` ints.

    Parameters
    ----------
    graph:
        NetworkX MultiDiGraph from :func:`discover_network`. Nodes are
        expected to be ``'X{k}'`` labels matching ``node_order``.
    node_order:
        Sequence mapping column index to node label.
    """
    index = {node: k for k, node in enumerate(node_order)}
    edges = set()
    for u, v, data in graph.edges(data=True):
        lag = int(data.get("lag", 0))
        if lag < 1:
            continue
        edges.add((int(index[u]), int(index[v]), lag))
    return edges


def normalize_pcmci_edges(graph) -> set:
    """Normalize a :func:`pcmci_to_networkx` graph to ``(source, target, lag)``.

    Only directed links at lag >= 1 are counted as discoveries;
    undirected / conflicting / contemporaneous links are ignored.
    """
    edges = set()
    for u, v, data in graph.edges(data=True):
        if data.get("link_type", "directed") != "directed":
            continue
        lag = int(data.get("lag", 0))
        if lag < 1:
            continue
        edges.add((int(u), int(v), lag))
    return edges


def score_edges(truth: set, predicted: set) -> dict:
    """Score ``predicted`` against ``truth`` with exact-match counting.

    Returns overall counts plus a cross-link / self-lag breakdown
    (``cross_*`` is the main metric; ``self_*`` is reported separately).
    """
    tp = len(truth & predicted)
    fp = len(predicted - truth)
    fn = len(truth - predicted)
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    out = {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "n_truth": len(truth),
        "n_predicted": len(predicted),
    }
    truth_self = {e for e in truth if e[0] == e[1]}
    predicted_self = {e for e in predicted if e[0] == e[1]}
    cross_truth = truth - truth_self
    cross_predicted = predicted - predicted_self
    cross_tp = len(cross_truth & cross_predicted)
    cross_fp = len(cross_predicted - cross_truth)
    cross_fn = len(cross_truth - cross_predicted)
    cross_precision = cross_tp / (cross_tp + cross_fp) if (cross_tp + cross_fp) else 0.0
    cross_recall = cross_tp / (cross_tp + cross_fn) if (cross_tp + cross_fn) else 0.0
    cross_f1 = (
        2 * cross_precision * cross_recall / (cross_precision + cross_recall)
        if (cross_precision + cross_recall)
        else 0.0
    )
    out.update(
        {
            "cross_TP": cross_tp,
            "cross_FP": cross_fp,
            "cross_FN": cross_fn,
            "cross_precision": cross_precision,
            "cross_recall": cross_recall,
            "cross_F1": cross_f1,
            "self_TP": len(truth_self & predicted_self),
            "self_FP": len(predicted_self - truth_self),
            "self_FN": len(truth_self - predicted_self),
        }
    )
    return out
