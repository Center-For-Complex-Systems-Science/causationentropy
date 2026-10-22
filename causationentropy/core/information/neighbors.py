import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.distance import cdist

_METRIC_TO_MINKOWSKI_P = {
    "euclidean": 2.0,
    "minkowski": 2.0,
    "cityblock": 1.0,
    "manhattan": 1.0,
    "chebyshev": np.inf,
}


def supports_kd_tree(metric):
    return metric in _METRIC_TO_MINKOWSKI_P


def kth_neighbor_distance(X, k, metric="euclidean", kd_tree=True):
    X = np.asarray(X)

    if kd_tree and supports_kd_tree(metric):
        p = _METRIC_TO_MINKOWSKI_P[metric]
        tree = cKDTree(X)
        dist, _ = tree.query(X, k=k + 1, p=p)
        return dist[:, k]

    D = np.sort(cdist(X, X, metric=metric), axis=1)
    return D[:, k]


def radius_neighbor_counts(X, radii, metric="euclidean", kd_tree=True):
    X = np.asarray(X)
    radii = np.asarray(radii)

    if kd_tree and supports_kd_tree(metric):
        p = _METRIC_TO_MINKOWSKI_P[metric]
        tree = cKDTree(X)
        candidates = tree.query_ball_point(X, r=radii, p=p)
        counts = np.empty(X.shape[0], dtype=int)
        for i, idx in enumerate(candidates):
            idx = np.asarray(idx)
            d = cdist(X[i : i + 1], X[idx], metric=metric)[0]
            counts[i] = np.sum(d < radii[i]) - 1
        return counts

    D = cdist(X, X, metric=metric)
    return np.sum(D < radii[:, None], axis=1) - 1


def knn_indices(X, k, metric="euclidean", kd_tree=True):
    X = np.asarray(X)

    if kd_tree and supports_kd_tree(metric):
        p = _METRIC_TO_MINKOWSKI_P[metric]
        tree = cKDTree(X)
        _, idx = tree.query(X, k=k + 1, p=p)
        return idx[:, 1 : k + 1]

    D = cdist(X, X, metric=metric)
    return np.argsort(D, axis=1)[:, 1 : k + 1]
