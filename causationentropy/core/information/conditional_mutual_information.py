import functools
import hashlib

import numpy as np
from scipy.spatial.distance import cdist

from causationentropy.core.information.entropy import (
    geometric_knn_entropy,
    kde_entropy,
    poisson_entropy,
    poisson_joint_entropy,
)
from causationentropy.core.information.mutual_information import (
    gaussian_mutual_information,
    geometric_knn_mutual_information,
    kde_mutual_information,
    knn_mutual_information,
)


def _array_hash(arr, metric="euclidean"):
    """Create a hashable key for numpy arrays to use with LRU cache."""
    return hashlib.md5(f"{arr.tobytes()}{metric}".encode()).hexdigest()


def estimate_cache_size(n_vars, max_lag, n_samples):
    """
    Estimate optimal LRU cache size for distance matrices.

    Parameters
    ----------
    n_vars : int
        Number of variables in the dataset
    max_lag : int
        Maximum lag considered
    n_samples : int
        Number of time samples

    Returns
    -------
    cache_size : int
        Recommended cache size
    memory_mb : float
        Estimated memory usage in MB
    """
    # Number of unique predictor combinations per variable
    predictors_per_var = n_vars * max_lag

    # Forward selection: at most predictors_per_var evaluations per variable
    # Backward elimination: at most len(selected) evaluations per variable
    # Conservative estimate: 2 * predictors_per_var operations per variable
    operations_per_var = 2 * predictors_per_var

    # Total operations across all variables
    total_operations = n_vars * operations_per_var

    # Distance matrices have unique combinations:
    # X, Y, Z, XZ, YZ, XYZ combinations
    # Conservative estimate: 6 unique combinations per operation
    unique_matrices = min(total_operations * 6, 1000)  # Cap at reasonable limit

    # Memory estimation (8 bytes per float64 element)
    avg_matrix_size = n_samples * n_samples * 8  # bytes
    memory_bytes = unique_matrices * avg_matrix_size
    memory_mb = memory_bytes / (1024 * 1024)

    # Recommend cache size (balance memory vs hit rate)
    if memory_mb < 100:
        cache_size = unique_matrices
    elif memory_mb < 500:
        cache_size = int(unique_matrices * 0.5)
    else:
        cache_size = min(200, int(unique_matrices * 0.2))

    return cache_size, memory_mb


# Create a configurable cache - default size, can be updated
_distance_cache_size = 128


def set_distance_cache_size(size):
    """Set the cache size for distance matrix computations."""
    global _distance_cache_size
    _distance_cache_size = size


def get_cache_stats():
    """
    Get statistics about the current cache usage.

    Returns
    -------
    stats : dict
        Dictionary containing cache statistics:
        - distance_cache_size: Current number of cached distance matrices
        - distance_cache_limit: Maximum cache size for distance matrices
        - detcorr_cache_size: Current number of cached determinants
        - detcorr_cache_limit: Maximum cache size for determinants
    """
    return {
        "distance_cache_size": len(_distance_cache),
        "distance_cache_limit": _distance_cache_size,
        "detcorr_cache_size": len(_detcorr_cache),
        "detcorr_cache_limit": 128,
    }


def clear_caches():
    """Clear all caches to free memory."""
    _distance_cache.clear()
    _detcorr_cache.clear()


def configure_cache_for_discovery(data_shape, max_lag=5, information_method="knn"):
    """
    Configure optimal cache settings for causal discovery.

    This is a convenience function that automatically sets up caching
    based on your data and analysis parameters.

    Parameters
    ----------
    data_shape : tuple
        Shape of your data (n_samples, n_variables)
    max_lag : int, default=5
        Maximum lag for causal discovery
    information_method : str, default='knn'
        Information estimation method ('knn', 'geometric_knn', 'gaussian', etc.)

    Returns
    -------
    config : dict
        Cache configuration details including recommended settings
    """
    n_samples, n_vars = data_shape

    # Estimate cache requirements
    cache_size, memory_mb = estimate_cache_size(n_vars, max_lag, n_samples)

    # Adjust based on information method
    if information_method in ["knn", "geometric_knn"]:
        # These methods are distance-heavy, benefit most from caching
        cache_multiplier = 1.0
    elif information_method == "gaussian":
        # Less distance computation, smaller cache sufficient
        cache_multiplier = 0.3
    elif information_method == "kde":
        # Moderate distance usage
        cache_multiplier = 0.6
    else:
        cache_multiplier = 0.5

    # Apply multiplier and ensure reasonable bounds
    adjusted_cache_size = max(16, min(1000, int(cache_size * cache_multiplier)))

    # Set the cache size
    set_distance_cache_size(adjusted_cache_size)

    # Clear existing caches to start fresh
    clear_caches()

    config = {
        "data_shape": data_shape,
        "max_lag": max_lag,
        "information_method": information_method,
        "cache_size": adjusted_cache_size,
        "estimated_memory_mb": memory_mb * cache_multiplier,
        "cache_efficiency": (
            "high" if information_method in ["knn", "geometric_knn"] else "moderate"
        ),
    }

    print(f"Cache configured for {information_method} method:")
    print(f"  Cache size: {adjusted_cache_size}")
    print(f"  Estimated memory: {config['estimated_memory_mb']:.1f} MB")
    print(f"  Cache efficiency: {config['cache_efficiency']}")

    return config


# Global cache for distance matrices
_distance_cache = {}


def cached_cdist(data, metric="euclidean"):
    """
    Cached distance matrix computation.

    Parameters
    ----------
    data : array-like of shape (n_samples, n_features)
        Input data matrix
    metric : str, default='euclidean'
        Distance metric

    Returns
    -------
    distances : ndarray of shape (n_samples, n_samples)
        Pairwise distance matrix
    """
    data = np.asarray(data)

    # Create a cache key from data content and parameters
    data_hash = _array_hash(data, metric)

    # Check if result is already cached
    if data_hash in _distance_cache:
        cached_result = _distance_cache[data_hash]
        # Verify cached result has correct shape
        if cached_result.shape[0] != data.shape[0]:
            # Cache collision detected, remove invalid entry
            del _distance_cache[data_hash]
        else:
            return cached_result

    # Compute distance matrix
    result = cdist(data, data, metric=metric)

    # Manage cache size
    if len(_distance_cache) >= _distance_cache_size:
        # Remove oldest entry (simple FIFO, could be improved to LRU)
        oldest_key = next(iter(_distance_cache))
        del _distance_cache[oldest_key]

    # Cache the result
    _distance_cache[data_hash] = result

    return result


# Global cache for correlation determinants
_detcorr_cache = {}


def cached_detcorr(A):
    """
    Cached correlation determinant computation.

    Parameters
    ----------
    A : array-like
        Input data matrix

    Returns
    -------
    det : float
        Log determinant of correlation matrix
    """
    A = np.asarray(A)
    data_hash = _array_hash(A)

    # Check if result is already cached
    if data_hash in _detcorr_cache:
        return _detcorr_cache[data_hash]

    # Compute correlation determinant
    C = np.corrcoef(A.T)
    result = float(C) if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]

    # Manage cache size (keep it smaller for detcorr)
    if len(_detcorr_cache) >= 128:
        oldest_key = next(iter(_detcorr_cache))
        del _detcorr_cache[oldest_key]

    # Cache the result
    _detcorr_cache[data_hash] = result

    return result


def gaussian_conditional_mutual_information(X, Y, Z=None):
    r"""
    Compute conditional mutual information for multivariate Gaussian variables.

    For multivariate Gaussian variables, the conditional mutual information has
    a closed-form expression using covariance matrix determinants:

    .. math::

        I(X; Y | Z) = \frac{1}{2} \log \frac{|\Sigma_{XZ}| |\Sigma_{YZ}|}{|\Sigma_Z| |\Sigma_{XYZ}|}

    This can also be expressed as:

    .. math::

        I(X; Y | Z) = \frac{1}{2} [\log |\Sigma_{XZ}| + \log |\Sigma_{YZ}| - \log |\Sigma_Z| - \log |\Sigma_{XYZ}|]

    where :math:`\Sigma_{\cdot}` denotes the covariance matrix of the subscripted variables.

    Parameters
    ----------
    X : array-like of shape (N, k_x)
        First variable with N samples and k_x features.
    Y : array-like of shape (N, k_y)
        Second variable with N samples and k_y features.
    Z : array-like of shape (N, k_z) or None
        Conditioning variable with N samples and k_z features.
        If None, computes marginal mutual information I(X;Y).

    Returns
    -------
    I : float
        Conditional mutual information in nats.

    Notes
    -----
    This implementation uses log-determinants of correlation matrices for
    numerical stability, employing the signed log-determinant function
    to handle potential numerical issues.

    The Gaussian assumption implies that:
    - All conditional dependencies are captured by linear relationships
    - Higher-order moments beyond covariance carry no information
    - The estimator is exact under Gaussianity

    For non-Gaussian data, this estimator provides a lower bound on the
    true conditional mutual information.
    """
    if Z is None:
        return gaussian_mutual_information(X, Y)

    SZ = cached_detcorr(Z)
    SXZ = cached_detcorr(np.hstack((X, Z)))
    SYZ = cached_detcorr(np.hstack((Y, Z)))
    SXYZ = cached_detcorr(np.hstack((X, Y, Z)))

    return 0.5 * (SXZ + SYZ - SZ - SXYZ)


def kde_conditional_mutual_information(
    X, Y, Z, bandwidth="silverman", kernel="gaussian"
):
    """
    Estimate conditional mutual information using Kernel Density Estimation.

    This function computes conditional mutual information using the entropy decomposition:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where each entropy term is estimated using kernel density estimation.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, reduces to marginal mutual information.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE.
    kernel : str, default='gaussian'
        Kernel function for density estimation.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    The KDE approach can capture nonlinear conditional dependencies but suffers from:
    - Curse of dimensionality for high-dimensional conditioning sets
    - Bandwidth selection sensitivity
    - Computational complexity scaling with sample size

    Consider k-NN methods for high-dimensional problems or large datasets.
    """
    if Z is None:
        I = kde_mutual_information(X, Y, bandwidth=bandwidth, kernel=kernel)
    else:

        XZ = np.hstack((X, Z))
        YZ = np.hstack((Y, Z))
        XYZ = np.hstack((X, Y, Z))

        # Compute the entropies
        Hz = kde_entropy(Z, bandwidth=bandwidth, kernel=kernel)
        Hxz = kde_entropy(XZ, bandwidth=bandwidth, kernel=kernel)
        Hyz = kde_entropy(YZ, bandwidth=bandwidth, kernel=kernel)
        Hxyz = kde_entropy(XYZ, bandwidth=bandwidth, kernel=kernel)
        I = Hxz + Hyz - Hxyz - Hz

    return I


def knn_conditional_mutual_information(X, Y, Z, metric="euclidean", k=1):
    """
    Estimate conditional mutual information using k-nearest neighbor method.

    This function implements conditional mutual information estimation using
    the relationship:

    .. math::

        I(X; Y | Z) = I(X, Y) - I(X, Y; Z)

    where both mutual information terms are estimated using the KSG k-NN estimator.

    The approach leverages the fact that:

    .. math::

        I(X; Y | Z) = I(X; Y) - I(X; Y | Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='euclidean'
        Distance metric for k-NN calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Notes
    -----
    This implementation uses the decomposition approach rather than direct
    conditional MI estimation. The accuracy depends on:

    - Quality of marginal MI estimates
    - Dimensionality of the joint space
    - Sample size relative to effective dimensionality

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., Grassberger, P. Estimating mutual information.
           Physical Review E 69, 066138 (2004).
    """
    if Z is None:
        return knn_mutual_information(
            X, Y, metric=metric, k=k
        )  # np.max([self.MutualInfo_KNN(X,self.Y),0])
    else:
        XY = np.concatenate((X, Y), axis=1)
        MIXYZ = knn_mutual_information(XY, Z, metric=metric, k=k)
        MIXY = knn_mutual_information(X, Y, metric=metric, k=k)

        return MIXY - MIXYZ


def geometric_knn_conditional_mutual_information(X, Y, Z, metric="euclidean", k=1):
    """
    Estimate conditional mutual information using geometric k-nearest neighbor method.

    This function applies the geometric k-NN entropy estimator to compute
    conditional mutual information via the entropy decomposition:

    .. math::

        I(X; Y | Z) = H_{\text{geom}}(X, Z) + H_{\text{geom}}(Y, Z) - H_{\text{geom}}(Z) - H_{\text{geom}}(X, Y, Z)

    The geometric correction accounts for local manifold structure, providing
    improved estimates for data with non-uniform density or intrinsic dimensionality
    lower than the ambient space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information.
    metric : str, default='euclidean'
        Distance metric for neighbor calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated conditional mutual information using geometric k-NN method.

    Notes
    -----
    The geometric approach is particularly effective for:
    - Data on lower-dimensional manifolds
    - Non-uniform density distributions
    - Cases where local geometric structure is important

    The method accounts for the effective local dimensionality through
    geometric corrections to the standard k-NN entropy estimates.

    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of
           entropy and mutual information. Chaos 28, 033113 (2018).
    """

    if Z is None:
        return geometric_knn_mutual_information(X, Y)
    YZdist = cached_cdist(np.hstack((Y, Z)), metric=metric)
    XZdist = cached_cdist(np.hstack((X, Z)), metric=metric)
    XYZdist = cached_cdist(np.hstack((X, Y, Z)), metric=metric)
    Zdist = cached_cdist(Z, metric=metric)
    HZ = geometric_knn_entropy(Z, Zdist, k)
    HXZ = geometric_knn_entropy(np.hstack((X, Z)), XZdist, k)
    HYZ = geometric_knn_entropy(np.hstack((Y, Z)), YZdist, k)
    HXYZ = geometric_knn_entropy(np.hstack((X, Y, Z)), XYZdist, k)
    return HXZ + HYZ - HXYZ - HZ


def poisson_conditional_mutual_information(X, Y, Z):
    """
    Estimate conditional mutual information for multivariate Poisson distributions.

    This function computes conditional mutual information for discrete count data
    assuming Poisson distributions. The estimation uses the covariance structure
    of the multivariate Poisson distribution:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    where entropies are computed using Poisson-specific formulations that account
    for the discrete nature and parameter structure of Poisson variables.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        Count data from first Poisson variables.
    Y : array-like of shape (n_samples, n_features_y)
        Count data from second Poisson variables.
    Z : array-like of shape (n_samples, n_features_z) or None
        Count data from conditioning Poisson variables.
        If None, computes marginal mutual information.

    Returns
    -------
    I : float
        Estimated conditional mutual information for Poisson data.

    Notes
    -----
    This implementation is specifically designed for discrete count data where:
    - Variables follow Poisson distributions
    - Dependencies are captured through covariance structure
    - Joint distributions maintain Poisson-like properties

    Applications include:
    - Gene expression count data
    - Event occurrence data
    - Discrete interaction networks
    - Epidemiological count models

    References
    ----------
    .. [1] Fish, A., Sun, J., Bollt, E. Interaction networks from discrete event data by
           Poisson multivariate mutual information estimation and information flow with
           applications from gene expression data. (In preparation)
    """

    if Z is None:
        SXY = np.corrcoef(X.T, Y.T)
        l_est = SXY - np.diag(np.diag(SXY))
        np.fill_diagonal(SXY, np.diagonal(SXY) - np.sum(l_est, axis=0))
        Dcov = np.diag(SXY) + np.sum(l_est, axis=0)
        TF = poisson_joint_entropy(SXY)
        FT = np.sum(poisson_entropy(Dcov))

        return FT - TF
    else:
        SzX = X.shape[1]
        SzY = Y.shape[1]
        SzZ = Z.shape[1]
        indX = np.arange(SzX)
        indY = np.arange(SzY) + SzX
        indZ = np.arange(SzZ) + SzX + SzY
        XYZ = np.concatenate((X, Y, Z), axis=1)
        SXYZ = np.corrcoef(XYZ.T)
        SS = SXYZ
        Sa = SXYZ - np.diag(np.diag(SXYZ))
        np.fill_diagonal(SS, np.diagonal(SS) - Sa)
        SS[0:SzX, 0:SzX] = SS[0:SzX, 0:SzX] + SXYZ[0:SzX, SzX : SzX + SzY]
        SS[SzX : SzX + SzY, SzX : SzX + SzY] = (
            SS[SzX : SzX + SzY, SzX : SzX + SzY] + SXYZ[SzX : SzX + SzY, 0:SzX]
        )
        S_est1 = SS[np.concatenate((indY, indZ)), :][:, np.concatenate((indY, indZ))]
        S_est2 = SS[np.concatenate((indX, indZ)), :][:, np.concatenate((indX, indZ))]
        HYZ = poisson_joint_entropy(S_est1)
        SindZ = SS[indZ, :][:, indZ]
        HZ = poisson_joint_entropy(SindZ)
        HXYZ = poisson_joint_entropy(SXYZ - np.diag(Sa))
        HXZ = poisson_joint_entropy(S_est2)
        H_YZ = HYZ - HZ
        H_XYZ = HXYZ - HXZ
        return H_XYZ - H_YZ


def conditional_mutual_information(
    X,
    Y,
    Z=None,
    method="gaussian",
    metric="euclidean",
    k=6,
    bandwidth="silverman",
    kernel="gaussian",
):
    """
    Compute conditional mutual information using specified estimation method.

    This function provides a unified interface for computing conditional mutual information
    I(X;Y|Z) using various estimation approaches. The choice of method depends on the
    data type, dimensionality, and distributional assumptions.

    Conditional mutual information quantifies the information shared between X and Y
    when conditioning on Z:

    .. math::

        I(X; Y | Z) = H(X | Z) - H(X | Y, Z)

    Equivalently:

    .. math::

        I(X; Y | Z) = H(X, Z) + H(Y, Z) - H(Z) - H(X, Y, Z)

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    Z : array-like of shape (n_samples, n_features_z) or None
        Conditioning variable. If None, computes marginal mutual information I(X;Y).
    method : str, default='gaussian'
        Estimation method. Available options:

        - 'gaussian': Assumes multivariate Gaussian distributions
        - 'kde' or 'kernel_density': Kernel density estimation
        - 'knn': k-nearest neighbor (KSG) estimator
        - 'geometric_knn': Geometric k-NN with manifold corrections
        - 'poisson': For discrete count data with Poisson assumptions

    metric : str, default='euclidean'
        Distance metric for k-NN based methods.
    k : int, default=1
        Number of nearest neighbors for k-NN methods.
    bandwidth : str or float, default='silverman'
        Bandwidth parameter for KDE methods.
    kernel : str, default='gaussian'
        Kernel function for KDE methods.

    Returns
    -------
    I : float
        Estimated conditional mutual information in nats.

    Raises
    ------
    ValueError
        If an unsupported method is specified.

    Notes
    -----
    **Method Selection Guidelines:**

    - **Gaussian**: Best for linear relationships, exact under Gaussianity
    - **KDE**: Good for smooth nonlinear dependencies, curse of dimensionality
    - **k-NN**: Robust for moderate dimensions, adapts to local density
    - **Geometric k-NN**: Effective for manifold data with intrinsic structure
    - **Poisson**: Specifically for discrete count data
    - **Histogram**: Simple baseline, sensitive to binning

    **Computational Complexity:**
    - Gaussian: O(n³) for matrix operations
    - KDE: O(n²) for density evaluation
    - k-NN: O(n² log n) for neighbor finding

    **Sample Size Requirements:**
    - Increase with dimensionality and complexity of dependencies
    - k-NN methods generally require fewer samples than KDE
    - Parametric methods (Gaussian) most sample-efficient when assumptions hold

    Examples
    --------
    >>> import numpy as np
    >>> from causationentropy.core.information.conditional_mutual_information import conditional_mutual_information
    >>>
    >>> # Generate sample data
    >>> n = 1000
    >>> X = np.random.randn(n, 2)
    >>> Y = np.random.randn(n, 1)
    >>> Z = np.random.randn(n, 1)
    >>>
    >>> # Compute conditional MI using different methods
    >>> cmi_gauss = conditional_mutual_information(X, Y, Z, method='gaussian')
    >>> cmi_knn = conditional_mutual_information(X, Y, Z, method='knn', k=3)
    >>>
    >>> print(f"Gaussian CMI: {cmi_gauss:.3f}")
    >>> print(f"k-NN CMI: {cmi_knn:.3f}")
    """
    if method == "gaussian":
        return gaussian_conditional_mutual_information(X, Y, Z)

    elif method == "kde" or method == "kernel_density":
        return kde_conditional_mutual_information(
            X, Y, Z, bandwidth=bandwidth, kernel=kernel
        )

    elif method == "knn":
        return knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "geometric_knn":
        return geometric_knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "poisson":
        return poisson_conditional_mutual_information(X, Y, Z)

    else:
        supported_methods = [
            "gaussian",
            "kde",
            "kernel_density",
            "knn",
            "geometric_knn",
            "poisson",
        ]
        raise ValueError(
            f"Method '{method}' unavailable. Supported methods: {supported_methods}"
        )
