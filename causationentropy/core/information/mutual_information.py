import warnings

import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

from causationentropy.core.information.entropy import geometric_knn_entropy, kde_entropy
from causationentropy.core.linalg import correlation_log_determinant


def gaussian_mutual_information(X, Y):
    r"""
    Compute mutual information for multivariate Gaussian variables using log-determinants.

    For multivariate Gaussian random variables, the mutual information has a closed-form
    expression in terms of the covariance matrices:

    .. math::

        I(X; Y) = \frac{1}{2} \log \frac{|\Sigma_X| |\Sigma_Y|}{|\Sigma_{XY}|}

    where :math:`\Sigma_X`, :math:`\Sigma_Y` are the covariance matrices of X and Y,
    and :math:`\Sigma_{XY}` is the joint covariance matrix of the concatenated vector [X, Y].

    This implementation uses correlation matrices and their log-determinants for
    numerical stability.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First multivariate Gaussian variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second multivariate Gaussian variable. Must have the same number of samples as X.

    Returns
    -------
    I : float
        Mutual information in nats (natural units).

    Notes
    -----
    This estimator is exact for multivariate Gaussian data and provides the
    theoretical benchmark for other mutual information estimators.

    The Gaussian assumption implies:
    - All marginal and joint distributions are multivariate normal
    - Linear relationships capture all dependencies
    - Higher-order moments beyond covariance are uninformative

    For non-Gaussian data, this estimator captures only linear dependencies
    and may underestimate the true mutual information.
    """

    SX = correlation_log_determinant(X)
    SY = correlation_log_determinant(Y)
    SXY = correlation_log_determinant(np.hstack((X, Y)))

    mi = 0.5 * (SX + SY - SXY)
    return mi


def kde_mutual_information(X, Y, bandwidth="silverman", kernel="gaussian"):
    """
    Estimate mutual information using Kernel Density Estimation.

    This function computes mutual information using the relationship:

    .. math::

        I(X; Y) = H(X) + H(Y) - H(X, Y)

    where each entropy term is estimated using KDE. The joint entropy H(X,Y)
    is computed on the concatenated space [X, Y].

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    bandwidth : str or float, default='silverman'
        Bandwidth selection method for kernel density estimation.
    kernel : str, default='gaussian'
        Kernel function type.

    Returns
    -------
    I : float
        Estimated mutual information in nats.

    Notes
    -----
    The KDE approach can capture nonlinear dependencies but is sensitive to:
    - Bandwidth selection (affects bias-variance tradeoff)
    - Curse of dimensionality for high-dimensional data
    - Sample size requirements for reliable density estimates

    Consider using k-NN methods for high-dimensional data or small samples.
    """
    XY = np.hstack((X, Y))
    Hx = kde_entropy(X, bandwidth=bandwidth, kernel=kernel)
    Hy = kde_entropy(Y, bandwidth=bandwidth, kernel=kernel)
    Hxy = kde_entropy(XY, bandwidth=bandwidth, kernel=kernel)

    mi = Hx + Hy - Hxy
    return mi


def knn_mutual_information(X, Y, metric="euclidean", k=1):
    r"""
    Estimate mutual information using k-nearest neighbor (KSG) method.

    This function implements the Kraskov-Stögbauer-Grassberger estimator,
    which uses k-nearest neighbor statistics to estimate mutual information:

    .. math::

        I(X; Y) = \psi(k) + \psi(N) - \langle \psi(n_x + 1) + \psi(n_y + 1) \rangle

    where :math:`\psi` is the digamma function, :math:`N` is the total number of samples,
    :math:`n_x` and :math:`n_y` are the numbers of neighbors in the marginal spaces
    within the distance to the k-th neighbor in the joint space.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    metric : str, default='euclidean'
        Distance metric for neighborhood calculations.
    k : int, default=1
        Number of nearest neighbors to consider.

    Returns
    -------
    I : float
        Estimated mutual information in nats.

    Notes
    -----
    The KSG estimator:

    - Is asymptotically consistent
    - Adapts to local density variations
    - Works well for continuous data
    - Can handle moderate dimensionality

    Choice of k involves bias-variance tradeoff:
    - Small k: Lower bias, higher variance
    - Large k: Higher bias, lower variance

    References
    ----------
    .. [1] Kraskov, A., Stögbauer, H., Grassberger, P. Estimating mutual information.
           Physical Review E 69, 066138 (2004).
    """
    # construct the joint space
    n = X.shape[0]
    JS = np.column_stack((X, Y))

    # Find the K^th smallest distance in the joint space
    D = np.sort(cdist(JS, JS, metric=metric), axis=1)[:, k]
    epsilon = D

    # Count neighbors within epsilon in marginal spaces
    Dx = cdist(X, X, metric=metric)
    nx = np.sum(Dx < epsilon[:, None], axis=1) - 1
    Dy = cdist(Y, Y, metric=metric)
    ny = np.sum(Dy < epsilon[:, None], axis=1) - 1

    # KSG Estimation formula
    I1a = digamma(k)
    I1b = digamma(n)
    I1 = I1a + I1b
    I2 = -np.mean(digamma(nx + 1) + digamma(ny + 1))
    mi = I1 + I2
    return mi


def geometric_knn_mutual_information(X, Y, metric="euclidean", k=1):
    """
    Estimate mutual information using geometric k-nearest neighbor method.

    This function applies the geometric k-NN entropy estimator to compute
    mutual information via the entropy decomposition:

    .. math::

        I(X; Y) = H_{\text{geom}}(X) + H_{\text{geom}}(Y) - H_{\text{geom}}(X, Y)

    The geometric correction accounts for local manifold structure and
    provides improved estimates for data with non-uniform density distributions.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features_x)
        First variable.
    Y : array-like of shape (n_samples, n_features_y)
        Second variable.
    metric : str, default='euclidean'
        Distance metric for neighbor calculations.
    k : int, default=1
        Number of nearest neighbors.

    Returns
    -------
    I : float
        Estimated mutual information using geometric k-NN method.

    Notes
    -----
    This estimator is particularly effective for:
    - Data lying on lower-dimensional manifolds
    - Non-uniform density distributions
    - Cases where local geometry matters

    The geometric correction helps account for the intrinsic dimensionality
    of the data, potentially providing more accurate estimates than standard k-NN methods.

    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of
           entropy and mutual information. Chaos 28, 033113 (2018).
    """
    Xdist = cdist(X, X, metric=metric)
    Ydist = cdist(Y, Y, metric=metric)
    XYdist = cdist(np.hstack((X, Y)), np.hstack((X, Y)), metric=metric)

    HX = geometric_knn_entropy(X, Xdist, k)
    HY = geometric_knn_entropy(Y, Ydist, k)
    HXY = geometric_knn_entropy(np.hstack((X, Y)), XYdist, k)

    mi = HX + HY - HXY

    # Safety check: return 0 if result is NaN or infinite
    if not np.isfinite(mi):
        warnings.warn("NaN result in geometric_knn_mutual_information. Returning 0.0")
        return 0.0

    # Ensure non-negativity
    return max(0.0, mi)
