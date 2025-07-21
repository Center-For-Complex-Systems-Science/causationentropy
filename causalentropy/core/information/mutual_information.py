import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma
from causalentropy.core.information.entropy import kde_entropy, geometric_knn_entropy
from causalentropy.core.linalg import correlation_log_determinant


def gaussian_mutual_information(X, Y):
    """
    I(X;Y) for (multivariate) Gaussian vectors X, Y using log-determinants.
    X, Y must each be 2-D with the **same number of rows** (samples).
    """

    SX = correlation_log_determinant(X)
    SY = correlation_log_determinant(Y)
    SXY = correlation_log_determinant(np.hstack((X, Y)))

    return 0.5 * (SX + SY - SXY)


def kde_mutual_information(X, Y, bandwidth='silverman', kernel='gaussian'):
    XY = np.hstack((X, Y))
    Hx = kde_entropy(X, bandwidth=bandwidth, kernel=kernel)
    Hy = kde_entropy(Y, bandwidth=bandwidth, kernel=kernel)
    Hxy = kde_entropy(XY, bandwidth=bandwidth, kernel=kernel)

    return Hx + Hy - Hxy


def knn_mutual_information(X, Y, metric='euclidean', k=1):
    # construct the joint space
    n = X.shape[0]
    JS = np.column_stack((X, Y))

    # Find the K^th smallest distance in the joint space
    D = np.sort(cdist(JS, JS, metric=metric, p=k + 1), axis=1)[:, k]
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
    I2 = - np.mean(digamma(nx + 1) + digamma(ny + 1))
    return I1 + I2


def geometric_knn_mutual_information(X, Y, metric='euclidean', k=1):
    """A method for estimating Mutual information (which will be
    needed in Causation entropy) which comes from the paper
    'Geometric k-nearest neighbor estimation of entropy and mutual information'
    by Lord, Sun and Bollt
    """
    Xdist = cdist(X, X, metric=metric)
    Ydist = cdist(Y, Y, metric=metric)
    XYdist = cdist(np.hstack((X, Y)), np.hstack((X, Y)), metric=metric)

    HX = geometric_knn_entropy(X, k, Xdist)
    HY = geometric_knn_entropy(Y, k, Ydist)
    HXY = geometric_knn_entropy(np.hstack((X, Y)), k, XYdist)

    return HX + HY - HXY


def negative_binomial_mutual_information(X, Y, Z):
    raise NotImplemented()


def hawkes_mutual_information(X, Y, Z):
    raise NotImplemented()


def von_mises_mutual_information(X, Y, Z):
    raise NotImplemented()


def laplace_mutual_information(X, Y, Z):
    raise NotImplemented()


def histogram_mutual_information(X, Y, Z):
    raise NotImplemented()
