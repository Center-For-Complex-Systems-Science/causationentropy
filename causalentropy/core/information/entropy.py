import numpy as np
from scipy.special import gamma
from sklearn.neighbors import KernelDensity
import scipy
from scipy.stats import nbinom
from scipy.special import i0, i1


def l2dist(a, b):
    r"""
    Compute the Euclidean (L2) distance between two points.
    
    .. math::
        
        d(a, b) = ||a - b||_2 = \sqrt{\sum_{i=1}^{n} (a_i - b_i)^2}
        
    Parameters
    ----------
    a, b : array-like
        Input points or vectors.
        
    Returns
    -------
    distance : float
        Euclidean distance between a and b.
    """
    return np.linalg.norm(a - b)


def hyperellipsoid_check(svd_Yi, Z_i):
    """
    Check if points lie within a hyperellipsoid defined by SVD components.
    
    This function determines whether points in Z_i fall within the unit
    hyperellipsoid defined by the singular value decomposition of Yi.
    
    Parameters
    ----------
    svd_Yi : tuple
        SVD decomposition (U, S, Vt) of the reference matrix.
    Z_i : array-like
        Points to test for inclusion in the hyperellipsoid.
        
    Returns
    -------
    inside : bool
        True if all points lie within the hyperellipsoid, False otherwise.
        
    Notes
    -----
    This is used in the geometric k-NN entropy estimation to assess
    the local geometric configuration of nearest neighbors.
    """
    # U, S, Vt = svd_Yi
    # transformed = np.dot(Z_i, Vt.T) / S
    # return np.sum(transformed ** 2) <= 1
    U, S, Vt = svd_Yi
    r = len(S)                 # local rank
    transformed = (Z_i @ Vt.T[:, :r]) / S
    return (transformed**2).sum() <= 1


def kde_entropy(X, bandwidth='silverman', kernel='gaussian'):
    r"""
    Estimate entropy using Kernel Density Estimation (KDE).
    
    This function computes the differential entropy of a continuous random variable
    using kernel density estimation. The entropy is defined as:
    
    .. math::
        
        H(X) = -\int f(x) \log f(x) \, dx
        
    where :math:`f(x)` is the probability density function estimated via KDE:
    
    .. math::
        
        \hat{f}(x) = \frac{1}{nh} \sum_{i=1}^{n} K\left(\frac{x - x_i}{h}\right)
        
    with kernel function :math:`K` and bandwidth :math:`h`.
    
    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Input data for entropy estimation.
    bandwidth : str or float, default='silverman'
        Bandwidth selection method or explicit bandwidth value.
        If 'silverman', uses Silverman's rule of thumb.
    kernel : str, default='gaussian'
        Kernel function type. Options include 'gaussian', 'tophat', 'epanechnikov',
        'exponential', 'linear', 'cosine'.
        
    Returns
    -------
    H : float
        Estimated differential entropy in nats (natural units).
        
    Notes
    -----
    The KDE entropy estimator can suffer from boundary effects and may be biased
    for small sample sizes. The choice of bandwidth critically affects the estimate:
    
    - Too small: Undersmoothed, entropy overestimated
    - Too large: Oversmoothed, entropy underestimated
    
    Silverman's rule provides a reasonable default bandwidth for Gaussian-like data.
    """
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
    log_density = np.exp(kde.score_samples(X))
    Hx = -np.sum(np.log(log_density)) / len(log_density)
    return Hx


def geometric_knn_entropy(X, Xdist, k=1):
    r"""
    Estimate entropy using geometric k-nearest neighbor method.
    
    This function implements the geometric k-NN entropy estimator from Lord, Sun, and Bollt.
    The method estimates differential entropy by analyzing the geometric properties of
    k-nearest neighbor configurations in the data space.
    
    The entropy estimate is given by:
    
    .. math::
        
        H(X) = \log N + \log \frac{\pi^{d/2}}{\Gamma(1 + d/2)} + \frac{d}{N} \sum_{i=1}^{N} \log \rho_i + \text{geometric correction}
        
    where :math:`N` is the sample size, :math:`d` is the dimension, :math:`\rho_i` is the
    distance to the k-th nearest neighbor of point :math:`i`, and the geometric correction
    accounts for the local geometry of the nearest neighbor configuration.
    
    Parameters
    ----------
    X : array-like of shape (N, d)
        Input data matrix where N is the number of samples and d is the dimensionality.
    Xdist : array-like of shape (N, N)
        Pairwise distance matrix between all points in X.
    k : int, default=1
        Number of nearest neighbors to consider for entropy estimation.
        
    Returns
    -------
    H_X : float
        Estimated differential entropy using the geometric k-NN method.
        
    Notes
    -----
    This estimator is particularly effective for:
    
    - High-dimensional data where traditional methods may fail
    - Data with non-uniform density distributions
    - Cases where the underlying geometry is important
    
    The geometric correction term accounts for the local dimensionality and shape
    of the data manifold, making this estimator more robust than standard k-NN methods.
    
    References
    ----------
    .. [1] Lord, W.M., Sun, J., Bollt, E.M. Geometric k-nearest neighbor estimation of 
           entropy and mutual information. Chaos 28, 033113 (2018).
    """
    N, d = X.shape
    Xknn = np.zeros((N, k), dtype=int)

    for i in range(N):
        Xknn[i, :] = np.argsort(Xdist[i, :])[1:k + 1]
    H_X = np.log(N) + np.log(np.pi ** (d / 2) / gamma(1 + d / 2))
    H_X += d / N * np.sum([np.log(l2dist(X[i, :], X[Xknn[i, k - 1], :])) for i in range(N)])
    H_X += 1 / N * np.sum(
        [-np.log(max(1, np.sum([hyperellipsoid_check(np.linalg.svd(Y_i), Z_i[j, :]) for j in range(k)])))
         + np.sum([np.log(sing_Yi[l] / sing_Yi[0]) for l in range(d)])
         for i in range(N)
         for Y_i in [X[np.append([i], Xknn[i, :]), :] - np.mean(X[np.append([i], Xknn[i, :]), :], axis=0)]
         for svd_Yi in [np.linalg.svd(Y_i)]
         for sing_Yi in [svd_Yi[1]]
         for Z_i in [X[Xknn[i, :], :] - X[i, :]]
         ])
    return H_X


def poisson_entropy(lambdas):
    r"""
    Estimate entropy for Poisson-distributed random variables.
    
    This function computes the entropy of Poisson random variables with given rate
    parameters. For a Poisson random variable X with parameter λ, the entropy is:
    
    .. math::
        
        H(X) = -\sum_{k=0}^{\infty} P(X = k) \log P(X = k)
        
    where :math:`P(X = k) = \frac{\lambda^k e^{-\lambda}}{k!}`.
    
    The summation is truncated when the cumulative probability reaches a specified
    tolerance to ensure numerical stability.
    
    Parameters
    ----------
    lambdas : array-like
        Rate parameters for the Poisson distributions. Can be scalar or array.
        Values are automatically converted to absolute values.
        
    Returns
    -------
    est : float or array-like
        Estimated entropy values in nats. Shape matches the input lambdas.
        
    Notes
    -----
    This implementation:
    
    - Uses adaptive truncation based on cumulative probability mass
    - Handles numerical stability by setting log(0) terms to zero
    - Returns real values even if complex arithmetic is used internally
    
    The estimator is particularly useful for count data and discrete event processes
    where Poisson assumptions are appropriate.
    
    References
    ----------
    .. [1] Fish, A., Bollt, E. Interaction networks from discrete event data by Poisson 
           multivariate mutual information estimation and information flow with applications 
           from gene expression data. (In preparation)
    """
    lambdas = np.abs(lambdas)
    First = np.exp(-lambdas)
    Psum = First
    P = [np.matrix(First)]
    counter = 0
    small = 1
    i = 1
    while np.max(1 - Psum) > 1e-16 and small > 1e-75:
        counter = counter + 1
        prob = scipy.stats.poisson.pmf(i, lambdas)
        Psum = Psum + prob
        P.append(np.matrix(prob))
        if i >= np.max(lambdas):
            small = np.min(prob)

        i = i + 1

    P = np.array(P).squeeze()
    est_a = P * np.log(P)
    est_a[np.isinf(est_a)] = 0
    est_a[np.isnan(est_a)] = 0
    try:
        est = -np.sum(est_a, axis=0)
    except:
        est = -np.sum(est_a)
    return np.real(est)


def poisson_joint_entropy(Cov):
    r"""
    Estimate joint entropy for multivariate Poisson distributions.
    
    This function computes the joint entropy of a multivariate Poisson distribution
    using the covariance matrix structure. The joint entropy decomposes into:
    
    .. math::
        
        H(\mathbf{X}) = \sum_{i} H(X_i) + \sum_{i<j} \text{Cov}(X_i, X_j)
        
    where the first term represents marginal entropies and the second captures
    the interaction effects through covariances.
    
    Parameters
    ----------
    Cov : array-like of shape (n, n)
        Covariance matrix of the multivariate Poisson distribution.
        Diagonal elements represent marginal variances (= means for Poisson).
        Off-diagonal elements represent covariances between variables.
        
    Returns
    -------
    joint_entropy : float
        Estimated joint entropy of the multivariate Poisson distribution.
        
    Notes
    -----
    This decomposition assumes a specific form for multivariate Poisson distributions
    where the interaction structure is captured through the covariance terms.
    
    The method:
    
    1. Computes marginal entropies using diagonal elements (Poisson parameters)
    2. Adds covariance contributions from off-diagonal elements
    
    This approach is computationally efficient for high-dimensional Poisson models.
    """
    T = np.triu(Cov, 1)
    T = np.matrix(T)
    U = np.matrix(np.diag(Cov))
    Ent1 = np.sum(poisson_entropy(U))
    Ent2 = np.sum(T)
    return Ent1 + Ent2