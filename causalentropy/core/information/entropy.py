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
    U, S, Vt = svd_Yi
    transformed = np.dot(Z_i, Vt.T) / S
    return np.sum(transformed ** 2) <= 1


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


def negative_binomial_entropy(r, p, max_k=None, tol=1e-12, base=np.e):
    r"""
    Compute entropy of a negative binomial distribution.
    
    The negative binomial distribution models the number of failures before the
    r-th success in a sequence of independent Bernoulli trials with success probability p.
    The entropy is calculated as:
    
    .. math::
        
        H(X) = -\sum_{k=0}^{\infty} P(X = k) \log P(X = k)
        
    where :math:`P(X = k) = \binom{k+r-1}{k} p^r (1-p)^k` for k = 0, 1, 2, ...
    
    Parameters
    ----------
    r : float
        Number of successes parameter. Must be positive.
    p : float
        Success probability parameter. Must satisfy 0 < p < 1.
    max_k : int, optional
        Maximum value of k for truncation. If None, determined adaptively
        based on the cumulative probability mass.
    tol : float, default=1e-12
        Tolerance for adaptive truncation. Summation continues until
        the tail probability falls below this threshold.
    base : float, default=np.e
        Logarithm base for entropy calculation. Default is natural logarithm.
        
    Returns
    -------
    H : float
        Entropy of the negative binomial distribution.
        
    Raises
    ------
    ValueError
        If parameters are outside valid ranges.
        
    Notes
    -----
    The negative binomial distribution generalizes the geometric distribution
    and is commonly used for modeling overdispersed count data.
    
    For computational efficiency, the infinite sum is truncated adaptively
    based on the tail probability mass.
    """
    # basic checks
    if not (0 < p < 1):
        raise ValueError("p must satisfy 0 < p < 1.")
    if r <= 0:
        raise ValueError("r must be positive.")

    # choose a truncation if none supplied
    if max_k is None:
        # quantile might return inf for extremely small tol; guard for that
        q = nbinom.ppf(1 - tol, r, p)
        max_k = int(q if np.isfinite(q) else np.ceil(r * (1 - p) / p * 10))

    ks = np.arange(max_k + 1)  # 0 … max_k
    pmf = nbinom.pmf(ks, r, p)

    # numerical entropy (ignore zero-mass terms to avoid log(0))
    mask = pmf > 0
    H = -np.sum(pmf[mask] * np.log(pmf[mask]))

    # convert to requested base
    if base != np.e:
        H /= np.log(base)
    return H


def hawkes_entropy(events, mu, alpha, beta, T=None, base=np.e):
    r"""
    Compute entropy of a univariate Hawkes point process.
    
    The Hawkes process is a self-exciting point process where the intensity function
    depends on the history of past events:
    
    .. math::
        
        \lambda(t) = \mu + \alpha \sum_{t_i < t} e^{-\beta(t - t_i)}
        
    The entropy is computed using the log-likelihood of the observed event sequence:
    
    .. math::
        
        H = -\left[ \sum_{i=1}^{n} \log \lambda(t_i) - \int_0^T \lambda(s) ds \right]
        
    Parameters
    ----------
    events : array-like of shape (n,)
        Strictly increasing sequence of event times.
    mu : float
        Background intensity parameter. Must be positive.
    alpha : float
        Self-excitation parameter. Must be non-negative.
    beta : float
        Decay parameter. Must be positive.
    T : float, optional
        Observation window end time. If None, uses the last event time.
    base : float, default=np.e
        Logarithm base for entropy calculation.
        
    Returns
    -------
    H : float
        Entropy of the Hawkes process.
        
    Raises
    ------
    ValueError
        If event times are not strictly increasing or parameters are invalid.
        
    Notes
    -----
    The Hawkes process is widely used in finance, neuroscience, and seismology
    to model clustered or self-exciting events.
    
    The parameter constraints ensure:
    - μ > 0: Positive background rate
    - α ≥ 0: Non-negative self-excitation
    - β > 0: Positive decay rate
    
    For stability, typically α < β to ensure stationarity.
    """
    events = np.asfarray(events)
    if events.ndim != 1 or events.size == 0:
        raise ValueError("events must be a non-empty 1-D array.")
    if np.any(np.diff(events) <= 0):
        raise ValueError("events must be strictly increasing.")
    if mu <= 0 or alpha < 0 or beta <= 0:
        raise ValueError("require mu>0, alpha≥0, beta>0")

    if T is None:
        T = float(events[-1])
    elif T < events[-1]:
        raise ValueError("T must be ≥ last event time.")

    n = events.size
    lambdas = np.empty(n)

    s = 0.0
    last_t = 0.0
    for i, t in enumerate(events):
        decay = np.exp(-beta * (t - last_t))
        s *= decay
        lambdas[i] = mu + alpha * s
        s += 1.0
        last_t = t

    ll_sum = np.sum(np.log(lambdas))
    integral = mu * T + (alpha / beta) * np.sum(1.0 - np.exp(-beta * (T - events)))
    H = -(ll_sum - integral)

    if base != np.e:
        H /= np.log(base)
    return H


def von_mises_entropy(kappa, base=np.e):
    r"""
    Compute entropy of the von Mises (circular normal) distribution.
    
    The von Mises distribution is the circular analogue of the normal distribution,
    defined on the circle [0, 2π). Its entropy is given by:
    
    .. math::
        
        H = \log(2\pi I_0(\kappa)) - \kappa \frac{I_1(\kappa)}{I_0(\kappa)}
        
    where :math:`I_0` and :math:`I_1` are modified Bessel functions of the first kind,
    and κ is the concentration parameter.
    
    Parameters
    ----------
    kappa : float or array-like
        Concentration parameter(s). Must be non-negative.
        κ = 0 corresponds to the uniform circular distribution.
        Large κ indicates high concentration around the mean direction.
    base : float, default=np.e
        Logarithm base for entropy calculation.
        
    Returns
    -------
    H : float or array-like
        Entropy value(s). Returns scalar if kappa is scalar, array otherwise.
        
    Raises
    ------
    ValueError
        If kappa contains negative values.
        
    Notes
    -----
    Special cases:
    
    - κ = 0: Uniform distribution, H = log(2π)
    - κ → ∞: Concentrated distribution, H → -∞
    
    The von Mises distribution is commonly used for:
    
    - Directional data (wind directions, animal orientations)
    - Periodic phenomena with preferred phases
    - Angular measurements with measurement error
    """
    kappa = np.asfarray(kappa)
    if np.any(kappa < 0):
        raise ValueError("kappa must be ≥ 0")

    two_pi = 2.0 * np.pi
    I0 = i0(kappa)
    with np.errstate(divide='ignore', invalid='ignore'):
        H = np.log(two_pi * I0) - kappa * i1(kappa) / I0
    H = np.where(kappa == 0, np.log(two_pi), H)

    if base != np.e:
        H /= np.log(base)

    return H.item() if H.size == 1 else H


def laplace_entropy(b, base=np.e):
    r"""
    Compute entropy of the Laplace (double exponential) distribution.
    
    The Laplace distribution has probability density function:
    
    .. math::
        
        f(x) = \frac{1}{2b} \exp\left(-\frac{|x - \mu|}{b}\right)
        
    The entropy is independent of the location parameter μ and equals:
    
    .. math::
        
        H = 1 + \log(2b)
        
    Parameters
    ----------
    b : float or array-like
        Scale parameter(s). Must be positive.
    base : float, default=np.e
        Logarithm base for entropy calculation.
        
    Returns
    -------
    H : float or array-like
        Entropy value(s). Returns scalar if b is scalar, array otherwise.
        
    Raises
    ------
    ValueError
        If b contains non-positive values.
        
    Notes
    -----
    The Laplace distribution:
    
    - Has heavier tails than the normal distribution
    - Is the maximum entropy distribution for a given mean absolute deviation
    - Arises naturally in robust statistics and sparse modeling
    
    The entropy scales logarithmically with the scale parameter b.
    """
    b = np.asfarray(b)
    if np.any(b <= 0):
        raise ValueError("scale parameter b must be positive")

    H = 1.0 + np.log(2.0 * b)  # nats
    if base != np.e:
        H /= np.log(base)
    return H.item() if H.size == 1 else H


def histogram_entropy(x, bins='auto', base=np.e):
    r"""
    Estimate entropy using histogram-based probability estimation.
    
    This function computes entropy by constructing a histogram of the data
    and treating the relative frequencies as probability estimates:
    
    .. math::
        
        H = -\sum_{i=1}^{m} p_i \log p_i
        
    where :math:`p_i = n_i/n` is the proportion of data points in bin i,
    and m is the number of non-empty bins.
    
    Parameters
    ----------
    x : array-like
        Input data for entropy estimation. Will be flattened to 1-D.
    bins : int, sequence, or str, default='auto'
        Binning specification passed to numpy.histogram.
        Common options: 'auto', 'sqrt', 'log', or explicit bin edges.
    base : float, default=np.e
        Logarithm base for entropy calculation.
        
    Returns
    -------
    H : float
        Estimated entropy based on histogram probabilities.
        
    Raises
    ------
    ValueError
        If x is empty.
        
    Notes
    -----
    The histogram method provides a simple non-parametric entropy estimator,
    but several considerations apply:
    
    **Advantages:**
    - Simple and intuitive
    - No distributional assumptions
    - Computationally efficient
    
    **Limitations:**
    - Sensitive to bin choice (number and placement)
    - Can be biased for small samples
    - Curse of dimensionality for multivariate data
    
    **Bin Selection Guidelines:**
    - Too few bins: Underestimate entropy (oversmoothing)
    - Too many bins: Overestimate entropy (undersmoothing)
    - 'auto' uses NumPy's automatic bin selection
    
    For multivariate entropy estimation, consider alternative methods
    like k-NN or KDE approaches.
    """
    x = np.asarray(x).ravel()
    if x.size == 0:
        raise ValueError("x must contain at least one value")

    counts, _ = np.histogram(x, bins=bins)
    probs = counts.astype(float)
    probs /= probs.sum()

    probs = probs[probs > 0]  # drop empty bins
    H = -np.sum(probs * np.log(probs))

    if base != np.e:
        H /= np.log(base)
    return H


