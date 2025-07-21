import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma
from causalentropy.core.information.entropy import kde_entropy, geometric_knn_entropy
from causalentropy.core.linalg import correlation_log_determinant


def gaussian_mutual_information(X, Y):
    """
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

    return 0.5 * (SX + SY - SXY)


def kde_mutual_information(X, Y, bandwidth='silverman', kernel='gaussian'):
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

    return Hx + Hy - Hxy


def knn_mutual_information(X, Y, metric='euclidean', k=1):
    """
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

    return HX + HY - HXY


def negative_binomial_mutual_information(X, Y, r_x, p_x, r_y, p_y):
    """
    Compute mutual information for negative binomial distributions.
    
    This function estimates mutual information between two negative binomial
    random variables using their marginal entropies and joint entropy:
    
    .. math::
        
        I(X; Y) = H(X) + H(Y) - H(X, Y)
        
    where each variable X ~ NB(r_x, p_x) and Y ~ NB(r_y, p_y).
    
    Parameters
    ----------
    X : array-like
        Observed data from the first negative binomial variable.
    Y : array-like
        Observed data from the second negative binomial variable.
    r_x, p_x : float
        Parameters of X's negative binomial distribution.
        r_x > 0 (number of failures), 0 < p_x < 1 (success probability).
    r_y, p_y : float
        Parameters of Y's negative binomial distribution.
        r_y > 0 (number of failures), 0 < p_y < 1 (success probability).
        
    Returns
    -------
    I : float
        Estimated mutual information in nats.
        
    Notes
    -----
    Current implementation assumes independence for joint entropy calculation,
    which results in zero mutual information. This serves as a baseline and
    should be extended to handle dependent negative binomial variables.
    
    Future improvements could include:
    - Copula-based joint modeling
    - Empirical joint entropy estimation
    - Parametric dependence structures
    
    The negative binomial distribution is commonly used for modeling:
    - Overdispersed count data
    - Number of failures before r successes
    - Contagion processes in epidemiology
    """
    from causalentropy.core.information.entropy import negative_binomial_entropy
    
    # For independent negative binomial variables, MI = H(X) + H(Y) - H(X,Y)
    # This is a simplified implementation assuming independence for joint entropy
    h_x = negative_binomial_entropy(r_x, p_x)
    h_y = negative_binomial_entropy(r_y, p_y)
    
    # Joint entropy approximation (assumes independence - could be improved)
    h_xy = h_x + h_y
    
    return h_x + h_y - h_xy  # Will be 0 for independent case


def hawkes_mutual_information(events_x, events_y, mu_x, alpha_x, beta_x, mu_y, alpha_y, beta_y, T=None):
    """
    Compute mutual information between two Hawkes point processes.
    
    This function estimates mutual information between two self-exciting point processes
    using their individual entropies:
    
    .. math::
        
        I(X; Y) = H(X) + H(Y) - H(X, Y)
        
    where X and Y are Hawkes processes with intensity functions:
    
    .. math::
        
        \lambda_X(t) = \mu_X + \alpha_X \sum_{t_i^X < t} e^{-\beta_X(t - t_i^X)}
        
        \lambda_Y(t) = \mu_Y + \alpha_Y \sum_{t_j^Y < t} e^{-\beta_Y(t - t_j^Y)}
    
    Parameters
    ----------
    events_x, events_y : array-like
        Strictly increasing sequences of event times for the two processes.
    mu_x, alpha_x, beta_x : float
        Parameters for the first Hawkes process:
        - mu_x: Background intensity (> 0)
        - alpha_x: Self-excitation strength (≥ 0)
        - beta_x: Decay rate (> 0)
    mu_y, alpha_y, beta_y : float
        Parameters for the second Hawkes process.
    T : float, optional
        Observation time horizon. If None, uses the maximum event time.
        
    Returns
    -------
    I : float
        Estimated mutual information in nats.
        
    Notes
    -----
    Current implementation assumes independence between processes for joint entropy,
    resulting in zero mutual information. This serves as a baseline implementation.
    
    Extensions for dependent Hawkes processes could include:
    - Cross-excitation terms in intensity functions
    - Multivariate Hawkes process modeling
    - Empirical cross-intensity estimation
    
    Applications include:
    - Financial market event dependencies
    - Neural spike train interactions
    - Social network activity cascades
    - Earthquake aftershock sequences
    """
    from causalentropy.core.information.entropy import hawkes_entropy
    
    # Calculate individual entropies
    h_x = hawkes_entropy(events_x, mu_x, alpha_x, beta_x, T=T)
    h_y = hawkes_entropy(events_y, mu_y, alpha_y, beta_y, T=T)
    
    # Joint entropy approximation (simplified - assumes independence)
    h_xy = h_x + h_y
    
    return h_x + h_y - h_xy  # Will be 0 for independent processes


def von_mises_mutual_information(X, Y, kappa_x, kappa_y):
    """
    Compute mutual information for von Mises (circular normal) distributions.
    
    This function estimates mutual information between two circular random variables
    following von Mises distributions using marginal entropies:
    
    .. math::
        
        I(X; Y) = H(X) + H(Y) - H(X, Y)
        
    where X ~ VM(μ_x, κ_x) and Y ~ VM(μ_y, κ_y).
    
    Parameters
    ----------
    X, Y : array-like
        Angular data samples from von Mises distributions, typically in [0, 2π).
    kappa_x, kappa_y : float
        Concentration parameters for the distributions (≥ 0).
        κ = 0 corresponds to uniform circular distribution.
        Large κ indicates high concentration around the mean direction.
        
    Returns
    -------
    I : float
        Estimated mutual information in nats.
        
    Notes
    -----
    Current implementation assumes independence for joint entropy calculation.
    
    The von Mises distribution is appropriate for:
    - Directional data (wind directions, animal orientations)
    - Periodic phenomena with preferred phases
    - Angular measurements with concentration around a mean direction
    
    Extensions for dependent circular variables could include:
    - Bivariate von Mises distributions
    - Empirical circular correlation measures
    - Copula-based circular dependence models
    """
    from causalentropy.core.information.entropy import von_mises_entropy
    
    # Calculate marginal entropies
    h_x = von_mises_entropy(kappa_x)
    h_y = von_mises_entropy(kappa_y)
    
    # Joint entropy approximation (simplified)
    h_xy = h_x + h_y
    
    return h_x + h_y - h_xy


def laplace_mutual_information(X, Y, b_x, b_y):
    """
    Compute mutual information for Laplace (double exponential) distributions.
    
    This function estimates mutual information between two Laplace-distributed
    random variables using their marginal entropies:
    
    .. math::
        
        I(X; Y) = H(X) + H(Y) - H(X, Y)
        
    where X ~ Laplace(μ_x, b_x) and Y ~ Laplace(μ_y, b_y).
    
    Parameters
    ----------
    X, Y : array-like
        Data samples from Laplace distributions.
    b_x, b_y : float
        Scale parameters for the distributions (> 0).
        Larger values indicate greater spread around the location parameter.
        
    Returns
    -------
    I : float
        Estimated mutual information in nats.
        
    Notes
    -----
    Current implementation assumes independence for joint entropy calculation.
    
    The Laplace distribution is characterized by:
    - Exponential tails (heavier than Gaussian)
    - Maximum entropy for given mean absolute deviation
    - Robustness to outliers
    
    Applications include:
    - Robust regression and sparse modeling
    - Signal processing with impulsive noise
    - Bayesian inference with L1 regularization
    
    Extensions could include:
    - Bivariate Laplace distributions
    - Asymmetric Laplace models
    - Copula-based dependence structures
    """
    from causalentropy.core.information.entropy import laplace_entropy
    
    # Calculate marginal entropies
    h_x = laplace_entropy(b_x)
    h_y = laplace_entropy(b_y)
    
    # Joint entropy approximation (simplified)
    h_xy = h_x + h_y
    
    return h_x + h_y - h_xy


def histogram_mutual_information(X, Y, bins='auto'):
    """
    Estimate mutual information using histogram-based entropy calculation.
    
    This function computes mutual information by constructing histograms for
    marginal and joint distributions:
    
    .. math::
        
        I(X; Y) = \sum_{i,j} p(x_i, y_j) \log \frac{p(x_i, y_j)}{p(x_i) p(y_j)}
        
    where probabilities are estimated from histogram bin frequencies.
    
    Parameters
    ----------
    X, Y : array-like
        Data samples for mutual information estimation.
    bins : int, sequence, or str, default='auto'
        Binning specification for histograms. Options:
        - int: Number of equal-width bins
        - sequence: Bin edges
        - str: Binning strategy ('auto', 'sqrt', 'log', etc.)
        
    Returns
    -------
    I : float
        Estimated mutual information in nats.
        
    Notes
    -----
    The histogram method provides a simple non-parametric estimator with
    several important considerations:
    
    **Advantages:**
    - Intuitive and easy to implement
    - No distributional assumptions
    - Captures nonlinear dependencies
    
    **Limitations:**
    - Sensitive to bin size selection
    - Curse of dimensionality for high-dimensional data
    - Can be biased for small samples
    
    **Bin Selection Guidelines:**
    - Too few bins: Undersmoothing, may miss dependencies
    - Too many bins: Oversmoothing, sparse joint histogram
    - Rule of thumb: ~√N bins for N samples
    
    For high-dimensional data or complex dependencies, consider k-NN or KDE methods.
    """
    from causalentropy.core.information.entropy import histogram_entropy
    import numpy as np
    
    # Calculate marginal entropies
    h_x = histogram_entropy(X, bins=bins)
    h_y = histogram_entropy(Y, bins=bins)
    
    # Joint histogram for joint entropy
    xy_joint = np.column_stack([X.flatten(), Y.flatten()])
    h_xy = histogram_entropy(xy_joint, bins=bins)
    
    return h_x + h_y - h_xy
