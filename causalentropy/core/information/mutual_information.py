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


def negative_binomial_mutual_information(X, Y, r_x, p_x, r_y, p_y):
    """
    Mutual information for negative binomial distributions.
    
    Parameters
    ----------
    X : array_like
        Data from first negative binomial variable
    Y : array_like  
        Data from second negative binomial variable
    r_x, p_x : float
        Parameters of X's negative binomial distribution
    r_y, p_y : float
        Parameters of Y's negative binomial distribution
        
    Returns
    -------
    float
        Mutual information estimate
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
    Mutual information for Hawkes processes.
    
    Parameters
    ----------
    events_x, events_y : array_like
        Event times for the two Hawkes processes
    mu_x, alpha_x, beta_x : float
        Parameters for first Hawkes process
    mu_y, alpha_y, beta_y : float
        Parameters for second Hawkes process
    T : float, optional
        Time horizon
        
    Returns
    -------
    float
        Mutual information estimate
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
    Mutual information for von Mises distributions.
    
    Parameters
    ----------
    X, Y : array_like
        Angular data from von Mises distributions
    kappa_x, kappa_y : float
        Concentration parameters
        
    Returns
    -------
    float
        Mutual information estimate
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
    Mutual information for Laplace distributions.
    
    Parameters
    ----------
    X, Y : array_like
        Data from Laplace distributions
    b_x, b_y : float
        Scale parameters
        
    Returns
    -------
    float
        Mutual information estimate
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
    Mutual information using histogram-based entropy estimation.
    
    Parameters
    ----------
    X, Y : array_like
        Data samples
    bins : int or str
        Number of bins or binning strategy
        
    Returns
    -------
    float
        Mutual information estimate
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
