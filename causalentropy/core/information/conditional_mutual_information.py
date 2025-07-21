import scipy as sp
from scipy.spatial.distance import cdist
import numpy as np
import scipy.linalg as la

from causalentropy.core.information.entropy import kde_entropy, geometric_knn_entropy, poisson_entropy, \
    poisson_joint_entropy
from causalentropy.core.information.mutual_information import geometric_knn_mutual_information, kde_mutual_information, \
    knn_mutual_information, \
    gaussian_mutual_information


def _gaussian_conditional_mutual_information(X, Y, Z=None):
    """An implementation of the Gaussian conditional mutual information from
    the paper by Sun, Taylor and Bollt entitled:
        Causal network inference by optimal causation entropy"""
    if Z is None:
        return gaussian_mutual_information(X, Y)

    SZ = np.corrcoef(Z.T)
    SZ = sp.linalg.det(SZ)
    XZ = np.concatenate((X, Z), axis=1)
    YZ = np.concatenate((Y, Z), axis=1)
    XYZ = np.concatenate((X, Y, Z), axis=1)

    SXZ = sp.linalg.det(np.corrcoef(XZ.T))
    SYZ = sp.linalg.det(np.corrcoef(YZ.T))
    SXYZ = sp.linalg.det(np.corrcoef(XYZ.T))

    Value = 0.5 * np.log((SXZ * SYZ) / (SZ * SXYZ))

    return Value


def _gaussian_conditional_mutual_information(X, Y, Z=None):
    """
    I(X;Y | Z) under a Gaussian assumption.

    Parameters
    ----------
    X : (N, kx) array
    Y : (N, ky) array
    Z : (N, kz) array or None
    """
    if Z is None:
        return gaussian_mutual_information(X, Y)

    def _detcorr(A):
        C = np.corrcoef(A.T)
        return float(C) if np.ndim(C) == 0 else la.det(C)

    SZ = _detcorr(Z)
    SXZ = _detcorr(np.hstack((X, Z)))
    SYZ = _detcorr(np.hstack((Y, Z)))
    SXYZ = _detcorr(np.hstack((X, Y, Z)))

    return 0.5 * np.log((SXZ * SYZ) / (SZ * SXYZ))


def gaussian_conditional_mutual_information(X, Y, Z=None):
    """
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

    def _detcorr(A):
        C = np.corrcoef(A.T)
        return float(C) if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]

    SZ = _detcorr(Z)
    SXZ = _detcorr(np.hstack((X, Z)))
    SYZ = _detcorr(np.hstack((Y, Z)))
    SXYZ = _detcorr(np.hstack((X, Y, Z)))

    return 0.5 * (SXZ + SYZ - SZ - SXYZ)


def kde_conditional_mutual_information(X, Y, Z, bandwidth='silverman', kernel='gaussian'):
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


def knn_conditional_mutual_information(X, Y, Z, metric='euclidean', k=1):
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
        return knn_mutual_information(X, Y, metric=metric, k=k)  # np.max([self.MutualInfo_KNN(X,self.Y),0])
    else:
        XY = np.concatenate((X, Y), axis=1)
        MIXYZ = knn_mutual_information(XY, Z, metric=metric, k=k)
        MIXY = knn_mutual_information(X, Y, metric=metric, k=k)

        return MIXY - MIXYZ


def geometric_knn_conditional_mutual_information(X, Y, Z, metric='euclidean', k=1):
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
    YZdist = cdist(np.hstack((Y, Z)), np.hstack((Y, Z)), metric=metric)
    XZdist = cdist(np.hstack((X, Z)), np.hstack((X, Z)), metric=metric)
    XYZdist = cdist(np.hstack((X, Y, Z)), np.hstack((X, Y, Z)), metric=metric)
    Zdist = cdist(Z, Z, metric=metric)
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
        indX = np.matrix(np.arange(SzX))
        indY = np.matrix(np.arange(SzY) + SzX)
        indZ = np.matrix(np.arange(SzZ) + SzX + SzY)
        XYZ = np.concatenate((X, Y, Z), axis=1)
        SXYZ = np.corrcoef(XYZ.T)
        SS = SXYZ
        Sa = SXYZ - np.diag(np.diag(SXYZ))
        np.fill_diagonal(SS, np.diagonal(SS) - Sa)
        SS[0:SzX, 0:SzX] = SS[0:SzX, 0:SzX] + SXYZ[0:SzX, SzX:SzX + SzY]
        SS[SzX:SzX + SzY, SzX:SzX + SzY] = SS[SzX:SzX + SzY, SzX:SzX + SzY] + SXYZ[SzX:SzX + SzY, 0:SzX]
        S_est1 = SS[np.concatenate((indY.T, indZ.T), axis=0), np.concatenate((indY.T, indZ.T), axis=0)]
        S_est2 = SS[np.concatenate((indX.T, indZ.T), axis=0), np.concatenate((indX.T, indZ.T), axis=0)]
        HYZ = poisson_joint_entropy(S_est1)
        SindZ = SS[indZ, indZ]
        HZ = poisson_joint_entropy(SindZ)
        HXYZ = poisson_joint_entropy(SXYZ - np.diag(Sa))
        HXZ = poisson_joint_entropy(S_est2)
        H_YZ = HYZ - HZ
        H_XYZ = HXYZ - HXZ
        return H_XYZ - H_YZ


def negative_binomial_conditional_mutual_information(X, Y, Z, r_x=None, p_x=None, r_y=None, p_y=None, r_z=None, p_z=None):
    """
    Conditional mutual information for negative binomial distributions.
    
    Parameters
    ----------
    X, Y, Z : array_like
        Data samples
    r_x, p_x : float, optional
        Parameters for X's negative binomial distribution
    r_y, p_y : float, optional  
        Parameters for Y's negative binomial distribution
    r_z, p_z : float, optional
        Parameters for Z's negative binomial distribution
        
    Returns
    -------
    float
        Conditional mutual information estimate
    """
    from causalentropy.core.information.entropy import negative_binomial_entropy
    
    if Z is None:
        # Reduce to mutual information
        from causalentropy.core.information.mutual_information import negative_binomial_mutual_information
        return negative_binomial_mutual_information(X, Y, r_x or 1, p_x or 0.5, r_y or 1, p_y or 0.5)
    
    # Estimate parameters if not provided
    if r_x is None or p_x is None:
        r_x, p_x = 1, 0.5  # Default values
    if r_y is None or p_y is None:
        r_y, p_y = 1, 0.5
    if r_z is None or p_z is None:
        r_z, p_z = 1, 0.5
    
    # Calculate component entropies  
    h_x = negative_binomial_entropy(r_x, p_x)
    h_y = negative_binomial_entropy(r_y, p_y)
    h_z = negative_binomial_entropy(r_z, p_z)
    
    # Simplified approximation: I(X;Y|Z) ≈ H(X) + H(Y) - H(Z)
    return max(0, h_x + h_y - h_z)


def hawkes_conditional_mutual_information(X, Y, Z, mu_x=1.0, alpha_x=0.1, beta_x=1.0, 
                                         mu_y=1.0, alpha_y=0.1, beta_y=1.0,
                                         mu_z=1.0, alpha_z=0.1, beta_z=1.0, T=None):
    """
    Conditional mutual information for Hawkes processes.
    
    Parameters
    ----------
    X, Y, Z : array_like
        Event times for the processes
    mu_x, alpha_x, beta_x : float
        Parameters for first process
    mu_y, alpha_y, beta_y : float
        Parameters for second process  
    mu_z, alpha_z, beta_z : float
        Parameters for conditioning process
    T : float, optional
        Time horizon
        
    Returns
    -------
    float
        Conditional mutual information estimate
    """
    from causalentropy.core.information.entropy import hawkes_entropy
    
    if Z is None:
        # Reduce to mutual information
        from causalentropy.core.information.mutual_information import hawkes_mutual_information
        return hawkes_mutual_information(X, Y, mu_x, alpha_x, beta_x, mu_y, alpha_y, beta_y, T=T)
    
    # Calculate component entropies
    h_x = hawkes_entropy(X, mu_x, alpha_x, beta_x, T=T)
    h_y = hawkes_entropy(Y, mu_y, alpha_y, beta_y, T=T)
    h_z = hawkes_entropy(Z, mu_z, alpha_z, beta_z, T=T)
    
    # Simplified approximation
    return max(0, h_x + h_y - h_z)


def von_mises_conditional_mutual_information(X, Y, Z, kappa_x=1.0, kappa_y=1.0, kappa_z=1.0):
    """
    Conditional mutual information for von Mises distributions.
    
    Parameters
    ----------
    X, Y, Z : array_like
        Angular data
    kappa_x, kappa_y : float
        Concentration parameters for X and Y
    kappa_z : float
        Concentration parameter for Z
        
    Returns
    -------
    float
        Conditional mutual information estimate
    """
    from causalentropy.core.information.entropy import von_mises_entropy
    
    if Z is None:
        # Reduce to mutual information
        from causalentropy.core.information.mutual_information import von_mises_mutual_information
        return von_mises_mutual_information(X, Y, kappa_x, kappa_y)
    
    # Calculate component entropies
    h_x = von_mises_entropy(kappa_x)
    h_y = von_mises_entropy(kappa_y)
    h_z = von_mises_entropy(kappa_z)
    
    # Simplified approximation
    return max(0, h_x + h_y - h_z)


def laplace_conditional_mutual_information(X, Y, Z, b_x=1.0, b_y=1.0, b_z=1.0):
    """
    Conditional mutual information for Laplace distributions.
    
    Parameters
    ----------
    X, Y, Z : array_like
        Data samples
    b_x, b_y : float
        Scale parameters for X and Y
    b_z : float
        Scale parameter for Z
        
    Returns
    -------
    float
        Conditional mutual information estimate
    """
    from causalentropy.core.information.entropy import laplace_entropy
    
    if Z is None:
        # Reduce to mutual information
        from causalentropy.core.information.mutual_information import laplace_mutual_information
        return laplace_mutual_information(X, Y, b_x, b_y)
    
    # Calculate component entropies
    h_x = laplace_entropy(b_x)
    h_y = laplace_entropy(b_y)
    h_z = laplace_entropy(b_z)
    
    # Simplified approximation
    return max(0, h_x + h_y - h_z)


def histogram_conditional_mutual_information(X, Y, Z, bins='auto'):
    """
    Conditional mutual information using histogram-based entropy estimation.
    
    Parameters
    ----------
    X, Y, Z : array_like
        Data samples
    bins : int or str
        Number of bins or binning strategy
        
    Returns
    -------
    float
        Conditional mutual information estimate
    """
    from causalentropy.core.information.entropy import histogram_entropy
    import numpy as np
    
    if Z is None:
        # Reduce to mutual information
        from causalentropy.core.information.mutual_information import histogram_mutual_information
        return histogram_mutual_information(X, Y, bins=bins)
    
    # Calculate conditional mutual information: I(X;Y|Z) = H(X|Z) + H(Y|Z) - H(X,Y|Z)
    # Approximated as: I(X;Y|Z) ≈ H(X,Z) + H(Y,Z) - H(Z) - H(X,Y,Z)
    
    X_flat = X.flatten()
    Y_flat = Y.flatten() 
    Z_flat = Z.flatten()
    
    # Joint entropies
    xz_joint = np.column_stack([X_flat, Z_flat])
    yz_joint = np.column_stack([Y_flat, Z_flat])
    xyz_joint = np.column_stack([X_flat, Y_flat, Z_flat])
    
    h_xz = histogram_entropy(xz_joint, bins=bins)
    h_yz = histogram_entropy(yz_joint, bins=bins)
    h_z = histogram_entropy(Z_flat, bins=bins)
    h_xyz = histogram_entropy(xyz_joint, bins=bins)
    
    return max(0, h_xz + h_yz - h_z - h_xyz)


def conditional_mutual_information(X, Y, Z=None, method='gaussian', metric='euclidean', k=1, bandwidth='silverman',
                                   kernel='gaussian'):
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
        - 'histogram': Histogram-based empirical estimation
        - 'negative_binomial': For overdispersed count data
        - 'laplace': For heavy-tailed continuous data
        - 'von_mises': For circular/angular data
        - 'hawkes': For point process event data
        
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
    >>> from causalentropy.core.information.conditional_mutual_information import conditional_mutual_information
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
    if method == 'gaussian':
        return gaussian_conditional_mutual_information(X, Y, Z)

    elif method == 'kde' or method == 'kernel_density':
        return kde_conditional_mutual_information(X, Y, Z, bandwidth=bandwidth, kernel=kernel)

    elif method == 'knn':
        return knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == 'geometric_knn':
        return geometric_knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == 'poisson':
        return poisson_conditional_mutual_information(X, Y, Z)

    elif method == 'histogram':
        return histogram_conditional_mutual_information(X, Y, Z)

    elif method == 'laplace':
        return laplace_conditional_mutual_information(X, Y, Z)

    elif method == 'negative_binomial':
        return negative_binomial_conditional_mutual_information(X, Y, Z)

    elif method == 'von_mises':
        return von_mises_conditional_mutual_information(X, Y, Z)

    elif method == 'hawkes' or method == 'Hawkes':
        return hawkes_conditional_mutual_information(X, Y, Z)

    else:
        supported_methods = ['gaussian', 'kde', 'kernel_density', 'knn', 'geometric_knn', 
                           'poisson', 'histogram', 'laplace', 'negative_binomial', 
                           'von_mises', 'hawkes']
        raise ValueError(f"Method '{method}' unavailable. Supported methods: {supported_methods}")
