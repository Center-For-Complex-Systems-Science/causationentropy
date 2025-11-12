import numpy as np
from scipy.spatial.distance import cdist
from scipy.special import digamma

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

    def _detcorr(A):
        C = np.corrcoef(A.T)
        # For 1D input, corrcoef returns scalar 1.0, and log(1.0) = 0.0
        return 0.0 if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]

    SZ = _detcorr(Z)
    SXZ = _detcorr(np.hstack((X, Z)))
    SYZ = _detcorr(np.hstack((Y, Z)))
    SXYZ = _detcorr(np.hstack((X, Y, Z)))

    cmi = 0.5 * (SXZ + SYZ - SZ - SXYZ)
    return cmi


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


def knn_conditional_mutual_information(X, Y, Z, metric="minkowski", k=1):
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
    metric : str, default='minkowski'
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
        return knn_mutual_information(X, Y, metric=metric, k=k)
    else:

        JS = np.column_stack((X, Y, Z))
        # Find the K-th smallest distance in the joint space
        if metric == "minkowski":
            D = np.sort(cdist(JS, JS, metric=metric, p=k + 1), axis=1)[:, k]
        else:
            D = np.sort(cdist(JS, JS, metric=metric), axis=1)[:, k]
        epsilon = D
        # Count neighbors within epsilon in marginal spaces
        Dxz = cdist(np.column_stack((X, Z)), np.column_stack((X, Z)), metric=metric)
        nxz = np.sum(Dxz < epsilon[:, None], axis=1) - 1
        Dyz = cdist(np.column_stack((Y, Z)), np.column_stack((Y, Z)), metric=metric)
        nyz = np.sum(Dyz < epsilon[:, None], axis=1) - 1
        Dz = cdist(Z, Z, metric=metric)
        nz = np.sum(Dz < epsilon[:, None], axis=1) - 1

        # VP Estimation formula
        I = digamma(k) - np.mean(digamma(nxz + 1) + digamma(nyz + 1) - digamma(nz + 1))
        return I


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
    YZdist = cdist(np.hstack((Y, Z)), np.hstack((Y, Z)), metric=metric)
    XZdist = cdist(np.hstack((X, Z)), np.hstack((X, Z)), metric=metric)
    XYZdist = cdist(np.hstack((X, Y, Z)), np.hstack((X, Y, Z)), metric=metric)
    Zdist = cdist(Z, Z, metric=metric)
    HZ = geometric_knn_entropy(Z, Zdist, k)
    HXZ = geometric_knn_entropy(np.hstack((X, Z)), XZdist, k)
    HYZ = geometric_knn_entropy(np.hstack((Y, Z)), YZdist, k)
    HXYZ = geometric_knn_entropy(np.hstack((X, Y, Z)), XYZdist, k)
    cmi = HXZ + HYZ - HXYZ - HZ
    return cmi


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
        SS[0:SzX, 0:SzX] = SS[0:SzX, 0:SzX] + SXYZ[0:SzX, SzX : SzX + SzY]
        SS[SzX : SzX + SzY, SzX : SzX + SzY] = (
            SS[SzX : SzX + SzY, SzX : SzX + SzY] + SXYZ[SzX : SzX + SzY, 0:SzX]
        )
        S_est1 = SS[
            np.concatenate((indY.T, indZ.T), axis=0),
            np.concatenate((indY.T, indZ.T), axis=0),
        ]
        S_est2 = SS[
            np.concatenate((indX.T, indZ.T), axis=0),
            np.concatenate((indX.T, indZ.T), axis=0),
        ]
        HYZ = poisson_joint_entropy(S_est1)
        SindZ = SS[indZ, indZ]
        HZ = poisson_joint_entropy(SindZ)
        HXYZ = poisson_joint_entropy(SXYZ - np.diag(Sa))
        HXZ = poisson_joint_entropy(S_est2)
        H_YZ = HYZ - HZ
        H_XYZ = HXYZ - HXZ
        cmi = H_XYZ - H_YZ
        return cmi


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
        cmi = gaussian_conditional_mutual_information(X, Y, Z)

    elif method == "kde" or method == "kernel_density":
        cmi = kde_conditional_mutual_information(
            X, Y, Z, bandwidth=bandwidth, kernel=kernel
        )

    elif method == "knn":
        cmi = knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "geometric_knn":
        cmi = geometric_knn_conditional_mutual_information(X, Y, Z, metric=metric, k=k)

    elif method == "poisson":
        cmi = poisson_conditional_mutual_information(X, Y, Z)

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

    # Ensure non-negativity: CMI is theoretically always >= 0,
    # but finite sample estimation can produce small negative values.
    # Only clamp finite values; preserve NaN/inf for error handling.
    if np.isfinite(cmi):
        return max(0.0, cmi)
    return cmi
