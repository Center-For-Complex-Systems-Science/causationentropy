import scipy
from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import scipy as sp
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.special import gamma as Gamma
from scipy.special import digamma as Digamma
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoCV
import numpy as np
import scipy.linalg as la

from causalentropy.core.information.entropy import kde_entropy, geometric_knn_entropy, poisson_entropy, poisson_joint_entropy
from causalentropy.core.information.mutual_information import geometric_knn_mutual_information, kde_mutual_information, knn_mutual_information, \
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
        return float(C) if np.ndim(C) == 0 else np.linalg.slogdet(C)[1]

    SZ = _detcorr(Z)
    SXZ = _detcorr(np.hstack((X, Z)))
    SYZ = _detcorr(np.hstack((Y, Z)))
    SXYZ = _detcorr(np.hstack((X, Y, Z)))

    return 0.5 * (SXZ + SYZ - SZ - SXYZ)

def kde_conditional_mutual_information(X, Y, Z, bandwidth='silverman', kernel='gaussian'):
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
    """KNN version (via the paper by Kraskov, Stogbauer and Grassberger called
    'Estimating mutual information'
    of Conditional mutual information for Causation entropy...
    """
    if Z is None:
        return knn_mutual_information(X, Y, metric=metric, k=k)  # np.max([self.MutualInfo_KNN(X,self.Y),0])
    else:
        XY = np.concatenate((X, Y), axis=1)
        MIXYZ = knn_mutual_information(XY, Z, metric=metric, k=k)
        MIXY = knn_mutual_information(X, Y, metric=metric, k=k)

        return MIXY - MIXYZ

def geometric_knn_conditional_mutual_information(X, Y, Z, metric='euclidean', k=1):
    """A method for estimating CMI (which will be
    needed in Causation entropy) which comes from the paper
    'Geometric k-nearest neighbor estimation of entropy and mutual information'
    by Lord, Sun and Bollt
    """

    if Z is None:
        return geometric_knn_mutual_information(X, Y)
    YZdist = cdist(np.hstack((Y, Z)), np.hstack((Y, Z)), metric=metric)
    XZdist = cdist(np.hstack((X, Z)), np.hstack((X, Z)), metric=metric)
    XYZdist = cdist(np.hstack((X, Y, Z)), np.hstack((X, Y, Z)), metric=metric)
    Zdist = cdist(Z, Z, metric=metric)
    HZ = geometric_knn_entropy(Z, k, Zdist)
    HXZ = geometric_knn_entropy(np.hstack((X, Z)), k, XZdist)
    HYZ = geometric_knn_entropy(np.hstack((Y, Z)), k, YZdist)
    HXYZ = geometric_knn_entropy(np.hstack((X, Y, Z)), k, XYZdist)
    return HXZ + HYZ - HXYZ - HZ

def poisson_conditional_mutual_information(X, Y, Z):
    """Estimate of conditional mutual information from Poisson marginals,
    from the paper by Fish, Sun and Bollt entitled:
        Interaction Networks from Discrete Event Data by Poisson Multivariate
        Mutual Information Estimation and Information Flow with Applications
        from Gene Expression Data"""

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


def negative_binomial_considtional_mutual_information(X, Y, Z):
    raise NotImplemented()


def hawkes_conditional_mutual_information(X, Y, Z):
    raise NotImplemented()


def von_mises_conditional_mutual_information(X, Y, Z):
    raise NotImplemented()


def laplace_conditional_mutual_information(X, Y, Z):
    raise NotImplemented()


def histogram_conditional_mutual_information(X, Y, Z):
    raise NotImplemented()

def conditional_mutual_information(X, Y, Z=None, method='gaussian', metric='euclidean', k=1, bandwidth='silverman', kernel='gaussian'):
    """Compute the CMI based upon whichever method"""
    if method == 'gaussian':
        return gaussian_conditional_mutual_information(X, Y, Z)

    elif method == 'kernel_density':
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

    elif method == 'negative_inomial':
        return negative_binomial_mutual_information(X, Y, Z)

    elif method == 'von_mises':
        return von_mises_conditional_mutual_information(X, Y, Z)

    elif method == 'Hawkes':
        return hawkes_conditional_mutual_information(X, Y, Z)

    else:
        raise ValueError(f"Method {method} unavailable.")