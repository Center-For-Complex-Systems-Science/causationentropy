import numpy as np
from scipy.special import gamma
from sklearn.neighbors import KernelDensity
import scipy
from scipy.stats import nbinom
from scipy.special import i0, i1


def l2dist(a, b):
    return np.linalg.norm(a - b)


def hyperellipsoid_check(svd_Yi, Z_i):
    # Check if Z_i lies within the hyperellipsoid defined by svd_Yi
    U, S, Vt = svd_Yi
    transformed = np.dot(Z_i, Vt.T) / S
    return np.sum(transformed ** 2) <= 1

def kde_entropy(X, bandwidth='silverman', kernel='gaussian'):
    """
    Parameters
    ----------
    X
    bandwidth
    kernel

    Returns
    -------

    """
    kde = KernelDensity(bandwidth=bandwidth, kernel=kernel).fit(X)
    log_density = np.exp(kde.score_samples(X))
    Hx = -np.sum(np.log(log_density)) / len(log_density)
    return Hx

def geometric_knn_entropy(X, Xdist, k=1):
    """A method for estimating entropy (which will be used for estimating mutual
    informations needed in Causation entropy) which comes from the paper
    'Geometric k-nearest neighbor estimation of entropy and mutual information'
    by Lord, Sun and Bollt
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
    """A method for esitmating the Poisson entropy that will be needed for
    computation of conditional mutual informations. For details see the paper
    by Fish and Bollt
    'Interaction networks from discrete event data by Poisson multivariate
    mutual information estimation and information flow with applications
    from gene expression data'
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
    """A method for esitmating the Poisson entropy that will be needed for
    computation of conditional mutual informations. For details see the paper
    by Fish and Bollt
    'Interaction networks from discrete event data by Poisson multivariate
    mutual information estimation and information flow with applications
    from gene expression data'
    """
    T = np.triu(Cov, 1)
    T = np.matrix(T)
    U = np.matrix(np.diag(Cov))
    Ent1 = np.sum(poisson_entropy(U))
    Ent2 = np.sum(T)
    return Ent1 + Ent2


def negative_binomial_entropy(r, p, max_k=None, tol = 1e-12, base = np.e):
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

    ks = np.arange(max_k + 1)          # 0 … max_k
    pmf = nbinom.pmf(ks, r, p)

    # numerical entropy (ignore zero-mass terms to avoid log(0))
    mask = pmf > 0
    H = -np.sum(pmf[mask] * np.log(pmf[mask]))

    # convert to requested base
    if base != np.e:
        H /= np.log(base)
    return H

def hawkes_entropy(events, mu, alpha, beta, T=None, base=np.e):
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
    b = np.asfarray(b)
    if np.any(b <= 0):
        raise ValueError("scale parameter b must be positive")

    H = 1.0 + np.log(2.0 * b)  # nats
    if base != np.e:
        H /= np.log(base)
    return H.item() if H.size == 1 else H

def histogram_entropy(x, bins='auto', base=np.e):
    x = np.asarray(x).ravel()
    if x.size == 0:
        raise ValueError("x must contain at least one value")

    counts, _ = np.histogram(x, bins=bins)
    probs = counts.astype(float)
    probs /= probs.sum()

    probs = probs[probs > 0]          # drop empty bins
    H = -np.sum(probs * np.log(probs))

    if base != np.e:
        H /= np.log(base)
    return H


def kolmogorov_sinai_entropy(orbit, system):
    raise NotImplemented()

def topological_entropy(system):
    raise NotImplemented()

def permutation_entropy(ts, m, delay):
    raise NotImplemented()

def approximate_entropy(ts, m, r):
    raise NotImplemented()

def sample_entropy(ts, m, r):
    raise NotImplemented()

def multiscale_entropy(ts, m, r, scales):
    raise NotImplemented()

def fuzzy_entropy(ts, m, r):
    raise NotImplemented()

def dispersion_entropy(ts, m, classes):
    raise NotImplemented()

def wavelet_entropy(ts, wavelet):
    raise NotImplemented()

def transfer_entropy(source, target, k, l):
    raise NotImplemented()

def entropy_rate_time_series(ts, order):
    raise NotImplemented()

def algorithmic_entropy(obj):
    raise NotImplemented()

# Graph Based Entropies

def shannon_graph_entropy(G):
    raise NotImplemented()

def von_neumann_graph_entropy(G):
    raise NotImplemented()

def laplacian_spectrum_entropy(G):
    raise NotImplemented()

def entropy_rate_random_walk(G):
    raise NotImplemented()

def communicability_entropy(G):
    raise NotImplemented()
