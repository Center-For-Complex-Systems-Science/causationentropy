import numpy as np
import networkx as nx


def logistic_map(X, r):
    return r * X * (1 - X)


def logisic_dynamics(r=3.99, sigma=0.1, seed=42):
    """Network coupled logistic map, r is the logistic map parameter
       and sigma is the coupling strength between oscillators"""

    rng = np.random.default_rng(seed)
    p = 0.1
    n = 20
    T = 100 # n and T are self attributes
    G = nx.erdos_renyi_graph(n, p, seed=seed)
    A = nx.adjacency_matrix(G)
    # Must adjust the adjacency matrix so that dynamics stay in [0,1]
    A = A / np.sum(A, axis=1)
    A = A.T

    # Since the row sums equal to 1 the Laplacian matrix is easy...
    L = np.eye(n) - A
    L = np.array(L)

    XY = np.zeros((T, n))
    XY[0, :] = rng.random(n)
    for i in range(1, T):
        XY[i, :] = logistic_map(XY[i - 1, :], r) - sigma * np.dot(L, logistic_map(XY[i - 1, :], r)).T

    return XY


def Gen_Stochastic_Gaussian(self, Epsilon=1e-1):
    """Linear stochastic Gaussian process"""

    if self.NetworkAdjacency is None:
        raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")

    if self.Rho is None:
        raise ValueError("Missing Rho, please set it using set_Rho")

    self.Gaussian_Epsilon = Epsilon
    R = 2 * (np.random.rand(self.n, self.n) - 0.5)
    A = np.array(self.NetworkAdjacency) * R
    A = A / np.max(np.abs(np.linalg.eigvals(A)))
    A = A * self.Rho
    self.Lin_Stoch_Gaussian_Adjacency = A
    XY = np.zeros((self.T, self.n))
    XY[0, :] = Epsilon * np.random.randn(1, self.n)
    for i in range(1, self.T):
        Xi = np.dot(A, np.matrix(XY[i - 1, :]).T) + Epsilon * np.random.randn(self.n, 1)
        XY[i, :] = Xi.T
    self.XY = XY


def Gen_Poisson_Data(self, Epsilon=1, noiseLevel=1):
    if self.NetworkAdjacency is None:
        raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
    p = int(self.n)
    noiseLam = noiseLevel * np.ones((p, 1))
    Zp = int(p * (p - 1) / 2)
    lambdas = Epsilon * np.ones((p + Zp, 1))
    P = self.PermutationMatrix()
    One_p = np.ones((p, 1))
    NK = self.nchoosek(p, 2)
    # Size = (p,p)
    Tr = np.matrix(self.NetworkAdjacency[NK[:, 0] - 1, NK[:, 1] - 1])
    I_p = np.eye(p)
    T_p = np.dot(One_p, Tr)
    PTP = np.array(P) * np.array(T_p)
    B = np.concatenate((I_p, PTP), axis=1)
    B = B.T
    Lambs = lambdas.T * np.ones((self.T, Zp + p))
    Y = np.random.poisson(Lambs)

    Lambs = noiseLam.T * np.ones((self.T, p))
    E = np.random.poisson(Lambs)

    X = np.dot(Y, B) + E
    return X


def Gen_Stochastic_Poisson_Pedeli(self, noiseLevel=0):
    """Stochastic Poisson process -
    Structure comes from the paper:
        'Some properties of multivariate INAR(1) processes'

    By Pedeli and Karlis

    """

    if self.NetworkAdjacency is None:
        raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")

    if self.Rho is None:
        raise ValueError("Missing Rho, please set it using set_Rho")

    p = self.n
    alpha = self.Rho * np.random.rand(p)
    self.PoissonAlphas = alpha
    X = np.zeros((self.T, self.n))
    X.astype('int32')
    T = self.T
    self.T = 1
    X[0, :] = self.Gen_Poisson_Data(noiseLevel=noiseLevel).astype('int32')

    for i in range(1, T):
        S = self.Gen_Poisson_Data(noiseLevel=noiseLevel)
        B = np.random.binomial(X[i - 1, :].astype('int32'), alpha)
        X[i, :] = B.astype('int32') + S.astype('int32')

    self.XY = X
    self.T = T


def Gen_Stochastic_Poisson_Armillotta(self, Lambda_0=1, Betas=np.array([0.2, 0.3, 0.2])):
    """Stochastic Poisson process which is generally known as
       an INteger AutoRegressive (INAR) Process from the paper
       on Network INAR or in this case PNAR...   The name of the paper is:
           Poisson network autoregression
           and it is written by Armillotta and Fokianos"""

    if self.NetworkAdjacency is None:
        raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")

    if len(Betas) != 3:
        raise ValueError(
            "The Armillotta version of Stochastic Poisson only currently allows for exactly 3 beta values, and thus only allows Tau = 1...")
    beta_0, beta_1, beta_2 = Betas
    A = self.NetworkAdjacency
    self.Armillotta_Poisson_Betas = Betas
    self.Armillotta_Poisson_Lambda_0 = Lambda_0
    Y_init = np.random.poisson(Lambda_0, (self.n, 1))
    # To ensure we don't get NaN's do below steps
    D = np.sum(A, axis=1)
    D[np.where(D == 0)] = 1
    C = A / D
    lam_0 = beta_0 + (beta_1 * np.dot(C, Y_init)).T + beta_2 * Y_init.T
    X = np.zeros((self.T, self.n))
    X[0, :] = np.random.poisson(lam_0)

    for i in range(1, self.T):
        lam_t = beta_0 + (beta_1 * np.dot(C, X[i - 1, :])) + beta_2 * X[i - 1, :]
        X[i, :] = np.random.poisson(lam_t)

    self.XY = X