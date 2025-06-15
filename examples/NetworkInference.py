# -*- coding: utf-8 -*-
"""
Created on Tue Apr 29 18:25:12 2025

@author: jefis
"""

from sklearn.neighbors import KernelDensity
from sklearn.linear_model import LinearRegression
import scipy as sp
from scipy.spatial import distance
from scipy.spatial.distance import cdist
from scipy.special import gamma as Gamma
from scipy.special import digamma as Digamma
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoCV
import scipy.stats as SStats
import numpy as np
import itertools
import copy
import datetime
import glob
from matplotlib import pyplot as plt
from matplotlib import rc
import time
from joblib import Parallel, delayed
import multiprocessing

"""Written by Jeremie Fish, 
   Last Update: May 22nd 2025

   Depending on method used please cite the appropriate papers..."""


class NetworkInference:
    """A class for NetworkInference. Version number 0.2"""

    def __init__(self):

        # Version
        self.VersionNum = 0.2

        # conditional (causation entropy) return stuff
        self.conditional_returns_set = 'existing_edges'

        # for potential multiprocessing
        self.num_processes = np.max([multiprocessing.cpu_count() - 1, 1])  # default to cpus-1 unless only 1 cpu...
        self.parallel_shuffles = False
        self.parallel_nodes = False

        # Time series stuff
        self.n = 100
        self.T = 200

        # Initialize time series to nothing, this must be entered or generated
        self.X = None
        self.Y = None
        self.Z = None
        self.XY = None

        # Network stuff
        self.NetworkAdjacency = None
        self.NetworkType = None
        self.Lin_Stoch_Gaussian_Adjacency = None
        self.Logistic_Adjacency = None

        # Rho, which is effectively how "noise dominated" the system is
        # It is the spectral radius of the input matrix
        self.Rho = None

        # Inference Stuff
        self.Overall_Inference_Method = 'Standard_oCSE'
        self.InferenceMethod_oCSE = 'Gaussian'
        self.KernelBandWidth = None
        self.KernelType = 'gaussian'
        self.KNN_K = 10
        self.KNN_dist_metric = 'minkowski'
        self.Tau = int(1)  # Tau is assumed to be 1 always but can be changed

        # oCSE stuff
        self.Num_Shuffles_oCSE = 100
        self.Forward_oCSE_alpha = 0.02
        self.Backward_oCSE_alpha = 0.02

        # Information Informed LASSO stuff
        self.IIkfold = 10  # For crossfold validation when num data points small
        self.II_InfCriterion = 'bic'  # can be aic or bic, bic does better
        self.max_num_lambdas = 100

        # Miscellaneous
        self.Pedeli_PoissonAlphas = None
        self.Correlation_XY = None
        self.Prctile = None
        self.Dict = None
        self.Sinit = None
        self.Sfinal = None
        # TO DO,Add all of the other listed methods. These will become available in future renditions of the software.
        self.Available_Inference_Methods = ['Standard_oCSE', 'Alternative_oCSE', 'InformationInformed_LASSO',
                                            'LASSO']  # ['Standard_oCSE', 'Alternative_oCSE', 'Lasso', 'Graphical_Lasso',
        # 'GLM_Lasso', 'Kernel_Lasso','Entropic_Regression','Convergent_Cross_Mapping',
        # 'PCMCI','PCMCIplus','LPMCI','Partial_Correlation','GPDC']
        self.DataType = 'Continuous'
        self.font = None

        # Dynamics stuff
        self.Gaussian_Epsilon = None
        self.Logistic_parameter_r = None
        self.Logistic_Coupling = None
        self.Armillotta_Poisson_Betas = None
        self.Armillotta_Poisson_Lambda_0 = None

        # For Error computation
        self.B = None  # The estimated Adjacency matrix
        self.TPR = None
        self.FPR = None
        self.AUC = None
        self.Xshuffle = False

    def set_conditional_returns_set(self, cond_ret_type):
        self.conditional_returns_set = cond_ret_type

    def set_KNN_dist_metric(self, metric):
        self.KNN_dist_metric = metric

    def set_parallel_nodes(self, truth_val):
        self.parallel_nodes = truth_val

    def set_parallel_shuffles(self, truth_val):
        self.parallel_shuffles = truth_val

    def set_num_processes(self, num):
        self.num_processes = num

    def set_II_InfCriterion(self, criterion):
        """Should be a string either 'aic' or 'bic' the only two options """
        self.II_InfCriterion = criterion

    def set_IIkfold(self, num):
        """Should be a positive integer!"""
        self.IIkfold = num

    def set_Xshuffle(self, TF=False):
        self.Xshuffle = TF

    def set_font(self, font):
        self.font = font

    def set_AUC(self, AUC):
        self.AUC = AUC

    def set_FPR(self, FPR):
        self.FPR = FPR

    def set_TPR(self, TPR):
        self.TPR = TPR

    def set_B(self, B):
        self.B = B

    def set_Tau(self, Tau):
        self.Tau = int(Tau)

    def set_DataType(self, Type):
        Types = ['Discrete', 'Continuous']
        if Type not in Types:
            raise ValueError("Sorry the only types allowed for DataType are: ", Types)

        self.DataType = Type

    def set_Available_Inference_Methods(self, Methods):
        self.Available_Inference_Methods = Methods

    def set_Armillotta_Poisson_Betas(self, Betas):
        self.Armillotta_Poisson_Betas = Betas

    def set_Armillotta_Poisson_Lambda_0(self, Lam0):
        self.Armillotta_Poisson_Lambda_0 = Lam0

    def set_Lin_Stoch_Gaussian_Adjacency(self, A):
        self.Lin_Stoch_Gaussian_Adjacency = A

    def set_Logistic_Adjacency(self, A):
        self.Logistic_Adjacency = A

    def set_Pedeli_PoissonAlphas(self, alphas):
        self.Pedeli_PoissonAlphas = alphas

    def set_Prctile(self, Prctile):
        self.Prctile = Prctile

    def set_Dict(self, Dict):
        self.Dict = Dict

    def set_Sinit(self, Sin):
        self.Sinit = Sin

    def set_Sfinal(self, Sfin):
        self.Sfinal = Sfin

    def set_Gaussian_Epsilon(self, Eps):
        self.Gaussian_Epsilon = Eps

    def set_Logistic_parameter_r(self, r):
        self.Logistic_parameter_r = r

    def set_Logistic_Coupling(self, Sigma):
        self.Logistic_Coupling = Sigma

    def set_n(self, n):
        self.n = n

    def set_T(self, T):
        self.T = T

    def set_X(self, X):
        """The time series (T x n) which is the set of predictors"""

        self.X = X
        self.T = X.shape[0]

    def set_Y(self, Y):
        """The time series (T x m) of target variable(s)
           (note m usually is equal to 1 but for generalized
           interactions between many nodes it may be useful
           to have m>1) """

        self.Y = Y
        if self.Y is not None:
            self.T = Y.shape[0]

    def set_Z(self, Z):
        """Set the conditioning set of variables. It is possible that
           you do not want to condition on anything in which case it
           is fine that Z is None"""
        self.Z = Z

    def set_XY(self, XY):

        self.XY = XY
        if self.XY is not None:
            self.T = XY.shape[0]
            self.n = XY.shape[1]

    def set_Forward_oCSE_alpha(self, alpha):
        self.Forward_oCSE_alpha = alpha

    def set_Num_Shuffles_oCSE(self, Num_Shuffles):
        self.Num_Shuffles_oCSE = Num_Shuffles

    def set_Backward_oCSE_alpha(self, alpha):
        self.Backward_oCSE_alpha = alpha

    def set_Correlation_XY(self, Correlation_XY):
        self.Correlation_XY = Correlation_XY

    def set_Overall_Inference_Method(self, Method):
        self.Overall_Inference_Method = Method

    def set_InferenceMethod_oCSE(self, Method):
        self.InferenceMethod_oCSE = Method

    def set_KernelBandWidth(self, Bandwidth):
        self.KernelBandWidth = Bandwidth

    def set_KernelType(self, Type):
        self.KernelType = Type

    def set_KNN_K(self, K):
        self.KNN_K = K

    def set_NetworkAdjacency(self, A):
        """Adjacency matrix assumed to be (n x n)"""
        self.NetworkAdjacency = A
        self.n = A.shape[0]

    def set_NetworkType(self, Type):
        self.NetworkType = Type

    def set_Rho(self, Rho):
        self.Rho = Rho

    def return_conditional_returns_set(self):
        return self.conditional_returns_set

    def return_KNN_dist_metric(self):
        return self.KNN_dist_metric

    def return_parallel_nodes(self):
        return self.parallel_nodes

    def return_parallel_shuffles(self):
        return self.parallel_shuffles

    def return_num_processes(self):
        return self.num_processes

    def return_Xshuffle(self):
        return self.Xshuffle

    def return_II_InfCriterion(self):
        return self.II_InfCriterion

    def return_IIkfold(self):
        return self.IIkfold

    def return_font(self):
        return self.font

    def return_DataType(self):
        return self.DataType

    def return_VersionNum(self):
        return self.VersionNum

    def return_Available_Inference_Methods(self):
        return self.Available_Inference_Methods

    def return_n(self):
        return self.n

    def return_T(self):
        return self.T

    def return_Num_Shuffles_oCSE(self):
        return self.Num_Shuffles_oCSE

    def return_Z(self):
        return self.Z

    def return_X(self):
        return self.X

    def return_Y(self):
        return self.Y

    def return_XY(self):
        return self.XY

    def return_NetworkAdjacency(self):
        return self.NetworkAdjacency

    def return_NetworkType(self):
        return self.NetworkType

    def return_Rho(self):
        return self.Rho

    def return_Overall_Inference_Method(self):
        return self.Overall_Inference_Method

    def return_InferenceMethod_oCSE(self):
        return self.InferenceMethod_oCSE

    def return_KernelBandWidth(self):
        return self.KernelBandWidth

    def return_KernelType(self):
        return self.KernelType

    def return_KNN_K(self):
        return self.KNN_K

    def return_Tau(self):
        return self.Tau

    def return_Forward_oCSE_alpha(self):
        return self.Forward_oCSE_alpha

    def return_Backward_oCSE_alpha(self):
        return self.Backward_oCSE_alpha

    def return_Pedeli_PoissonAlphas(self):
        return self.Pedeli_PoissonAlphas

    def return_Correlation_XY(self):
        return self.Correlation_XY

    def return_Prctile(self):
        return self.Prctile

    def return_Sinit(self):
        return self.Sinit

    def return_Sfinal(self):
        return self.Sfinal

    def return_Dict(self):
        return self.Dict

    def return_Lin_Stoch_Gaussian_Adjacency(self):
        return self.Lin_Stoch_Gaussian_Adjacency

    def return_Logistic_Adjacency(self):
        return self.Logistic_Adjacency

    def return_Gaussian_Epsilon(self):
        return self.Gaussian_Epsilon

    def return_Logistic_parameter_r(self):
        return self.Logistic_parameter_r

    def return_Logistic_Coupling(self):
        return self.Logistic_Coupling

    def return_Armillotta_Poisson_Betas(self):
        return self.Armillotta_Poisson_Betas

    def return_Armillotta_Poisson_Lambda_0(self):
        return self.Armillotta_Poisson_Lambda_0

    def return_B(self):
        return self.B

    def return_TPR(self):
        return self.TPR

    def return_FPR(self):
        return self.FPR

    def return_AUC(self):
        return self.AUC

    def save_state(self):
        VersionNum = self.VersionNum

        Save_Dict = {}
        Save_Dict['VersionNum'] = self.VersionNum
        Save_Dict['T'] = self.T
        Save_Dict['n'] = self.n
        Save_Dict['X'] = self.X
        Save_Dict['Y'] = self.Y
        Save_Dict['Z'] = self.Z
        Save_Dict['XY'] = self.XY
        Save_Dict['NetworkAdjacency'] = self.NetworkAdjacency
        Save_Dict['NetworkType'] = self.NetworkType
        Save_Dict['Lin_Stoch_Gaussian_Adjacency'] = self.Lin_Stoch_Gaussian_Adjacency
        Save_Dict['Logistic_Adjacency'] = self.Logistic_Adjacency
        Save_Dict['Rho'] = self.Rho
        Save_Dict['Overall_Inference_Method'] = self.Overall_Inference_Method
        Save_Dict['InferenceMethod_oCSE'] = self.InferenceMethod_oCSE
        Save_Dict['KernelBandWidth'] = self.KernelBandWidth
        Save_Dict['KernelType'] = self.KernelType
        Save_Dict['KNN_K'] = self.KNN_K
        Save_Dict['Tau'] = self.Tau
        Save_Dict['Num_Shuffles_oCSE'] = self.Num_Shuffles_oCSE
        Save_Dict['Forward_oCSE_alpha'] = self.Forward_oCSE_alpha
        Save_Dict['Backward_oCSE_alpha'] = self.Backward_oCSE_alpha
        Save_Dict['Pedeli_PoissonAlphas'] = self.Pedeli_PoissonAlphas
        Save_Dict['Correlation_XY'] = self.Correlation_XY
        Save_Dict['Prctile'] = self.Prctile
        Save_Dict['Dict'] = self.Dict
        Save_Dict['Sinit'] = self.Sinit
        Save_Dict['Sfinal'] = self.Sfinal
        Save_Dict['DataType'] = self.DataType
        Save_Dict['Gaussian_Epsilon'] = self.Gaussian_Epsilon
        Save_Dict['Logistic_parameter_r'] = self.Logistic_parameter_r
        Save_Dict['Logistic_Coupling'] = self.Logistic_Coupling
        Save_Dict['Armillotta_Poisson_Betas'] = self.Armillotta_Poisson_Betas
        Save_Dict['Armillotta_Poisson_Lambda_0'] = self.Armillotta_Poisson_Lambda_0
        Save_Dict['Available_Inference_Methods'] = self.Available_Inference_Methods
        Save_Dict['font'] = self.font
        Save_Dict['B'] = self.B
        Save_Dict['TPR'] = self.TPR
        Save_Dict['FPR'] = self.FPR
        Save_Dict['AUC'] = self.AUC
        Save_Dict['Xshuffle'] = self.Xshuffle
        Save_Dict['IIkfold'] = self.IIkfold
        Save_Dict['II_InfCriterion'] = self.II_InfCriterion
        Save_Dict['num_processes'] = self.num_processes
        Save_Dict['parallel_shuffles'] = self.parallel_shuffles
        Save_Dict['parallel_nodes'] = self.parallel_nodes
        Save_Dict['KNN_dist_metric'] = self.KNN_dist_metric
        Save_Dict['conditional_returns_set'] = self.condtional_returns_set

        Date = datetime.datetime.today().strftime('%Y-%m-%d')
        F = glob.glob('NetworkInference_version_' + str(VersionNum) + '_Date_' + Date + '*.npz')
        num = len(F) + 1

        np.savez('NetworkInference_version_' + str(VersionNum) + '_Date_' + Date + '_num_' + str(num), Save_Dict)

    def load_state(self, Date=None, Num=None, Version=None):
        if Date is None:
            # Default to today's date for attempt at loading
            Date = datetime.datetime.today().strftime('%Y-%m-%d')

        if Version is None:
            VersionNum = self.VersionNum

        if Num is None:
            # Try finding the local file
            F = glob.glob('NetworkInference_version_' + str(VersionNum) + '_Date_' + Date + '*.npz')
            if len(F) == 0:
                raise ValueError("Sorry no NetworkInference file with date: ", Date,
                                 " can be found in the current directory, please specify the date!")

            else:
                # If the number is unspecified, then always choose the most recent (largests number)
                NumList = []
                for i in range(len(F)):
                    StringNum = F[i].split('_')[-1].split('.npz')[0]
                    NumList.append(int(StringNum))

                Num = max(NumList)
                if len(F) > 1:
                    print("There were multiple files discovered for date: ", Date,
                          " the most recent one was used by default")
                    print("If you wish to use a different filenumber please specify the file number to use")

        Filename = "NetworkInference_version_" + str(VersionNum) + "_Date_" + Date + "_num_" + str(Num) + ".npz"
        npz = np.load(Filename, allow_pickle=True)
        Load_Dict = npz['arr_0'][()]
        # Now set all values to the loaded states
        self.set_T(Load_Dict['T'])
        self.set_n(Load_Dict['n'])
        self.set_X(Load_Dict['X'])
        self.set_Y(Load_Dict['Y'])
        self.set_Z(Load_Dict['Z'])
        self.set_XY(Load_Dict['XY'])
        self.set_NetworkAdjacency(Load_Dict['NetworkAdjacency'])
        self.set_NetworkType(Load_Dict['NetworkType'])
        self.set_Lin_Stoch_Gaussian_Adjacency(Load_Dict['Lin_Stoch_Gaussian_Adjacency'])
        self.set_Logistic_Adjacency(Load_Dict['Logistic_Adjacency'])
        self.set_Rho(Load_Dict['Rho'])
        self.set_Overall_Inference_Method(Load_Dict['Overall_Inference_Method'])
        self.set_InferenceMethod_oCSE(Load_Dict['InferenceMethod_oCSE'])
        self.set_KernelBandWidth(Load_Dict['KernelBandWidth'])
        self.set_KernelType(Load_Dict['KernelType'])
        self.set_KNN_K(Load_Dict['KNN_K'])
        self.set_Tau(Load_Dict['Tau'])
        self.set_Num_Shuffles_oCSE(Load_Dict['Num_Shuffles_oCSE'])
        self.set_Forward_oCSE_alpha(Load_Dict['Forward_oCSE_alpha'])
        self.set_Backward_oCSE_alpha(Load_Dict['Backward_oCSE_alpha'])
        self.set_Pedeli_PoissonAlphas(Load_Dict['Pedeli_PoissonAlphas'])
        self.set_Correlation_XY(Load_Dict['Correlation_XY'])
        self.set_Prctile(Load_Dict['Prctile'])
        self.set_Dict(Load_Dict['Dict'])
        self.set_Sinit(Load_Dict['Sinit'])
        self.set_Sfinal(Load_Dict['Sfinal'])
        self.set_DataType(Load_Dict['DataType'])
        self.set_Gaussian_Epsilon(Load_Dict['Gaussian_Epsilon'])
        self.set_Logistic_parameter_r(Load_Dict['Logistic_parameter_r'])
        self.set_Logistic_Coupling(Load_Dict['Logistic_Coupling'])
        self.set_Armillotta_Poisson_Betas(Load_Dict['Armillotta_Poisson_Betas'])
        self.set_Armillotta_Poisson_Lambda_0(Load_Dict['Armillotta_Poisson_Lambda_0'])
        self.set_Available_Inference_Methods(Load_Dict['Available_Inference_Methods'])
        self.set_font(Load_Dict['font'])
        self.set_B(Load_Dict['B'])
        self.set_TPR(Load_Dict['TPR'])
        self.set_FPR(Load_Dict['FPR'])
        self.set_AUC(Load_Dict['AUC'])
        self.set_Xshuffle(Load_Dict['Xshuffle'])
        self.set_IIkfold(Load_Dict['IIkfold'])
        self.set_II_InfCriterion(Load_Dict['II_InfCriterion'])
        self.set_num_processes(Load_Dict['num_processes'])
        self.set_parallel_shuffles(Load_Dict['parallel_shuffles'])
        self.set_parallel_nodes(Load_Dict['parallel_nodes'])
        self.set_KNN_dist_metric(Load_Dict['KNN_dist_metric'])
        self.set_conditional_returns_set(Load_Dict['conditional_returns_set'])

    def nchoosek(self, n, k):
        Range = range(1, n + 1)
        comb = list(itertools.combinations(Range, k))
        return np.array(comb)

    def PermutationMatrix(self):
        p = int(self.n)
        Zp = int(p * (p - 1) / 2)
        P = np.zeros((p, Zp))
        NK = self.nchoosek(p, 2)
        for i in range(p):
            Z = np.zeros(Zp)
            Wh = np.where(NK - 1 == i)
            Z[Wh[0]] = 1;
            P[i, :] = Z
        return P

    def sub2ind(self, Size, Arr):
        return np.ravel_multi_index(Arr, Size, order='F')

    def Logistic_Map(self, X, r):

        return r * X * (1 - X)

    def Gen_Logistic_Dynamics(self, r=3.99, sigma=0.1):
        """Network coupled logistic map, r is the logistic map parameter
           and sigma is the coupling strength between oscillators"""

        if self.NetworkAdjacency is None:
            raise ValueError("Missing adjacency matrix, please add this using set_NetworkAdjacency")
        T = self.T
        self.Logistic_parameter_r = r
        self.Logistic_Coupling = sigma
        A = self.NetworkAdjacency
        # Must adjust the adjacency matrix so that dynamics stay in [0,1]
        A = A / np.sum(A, axis=1)
        A = A.T
        A[np.isnan(A)] = 0
        A[np.isinf(A)] = 0
        # Since the row sums equal to 1 the Laplacian matrix is easy...
        L = np.eye(self.n) - A
        L = np.array(L)
        self.Logistic_Adjacency = A
        XY = np.zeros((T, self.n))
        XY[0, :] = np.random.rand(self.n)
        for i in range(1, T):
            XY[i, :] = self.Logistic_Map(XY[i - 1, :], r) - sigma * np.dot(L, self.Logistic_Map(XY[i - 1, :], r)).T

        self.XY = XY

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

    def Compute_CMI_Gaussian_Fast(self, X):
        """An implementation of the Gaussian conditional mutual information from
        the paper by Sun, Taylor and Bollt entitled:
            Causal network inference by optimal causation entropy"""
        # print(self.Xshuffle)
        # if self.Xshuffle:
        #    return self.Compute_CMI_Gaussian(X)
        # else:

        TupleX = X.shape
        TupleY = self.Y.shape
        # Fix this in a moment
        if not self.Xshuffle:
            self.XYInd = np.hstack((self.indX, self.indY))
        else:

            self.XYInd = np.hstack((self.shuffX, self.shuffY))

        if len(TupleX) == 1:
            SX = [[1]]

        else:
            if TupleX[1] == 1:
                SX = [[1]]
            else:
                if not self.Xshuffle:
                    SX = self.bigCorr[self.indX, :]
                    SX = SX[:, self.indX]
                else:
                    SX = self.shuffCorr[self.shuffX, :]
                    SX = SX[:, self.shuffX]

        if len(TupleY) == 1:
            SY = [[1]]
        else:
            if TupleY[1] == 1:
                SY = [[1]]
            else:
                if not self.Xshuffle:
                    SY = self.bigCorr[self.indY, :]
                    SY = SY[:, self.indY]
                else:
                    SY = self.shuffCorr[self.shuffY, :]
                    SY = SY[:, self.shuffY]

        SX = np.linalg.det(SX)
        SY = np.linalg.det(SY)
        if self.Z is None:
            if not self.Xshuffle:
                XY = self.bigCorr[self.XYInd, :]
            else:
                XY = self.shuffCorr[self.XYInd, :]
            XY = XY[:, self.XYInd]
            # print(XY)
            SXY = np.linalg.det(XY)

            return 0.5 * np.log((SX * SY) / SXY)

        else:
            if not self.Xshuffle:
                self.bigInd = np.hstack((self.indX, self.indY, self.indZ))
                self.XZInd = np.hstack((self.indX, self.indZ))
            else:
                # print("This is self.Z: ", self.Z)
                self.bigInd = np.hstack((self.shuffX, self.shuffY, self.shuffZ))
                self.XZInd = np.hstack((self.shuffX, self.shuffZ))
            TupleZ = self.Z.shape
            if len(TupleZ) == 1:
                SZ = [[1]]
            else:
                if TupleZ[1] == 1:
                    SZ = [[1]]
                else:
                    if not self.Xshuffle:
                        SZ = self.bigCorr[self.indZ, :]
                        SZ = SZ[:, self.indZ]
                    else:
                        SZ = self.shuffCorr[self.shuffZ, :]
                        SZ = SZ[:, self.shuffZ]

            SZ = np.linalg.det(SZ)
            if not self.Xshuffle:
                self.YZInd = np.hstack((self.indY, self.indZ))
                self.XYZInd = np.hstack((self.indX, self.indY, self.indZ))
                XZ = self.bigCorr[self.XZInd, :]
                XZ = XZ[:, self.XZInd]
                # print(XZ)
                YZ = self.bigCorr[self.YZInd, :]
                YZ = YZ[:, self.YZInd]
                XYZ = self.bigCorr[self.XYZInd, :]
                XYZ = XYZ[:, self.XYZInd]
            else:
                # print("This is self.Z: ", self.Z)
                self.YZInd = np.hstack((self.shuffY, self.shuffZ))
                self.XYZInd = np.hstack((self.shuffX, self.shuffY, self.shuffZ))
                XZ = self.shuffCorr[self.XZInd, :]
                XZ = XZ[:, self.XZInd]
                # print(XZ)
                YZ = self.shuffCorr[self.YZInd, :]
                YZ = YZ[:, self.YZInd]
                XYZ = self.shuffCorr[self.XYZInd, :]
                XYZ = XYZ[:, self.XYZInd]

            SXZ = np.linalg.det(XZ)
            SYZ = np.linalg.det(YZ)
            SXYZ = np.linalg.det(XYZ)

            return 0.5 * np.log((SXZ * SYZ) / (SZ * SXYZ))

    def Compute_CMI_Gaussian(self, X):
        """An implementation of the Gaussian conditional mutual information from
        the paper by Sun, Taylor and Bollt entitled:
            Causal network inference by optimal causation entropy"""

        TupleX = X.shape
        TupleY = self.Y.shape
        if len(TupleX) == 1:
            SX = [[1]]

        else:
            if TupleX[1] == 1:
                SX = [[1]]
            else:
                SX = np.corrcoef(X.T)

        if len(TupleY) == 1:
            SY = [[1]]
        else:
            if TupleY[1] == 1:
                SY = [[1]]
            else:
                SY = np.corrcoef(self.Y.T)

        if self.Z is None:
            SX = sp.linalg.det(SX)
            SY = sp.linalg.det(SY)
            # SXY = np.linalg.det(np.corrcoef(X.T,self.Y.T))
            SXY = sp.linalg.det(np.corrcoef(X.T, self.Y.T))
            return 0.5 * np.log((SX * SY) / SXY)


        else:
            TupleZ = self.Z.shape
            if len(TupleZ) == 1:
                SZ = [[1]]
            else:
                if TupleZ[1] == 1:
                    SZ = [[1]]
                else:
                    SZ = np.corrcoef(self.Z.T)

            # SZ = np.linalg.det(SZ)
            SZ = sp.linalg.det(SZ)
            # T1 = time.time()
            XZ = np.concatenate((X, self.Z), axis=1)
            YZ = np.concatenate((self.Y, self.Z), axis=1)
            XYZ = np.concatenate((X, self.Y, self.Z), axis=1)
            # T2 = time.time()-T1
            # print("This is concatenation time: ", T2)

            # SXZ = np.linalg.det(np.corrcoef(XZ.T))
            # SYZ = np.linalg.det(np.corrcoef(YZ.T))
            # SXYZ = np.linalg.det(np.corrcoef(XYZ.T))
            SXZ = sp.linalg.det(np.corrcoef(XZ.T))
            SYZ = sp.linalg.det(np.corrcoef(YZ.T))
            SXYZ = sp.linalg.det(np.corrcoef(XYZ.T))

            Value = 0.5 * np.log((SXZ * SYZ) / (SZ * SXYZ))

            return Value

    def Compute_CMI_Hawkes(self, X):
        Blah = 1

    def Compute_CMI_VonMises(self, X):
        Blah = 1

    def Compute_CMI_Laplace(self, X):
        Blah = 1

    def Compute_CMI_Histogram(self, X):
        Blah = 1

    def Compute_Entropy_KernelDensity(self, X):
        if self.KernelBandWidth is None:
            # This is approximately the optimal bandwidth according to Silverman

            BandWidth = 'silverman'
        else:
            BandWidth = self.KernelBandWidth
        kdeX = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(X)
        densityX = np.exp(kdeX.score_samples(X))
        Hx = -np.sum(np.log(densityX)) / len(densityX)
        return Hx

    def Compute_MI_KernelDensity(self, X):
        if self.KernelBandWidth is None:
            # This is approximately the optimal bandwidth according to Silverman

            BandWidth = 'silverman'
        else:
            BandWidth = self.KernelBandWidth

        XY = np.hstack((X, self.Y))
        # Set up the kde computation
        kdeX = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(X)
        kdeY = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(self.Y)
        kdeXY = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(XY)
        # Get the densities
        densityX = np.exp(kdeX.score_samples(X))
        densityY = np.exp(kdeY.score_samples(self.Y))
        densityXY = np.exp(kdeXY.score_samples(XY))
        # Compute the entropies
        Hx = -np.sum(np.log(densityX)) / len(densityX)
        Hy = -np.sum(np.log(densityY)) / len(densityY)
        Hxy = -np.sum(np.log(densityXY)) / len(densityXY)

        return Hx + Hy - Hxy

    def Compute_CMI_KernelDensity(self, X):

        if self.Z is None:
            I = self.Compute_MI_KernelDensity(X)
        else:
            if self.KernelBandWidth is None:
                # This is approximately the optimal bandwidth according to Silverman

                BandWidth = 'silverman'
            else:
                BandWidth = self.KernelBandWidth

            # Find all necessary joint variables
            # print(X.shape)
            # print(self.Z.shape)
            XZ = np.hstack((X, self.Z))
            YZ = np.hstack((self.Y, self.Z))
            XYZ = np.hstack((X, self.Y, self.Z))
            # Setup the kde computation
            kdeZ = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(self.Z)
            kdeXZ = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(XZ)
            kdeYZ = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(YZ)
            kdeXYZ = KernelDensity(bandwidth=BandWidth, kernel=self.KernelType).fit(XYZ)
            # Compute the densities
            densityZ = np.exp(kdeZ.score_samples(self.Z))
            densityXZ = np.exp(kdeXZ.score_samples(XZ))
            densityYZ = np.exp(kdeYZ.score_samples(YZ))
            densityXYZ = np.exp(kdeXYZ.score_samples(XYZ))
            # Compute the entropies
            Hz = -np.sum(np.log(densityZ)) / len(densityZ)
            Hxz = -np.sum(np.log(densityXZ)) / len(densityXZ)
            Hyz = -np.sum(np.log(densityYZ)) / len(densityYZ)
            Hxyz = -np.sum(np.log(densityXYZ)) / len(densityXYZ)
            I = Hxz + Hyz - Hxyz - Hz

        return I

    def MutualInfo_KNN(self, X, Y):
        # construct the joint space
        n = X.shape[0]
        JS = np.column_stack((X, Y))
        # n = JS.shape[0]
        # Find the K^th smallest distance in the joint space
        D = np.sort(cdist(JS, JS, metric=self.KNN_dist_metric, p=self.KNN_K + 1), axis=1)[:, self.KNN_K]
        epsilon = D

        # Count neighbors within epsilon in marginal spaces
        Dx = cdist(X, X, metric=self.KNN_dist_metric)
        nx = np.sum(Dx < epsilon[:, None], axis=1) - 1
        Dy = cdist(Y, Y, metric=self.KNN_dist_metric)
        ny = np.sum(Dy < epsilon[:, None], axis=1) - 1

        # KSG Estimation formula
        I1a = Digamma(self.KNN_K)
        I1b = Digamma(n)
        I1 = I1a + I1b
        I2 = - np.mean(Digamma(nx + 1) + Digamma(ny + 1))
        return I1 + I2

    def Compute_CMI_KNN_v2(self, X):
        if self.Z is None:
            return self.MutualInfo_KNN(X, self.Y)
        # Construct the joint space
        JS = np.column_stack((X, self.Y, self.Z))
        # Find the K^th smallest distance in the joint space
        D = np.sort(cdist(JS, JS, metric=self.KNN_dist_metric, p=self.KNN_K + 1), axis=1)[:, self.KNN_K]
        epsilon = D
        # Count neighbors within epsilon in marginal spaces
        Dxz = cdist(np.column_stack((X, self.Z)), np.column_stack((X, self.Z)), metric=self.KNN_dist_metric)
        nxz = np.sum(Dxz < epsilon[:, None], axis=1) - 1
        Dyz = cdist(np.column_stack((self.Y, self.Z)), np.column_stack((self.Y, self.Z)), metric=self.KNN_dist_metric)
        nyz = np.sum(Dyz < epsilon[:, None], axis=1) - 1
        Dz = cdist(self.Z, self.Z, metric=self.KNN_dist_metric)
        nz = np.sum(Dz < epsilon[:, None], axis=1) - 1
        # VP Estimation formula
        I = Digamma(self.KNN_K) - np.mean(Digamma(nxz + 1) + Digamma(nyz + 1) - Digamma(nz + 1))
        return I

    def Compute_CMI_KNN(self, X):
        """KNN version (via the paper by Kraskov, Stogbauer and Grassberger called
        'Estimating mutual information'
        of Conditional mutual information for Causation entropy...
        """
        if self.Z is None:
            return self.MutualInfo_KNN(X, self.Y)  # np.max([self.MutualInfo_KNN(X,self.Y),0])
        else:
            XcatY = np.concatenate((X, self.Y), axis=1)
            MIXYZ = self.MutualInfo_KNN(XcatY, self.Z)
            MIXY = self.MutualInfo_KNN(X, self.Y)

            return MIXY - MIXYZ  # np.max([MIXY-MIXYZ,0])

    def l2dist(self, a, b):
        return np.linalg.norm(a - b)

    def hyperellipsoid_check(self, svd_Yi, Z_i):
        # Check if Z_i lies within the hyperellipsoid defined by svd_Yi
        U, S, Vt = svd_Yi
        transformed = np.dot(Z_i, Vt.T) / S
        return np.sum(transformed ** 2) <= 1

    def Entropy_GKNN(self, X, k, Xdist):
        """A method for estimating entropy (which will be used for estimating mutual
        informations needed in Causation entropy) which comes from the paper
        'Geometric k-nearest neighbor estimation of entropy and mutual information'
        by Lord, Sun and Bollt
        """
        k = self.KNN_K
        N, d = X.shape
        Xknn = np.zeros((N, k), dtype=int)

        for i in range(N):
            Xknn[i, :] = np.argsort(Xdist[i, :])[1:k + 1]
        H_X = np.log(N) + np.log(np.pi ** (d / 2) / Gamma(1 + d / 2))
        H_X += d / N * np.sum([np.log(self.l2dist(X[i, :], X[Xknn[i, k - 1], :])) for i in range(N)])
        H_X += 1 / N * np.sum(
            [-np.log(max(1, np.sum([self.hyperellipsoid_check(np.linalg.svd(Y_i), Z_i[j, :]) for j in range(k)])))
             + np.sum([np.log(sing_Yi[l] / sing_Yi[0]) for l in range(d)])
             for i in range(N)
             for Y_i in [X[np.append([i], Xknn[i, :]), :] - np.mean(X[np.append([i], Xknn[i, :]), :], axis=0)]
             for svd_Yi in [np.linalg.svd(Y_i)]
             for sing_Yi in [svd_Yi[1]]
             for Z_i in [X[Xknn[i, :], :] - X[i, :]]
             ])
        return H_X

    def MI_GKNN(self, X, Y):
        """A method for estimating Mutual information (which will be
        needed in Causation entropy) which comes from the paper
        'Geometric k-nearest neighbor estimation of entropy and mutual information'
        by Lord, Sun and Bollt
        """
        Xdist = cdist(X, X, metric='euclidean')
        Ydist = cdist(Y, Y, metric='euclidean')
        XYdist = cdist(np.hstack((X, Y)), np.hstack((X, Y)), metric='euclidean')

        HX = self.Entropy_GKNN(X, self.KNN_K, Xdist)
        HY = self.Entropy_GKNN(Y, self.KNN_K, Ydist)
        HXY = self.Entropy_GKNN(np.hstack((X, Y)), self.KNN_K, XYdist)

        return HX + HY - HXY

    def Compute_CMI_Geometric_KNN(self, X):
        """A method for estimating CMI (which will be
        needed in Causation entropy) which comes from the paper
        'Geometric k-nearest neighbor estimation of entropy and mutual information'
        by Lord, Sun and Bollt
        """

        if self.Z is None:
            return self.MI_GKNN(X, self.Y)
        YZdist = cdist(np.hstack((self.Y, self.Z)), np.hstack((self.Y, self.Z)), metric='euclidean')
        XZdist = cdist(np.hstack((X, self.Z)), np.hstack((X, self.Z)), metric='euclidean')
        XYZdist = cdist(np.hstack((X, self.Y, self.Z)), np.hstack((X, self.Y, self.Z)), metric='euclidean')
        Zdist = cdist(self.Z, self.Z, metric='euclidean')
        HZ = self.Entropy_GKNN(self.Z, self.KNN_K, Zdist)
        HXZ = self.Entropy_GKNN(np.hstack((X, self.Z)), self.KNN_K, XZdist)
        HYZ = self.Entropy_GKNN(np.hstack((self.Y, self.Z)), self.KNN_K, YZdist)
        HXYZ = self.Entropy_GKNN(np.hstack((X, self.Y, self.Z)), self.KNN_K, XYZdist)
        return HXZ + HYZ - HXYZ - HZ

    def Compute_CMI_NegativeBinomial(self, X):
        Blah = 1

    def PoissEnt(self, Lambdas):
        """A method for esitmating the Poisson entropy that will be needed for
        computation of conditional mutual informations. For details see the paper
        by Fish and Bollt
        'Interaction networks from discrete event data by Poisson multivariate
        mutual information estimation and information flow with applications
        from gene expression data'
        """
        Lambdas = np.abs(Lambdas)
        First = np.exp(-Lambdas)
        Psum = First
        P = [np.matrix(First)]
        counter = 0
        small = 1
        i = 1
        while np.max(1 - Psum) > 1e-16 and small > 1e-75:
            counter = counter + 1
            prob = SStats.poisson.pmf(i, Lambdas)
            Psum = Psum + prob
            P.append(np.matrix(prob))
            if i >= np.max(Lambdas):
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

    def PoissonJointEntropy_New(self, Cov):
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
        Ent1 = np.sum(self.PoissEnt(U))
        Ent2 = np.sum(T)
        return Ent1 + Ent2

    def Compute_CMI_Poisson(self, X):
        """Estimate of conditional mutual information from Poisson marginals,
        from the paper by Fish, Sun and Bollt entitled:
            Interaction Networks from Discrete Event Data by Poisson Multivariate
            Mutual Information Estimation and Information Flow with Applications
            from Gene Expression Data"""

        if self.Z is None:
            SXY = np.corrcoef(X.T, self.Y.T)
            l_est = SXY - np.diag(np.diag(SXY))
            np.fill_diagonal(SXY, np.diagonal(SXY) - np.sum(l_est, axis=0))
            Dcov = np.diag(SXY) + np.sum(l_est, axis=0)
            TF = self.PoissonJointEntropy_New(SXY)
            FT = np.sum(self.PoissEnt(Dcov))
            # print(FT-TF)

            return FT - TF
        else:
            SzX = X.shape[1]
            SzY = self.Y.shape[1]
            SzZ = self.Z.shape[1]
            indX = np.matrix(np.arange(SzX))
            indY = np.matrix(np.arange(SzY) + SzX)
            indZ = np.matrix(np.arange(SzZ) + SzX + SzY)
            # print(indX,indY,indZ)
            XYZ = np.concatenate((X, self.Y, self.Z), axis=1)
            SXYZ = np.corrcoef(XYZ.T)
            SS = SXYZ
            Sa = SXYZ - np.diag(np.diag(SXYZ))
            np.fill_diagonal(SS, np.diagonal(SS) - Sa)
            SS[0:SzX, 0:SzX] = SS[0:SzX, 0:SzX] + SXYZ[0:SzX, SzX:SzX + SzY]
            SS[SzX:SzX + SzY, SzX:SzX + SzY] = SS[SzX:SzX + SzY, SzX:SzX + SzY] + SXYZ[SzX:SzX + SzY, 0:SzX]
            S_est1 = SS[np.concatenate((indY.T, indZ.T), axis=0), np.concatenate((indY.T, indZ.T), axis=0)]
            S_est2 = SS[np.concatenate((indX.T, indZ.T), axis=0), np.concatenate((indX.T, indZ.T), axis=0)]
            HYZ = self.PoissonJointEntropy_New(S_est1)
            SindZ = SS[indZ, indZ]
            HZ = self.PoissonJointEntropy_New(SindZ)
            HXYZ = self.PoissonJointEntropy_New(SXYZ - np.diag(Sa))
            HXZ = self.PoissonJointEntropy_New(S_est2)
            H_YZ = HYZ - HZ
            H_XYZ = HXYZ - HXZ
            return H_XYZ - H_YZ

    def Compute_CMI(self, X):
        """Compute the CMI based upon whichever method"""
        if self.InferenceMethod_oCSE == 'Gaussian':
            return self.Compute_CMI_Gaussian_Fast(X)

        elif self.InferenceMethod_oCSE == 'KernelDensity':
            return self.Compute_CMI_KernelDensity(X)

        elif self.InferenceMethod_oCSE == 'KNN':
            return self.Compute_CMI_KNN(X)

        elif self.InferenceMethod_oCSE == 'Poisson':
            return self.Compute_CMI_Poisson(X)

        elif self.InferenceMethod_oCSE == 'GeometricKNN':
            return self.Compute_CMI_Geometric_KNN(X)

        elif self.InferenceMethod_oCSE == 'Histogram':
            return self.Compute_CMI_Histogram(X)

        elif self.InferenceMethod_oCSE == 'Laplace':
            return self.Compute_CMI_Laplace(X)

        elif self.InferenceMethod_oCSE == 'NegativeBinomial':
            return self.Compute_CMI_NegativeBinomial(X)

        elif self.InferenceMethod_oCSE == 'VonMises':
            return self.Compute_CMI_VonMises(X)

        elif self.InferenceMethod_oCSE == 'Hawkes':
            return self.Compute_CMI_Hawkes(X)

        else:
            raise ValueError("Sorry oCSE inference method: ", self.InferenceMethod_oCSE, "is not available.")

    def Standard_Forward_oCSE(self):
        NotStop = True
        n = self.X.shape[1]
        TestVariables = np.arange(n)
        S = []
        loopnum = 0
        while NotStop:
            loopnum = loopnum + 1
            # print(loopnum)
            SetCheck = np.setdiff1d(TestVariables, S)
            m = len(SetCheck)
            if m == 0:
                NotStop = False
                break
            Ents = np.zeros(m)
            for i in range(m):
                X = self.X[:, [SetCheck[i]]]
                self.indX = np.array(SetCheck[i])

                Ents[i] = self.Compute_CMI(X)

            Argmax = Ents.argmax()
            X = self.X[:, [SetCheck[Argmax]]]
            if not self.parallel_shuffles:
                Dict = self.Standard_Shuffle_Test_oCSE(X, Ents[Argmax], self.Forward_oCSE_alpha)
            else:
                Dict = self.Parallel_Shuffle_Test_oCSE(X, Ents[Argmax], self.Forward_oCSE_alpha)

            if Dict['Pass']:
                S.append(SetCheck[Argmax])
                self.indZ = np.array(S)
                if len(S) == 1:
                    self.Z = self.X[:, [SetCheck[Argmax]]]
                else:
                    self.Z = np.concatenate((self.Z, self.X[:, [SetCheck[Argmax]]]), axis=1)
            else:
                NotStop = False

        return S

    def Alternative_Forward_oCSE(self):
        NotStop = True

        n = self.X.shape[1]
        TestVariables = np.arange(n)
        S = []
        loopnum = 0
        while NotStop:
            loopnum = loopnum + 1
            # print(loopnum)
            SetCheck = np.setdiff1d(TestVariables, S)
            m = len(SetCheck)
            if m == 0:
                NotStop = False
                break
            Ents = np.zeros(m)
            Passes = np.zeros(m, dtype=bool)
            for i in range(m):
                X = self.X[:, [SetCheck[i]]]

                Ents[i] = self.Compute_CMI(X)
                if not self.parallel_shuffles:
                    Dict = self.Standard_Shuffle_Test_oCSE(X, Ents[i], self.Forward_oCSE_alpha)
                else:
                    Dict = self.Parallel_Shuffle_Test_oCSE(X, Ents[i], self.Forward_oCSE_alpha)
                Passes[i] = Dict['Pass']

            NewEnts = Ents[Passes]
            if len(NewEnts) > 0:
                Wh = np.where(Passes)
                Wh = Wh[0]
                Argmax = Wh[NewEnts.argmax()]
                S.append(SetCheck[Argmax])
                if len(S) == 1:
                    self.Z = self.X[:, [SetCheck[Argmax]]]
                else:
                    self.Z = np.concatenate((self.Z, self.X[:, [SetCheck[Argmax]]]), axis=1)
            else:
                NotStop = False

        return S

    def Standard_Backward_oCSE(self, S):
        # Reset the conditioning matrix
        self.Z = None
        RP = np.random.permutation(len(S))
        Sn = copy.deepcopy(S)
        for i in range(len(S)):
            X = self.X[:, [S[RP[i]]]]

            Inds = np.setdiff1d(Sn, S[RP[i]])
            # print(Inds)
            Z = self.X[:, Inds]
            self.indZ = Inds
            self.indX = S[RP[i]]
            # if len(Inds)<=1:
            #    Z = self.X[:,[S[Inds]]]
            # else:
            #    Z = self.X[:,S[Inds]]
            self.Z = Z
            if Z.shape[1] == 0:
                self.Z = None
            Ent = self.Compute_CMI(X)
            # T1 = time.time()
            if not self.parallel_shuffles:
                Dict = self.Standard_Shuffle_Test_oCSE(X, Ent, self.Backward_oCSE_alpha)
            else:
                Dict = self.Parallel_Shuffle_Test_oCSE(X, Ent, self.Backward_oCSE_alpha)
            # T2 = time.time()-T1
            # print("This is how much time in shuffle test (backward): ",T2)
            if not Dict['Pass']:
                Sn = np.setdiff1d(Sn, S[RP[i]])

        return Sn

    def Parallel_input_function(self, X, T, TupleX):
        """For parallelization (simple for loop stuff)"""
        RP = np.random.permutation(T)
        if len(TupleX) > 1:
            Xshuff = X[RP, :]
        else:
            Xshuff = X[RP]
        # T1 = time.perf_counter_ns()
        if self.Z is not None:
            Size = np.sum([Xshuff.shape[1], self.Y.shape[1], self.Z.shape[1]])
            Cat = np.concatenate((Xshuff, self.Y, self.Z), axis=1)
            Arr = np.arange(Size)
            self.shuffX = Arr[0:Xshuff.shape[1]]
            self.shuffY = Arr[self.shuffX[-1] + 1:Xshuff.shape[1] + self.Y.shape[1]]
            self.shuffZ = Arr[self.shuffY[-1] + 1:self.shuffY[-1] + Xshuff.shape[1] + self.Z.shape[1]]
            # print("This is shuffZ: ", self.shuffZ)
        else:
            Size = np.sum([Xshuff.shape[1], self.Y.shape[1]])
            Cat = np.concatenate((Xshuff, self.Y), axis=1)
            Arr = np.arange(Size)
            self.shuffX = Arr[0:Xshuff.shape[1]]
            self.shuffY = Arr[self.shuffX[-1] + 1:Xshuff.shape[1] + self.Y.shape[1]]

        self.shuffCorr = np.corrcoef(Cat.T)

        return self.Compute_CMI(Xshuff)

    def Parallel_Shuffle_Test_oCSE(self, X, Ent, alpha):
        self.Xshuffle = True
        # print("This is Y shape: ", self.Y.shape)
        ns = self.Num_Shuffles_oCSE
        T = X.shape[0]
        # Ents = np.zeros(ns)
        TupleX = X.shape
        # print("This is X shape: ", TupleX)
        Ents = Parallel(n_jobs=self.num_processes)(
            delayed(self.Parallel_input_function)(X, T, TupleX) for i in range(ns))
        Ents = np.array(Ents)
        Prctile = int(100 * np.floor(ns * (1 - alpha)) / ns)
        self.Prctile = Prctile
        # print(Ents)
        # print(Ents[Ents>=np.percentile(Ents,Prctile)])
        try:
            Threshold = np.min(Ents[Ents >= np.percentile(Ents, Prctile)])
        except ValueError:
            Threshold = 0
        Dict = {'Threshold': Threshold}
        Dict['Value'] = Ent
        Dict['Pass'] = False
        if Ent >= Threshold:
            Dict['Pass'] = True

        self.Dict = Dict
        self.Xshuffle = False

        return Dict

    def Standard_Shuffle_Test_oCSE(self, X, Ent, alpha):
        """Implementation of the shuffle test (or permutation test) for
        oCSE. See the paper by Sun, Taylor and Bollt entitled:
            Causal network inference by optimal causation entropy

            for details.
            """
        self.Xshuffle = True
        # print("This is Y shape: ", self.Y.shape)
        ns = self.Num_Shuffles_oCSE
        T = X.shape[0]
        Ents = np.zeros(ns)
        TupleX = X.shape
        # print("This is X shape: ", TupleX)
        for i in range(ns):
            RP = np.random.permutation(T)
            if len(TupleX) > 1:
                Xshuff = X[RP, :]
            else:
                Xshuff = X[RP]
            # T1 = time.perf_counter_ns()
            if self.Z is not None:
                Size = np.sum([Xshuff.shape[1], self.Y.shape[1], self.Z.shape[1]])
                Cat = np.concatenate((Xshuff, self.Y, self.Z), axis=1)
                Arr = np.arange(Size)
                self.shuffX = Arr[0:Xshuff.shape[1]]
                self.shuffY = Arr[self.shuffX[-1] + 1:Xshuff.shape[1] + self.Y.shape[1]]
                self.shuffZ = Arr[self.shuffY[-1] + 1:self.shuffY[-1] + Xshuff.shape[1] + self.Z.shape[1]]
                # print("This is shuffZ: ", self.shuffZ)
            else:
                Size = np.sum([Xshuff.shape[1], self.Y.shape[1]])
                Cat = np.concatenate((Xshuff, self.Y), axis=1)
                Arr = np.arange(Size)
                self.shuffX = Arr[0:Xshuff.shape[1]]
                self.shuffY = Arr[self.shuffX[-1] + 1:Xshuff.shape[1] + self.Y.shape[1]]

            self.shuffCorr = np.corrcoef(Cat.T)

            Ents[i] = self.Compute_CMI(Xshuff)
            # print((time.perf_counter_ns()-T1)*10**-9)

        Prctile = int(100 * np.floor(ns * (1 - alpha)) / ns)
        self.Prctile = Prctile
        # print(Ents)
        # print(Ents[Ents>=np.percentile(Ents,Prctile)])
        try:
            Threshold = np.min(Ents[Ents >= np.percentile(Ents, Prctile)])
        except ValueError:
            Threshold = 0
        Dict = {'Threshold': Threshold}
        Dict['Value'] = Ent
        Dict['Pass'] = False
        if Ent >= Threshold:
            Dict['Pass'] = True

        self.Dict = Dict
        self.Xshuffle = False

        return Dict

    def Standard_oCSE(self):

        """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""
        if self.X is None:
            raise ValueError("Missing the potential predictors please add this using set_X")

        if self.Y is None:
            raise ValueError("Missing the target(s) please add using set_Y")

        # Ensure initially Z is None
        self.Z = None

        # Set this to avoid potential issues
        self.Correlation_XY = None

        # Find the initial set of potential predictors

        S = self.Standard_Forward_oCSE()
        # print("This is S: ", S)

        self.Sinit = S

        # The final set of predictors after removing spurious edges

        S = self.Standard_Backward_oCSE(S)

        self.Sfinal = S
        # print(S)

        # Reset conditioning set in case other methods need it
        self.Z = None

        return S

    def Alternative_oCSE(self):

        """Run the standard version of the oCSE algorithm. Note defaults to the
           KernelDensity plugin estimator if the method is not specified"""
        if self.X is None:
            raise ValueError("Missing the potential predictors please add this using set_X")

        if self.Y is None:
            raise ValueError("Missing the target(s) please add using set_Y")

        # Ensure initially Z is None
        self.Z = None

        # Set this to avoid potential issues
        self.Correlation_XY = None

        # Find the initial set of potential predictors
        S = self.Alternative_Forward_oCSE()
        self.Sinit = S

        # The final set of predictors after removing spurious edges
        S = self.Standard_Backward_oCSE(S)

        self.Sfinal = S

        # Reset conditioning set in case other methods need it
        self.Z = None

        return S

    def par_nodes_oCSE(self, XY_1, XY_2, i):
        self.Y = XY_2[:, [i]]
        self.indY = np.array([self.n])
        self.X = XY_1
        XY = np.hstack((self.X, self.Y))
        self.bigCorr = np.corrcoef(XY.T)
        self.indZ = None
        S = self.Standard_oCSE()

        return S

    def par_nodes_alternative_oCSE(self, XY_1, XY_2, i):
        self.Y = XY_2[:, [i]]
        self.X = XY_1
        S = self.Alternative_oCSE()
        return S

    def par_nodes_iil(self, XY_1, XY_2, i):
        self.NodeNum = i
        self.Y = XY_2[:, [i]]
        self.X = XY_1
        self.indY = np.array([self.n])
        XY = np.hstack((self.X, self.Y))
        self.bigCorr = np.corrcoef(XY.T)
        S = self.II_Lasso()
        return S

    def par_nodes_lasso(self, XY_1, XY_2, i):
        self.Y = XY_2[:, [i]]
        self.X = XY_1
        S = self.Lasso()
        return S

    def remove_linearly_dependent_variables(self, matrix):
        q, r = np.linalg.qr(matrix)
        independent_cols = np.where(np.abs(np.diag(r)) > 1e-11)[0]  # Use a tolerance
        return independent_cols

    def conditional_returns(self, remove_dependence=False):
        if self.NetworkAdjacency is None:
            raise ValueError("Adjacency matrix must be added to run conditional_returns. Use set_NetworkAdjacency")

        A = self.NetworkAdjacency
        XY = self.XY
        XY_1 = XY[0:self.T - self.Tau, :]
        XY_2 = XY[self.Tau:, :]

        Conditionals = {}
        Conditionals['Order'] = '(i_(t+tau),j_t)'
        for i in range(self.n):
            print("Estimating conditionals for edges in node number: ", i)
            self.Y = XY_2[:, [i]]
            self.indY = np.array([self.n])
            self.X = XY_1
            XY = np.hstack((self.X, self.Y))
            self.bigCorr = np.corrcoef(XY.T)

            for j in range(self.n):
                if self.conditional_returns_set == 'existing_edges':
                    if not remove_dependence:
                        Set = np.where(A[i, :] != 0)
                        self.indZ = Set[0]
                    else:
                        Set = np.where(A[i, :] != 0)
                        self.indZ = Set[0]
                        # check dependence
                        Zcheck = XY[:, self.indZ]
                        cor = np.corrcoef(Zcheck.T)
                        Zcheck = None
                        c = np.linalg.det(cor)
                        if c == 0:
                            indepSet = self.remove_linearly_dependent_variables(cor)
                            self.indZ = self.indZ[indepSet]
                            cor = None
                        else:
                            cor = None


                elif self.conditional_returns_set == 'all_but_one':
                    Set = np.setdiff1d(np.arange(self.n), np.array([j]))
                    self.indZ = Set

                else:
                    raise ValueError(
                        "conditional_returns_set must be one of the following: 'existing_edges', 'all_but_one'")

                self.Z = XY[:, self.indZ]
                X = XY[:, [j]]
                self.indX = np.array([j])
                Ent = self.Compute_CMI(X)
                InnerCond = {}
                InnerCond['CondSet'] = self.indZ
                InnerCond['CauseEnt'] = Ent
                Conditionals[(i, j)] = InnerCond

        return Conditionals

    def Estimate_Network(self):
        """A method for estimating the full network structure from data. """
        # Initialize...
        self.Y = None
        self.Z = None
        self.X = None
        Method = self.Overall_Inference_Method
        MethodList = self.Available_Inference_Methods
        if Method not in MethodList:
            raise ValueError("Sorry the Method: ", Method,
                             " is not currently implemented, the only available methods are: ",
                             self.Available_Inference_Methods)

        if Method == 'Standard_oCSE':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            if not self.parallel_nodes:
                for i in range(self.n):
                    print("Estimating edges for node number: ", i)
                    self.Y = XY_2[:, [i]]
                    self.indY = np.array([self.n])
                    self.X = XY_1
                    XY = np.hstack((self.X, self.Y))
                    self.bigCorr = np.corrcoef(XY.T)
                    self.indZ = None
                    S = self.Standard_oCSE()
                    B[i, S] = 1
            else:
                print("Estimating in parallel with ", self.num_processes, " processes.")
                results = Parallel(n_jobs=self.num_processes, verbose=11)(
                    delayed(self.par_nodes_oCSE)(XY_1, XY_2, i) for i in range(self.n))
                for i in range(len(results)):
                    B[i, results[i]] = 1

        elif Method == 'Alternative_oCSE':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            if not self.parallel_nodes:
                for i in range(self.n):
                    print("Estimating edges for node number: ", i)
                    self.Y = XY_2[:, [i]]
                    self.X = XY_1
                    S = self.Alternative_oCSE()
                    B[i, S] = 1
            else:
                print("Estimating in parallel with ", self.num_processes, " processes.")
                results = Parallel(n_jobs=self.num_processes, verbose=11)(
                    delayed(self.par_nodes_alternative_oCSE)(XY_1, XY_2, i) for i in range(self.n))
                for i in range(len(results)):
                    B[i, results[i]] = 1

        elif Method == 'InformationInformed_LASSO':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            if not self.parallel_nodes:
                for i in range(self.n):
                    print("Estimating edges for node number: ", i)
                    self.NodeNum = i
                    self.Y = XY_2[:, [i]]
                    self.X = XY_1
                    self.indY = np.array([self.n])
                    XY = np.hstack((self.X, self.Y))
                    self.bigCorr = np.corrcoef(XY.T)
                    S = self.II_Lasso()
                    B[i, S] = 1
            else:
                print("Estimating in parallel with ", self.num_processes, " processes.")
                results = Parallel(n_jobs=self.num_processes, verbose=11)(
                    delayed(self.par_nodes_iil)(XY_1, XY_2, i) for i in range(self.n))
                for i in range(len(results)):
                    B[i, results[i]] = 1

        elif Method == 'LASSO':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            if not self.parallel_nodes:
                for i in range(self.n):
                    print("Estimating edges for node number: ", i)
                    self.Y = XY_2[:, [i]]
                    self.X = XY_1
                    S = self.Lasso()
                    B[i, S] = 1
            else:
                print("Estimating in parallel with ", self.num_processes, " processes.")
                results = Parallel(n_jobs=self.num_processes, verbose=11)(
                    delayed(self.par_nodes_lasso)(XY_1, XY_2, i) for i in range(self.n))
                for i in range(len(results)):
                    B[i, results[i]] = 1
        else:
            # Do nothing
            B = []

        self.B = B
        return B

    def Lasso(self):
        n = self.n
        if self.X.shape[0] > n + 1:
            Lass = LassoLarsIC(criterion=self.II_InfCriterion, max_iter=self.max_num_lambdas).fit(self.X,
                                                                                                  self.Y.flatten())
        else:
            if not self.parallel_nodes:
                Lass = LassoCV(cv=self.IIkfold, n_alphas=self.max_num_lambdas).fit(self.X, self.Y.flatten())
            else:
                Lass = LassoCV(cv=self.IIkfold, n_alphas=self.max_num_lambdas, n_jobs=self.num_processes).fit(self.X,
                                                                                                              self.Y.flatten())
            # est_var = self.estimate_noise_variance(Type='Lasso')
            # Lass = LassoLarsIC(criterion=self.II_InfCriterion,noise_variance=est_var[0]).fit(self.X, self.Y.flatten())
        S = np.where(Lass.coef_ != 0)
        return S

    def II_Lasso(self):
        n = self.n
        Set = np.arange(n)
        self.CMI_matrix = np.zeros(n)
        for i in range(n):
            self.indX = np.array([i])
            self.indZ = np.setdiff1d(Set, self.indX)
            if self.X.shape[0] > n:

                self.Z = self.X[:, self.indZ]
            else:
                # rp =np.random.permutation(self.indZ)
                # self.indZ = rp[0:self.X.shape[0]-2]
                # self.Z = self.X[:,self.indZ]
                # Xnew = self.X[:,[i]]
                # self.indX = np.array([0])
                # XY = np.hstack((Xnew,self.Z,self.Y))
                # self.bigCorr = np.corrcoef(XY.T)
                # shp = XY.shape[1]
                # rng = np.arange(shp)
                # self.indY = np.array([shp-1])
                # self.indZ = rng[1:-1]
                self.indZ = None
                self.Z = None

            CMI = self.Compute_CMI_Gaussian_Fast(self.X)
            if np.isnan(CMI) or np.isinf(CMI):
                CMI = 1e-100
            self.CMI_matrix[i] = CMI

        if self.X.shape[0] > n + 1:
            LLIC = LassoLarsIC(criterion=self.II_InfCriterion, max_iter=self.max_num_lambdas).fit(
                self.X * self.CMI_matrix, self.Y.flatten())
        else:
            if not self.parallel_nodes:
                LLIC = LassoCV(cv=self.IIkfold, n_alphas=self.max_num_lambdas).fit(self.X * self.CMI_matrix,
                                                                                   self.Y.flatten())
            else:
                LLIC = LassoCV(cv=self.IIkfold, n_alphas=self.max_num_lambdas, n_jobs=self.num_processes).fit(
                    self.X * self.CMI_matrix, self.Y.flatten())
                # est_var = self.estimate_noise_variance()
            # LLIC = LassoLarsIC(criterion=self.II_InfCriterion,noise_variance=est_var[0]).fit(self.X*self.CMI_matrix, self.Y.flatten())
        S = np.where(LLIC.coef_ != 0)
        # print(self.CMI_matrix)
        return S

    def estimate_noise_variance(self, Type='LLIC'):
        ols_model = LinearRegression()
        if Type == 'LLIC':
            ols_model.fit(self.X * self.CMI_matrix, self.Y)
            y_pred = ols_model.predict(self.X * self.CMI_matrix)
        elif Type == 'Lasso':
            ols_model.fit(self.X, self.Y)
            y_pred = ols_model.predict(self.X)

        return np.sum((self.Y - y_pred) ** 2) / (
            np.abs(self.X.shape[0] - self.X.shape[1] - ols_model.intercept_)
        )

    def return_CMI_Mat(self):
        return self.CMI_matrix

    def Compute_TPR_FPR(self):
        if self.NetworkAdjacency is None:
            raise ValueError(
                "Sorry cannot compute error if NetworkAdjacency is not specified, please set using set_NetworkAdjacency")

        if self.B is None:
            raise ValueError(
                "Sorry cannot compute error if B (estimated network) is not specified, please set using set_B")

        A = self.NetworkAdjacency
        B = self.B
        n = self.n
        TPR = 1 - (np.sum((A - B) > 0)) / np.sum(A)
        FPR = np.sum((A - B) < 0) / (n * (n - 1) - np.sum(A))
        self.TPR = TPR
        self.FPR = FPR
        return (TPR, FPR)

    def Compute_AUC(self, TPRs, FPRs):
        """Estimate the area under the curve (AUC) using the trapezoidal rule
           for integration..."""

        AUC = np.trapz(TPRs, FPRs)
        self.AUC = AUC
        return AUC

    def Plot_ROC(self, TPRs, FPRs):
        if self.font is None:
            font = {'family': 'normal',
                    'weight': 'bold',
                    'size': 22}
        else:
            font = self.font
        rc('font', **font)

        plt.plot(FPRs, TPRs)
        plt.xlabel('False Positive Rate (FPR)')
        plt.ylabel('True Positive Rate (TPR)')
        plt.title('ROC')
        plt.xlim(0, 1)
        plt.ylim(0, 1)

        AUC = self.Compute_AUC(TPRs, FPRs)
        plt.text(0.4, 0.1, 'AUC = ' + format(AUC, '.4f'))
