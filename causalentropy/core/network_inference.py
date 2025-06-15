from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import LassoCV
import numpy as np
import itertools
import copy


class NetworkInference:
    """A class for NetworkInference. Version number 0.2"""

    def sub2ind(self, Size, Arr):
        return np.ravel_multi_index(Arr, Size, order='F')

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

    def Estimate_Network(self, X, Y, Z):
        # Main method
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

        elif Method == 'Alternative_oCSE':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            for i in range(self.n):
                print("Estimating edges for node number: ", i)
                self.Y = XY_2[:, [i]]
                self.X = XY_1
                S = self.Alternative_oCSE()
                B[i, S] = 1

        elif Method == 'InformationInformed_LASSO':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
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

        elif Method == 'LASSO':
            XY = self.XY
            XY_1 = XY[0:self.T - self.Tau, :]
            XY_2 = XY[self.Tau:, :]
            B = np.zeros((self.n, self.n))
            for i in range(self.n):
                print("Estimating edges for node number: ", i)
                self.Y = XY_2[:, [i]]
                self.X = XY_1
                S = self.Lasso()
                B[i, S] = 1
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