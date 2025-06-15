import numpy as np

def _shuffle_test(X, Y, Ent, Z=None, alpha=0.05, n_bootstraps=500):
    """Implementation of the shuffle test (or permutation test) for
    oCSE. See the paper by Sun, Taylor and Bollt entitled:
        Causal network inference by optimal causation entropy

        for details.
        """


    T = X.shape[0]
    Ents = np.zeros(n_bootstraps)
    TupleX = X.shape

    for i in range(n_bootstraps):
        RP = np.random.permutation(T)
        if len(TupleX) > 1:
            Xshuff = X[RP, :]
        else:
            Xshuff = X[RP]
        # T1 = time.perf_counter_ns()
        if Z is not None:
            Size = np.sum([Xshuff.shape[1], Y.shape[1], Z.shape[1]])
            Cat = np.concatenate((Xshuff, Y, Z), axis=1)
            Arr = np.arange(Size)
            shuffX = Arr[0:Xshuff.shape[1]]
            shuffY = Arr[shuffX[-1] + 1:Xshuff.shape[1] + Y.shape[1]]
            shuffZ = Arr[shuffY[-1] + 1:shuffY[-1] + Xshuff.shape[1] + Z.shape[1]]
            # print("This is shuffZ: ", self.shuffZ)
        else:
            Size = np.sum([Xshuff.shape[1], Y.shape[1]])
            Cat = np.concatenate((Xshuff, Y), axis=1)
            Arr = np.arange(Size)
            shuffX = Arr[0:Xshuff.shape[1]]
            shuffY = Arr[shuffX[-1] + 1:Xshuff.shape[1] + Y.shape[1]]

        shuffCorr = np.corrcoef(Cat.T)

        Ents[i] = conditional_mutual_information(Xshuff)


    Prctile = int(100 * np.floor(n_bootstraps * (1 - alpha)) / n_bootstraps)

    try:
        Threshold = np.min(Ents[Ents >= np.percentile(Ents, Prctile)])
    except ValueError:
        Threshold = 0
    Dict = {'Threshold': Threshold}
    Dict['Value'] = Ent
    Dict['Pass'] = False
    if Ent >= Threshold:
        Dict['Pass'] = True

    return Dict

def backward(X, Y, S, Z=None, alpha=0.05):
    # Reset the conditioning matrix
    Z = None
    RP = np.random.permutation(len(S))
    Sn = copy.deepcopy(S)
    for i in range(len(S)):
        X = X[:, [S[RP[i]]]]

        Inds = np.setdiff1d(Sn, S[RP[i]])

        Z = X[:, Inds]
        indZ = Inds
        indX = S[RP[i]]

        Ent = conditional_mutual_information(X)

        Dict = shuffle_test(X, Y, Ent, alpha)

        if not Dict['Pass']:
            Sn = np.setdiff1d(Sn, S[RP[i]])

    return Sn

def forward(X, Y, alpha=0.05):
    n = X.shape[1]
    TestVariables = np.arange(n)
    S = []
    NotStop = True
    while NotStop:

        SetCheck = np.setdiff1d(TestVariables, S)
        m = len(SetCheck)
        if m == 0:
            NotStop = False
            break
        Ents = np.zeros(m)
        for i in range(m):
            print(X)
            X = X[:, [SetCheck[i]]]
            indX = np.array(SetCheck[i])

            Ents[i] = conditional_mutual_information(X, Y)

        Argmax = Ents.argmax()
        X = X[:, [SetCheck[Argmax]]]

        Dict = shuffle_test(X, Y, Ents[Argmax], alpha)
        if Dict['Pass']:
            S.append(SetCheck[Argmax])
            indZ = np.array(S)
            if len(S) == 1:
                Z = X[:, [SetCheck[Argmax]]]
            else:
                Z = np.concatenate((Z, X[:, [SetCheck[Argmax]]]), axis=1)
        else:
            NotStop = False

    return S