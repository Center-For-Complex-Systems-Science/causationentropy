"""Equation discovery with entropic regression (AlMomani, Sun and Bollt, 2020)."""

from itertools import combinations_with_replacement

import numpy as np

from causationentropy.core.discovery import shuffle_test
from causationentropy.core.information.conditional_mutual_information import (
    conditional_mutual_information,
)


def polynomial_library(X, degree=2, include_bias=False, feature_names=None):
    r"""
    Build a library of monomial basis functions of the state variables.

    For :math:`d` state variables the library holds every monomial
    :math:`x_0^{p_0} x_1^{p_1} \cdots x_{d-1}^{p_{d-1}}` with
    :math:`1 \le p_0 + \dots + p_{d-1} \le` ``degree`` (plus the constant 1 when
    ``include_bias=True``), i.e. the standard polynomial basis of Eq. (3) in
    AlMomani, Sun and Bollt (2020).

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        State samples. A 1-D array is treated as a single variable.
    degree : int, default=2
        Highest total degree of the monomials. Must be at least 1.
    include_bias : bool, default=False
        If True, the first column is the constant 1.
    feature_names : list of str or None, default=None
        Names of the state variables, used in the term names. Defaults to
        ``['x0', 'x1', ...]``.

    Returns
    -------
    Phi : np.ndarray of shape (n_samples, n_terms)
        Basis functions evaluated at each sample.
    names : list of str
        Name of each column of ``Phi``, e.g. ``'x0'``, ``'x0^2'``, ``'x0 x1'``.

    Raises
    ------
    ValueError
        If ``degree < 1`` or ``feature_names`` has the wrong length.

    Examples
    --------
    >>> import numpy as np
    >>> Phi, names = polynomial_library(np.array([[2.0, 3.0]]), degree=2)
    >>> names
    ['x0', 'x1', 'x0^2', 'x0 x1', 'x1^2']
    >>> Phi.tolist()
    [[2.0, 3.0, 4.0, 6.0, 9.0]]

    References
    ----------
    A. A. R. AlMomani, J. Sun, and E. Bollt, "How entropic regression beats
    the outliers problem in nonlinear system identification," Chaos 30,
    013107 (2020). arXiv:1905.08061.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if degree < 1:
        raise ValueError(f"degree must be at least 1, got {degree}.")
    n_samples, n_features = X.shape
    if feature_names is None:
        feature_names = [f"x{i}" for i in range(n_features)]
    if len(feature_names) != n_features:
        raise ValueError(
            f"feature_names has {len(feature_names)} names for {n_features} variables."
        )

    columns, names = [], []
    if include_bias:
        columns.append(np.ones(n_samples))
        names.append("1")
    for d in range(1, degree + 1):
        for combo in combinations_with_replacement(range(n_features), d):
            columns.append(np.prod(X[:, combo], axis=1))
            powers = [(i, combo.count(i)) for i in sorted(set(combo))]
            names.append(
                " ".join(
                    feature_names[i] if p == 1 else f"{feature_names[i]}^{p}"
                    for i, p in powers
                )
            )
    return np.column_stack(columns), names


def central_difference(X, dt):
    r"""
    Estimate time derivatives with second-order central differences.

    .. math::

        \dot x(t_k) \approx \frac{x(t_{k+1}) - x(t_{k-1})}{2\,\Delta t},

    as used for the vector field estimates in AlMomani, Sun and Bollt (2020).
    The first and last samples have no central difference, so ``n_samples - 2``
    rows are returned; they belong to the states ``X[1:-1]``.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
        Uniformly sampled states. A 1-D array is treated as a single variable.
    dt : float
        Sampling interval. Must be positive.

    Returns
    -------
    np.ndarray of shape (n_samples - 2, n_features)
        Derivative estimates at ``X[1:-1]``.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.arange(5) * 0.5
    >>> central_difference(t**2, dt=0.5).ravel().tolist()
    [1.0, 2.0, 3.0]

    References
    ----------
    A. A. R. AlMomani, J. Sun, and E. Bollt, "How entropic regression beats
    the outliers problem in nonlinear system identification," Chaos 30,
    013107 (2020). arXiv:1905.08061.
    """
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    if dt <= 0:
        raise ValueError(f"dt must be positive, got {dt}.")
    if X.shape[0] < 3:
        raise ValueError("central_difference needs at least 3 samples.")
    return (X[2:] - X[:-2]) / (2.0 * dt)


def _least_squares(Phi, f, S, intercept):
    """Least-squares fit of ``f`` on the columns ``S``; returns (coef, b0, output)."""
    if not S:
        b0 = float(np.mean(f)) if intercept else 0.0
        return np.zeros(0), b0, np.full(f.shape, b0)
    A = Phi[:, S]
    if intercept:
        A = np.column_stack([A, np.ones(len(f))])
    sol = np.linalg.lstsq(A, f, rcond=None)[0]
    coef, b0 = (sol[:-1], float(sol[-1])) if intercept else (sol, 0.0)
    return coef, b0, A @ sol


def entropic_regression(
    f,
    Phi,
    alpha=0.05,
    n_shuffles=100,
    information="knn",
    k=2,
    metric=None,
    bandwidth="silverman",
    fit_intercept=False,
    rng=None,
):
    r"""
    Select a sparse set of basis functions for ``f`` with entropic regression.

    Entropic regression (ER), introduced by AlMomani, Sun and Bollt (2020),
    solves the linear inverse problem
    :math:`f = \Phi a + \xi` by greedy forward selection and backward elimination,
    like optimal causation entropy, but scores each basis function by the
    information its least-squares model output carries about ``f``. Write
    :math:`R(\Phi, f, S) = \Phi_S^\dagger f` for the least-squares coefficients on
    the columns :math:`S` and :math:`z_S = \Phi_S R(\Phi, f, S)` for the model output.

    *Forward stage.* Starting from :math:`S = \emptyset`, add the column

    .. math::

        p = \arg\max_{j \notin S} I(z_{S \cup \{j\}}; f \mid z_S)

    while that value passes a shuffle test (for :math:`S = \emptyset` the
    conditioning is dropped and the mutual information is used).

    *Backward stage.* Repeatedly find the column

    .. math::

        p = \arg\min_{j \in S} I(z_S; f \mid z_{S \setminus \{j\}})

    and remove it while that value fails the shuffle test.

    The coefficients are the least-squares solution on the final set.

    Parameters
    ----------
    f : array-like of shape (n_samples,)
        Target values, e.g. one component of the estimated vector field.
    Phi : array-like of shape (n_samples, n_terms)
        Basis functions evaluated at each sample.
    alpha : float, default=0.05
        Significance level of the shuffle tests. The paper's confidence
        parameter corresponds to ``1 - alpha``.
    n_shuffles : int, default=100
        Number of permutations per shuffle test.
    information : str, default='knn'
        Conditional mutual information estimator, passed as ``method`` to
        :func:`~causationentropy.core.information.conditional_mutual_information.conditional_mutual_information`.
    k : int, default=2
        Number of neighbours for the k-NN estimators (the paper uses k = 2).
    metric : str or None, default=None
        Distance metric for the k-NN estimators. None uses ``'chebyshev'`` for
        ``information='knn'``, as in the KSG estimator, so the first forward step
        (no conditioning) and the later steps use the same distance; for the other
        estimators None picks the estimator default.
    bandwidth : str or float, default='silverman'
        Bandwidth for the KDE estimator.
    fit_intercept : bool, default=False
        If True, every least-squares fit also includes a constant term. The
        constant is always kept and is not selected by ER (see Notes).
    rng : int, np.random.Generator or None, default=None
        Seed or generator for the shuffle tests.

    Returns
    -------
    support : list of int
        Selected column indices of ``Phi``, in increasing order.
    coef : np.ndarray of shape (n_terms,)
        Least-squares coefficients on ``support``; zero for all other columns.
    intercept : float
        Fitted constant (0.0 when ``fit_intercept=False``).

    Notes
    -----
    The forward stage uses the dynamic tolerance of the paper (SI.3, Algorithm 2):
    at each step the best candidate is accepted only if its conditional mutual
    information exceeds the :math:`(1-\alpha)` percentile of the values obtained
    after permuting ``f``. The backward stage applies the same test to the
    weakest term. Like the shuffle test in
    :func:`~causationentropy.core.discovery.discover_network`, the permutation
    breaks the dependence of ``f`` on the conditioning output as well, so it is an
    approximate test.

    Before every estimate, ``f`` and the model outputs are divided by the
    standard deviation of ``f``. Dividing all arguments by the same number does
    not change the k-NN neighbour counts, so the estimates are unchanged; it only
    sets the scale for the constant-output check.

    Mutual information does not change when a constant is added to a variable,
    so a constant basis column can never pass the information criterion. A
    constant offset is therefore handled with ``fit_intercept`` instead of a
    column of ones in ``Phi``.

    When the conditioning model output already fits ``f`` exactly (relative
    residual below 1e-10), the conditional mutual information is set to zero:
    ``f`` is then a function of the conditioning output, so the true value is
    zero, but the k-NN estimate on nearly identical vectors is not. This stops the
    forward stage and lets the backward stage remove redundant terms on exact
    data. With noise-free states and numerical derivatives, ER can still add terms
    with tiny coefficients that fit the finite-difference error; exact derivatives
    (``X_dot``) or a smaller time step avoid this.

    The authors' MATLAB package ERFit (version 1.1.0) makes different default
    choices: a static tolerance, computed once per equation as the 0.99 quantile
    of :math:`I(f; f_\pi)` over 200 permutations :math:`\pi`; the constant term
    always added back after the backward stage and dropped again if its
    coefficient is smaller than the time step; and extra forward stopping rules.
    ``alpha=0.01`` and ``n_shuffles=200`` use the same percentile and number of
    permutations here, in the dynamic test.

    Each shuffle test costs ``n_shuffles`` estimates. The k-NN estimator builds
    full distance matrices, so the cost per estimate grows as
    :math:`O(n^2)` in the number of samples.

    References
    ----------
    A. A. R. AlMomani, J. Sun, and E. Bollt, "How entropic regression beats
    the outliers problem in nonlinear system identification," Chaos 30,
    013107 (2020). arXiv:1905.08061.
    """
    f = np.asarray(f, dtype=float).ravel()
    Phi = np.asarray(Phi, dtype=float)
    if Phi.ndim == 1:
        Phi = Phi.reshape(-1, 1)
    if Phi.shape[0] != f.size:
        raise ValueError(f"Phi has {Phi.shape[0]} rows but f has {f.size} samples.")
    rng = np.random.default_rng(rng)
    if metric is None and information == "knn":
        metric = "chebyshev"
    scale = np.std(f)
    if scale == 0:
        scale = 1.0

    def col(v):
        return None if v is None else (v / scale).reshape(-1, 1)

    F = col(f)

    def cmi(out, cond):
        if np.ptp(out) <= 1e-12 * scale:
            return 0.0  # a constant output carries no information about f
        if cond is not None and np.linalg.norm(f - cond) <= 1e-10 * np.linalg.norm(f):
            return 0.0  # f is a function of cond, so I(out; f | cond) = 0
        return conditional_mutual_information(
            col(out),
            F,
            col(cond),
            method=information,
            metric=metric,
            k=k,
            bandwidth=bandwidth,
        )

    def significant(out, cond, value):
        if value <= 0.0:
            return False
        result = shuffle_test(
            F,
            col(out),
            col(cond),
            value,
            alpha=alpha,
            n_shuffles=n_shuffles,
            rng=rng,
            information=information,
            metric=metric,
            k_means=k,
            bandwidth=bandwidth,
        )
        return bool(result["Pass"])

    def output(S):
        return _least_squares(Phi, f, S, fit_intercept)[2] if S else None

    # Forward ER (Algorithm 1, lines 5-14).
    S = []
    while len(S) < Phi.shape[1]:
        z = output(S)
        candidates = [j for j in range(Phi.shape[1]) if j not in S]
        outs = [output(S + [j]) for j in candidates]
        values = [cmi(o, z) for o in outs]
        best = int(np.argmax(values))
        if not significant(outs[best], z, values[best]):
            break
        S.append(candidates[best])

    # Backward ER (Algorithm 1, lines 16-27).
    while S:
        full = output(S)
        values = [cmi(full, output([i for i in S if i != j])) for j in S]
        worst = int(np.argmin(values))
        cond = output([i for i in S if i != S[worst]])
        if significant(full, cond, values[worst]):
            break
        S.pop(worst)

    S = sorted(S)
    coef = np.zeros(Phi.shape[1])
    sol, intercept, _ = _least_squares(Phi, f, S, fit_intercept)
    coef[S] = sol
    return S, coef, intercept


class EntropicRegression:
    r"""
    Discover the equations of a dynamical system with entropic regression.

    For states :math:`x(t) \in \mathbb{R}^d` with :math:`\dot x = F(x)`, each
    component :math:`F_i` is written in a library of basis functions,
    :math:`F_i(x) = \sum_k a_{ik} \phi_k(x)`, and the sparse coefficients
    :math:`a_{ik}` are found separately for every :math:`i` with
    :func:`entropic_regression` [1]_. The default library is the monomials up to
    ``degree`` (:func:`polynomial_library`), and derivatives are estimated with
    :func:`central_difference` unless they are given.

    Parameters
    ----------
    degree : int, default=2
        Highest total degree of the polynomial library.
    fit_intercept : bool, default=False
        Fit a constant term in every equation (see :func:`entropic_regression`).
    alpha : float, default=0.05
        Significance level of the shuffle tests.
    n_shuffles : int, default=100
        Number of permutations per shuffle test.
    information : str, default='knn'
        Conditional mutual information estimator.
    k : int, default=2
        Number of neighbours for the k-NN estimators.
    metric : str or None, default=None
        Distance metric for the k-NN estimators (see :func:`entropic_regression`).
    bandwidth : str or float, default='silverman'
        Bandwidth for the KDE estimator.
    random_state : int, np.random.Generator or None, default=None
        Seed for the shuffle tests.

    Attributes
    ----------
    coef_ : np.ndarray of shape (n_features, n_terms)
        Coefficient of each library term (columns) in each equation (rows).
    intercept_ : np.ndarray of shape (n_features,)
        Constant term of each equation (zero unless ``fit_intercept=True``).
    support_ : list of list of int
        Selected library columns for each equation.
    terms_ : list of str
        Names of the library terms.
    feature_names_ : list of str
        Names of the state variables.

    Examples
    --------
    >>> import numpy as np
    >>> t = np.linspace(0, 4, 201)
    >>> X = np.column_stack([np.exp(-0.5 * t), np.exp(-2.0 * t)])
    >>> X_dot = np.column_stack([-0.5 * X[:, 0], -2.0 * X[:, 1]])
    >>> model = EntropicRegression(degree=2, n_shuffles=50, random_state=0)
    >>> model = model.fit(X, X_dot=X_dot)
    >>> print("\n".join(model.equations(precision=2)))
    x0' = -0.50 x0
    x1' = -2.00 x1

    References
    ----------
    .. [1] A. A. R. AlMomani, J. Sun, and E. Bollt, "How entropic regression beats
           the outliers problem in nonlinear system identification," Chaos 30,
           013107 (2020). arXiv:1905.08061.
    """

    def __init__(
        self,
        degree=2,
        fit_intercept=False,
        alpha=0.05,
        n_shuffles=100,
        information="knn",
        k=2,
        metric=None,
        bandwidth="silverman",
        random_state=None,
    ):
        self.degree = degree
        self.fit_intercept = fit_intercept
        self.alpha = alpha
        self.n_shuffles = n_shuffles
        self.information = information
        self.k = k
        self.metric = metric
        self.bandwidth = bandwidth
        self.random_state = random_state

    def fit(self, X, dt=None, X_dot=None, feature_names=None):
        """
        Fit one sparse equation per state variable.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Uniformly sampled states. A 1-D array is treated as a single variable.
        dt : float or None, default=None
            Sampling interval, used to estimate derivatives with
            :func:`central_difference` when ``X_dot`` is None. The first and
            last samples are then dropped.
        X_dot : array-like of shape (n_samples, n_features) or None, default=None
            Known derivatives (or any target with one column per state variable,
            e.g. the next state of a map). Takes precedence over ``dt``.
        feature_names : list of str or None, default=None
            Names of the state variables. Defaults to ``['x0', 'x1', ...]``.

        Returns
        -------
        self : EntropicRegression
        """
        X = np.asarray(X, dtype=float)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X_dot is None:
            if dt is None:
                raise ValueError("Pass either dt or X_dot.")
            X_dot = central_difference(X, dt)
            X = X[1:-1]
        else:
            X_dot = np.asarray(X_dot, dtype=float)
            if X_dot.ndim == 1:
                X_dot = X_dot.reshape(-1, 1)
            if X_dot.shape != X.shape:
                raise ValueError(
                    f"X_dot has shape {X_dot.shape} but X has shape {X.shape}."
                )

        Phi, self.terms_ = polynomial_library(
            X, self.degree, feature_names=feature_names
        )
        self.feature_names_ = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(X.shape[1])]
        )
        rng = np.random.default_rng(self.random_state)
        self.coef_ = np.zeros((X.shape[1], Phi.shape[1]))
        self.intercept_ = np.zeros(X.shape[1])
        self.support_ = []
        for i in range(X.shape[1]):
            S, coef, b0 = entropic_regression(
                X_dot[:, i],
                Phi,
                alpha=self.alpha,
                n_shuffles=self.n_shuffles,
                information=self.information,
                k=self.k,
                metric=self.metric,
                bandwidth=self.bandwidth,
                fit_intercept=self.fit_intercept,
                rng=rng,
            )
            self.support_.append(S)
            self.coef_[i] = coef
            self.intercept_[i] = b0
        return self

    def predict(self, X):
        """
        Evaluate the discovered vector field at the states ``X``.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            States at which to evaluate the equations.

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
            Predicted derivatives.
        """
        Phi, _ = polynomial_library(X, self.degree)
        return Phi @ self.coef_.T + self.intercept_

    def equations(self, precision=3):
        """
        Return the discovered equations as strings, one per state variable.

        Parameters
        ----------
        precision : int, default=3
            Number of decimals for the coefficients.

        Returns
        -------
        list of str
            For example ``"x0' = -10.000 x0 + 10.000 x1"``.
        """
        lines = []
        for i, name in enumerate(self.feature_names_):
            parts = []
            if self.fit_intercept:
                parts.append(f"{self.intercept_[i]:.{precision}f}")
            for j in self.support_[i]:
                parts.append(f"{self.coef_[i, j]:.{precision}f} {self.terms_[j]}")
            rhs = " + ".join(parts).replace("+ -", "- ") if parts else "0"
            lines.append(f"{name}' = {rhs}")
        return lines
