# Contributing to CausationEntropy

Thank you for your interest in contributing to CausationEntropy! This guide will help you get started, whether you're a student implementing new causal discovery methods or a researcher adding novel information-theoretic estimators.

## Table of Contents

- [For Students and Researchers](#for-students-and-researchers)
- [Development Setup](#development-setup)
- [Implementing New Methods](#implementing-new-methods)
- [Code Style and Standards](#code-style-and-standards)
- [Testing](#testing)
- [Submitting for Publication](#submitting-for-publication)

## For Students and Researchers

If you're implementing a new causal discovery algorithm or information-theoretic estimator for research or coursework, this section is for you.

### Quick Start for Method Development

The typical workflow for adding new methods:

1. **Information Theory Layer**: Implement entropy/mutual information estimators
2. **Discovery Layer**: Add your causal discovery algorithm
3. **Integration**: Hook into the main discovery interface
4. **Validation**: Test against synthetic data and existing methods

## Development Setup

1. **Clone and setup**:
   ```bash
   git clone https://github.com/Center-For-Complex-Systems-Science/causationentropy.git
   cd causationentropy
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -e .[dev,docs,plotting]
   ```

2. **Verify installation**:
   ```bash
   pytest causationentropy/tests/
   ```

## Implementing New Methods

### Code Style Requirement: No **kwargs

**CRITICAL**: This codebase does NOT use `**kwargs` or `*args` in function signatures. All parameters must be explicitly defined with proper type hints. This is a strict requirement that:

- Ensures type checking works correctly with mypy
- Makes the API self-documenting and clear
- Prevents parameter passing errors
- Improves IDE autocomplete and documentation generation

When adding new methods, always spell out every parameter explicitly in the function signature. Do not use variable-length argument patterns.

### Step 1: Information Theory Estimators

All information-theoretic measures go in `causationentropy/core/information/`.

#### Adding New Entropy Estimators

Create or modify `causationentropy/core/information/entropy.py`:

```python
def your_entropy_estimator(
    x: np.ndarray,
    bandwidth: str = "silverman",
    k: int = 5,
    metric: str = "euclidean"
) -> float:
    """
    Compute entropy using your method.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input data.
    bandwidth : str, default='silverman'
        Bandwidth parameter for KDE-based methods.
    k : int, default=5
        Number of neighbors for k-NN based methods.
    metric : str, default='euclidean'
        Distance metric for k-NN methods.

    Returns
    -------
    float
        Entropy estimate.

    References
    ----------
    .. [1] Your paper citation here
    """
    # Your implementation
    pass

# Register your estimator
ENTROPY_ESTIMATORS = {
    'gaussian': gaussian_entropy,
    'knn': knn_entropy,
    'your_method': your_entropy_estimator,  # Add this line
}
```

#### Adding Mutual Information Estimators

Modify `causationentropy/core/information/mutual_information.py`:

```python
def your_mutual_information(
    x: np.ndarray,
    y: np.ndarray,
    bandwidth: str = "silverman",
    k: int = 5,
    metric: str = "euclidean"
) -> float:
    """
    Compute mutual information I(X; Y) using your method.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        First variable.
    y : np.ndarray, shape (n_samples,)
        Second variable.
    bandwidth : str, default='silverman'
        Bandwidth parameter for KDE-based methods.
    k : int, default=5
        Number of neighbors for k-NN based methods.
    metric : str, default='euclidean'
        Distance metric for k-NN methods.

    Returns
    -------
    float
        Mutual information estimate.

    Notes
    -----
    Implementation details about your method.

    References
    ----------
    .. [1] Your publication reference
    """
    # Your implementation
    pass

# Register in the estimator dictionary
MI_ESTIMATORS = {
    'gaussian': gaussian_mutual_information,
    'knn': knn_mutual_information,
    'your_method': your_mutual_information,  # Add this
}
```

#### Adding Conditional Mutual Information

Modify `causationentropy/core/information/conditional_mutual_information.py`:

```python
def your_conditional_mi(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    bandwidth: str = "silverman",
    k: int = 5,
    metric: str = "euclidean"
) -> float:
    """
    Compute conditional mutual information I(X; Y | Z) using your method.

    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        First variable.
    y : np.ndarray, shape (n_samples,)
        Second variable.
    z : np.ndarray, shape (n_samples, n_conditions)
        Conditioning variables.
    bandwidth : str, default='silverman'
        Bandwidth parameter for KDE-based methods.
    k : int, default=5
        Number of neighbors for k-NN based methods.
    metric : str, default='euclidean'
        Distance metric for k-NN methods.

    Returns
    -------
    float
        Conditional mutual information estimate.

    References
    ----------
    .. [1] Your method paper
    .. [2] Related work citations
    """
    # Your implementation
    pass

# Add to the registry
CMI_ESTIMATORS = {
    'gaussian': gaussian_conditional_mi,
    'geometric_knn': geometric_knn_conditional_mi,
    'your_method': your_conditional_mi,  # Add this
}
```

### Step 2: Causal Discovery Algorithm

Create your discovery method in `causationentropy/core/discovery.py` or a new file.

**IMPORTANT**: Do NOT use `**kwargs` in your function signatures. All parameters must be explicitly defined. This is a core principle of the codebase to ensure type safety and API clarity.

```python
def your_optimal_causation_entropy(
    X: np.ndarray,
    Y: np.ndarray,
    rng: np.random.Generator,
    alpha_forward: float = 0.05,
    alpha_backward: float = 0.05,
    n_shuffles: int = 200,
    information: str = "gaussian",
    metric: str = "euclidean",
    k_means: int = 5,
    bandwidth: str = "silverman",
) -> list:
    """
    Execute your custom optimal causation entropy variant.

    This function should implement your causal discovery algorithm following the
    oCSE framework. It receives lagged predictor matrix X and target variable Y,
    and returns indices of selected predictors.

    Parameters
    ----------
    X : array-like of shape (T, n_features)
        Lagged predictor matrix where n_features = n_variables * max_lag.
    Y : array-like of shape (T, 1)
        Target variable column.
    rng : numpy.random.Generator
        Random number generator for reproducible results.
    alpha_forward : float, default=0.05
        Significance level for forward selection phase.
    alpha_backward : float, default=0.05
        Significance level for backward elimination phase.
    n_shuffles : int, default=200
        Number of permutations for statistical testing.
    information : str, default='gaussian'
        Information measure estimator type.
    metric : str, default='euclidean'
        Distance metric for k-NN estimators.
    k_means : int, default=5
        Number of neighbors for k-NN estimators.
    bandwidth : str, default='silverman'
        Bandwidth for KDE estimators.

    Returns
    -------
    S : list of int
        Indices of selected predictor variables from X that have causal
        relationships with Y.

    Notes
    -----
    Your algorithm should:
    1. Implement forward selection to identify candidate predictors
    2. Apply backward elimination to remove spurious relationships
    3. Use permutation tests via shuffle_test() for statistical significance
    4. Return only the indices of significant predictors

    Examples
    --------
    >>> import numpy as np
    >>> X = np.random.randn(100, 15)  # 100 samples, 15 lagged features
    >>> Y = np.random.randn(100, 1)
    >>> rng = np.random.default_rng(42)
    >>> S = your_optimal_causation_entropy(X, Y, rng)
    >>> print(f"Selected {len(S)} predictors")

    References
    ----------
    .. [1] Your algorithm paper
    """
    # Your algorithm implementation
    # Example structure:

    # Forward selection phase
    S = your_forward_selection(X, Y, rng, alpha_forward, n_shuffles,
                                information, metric, k_means, bandwidth)

    # Backward elimination phase
    S = your_backward_elimination(X, Y, S, rng, alpha_backward, n_shuffles,
                                   information, metric, k_means, bandwidth)

    return S
```

### Step 3: Integration with Main Interface

**IMPORTANT**: When integrating your method into `discover_network`, you must explicitly specify all parameters. The codebase does NOT use `**kwargs` - all parameters must be spelled out explicitly. This ensures type safety and makes the API clear.

Modify the main `discover_network` function in `causationentropy/core/discovery.py`:

```python
def discover_network(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = 'standard',  # Add your method name here
    information: str = "gaussian",
    max_lag: int = 5,
    alpha_forward: float = 0.05,
    alpha_backward: float = 0.05,
    metric: str = "euclidean",
    bandwidth: str = "silverman",
    k_means: int = 5,
    n_shuffles: int = 200,
    n_jobs: int = -1,
) -> nx.MultiDiGraph:
    """Main discovery interface."""

    rng = np.random.default_rng(42)

    # Validate method
    if method not in ["standard", "alternative", "information_lasso", "lasso", "your_method"]:
        raise NotImplementedError(f"discover_network: method={method} not supported.")

    # Process data and create lagged features
    # ... (data preprocessing code here)

    # Method dispatcher - handles each method inline
    for i in range(n):  # Loop over each target variable
        Y = Y_all[:, [i]]

        if method == 'standard':
            # Standard oCSE: create initial conditioning set from lagged target
            Z_init = []
            for tau in range(1, max_lag + 1):
                Z_init.append(series[max_lag - tau : T - tau, i])
            Z_init = np.column_stack(Z_init)
            S = standard_optimal_causation_entropy(
                X_lagged, Y, Z_init, rng,
                alpha_forward, alpha_backward, n_shuffles,
                information, metric, k_means, bandwidth
            )
        elif method == 'alternative':
            # Alternative oCSE: no initial conditioning set
            S = alternative_optimal_causation_entropy(
                X_lagged, Y, rng,
                alpha_forward, alpha_backward, n_shuffles,
                information, metric, k_means, bandwidth
            )
        elif method == 'information_lasso':
            # Information-theoretic LASSO variant
            S = information_lasso_optimal_causation_entropy(X_lagged, Y, rng)
        elif method == 'lasso':
            # Pure LASSO-based selection
            S = lasso_optimal_causation_entropy(X_lagged, Y, rng)
        elif method == 'your_method':
            # Your custom method - explicitly pass all needed parameters
            S = your_optimal_causation_entropy(
                X_lagged, Y, rng,
                alpha_forward, alpha_backward, n_shuffles,
                information, metric, k_means, bandwidth
            )

        # Add edges to graph for selected predictors
        # ... (edge creation code here)

    return G
```

**Key Points:**
- No `**kwargs` - all parameters must be explicitly defined in the function signature
- Each method receives the specific parameters it needs, spelled out completely
- This ensures type checking works correctly and makes the API self-documenting

### Step 4: Testing Your Implementation

Create comprehensive tests in `causationentropy/tests/`:

```python
# causationentropy/tests/test_your_method.py
import pytest
import numpy as np
import pandas as pd
import networkx as nx
from causationentropy.core.discovery import your_discovery_method
from causationentropy.core.information.entropy import your_entropy_estimator

class TestYourMethod:
    def test_entropy_estimator(self):
        """Test your entropy estimator."""
        # Test with known data
        x = np.random.normal(0, 1, (100, 2))
        entropy = your_entropy_estimator(x)

        assert entropy > 0  # Entropy should be positive
        assert np.isfinite(entropy)  # Should be finite

    def test_discovery_method(self):
        """Test your discovery method."""
        # Create synthetic data with known structure
        n_samples, n_vars = 100, 3
        data = np.random.normal(0, 1, (n_samples, n_vars))

        # Run discovery
        network = your_discovery_method(data)

        # Basic validity checks
        assert network.number_of_nodes() == n_vars
        assert isinstance(network, nx.MultiDiGraph)

        # Check edge attributes
        for u, v, data in network.edges(data=True):
            assert 'lag' in data
            assert 'cmi' in data
            assert 'p_value' in data
            assert data['lag'] >= 1  # Lags should be positive
            assert 0 <= data['p_value'] <= 1  # Valid p-value range

    def test_integration(self):
        """Test integration with main interface."""
        from causationentropy.core.discovery import discover_network

        data = pd.DataFrame(np.random.normal(0, 1, (50, 3)))
        network = discover_network(data, method='your_method')

        assert hasattr(network, 'nodes')
        assert hasattr(network, 'edges')
        assert isinstance(network, nx.MultiDiGraph)
```

Run your tests:
```bash
pytest causationentropy/tests/test_your_method.py -v
```

## Code Style and Standards

### Documentation Requirements

- **All functions** must have NumPy-style docstrings
- **Include mathematical formulations** in LaTeX for algorithms
- **Cite relevant papers** in References section
- **Provide examples** for main functions

### Example Documentation:

```python
def your_algorithm(data: np.ndarray, alpha: float = 0.05) -> nx.DiGraph:
    r"""
    Discover causal networks using Your Novel Method (YNM).
    
    The algorithm works by optimizing the following objective:
    
    .. math::
        \mathcal{L} = \sum_{i,j} I(X_i^{(t)}; X_j^{(t-\tau)} | \mathbf{Z}_{ij})
    
    where :math:`I(\cdot; \cdot | \cdot)` is conditional mutual information.
    
    Parameters
    ----------
    data : np.ndarray, shape (n_samples, n_variables)
        Multivariate time series data.
    alpha : float, default=0.05
        Significance level for hypothesis testing.
        
    Returns
    -------
    nx.DiGraph
        Directed graph representing causal relationships.
        
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, (100, 3))  
    >>> G = your_algorithm(data, alpha=0.01)
    >>> print(f"Found {G.number_of_edges()} edges")
    
    References
    ----------
    .. [1] Smith, J. et al. "Your Novel Method for Causal Discovery." 
           Journal of Causal Inference, 2024.
    .. [2] Related work citation here.
    """
```

### Code Quality

Run these before submitting:
```bash
black causationentropy/          # Code formatting
isort causationentropy/          # Import sorting
flake8 causationentropy/         # Linting
mypy causationentropy/           # Type checking (allowed to fail)
pytest --cov=causationentropy    # Test with coverage
```

## Submitting for Publication

### Preparing Your Contribution for Academic Publication

1. **Create a comprehensive example**:
   ```python
   # examples/your_method_example.py
   """
   Example demonstrating Your Novel Method for causal discovery.

   This example shows how to use YNM on both synthetic and real data,
   comparing results with existing methods like standard oCSE.
   """
   import numpy as np
   from causationentropy.core.discovery import discover_network
   from causationentropy.datasets.synthetic import logistic_dynamics

   # Generate synthetic data
   data, true_adjacency = logistic_dynamics()

   # Run your method
   network = discover_network(data, method='your_method', max_lag=3)

   # Compare with standard oCSE
   network_standard = discover_network(data, method='standard', max_lag=3)
   ```

2. **Add a detailed notebook**:
   ```bash
   # Create notebooks/your_method_tutorial.ipynb
   # Include:
   # - Method explanation
   # - Mathematical background  
   # - Performance comparisons
   # - Visualization of results
   ```

3. **Document computational complexity**:
   ```python
   def your_algorithm(data):
       """
       Time complexity: O(T * n^2 * τ_max * n_shuffles) where:
           - T = number of time points
           - n = number of variables
           - τ_max = maximum lag
           - n_shuffles = number of permutations
       Space complexity: O(T * n * τ_max)

       Note: The oCSE algorithm is computationally intensive - users should
       be patient when running on large datasets.
       """
   ```

4. **Benchmark against existing methods**:
   ```python
   # causationentropy/tests/test_benchmarks.py
   import time
   import numpy as np
   from causationentropy.core.discovery import discover_network
   from causationentropy.datasets.synthetic import logistic_dynamics

   def test_method_comparison():
       """Compare your method with standard approaches."""
       # Generate synthetic data with ground truth
       data, true_adjacency = logistic_dynamics()

       # Run multiple methods and compare
       methods = ['standard', 'alternative', 'your_method']
       results = {}

       for method in methods:
           start_time = time.time()
           network = discover_network(data, method=method, max_lag=3)
           elapsed_time = time.time() - start_time

           results[method] = {
               'num_edges': network.number_of_edges(),
               'runtime': elapsed_time,
               'network': network
           }

       # Compare accuracy, runtime, etc.
       print(f"Results: {results}")
   ```

### Publication Checklist

Before submitting your work:

- [ ] **Implementation complete** with all information estimators
- [ ] **Tests pass** with >90% coverage for new code
- [ ] **Documentation** includes mathematical formulation
- [ ] **Example notebook** demonstrates usage
- [ ] **Benchmarking** against existing methods
- [ ] **Code formatted** and linted (black, flake8)
- [ ] **Citations** properly formatted in docstrings
- [ ] **Performance analysis** documented

### Repository Structure for Your Method

```
causationentropy/
├── causationentropy/
│   ├── core/
│   │   ├── information/
│   │   │   ├── entropy.py                    # Add your entropy estimator
│   │   │   ├── mutual_information.py         # Add your MI estimator
│   │   │   └── conditional_mutual_information.py # Add your CMI estimator
│   │   ├── discovery.py                      # Add your discovery method
│   │   ├── stats.py                          # Statistical utilities
│   │   ├── linalg.py                         # Linear algebra utilities
│   │   └── plotting.py                       # Visualization tools
│   ├── datasets/
│   │   └── synthetic.py                      # Synthetic data generators
│   ├── graph/
│   │   └── utils.py                          # Graph conversion utilities
│   └── tests/
│       └── test_your_method.py               # Comprehensive tests
├── examples/
│   └── your_method_example.py                # Usage example
├── notebooks/
│   └── your_method_tutorial.ipynb            # Tutorial notebook
└── papers/                                   # Add your paper PDF here
    └── your_method.pdf
```

## Getting Help

- **Questions about implementation**: Create a GitHub issue
- **Mathematical questions**: Email kslote1@gmail.com  
- **Code reviews**: Open a draft pull request early for feedback
- **Publication guidance**: Discuss in GitHub Discussions

## Recognition

Student and researcher contributions are highlighted in:
- README acknowledgments
- Method-specific documentation  
- Release notes
- Academic citations when appropriate

Your contributions to causal discovery research are valuable - thank you for advancing the field! 🎓