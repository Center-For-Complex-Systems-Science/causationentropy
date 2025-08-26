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
   git clone https://github.com/kslote1/causalentropy.git
   cd causalentropy
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   pip install -e .[dev,docs,plotting]
   ```

2. **Verify installation**:
   ```bash
   pytest causalentropy/tests/
   ```

## Implementing New Methods

### Step 1: Information Theory Estimators

All information-theoretic measures go in `causalentropy/core/information/`. 

#### Adding New Entropy Estimators

Create or modify `causalentropy/core/information/entropy.py`:

```python
def your_entropy_estimator(x: np.ndarray, **kwargs) -> float:
    """
    Compute entropy using your method.
    
    Parameters
    ----------
    x : np.ndarray, shape (n_samples, n_features)
        Input data.
    **kwargs : dict
        Method-specific parameters.
        
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

Modify `causalentropy/core/information/mutual_information.py`:

```python
def your_mutual_information(x: np.ndarray, y: np.ndarray, **kwargs) -> float:
    """
    Compute mutual information I(X; Y) using your method.
    
    Parameters
    ----------
    x : np.ndarray, shape (n_samples,)
        First variable.
    y : np.ndarray, shape (n_samples,)
        Second variable.
    **kwargs : dict
        Method-specific parameters.
        
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

Modify `causalentropy/core/information/conditional_mutual_information.py`:

```python
def your_conditional_mi(
    x: np.ndarray, 
    y: np.ndarray, 
    z: np.ndarray, 
    **kwargs
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
    **kwargs : dict
        Method-specific parameters.
        
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

Create your discovery method in `causalentropy/core/discovery.py` or a new file:

```python
def your_discovery_method(
    data: Union[np.ndarray, pd.DataFrame],
    information: str = "your_method",
    max_lag: int = 5,
    alpha: float = 0.05,
    **kwargs
) -> nx.DiGraph:
    """
    Discover causal network using your algorithm.
    
    Parameters
    ----------
    data : array-like, shape (n_samples, n_variables)
        Time series data.
    information : str, default="your_method"
        Information estimator to use.
    max_lag : int, default=5
        Maximum time lag to consider.
    alpha : float, default=0.05
        Significance level for statistical tests.
    **kwargs : dict
        Additional algorithm parameters.
        
    Returns
    -------
    nx.DiGraph
        Discovered causal network.
        
    Notes
    -----
    Detailed description of your algorithm:
    1. Step 1 of your method
    2. Step 2 of your method
    3. etc.
    
    Examples
    --------
    >>> import numpy as np
    >>> data = np.random.normal(0, 1, (100, 3))
    >>> network = your_discovery_method(data)
    
    References
    ----------
    .. [1] Your algorithm paper
    """
    # Algorithm implementation
    network = nx.DiGraph()
    
    # Your discovery logic here
    # Use the information estimators you implemented above
    
    return network
```

### Step 3: Integration with Main Interface

Modify the main `discover_network` function in `causalentropy/core/discovery.py`:

```python
def discover_network(
    data: Union[np.ndarray, pd.DataFrame],
    method: str = 'standard',  # Add your method name here
    information: str = "gaussian",
    **kwargs
) -> nx.DiGraph:
    """Main discovery interface."""
    
    # Add your method to the dispatcher
    if method == 'standard':
        return _discover_standard(data, information, **kwargs)
    elif method == 'hawkes':
        return discover_network_hawkes(data, **kwargs)
    elif method == 'your_method':  # Add this
        return your_discovery_method(data, information, **kwargs)
    else:
        raise ValueError(f"Unknown method: {method}")
```

### Step 4: Testing Your Implementation

Create comprehensive tests in `causalentropy/tests/`:

```python
# causalentropy/tests/test_your_method.py
import pytest
import numpy as np
import pandas as pd
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
        assert isinstance(network, nx.DiGraph)
    
    def test_integration(self):
        """Test integration with main interface."""
        from causationentropy import discover_network
        
        data = pd.DataFrame(np.random.normal(0, 1, (50, 3)))
        network = discover_network(data, method='your_method')
        
        assert hasattr(network, 'nodes')
        assert hasattr(network, 'edges')
```

Run your tests:
```bash
pytest causalentropy/tests/test_your_method.py -v
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
black causalentropy/          # Code formatting
isort causalentropy/          # Import sorting  
flake8 causalentropy/         # Linting
mypy causalentropy/           # Type checking
pytest --cov=causalentropy    # Test with coverage
```

## Submitting for Publication

### Preparing Your Contribution for Academic Publication

1. **Create a comprehensive example**:
   ```python
   # examples/your_method_example.py
   """
   Example demonstrating Your Novel Method for causal discovery.
   
   This example shows how to use YNM on both synthetic and real data,
   comparing results with existing methods.
   """
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
       Time complexity: O(n^2 * p^2) where n=samples, p=variables
       Space complexity: O(n * p)
       """
   ```

4. **Benchmark against existing methods**:
   ```python
   # causalentropy/tests/test_benchmarks.py
   def test_method_comparison():
       """Compare your method with standard approaches."""
       # Generate synthetic data with ground truth
       # Run multiple methods
       # Compare accuracy, runtime, etc.
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
causalentropy/
├── causalentropy/core/information/
│   ├── entropy.py                    # Add your entropy estimator
│   ├── mutual_information.py         # Add your MI estimator  
│   └── conditional_mutual_information.py # Add your CMI estimator
├── causalentropy/core/
│   └── discovery.py                  # Add your discovery method
├── causalentropy/tests/
│   └── test_your_method.py           # Comprehensive tests
├── examples/
│   └── your_method_example.py        # Usage example
├── notebooks/
│   └── your_method_tutorial.ipynb    # Tutorial notebook
└── papers/                           # Add your paper PDF here
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