# CausationEntropy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/causationentropy/badge/?version=stable)](https://causationentropy.readthedocs.io/en/stable/?badge=stable)
[![codecov](https://codecov.io/gh/Center-For-Complex-Systems-Science/causationentropy/branch/main/graph/badge.svg)](https://app.codecov.io/gh/Center-For-Complex-Systems-Science/causationentropy)
[![Tests](https://github.com/kslote1/causationentropy/workflows/Tests/badge.svg)](https://github.com/kslote1/causationentropy/actions)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17047565.svg)](https://doi.org/10.5281/zenodo.17047565)


A Python library for discovering causal networks from time series data using **Optimal Causation Entropy (oCSE)**.

## Quick Start

See our Quick Start colab notebook: [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/Center-For-Complex-Systems-Science/causationentropy/blob/main/notebooks/Quickstart.ipynb)

## Overview

CausationEntropy implements state-of-the-art information-theoretic methods for causal discovery from multivariate time series. The library provides robust algorithms that can identify causal relationships while controlling for confounding variables and false discoveries.

### What it does

Given time series data, CausationEntropy finds which variables cause changes in other variables by:

1. **Predictive Testing**: Testing if knowing variable X at time t helps predict variable Y at time t+1
2. **Information Theory**: Using conditional mutual information to measure predictive relationships
3. **Statistical Control**: Rigorous statistical testing to avoid false discoveries
4. **Multiple Methods**: Supporting various information estimators and discovery algorithms

## Installation

### From PyPI (recommended)
```bash
pip install causationentropy
```

### Development Installation
```bash
git clone https://github.com/Center-For-Complex-Systems-Science/causationentropy.git
cd causationentropy
pip install -e .
```

##### Run the tests

```bash
python -m pytest causationentropy/tests/ --cov=causationentropy
```

With overage loally:
```bash
python -m pytest causationentropy/tests/ --cov=causationentropy --cov-report=xml --cov-report=term-missing -v
```

### Basic Usage

Get the relationships as a data frame:
```python
import pandas as pd
from causationentropy import discover_network
from causationentropy.graph import network_to_dataframe

# Load your time series data (variables as columns, time as rows)
data = pd.read_csv('data.csv')

# Discover causal network
network = discover_network(data, method='standard', max_lag=5)
df = network_to_dataframe(network)
df.head()
```

Plot the causal network:
```python
from causationentropy import discover_network
from causationentropy.core.plotting import plot_causal_network

# Load your time series data (variables as columns, time as rows)
data = pd.read_csv('data.csv')

# Discover causal network
network = discover_network(data, method='standard', max_lag=5)
fig, ax = plot_causal_network(network, save_path="network.png")
```
**Note:** This implementation of this algorithm runs in `O(n^2 T log T)` where `N` is the number of variables and `T` is the length of the time series. Application of this algorithm without optimizations is computationally intensive. When running this algorithm, please be patient. Optimizations of the algorithm are planned for a later release that leverage singular value decomposition and KD-Trees. However, these optimizations are not part of the original algorithm. Adding additional lags also contributes to additional performance degradations.

### Advanced Configuration

```python
from causationentropy import discover_network

# Configure discovery parameters
network = discover_network(
    data,
    method='standard',          # 'standard', 'alternative', 'information_lasso', or 'lasso'
    information='gaussian',     # 'gaussian', 'knn', 'kde', 'geometric_knn', or 'poisson'
    max_lag=5,                  # Maximum time lag to consider
    alpha_forward=0.05,         # Forward selection significance
    alpha_backward=0.05,        # Backward elimination significance
    n_shuffles=200              # Permutation test iterations
)
```

### Synthetic Data Example

```python
from causationentropy.datasets import synthetic
from causationentropy import discover_network

# Generate synthetic causal time series
data, true_network = synthetic.linear_stochastic_gaussian_process(
    n_variables=5, 
    n_samples=1000, 
    sparsity=0.3
)

# Discover network
discovered = discover_network(data)
```

## Key Features

- **Multiple Algorithms**: Standard, alternative, information lasso, and lasso variants of oCSE
- **Flexible Information Estimators**: Gaussian, k-NN, KDE, geometric k-NN, and Poisson methods  
- **Statistical Rigor**: Permutation-based significance testing with comprehensive test coverage
- **Synthetic Data**: Built-in generators for testing and validation
- **Visualization**: Network plotting and analysis tools
- 
## Mathematical Foundation

The algorithm uses **conditional mutual information** to quantify causal relationships:

$$I(X; Y | Z) = H(X | Z) + H(Y | Z) - H(X, Y | Z)$$

This measures how much variable X tells us about variable Y, beyond what we already know from conditioning set Z.

**Causal Discovery Rule**: Variable X causes Y if knowing X(t) significantly improves prediction of Y(t+1), even when controlling for all other relevant variables.

The algorithm implements a two-phase approach:
1. **Forward Selection**: Iteratively adds predictors that maximize conditional mutual information
2. **Backward Elimination**: Removes predictors that lose significance when conditioned on others

## Documentation

📚 **[Read the full documentation on ReadTheDocs](https://causationentropy.readthedocs.io/)**

- **[API Reference](https://causationentropy.readthedocs.io/en/latest/api/)**: Complete function and class documentation
- **[User Guide](https://causationentropy.readthedocs.io/en/latest/user_guide/)**: Detailed tutorials and examples
- **[Theory](https://causationentropy.readthedocs.io/en/latest/theory/)**: Mathematical background and algorithms
- **Examples**: Check the `notebooks/` directory
- **Research Papers**: See the `theory glossary` in the [documentation](https://causationentropy.readthedocs.io/en/latest/theory/index.html)

### Local Documentation

Build documentation locally:
```bash
cd docs/
make html
# Open docs/_build/html/index.html
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Citation

If you use this library in your research, please cite:

```bibtex
   @misc{slote2025causationentropy,
     author  = {Slote, Kevin and Fish, Jeremie and Bollt, Erik},
     title   = {CausationEntropy: A Python Library for Causal Discovery},
     url     = {https://github.com/Center-For-Complex-Systems-Science/causationentropy},
     doi     = {10.5281/zenodo.17047565}
   }
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/kslote1/causationentropy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kslote1/causationentropy/discussions)
- **Email**: kslote1@gmail.com

## Acknowledgments

This work builds upon fundamental research in information theory, causal inference, and time series analysis.
Special thanks to the open-source scientific Python community.

[Original Code](https://github.com/jefish003/NetworkInference)

## LLM Disclosure

Generative AI was used to help with doc strings, documentation, and unit tests.
