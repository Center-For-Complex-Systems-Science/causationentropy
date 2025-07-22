# CausationEntropy

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Documentation Status](https://readthedocs.org/projects/causalentropy/badge/?version=latest)](https://causalentropy.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/kslote1/causalentropy/branch/main/graph/badge.svg)](https://codecov.io/gh/kslote1/causalentropy)
[![Tests](https://github.com/kslote1/causalentropy/workflows/Tests/badge.svg)](https://github.com/kslote1/causalentropy/actions)

A Python library for discovering causal networks from time series data using **Optimal Causation Entropy (oCSE)**.

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
git clone https://github.com/kslote1/causalentropy.git
cd causalentropy
pip install -e .[dev,docs,plotting]
```

## Quick Start

### Basic Usage

```python
import numpy as np
import pandas as pd
from causalentropy import discover_network

# Load your time series data (variables as columns, time as rows)
data = pd.read_csv('your_data.csv')

# Discover causal network
network = discover_network(data, method='standard', max_lag=5)

# Examine results
print(f"Found {network.number_of_edges()} causal relationships")
for source, sink in network.edges(data=True):
    print(f"{source} → {sink}: {network[source][sink]}")
```

### Advanced Configuration

```python
# Configure discovery parameters
network = discover_network(
    data,
    method='standard',           # 'standard' or 'hawkes'
    information='gaussian',      # 'gaussian' or 'geometric_knn'
    max_lag=5,                  # Maximum time lag to consider
    alpha_forward=0.05,         # Forward selection significance
    alpha_backward=0.05,        # Backward elimination significance
    n_shuffles=200             # Permutation test iterations
)
```

### Synthetic Data Example

```python
from causalentropy.datasets import synthetic

# Generate synthetic causal time series
data, true_network = synthetic.generate_var_data(
    n_variables=5, 
    n_samples=1000, 
    sparsity=0.3
)

# Discover network
discovered = discover_network(data)

# Compare with ground truth
print(f"True edges: {true_network.number_of_edges()}")
print(f"Discovered edges: {discovered.number_of_edges()}")
```

## Key Features

- **Multiple Algorithms**: Standard oCSE and Hawkes process variants
- **Flexible Information Estimators**: Gaussian and geometric k-nearest neighbor methods  
- **Statistical Rigor**: Permutation-based significance testing with >90% test coverage
- **Synthetic Data**: Built-in generators for testing and validation
- **Visualization**: Network plotting and analysis tools
- **Performance**: Optimized implementations with parallel processing support

## Mathematical Foundation

The algorithm uses **conditional mutual information** to quantify causal relationships:

$$I(X; Y | Z) = H(X | Z) + H(Y | Z) - H(X, Y | Z)$$

This measures how much variable X tells us about variable Y, beyond what we already know from conditioning set Z.

**Causal Discovery Rule**: Variable X causes Y if knowing X(t) significantly improves prediction of Y(t+1), even when controlling for all other relevant variables.

The algorithm implements a two-phase approach:
1. **Forward Selection**: Iteratively adds predictors that maximize conditional mutual information
2. **Backward Elimination**: Removes predictors that lose significance when conditioned on others

## Documentation

📚 **[Read the full documentation on ReadTheDocs](https://causalentropy.readthedocs.io/)**

- **[API Reference](https://causalentropy.readthedocs.io/en/latest/api/)**: Complete function and class documentation
- **[User Guide](https://causalentropy.readthedocs.io/en/latest/user_guide/)**: Detailed tutorials and examples
- **[Theory](https://causalentropy.readthedocs.io/en/latest/theory/)**: Mathematical background and algorithms
- **Examples**: Check the `examples/` and `notebooks/` directories
- **Research Papers**: See the `papers/` directory for theoretical foundations

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
@misc{causationentropy2024,
  title={CausationEntropy: A Python Library for Causal Discovery},
  author={Kevin Slote},
  year={2024},
  url={https://github.com/kslote1/causalentropy}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE.txt) file for details.

## Support

- **Issues**: [GitHub Issues](https://github.com/kslote1/causalentropy/issues)
- **Discussions**: [GitHub Discussions](https://github.com/kslote1/causalentropy/discussions)
- **Email**: kslote1@gmail.com

## Acknowledgments

This work builds upon fundamental research in information theory, causal inference, and time series analysis. Special thanks to the open-source scientific Python community.
