# CausationEntropy

A Python library for discovering causal networks from time series data using **Optimal Causation Entropy (oCSE)**.

## What it does

Given time series data, CausationEntropy finds which variables cause changes in other variables by:
1. Testing if knowing variable X at time t helps predict variable Y at time t+1
2. Using information theory to measure predictive relationships
3. Statistical testing to avoid false discoveries

## Installation

```bash
pip install causationentropy
```

## Quick Example

```python
from causationentropy import discover_network
import pandas as pd

# Load your time series data
data = pd.read_csv('your_data.csv')

# Find causal relationships
network = discover_network(data)

# See the results
print(f"Found {network.number_of_edges()} causal relationships")
for source, sink in network.edges(data=True):
    print(f"{source}  {sink}")
```

## Key Math

The algorithm uses **conditional mutual information**:

$$I(X; Y | Z) = H(X | Z) + H(Y | Z) - H(X, Y | Z)$$

This measures how much X tells us about Y, beyond what we already know from Z.

**Causal Rule**: X causes Y if knowing X(t) significantly improves prediction of Y(t+1), even when controlling for other variables
