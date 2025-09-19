from . import core, datasets, tests

# Optionally, expose commonly used functions directly
from .core import discovery, estimators
from .core.discovery import discover_network
from .datasets import synthetic

__version__ = "0.1.0"
