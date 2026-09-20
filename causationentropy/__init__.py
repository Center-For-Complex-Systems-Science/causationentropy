# Copyright (c) 2024 Kevin Slote
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from . import core, datasets
from .core import discovery
from .core.discovery import discover_network
from .datasets import synthetic

__version__ = "0.1.0"
