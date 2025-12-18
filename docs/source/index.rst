.. Causation Entropy documentation master file

===============================
Causation Entropy Documentation
===============================

.. image:: _static/images/logo.jpeg
   :alt: Causation Entropy Logo
   :width: 200px
   :align: center

Welcome to the Causation Entropy documentation! This library provides tools for
analyzing causal relationships using information-theory based methods.

.. note::
   This is an active project. Check our `GitHub repository <https://github.com/Center-For-Complex-Systems-Science/causationentropy>`_
   for the latest updates.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install causationentropy

Basic usage:

.. code-block:: python

   from causationentropy import discover_network
   import numpy as np
   
   # Generate synthetic data
   data = np.random.randn(100, 5)  # 100 time points, 5 variables
   
   # Discover causal network
   network = discover_network(data, method='standard', information='gaussian')

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: API Reference
   :hidden:

   api/network_discovery
   api/information
   api/datasets
   api/linalg
   api/plotting
   api/stats


.. toctree::
   :maxdepth: 2
   :caption: Theory
   :hidden:

   theory/index

.. toctree::
   :maxdepth: 1
   :caption: Links
   :hidden:

   GitHub Repository <https://github.com/Center-For-Complex-Systems-Science/causationentropy>
   PyPI Package <https://pypi.org/project/causationentropy>

Please Cite
-----------

If you use Causation Entropy in your work, please cite:

.. code-block:: bibtex

   @misc{slote2025causationentropy,
     author  = {Slote, Kevin and Fish, Jeremie and Bollt, Erik},
     title   = {CausationEntropy: A Python Library for Causal Discovery},
     url     = {https://github.com/Center-For-Complex-Systems-Science/causationentropy},
     doi     = {10.5281/zenodo.17047565}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
