.. Optimal Causal Entropy documentation master file

====================================
Optimal Causal Entropy Documentation
====================================

.. image:: _static/images/logo.jpeg
   :alt: Optimal Causation Entropy Logo
   :width: 200px
   :align: center

Welcome to the Optimal Causation Entropy documentation! This library provides tools for
analyzing causal relationships using entropy-based methods.

.. note::
   This is an active project. Check our `GitHub repository <https://github.com/kslote1/causationentropy>`_ 
   for the latest updates.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install causationentropy

Basic usage:

.. code-block:: python

   from causationentropy.cs import discover_network
   
   # Your basic example here
   oc = OptimalCausalEntropy()
   result = oc.compute(data)

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   tutorials/index
   examples/index

.. toctree::
   :maxdepth: 2
   :caption: Reference
   :hidden:

   api/causationentropy
   theory/index

.. toctree::
   :maxdepth: 1
   :caption: Links
   :hidden:

   GitHub Repository <https://github.com/yourusername/causationentropy>
   PyPI Package <https://pypi.org/project/causationentropy>

Please Cite
-----------

If you use Optimal Causal Entropy in your work, please cite:

.. code-block:: bibtex

   @article{slote2025causationentropy,
     author  = {Slote, Kevin and Bollt, Erirk},
     title   = {Optimal Causal Entropy for Network Inference},
     journal = {Journal of Causal Inference},
     year    = {2025},
     volume  = {X},
     pages   = {1--20},
     doi     = {10.1234/causationentropy.2025}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`