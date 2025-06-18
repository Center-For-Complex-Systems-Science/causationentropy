.. Optimal Causal Entropy documentation master file

====================================
Optimal Causal Entropy Documentation
====================================

.. image:: _static/images/logo.jpeg
   :alt: Optimal Causal Entropy Logo
   :width: 200px
   :align: center

Welcome to the Optimal Causal Entropy documentation! This library provides tools for 
analyzing causal relationships using entropy-based methods.

.. note::
   This is an active project. Check our `GitHub repository <https://github.com/kslote1/causalentropy>`_ 
   for the latest updates.

Quick Start
-----------

Install the package:

.. code-block:: bash

   pip install causalentropy

Basic usage:

.. code-block:: python

   from causalentropy.cs import discover_network
   
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

   api/causalentropy
   theory/index

.. toctree::
   :maxdepth: 1
   :caption: Links
   :hidden:

   GitHub Repository <https://github.com/yourusername/causalentropy>
   PyPI Package <https://pypi.org/project/causalentropy>

Please Cite
-----------

If you use Optimal Causal Entropy in your work, please cite:

.. code-block:: bibtex

   @article{slote2025causalentropy,
     author  = {Slote, Kevin and Bollt, Erirk},
     title   = {Optimal Causal Entropy for Network Inference},
     journal = {Journal of Causal Inference},
     year    = {2025},
     volume  = {X},
     pages   = {1--20},
     doi     = {10.1234/causalentropy.2025}
   }


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`