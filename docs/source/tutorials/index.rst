========
Tutorials
========

This section contains practical examples of using the Causation Entropy library.

.. toctree::
   :maxdepth: 2

   basic_usage

Interactive Notebooks
====================

For interactive examples, check out our Jupyter notebooks:

.. toctree::
   :maxdepth: 1
   :glob:

   notebooks/*
Create examples/basic_usage.rst:
rst===========
Basic Usage
===========

This example demonstrates the fundamental usage of the library.

Simple Example
==============

.. code-block:: python

   import numpy as np
   from causationentropy import OptimalCausalEntropy
   
   # Generate sample data
   data = np.random.randn(100, 3)
   
   # Initialize the model
   oce = OptimalCausalEntropy()
   
   # Fit the model
   oce.fit(data)
   
   # Get results
   entropy = oce.compute_entropy()
   print(f"Causal entropy: {entropy}")

.. figure:: ../_static/images/diagrams/basic_flow.png
   :alt: Basic workflow diagram
   :width: 600px
   :align: center
   
   Basic workflow of the Causation Entropy method

Expected Output
===============

The output should look like this:

.. code-block:: text

   Causal entropy: 2.347
   Convergence achieved in 15 iterations