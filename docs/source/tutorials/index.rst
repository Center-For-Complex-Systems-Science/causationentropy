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

Basic Usage
===========

This example demonstrates the fundamental usage of the library.

Simple Example
==============

.. code-block:: python

   from causationentropy import discover_network

   # Load your time series data (variables as columns, time as rows)
   data = pd.read_csv('your_data.csv')

   # Discover causal network
   network = discover_network(data, method='standard', max_lag=5)

.. figure:: ../_static/images/diagrams/basic_flow.png
   :alt: Basic workflow diagram
   :width: 600px
   :align: center
   
   The `discover_network` method returns a NetworkX MultiDiGraph object.
