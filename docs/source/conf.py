# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

from __future__ import annotations
import importlib.metadata, os, sys
import pathlib, sys

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # …/causalentropy/
sys.path.insert(0, str(PROJECT_ROOT))

project = 'Optimal Causal Entropy'
copyright = '2025, Kevin Slote'
author = 'Kevin Slote'

version = '0.1.0'
release = '0.1.0'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

release   = importlib.metadata.version("causalentropy")  # pulls from code

extensions = [
    "sphinx.ext.autodoc",          # pull docstrings
    "sphinx.ext.napoleon",         # NumPy / Google style parsing
    "sphinx.ext.autosummary",      # creates stubs automatically
    "sphinx_autodoc_typehints",    # merge type hints into docs
    "myst_parser",                 # optional Markdown support
    "sphinx.ext.mathjax",          # LaTeX in HTML
    "sphinx_copybutton",           # nice copy-code buttons
]


templates_path = ['_templates']
exclude_patterns = []

language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
autodoc_typehints = "description"   # show hints in param tables
autosummary_generate = True         # build .rst stubs on the fly
html_theme = "sphinx_rtd_theme"
