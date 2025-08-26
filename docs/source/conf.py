# Conffrom __future__ import annotations
import importlib.metadata
import os
import sys
import pathlib

# Path configuration
PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]   # …/causationentropy/
sys.path.insert(0, str(PROJECT_ROOT))

# -- Project information -----------------------------------------------------
project = 'Optimal Causal Entropy'
copyright = '2025, Kevin Slote'
author = 'Kevin Slote'
version = '0.1.0'

# Get release version from installed package
try:
    release = importlib.metadata.version("causationentropy")  # pulls from code
except importlib.metadata.PackageNotFoundError:
    release = version  # fallback to version if package not installed

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.autodoc",          # pull docstrings
    "sphinx.ext.napoleon",         # NumPy / Google style parsing
    "sphinx.ext.autosummary",      # creates stubs automatically
    "sphinx.ext.mathjax",          # LaTeX in HTML
    "sphinx.ext.viewcode",         # add source code links
    "sphinx.ext.intersphinx",      # link to other docs
]

# Conditionally add extensions if available
try:
    import sphinx_autodoc_typehints
    extensions.append("sphinx_autodoc_typehints")
except ImportError:
    print("sphinx_autodoc_typehints not available - install with: pip install sphinx-autodoc-typehints")

try:
    import myst_parser
    extensions.append("myst_parser")
except ImportError:
    print("myst_parser not available - install with: pip install myst-parser")

try:
    import sphinx_copybutton
    extensions.append("sphinx_copybutton")
except ImportError:
    print("sphinx_copybutton not available - install with: pip install sphinx-copybutton")

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']
language = 'en'

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ['_static']

# Theme options
html_theme_options = {
    'logo_only': False,
    # 'display_version': True,  # Remove this line - not supported
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': '#2980b9',
    'collapse_navigation': False,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False
}

# Custom CSS
html_css_files = [
    'css/custom.css',
]

# -- Extension configuration -------------------------------------------------

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

# Type hints configuration (if available)
if "sphinx_autodoc_typehints" in extensions:
    autodoc_typehints = "description"   # show hints in param tables
    typehints_fully_qualified = False
    always_document_param_types = True

# Autosummary configuration
autosummary_generate = True         # build .rst stubs on the fly
autosummary_imported_members = True

# Napoleon configuration (Google/NumPy docstring style)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True

# Copy button configuration (if available)
if "sphinx_copybutton" in extensions:
    copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
    copybutton_prompt_is_regexp = True
    copybutton_exclude = '.linenos, .gp, .go'

# MyST parser configuration (if available)
if "myst_parser" in extensions:
    myst_enable_extensions = [
        "colon_fence",
        "deflist",
        "dollarmath",
        "fieldlist",
        "html_admonition",
        "html_image",
        "replacements",
        "smartquotes",
        "strikethrough",
        "substitution",
        "tasklist",
    ]

# Intersphinx mapping for cross-references
intersphinx_mapping = {
    'python': ('https://docs.python.org/3/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'scipy': ('https://docs.scipy.org/doc/scipy/', None),
    'matplotlib': ('https://matplotlib.org/stable/', None),
    'pandas': ('https://pandas.pydata.org/docs/', None),
    'sklearn': ('https://scikit-learn.org/stable/', None),
}

# MathJax configuration
mathjax3_config = {
    'tex': {
        'inlineMath': [['$', '$'], ['\\(', '\\)']],
        'displayMath': [['$$', '$$'], ['\\[', '\\]']],
    }
}

# -- Options for LaTeX output ------------------------------------------------
latex_elements = {
    'papersize': 'letterpaper',
    'pointsize': '10pt',
    'preamble': '',
    'figure_align': 'htbp',
}

latex_documents = [
    ('index', 'OptimalCausalEntropy.tex', 'Optimal Causal Entropy Documentation',
     'Kevin Slote', 'manual'),
]

# -- Options for manual page output ------------------------------------------
man_pages = [
    ('index', 'optimalcausationentropy', 'Optimal Causal Entropy Documentation',
     [author], 1)
]

# -- Options for Texinfo output ----------------------------------------------
texinfo_documents = [
    ('index', 'OptimalCausalEntropy', 'Optimal Causal Entropy Documentation',
     author, 'OptimalCausalEntropy', 'Library for optimal causal entropy analysis.',
     'Miscellaneous'),
]

# Print configuration summary
print(f"Sphinx configuration loaded for {project} v{release}")
print(f"Using theme: {html_theme}")
print(f"Extensions loaded: {extensions}")
print(f"Project root: {PROJECT_ROOT}")
