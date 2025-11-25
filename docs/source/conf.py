# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'torchTextClassifiers'
copyright = '2024-2025, Cédric Couralet, Meilame Tayebjee'
author = 'Cédric Couralet, Meilame Tayebjee'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',           # Auto-generate API docs from docstrings
    'sphinx.ext.napoleon',          # Support Google/NumPy style docstrings
    'sphinx.ext.viewcode',          # Add links to highlighted source code
    'sphinx.ext.intersphinx',       # Link to other project documentation
    'sphinx.ext.autosummary',       # Generate summary tables
    'sphinx_autodoc_typehints',     # Include type hints in documentation
    'sphinx_copybutton',            # Add copy button to code blocks
    'myst_parser',                  # Parse Markdown files
    'sphinx_design',                # Modern UI components (cards, grids, etc.)
    'nbsphinx',                     # Render Jupyter notebooks
    'sphinxcontrib.images'          # Allow zooming on images
]



templates_path = ['_templates']
exclude_patterns = []

# The suffix(es) of source filenames.
source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'pydata_sphinx_theme'
html_static_path = ['_static']
html_css_files = ['custom.css']

html_theme_options = {
    "github_url": "https://github.com/InseeFrLab/torchTextClassifiers",
    "logo": {
        "image_light": "_static/logo-ttc-light.svg",
        "image_dark": "_static/logo-ttc-dark.svg",
        "text": "torchTextClassifiers",
    },
    "show_nav_level": 2,
    "navigation_depth": 3,
    "show_toc_level": 2,
    "navbar_align": "left",
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "secondary_sidebar_items": ["page-toc", "edit-this-page"],
    "collapse_navigation": False,
    "navigation_with_keys": True,
}

# -- Extension configuration -------------------------------------------------

# Autodoc configuration
autodoc_default_options = {
    'members': True,
    'member-order': 'bysource',
    'special-members': '__init__',
    'undoc-members': True,
    'exclude-members': '__weakref__'
}

autodoc_typehints = 'description'
autodoc_typehints_description_target = 'documented'

# Mock imports for documentation (packages that aren't installed)
autodoc_mock_imports = ['transformers', 'tokenizers', 'datasets', 'captum']

# Napoleon configuration (for Google/NumPy style docstrings)
napoleon_google_docstring = True
napoleon_numpy_docstring = True
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = False
napoleon_type_aliases = None
napoleon_attr_annotations = True

# Intersphinx configuration (link to other documentation)
intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://numpy.org/doc/stable/', None),
    'lightning': ('https://lightning.ai/docs/pytorch/stable/', None),
}

# MyST parser configuration (for Markdown)
myst_enable_extensions = [
    "colon_fence",      # ::: for admonitions
    "deflist",          # Definition lists
    "html_image",       # HTML images
    "linkify",          # Auto-link URLs
    "replacements",     # Text replacements
    "smartquotes",      # Smart quotes
    "tasklist",         # Task lists
]

myst_heading_anchors = 3

# nbsphinx configuration (for Jupyter notebooks)
nbsphinx_execute = 'never'  # Don't execute notebooks during build
nbsphinx_allow_errors = True

# Autosummary configuration
autosummary_generate = True

# Copy button configuration
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# Syntax highlighting
pygments_style = 'sphinx'
