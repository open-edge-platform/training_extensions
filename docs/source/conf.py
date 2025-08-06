"""Configuration file for the Sphinx documentation builder.

For the full list of built-in configuration values, see the documentation:
https://www.sphinx-doc.org/en/master/usage/configuration.html

-- Project information -----------------------------------------------------
https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information
"""

# Copyright (C) 2022-2025 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import os
import sys

sys.path.insert(0, os.path.abspath("../../src"))

from otx import __version__

# ruff: noqa

# -- Project information ----------------------------------------------------- #

project = "OpenVINO™ Training Extensions"
copyright = "2024, OpenVINO™ Training Extensions Contributors"
author = "OpenVINO™ Training Extensions Contributors"
release = __version__

# -- General configuration --------------------------------------------------- #

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.napoleon",  # Support for NumPy and Google style docstrings
    "sphinx.ext.autodoc",
    "sphinx_copybutton",
    "sphinx.ext.autosummary",  # Create neat summary tables
    "sphinx.ext.viewcode",  # Find the source files
    "sphinx.ext.autosectionlabel",  # Refer sections its title
    "sphinx.ext.intersphinx",  # Generate links to the documentation
    "sphinx_tabs.tabs",
    "sphinx_design",
    "myst_parser",  # Enhanced markdown support
    "sphinx.ext.todo",  # Support for TODO items
    "sphinx.ext.coverage",  # Documentation coverage check
]

myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "html_image",
    "linkify",
    "tasklist",
]

suppress_warnings = [
    "ref.python",
    "autosectionlabel.*",
]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output ------------------------------------------------- #
# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_theme = "sphinx_book_theme"
html_static_path = ["_static"]
html_logo = "_static/logos/otx-logo.png"
html_favicon = "_static/logos/geti-favicon-64.png"

html_context = {
    "github_user": "open-edge-platform",
    "github_repo": "training_extensions",
    "github_version": "master",
    "doc_path": "docs/source/",
}

# -- Extension configuration -------------------------------------------------
autodoc_docstring_signature = True
autodoc_member_order = "bysource"
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
}
autodoc_member_order = "groupwise"
autodoc_default_options = {
    "members": True,
    "methods": True,
    "special-members": "__call__",
    "exclude-members": "_abc_impl",
    "show-inheritance": True,
}

autoclass_content = "both"

autosummary_generate = True  # Turn on sphinx.ext.autosummary
autosummary_ignore_module_all = False  # Summary list in __all__ no others
# autosummary_imported_members = True # document classes and functions imported in modules
