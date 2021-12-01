# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
import datetime

sys.path.insert(0, os.path.abspath(".."))

autodoc_mock_imports = ["ray"]

# -- Project information -----------------------------------------------------
project = "Phantom [BETA]"
year = datetime.datetime.now().year
copyright = "{}, AI Research, JP Morgan. Some icons used freely from Font Awesome (https://fontawesome.com/license/free)".format(
    year
)
author = "AI Research Group"
description = "A Multi-agent reinforcement-learning simulator framework."


# -- General configuration ---------------------------------------------------
# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.todo",
    "sphinx.ext.viewcode",
    "sphinx.ext.intersphinx",
    "sphinx_autodoc_typehints",
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# The master toctree document.
master_doc = "index"

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# The version info for the project you're documenting, acts as replacement for
# |version| and |release|, also used in various other places throughout the
# built documents.
#
# The short X.Y version.
version = "1.0"

# If true, the current module name will be prepended to all description
# unit titles (such as .. function::).
add_module_names = True

# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "sphinx_rtd_theme"
# html_theme = "alabaster"

# html_theme_options = {
#     "description": description,
#     "fixed_sidebar": True,
#     "show_powered_by": False,
# }

# html_sidebars = {"**": ["about.html", "localtoc.html", "searchbox.html"]}

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["img", "styles"]

# These paths are either relative to html_static_path
# or fully qualified paths (eg. https://...)
html_css_files = []

# The name of an image file (relative to this directory) to place at the top
# of the sidebar.
html_logo = "img/ai.png"

# The name of an image file (within the static path) to use as favicon of the
# docs.  This file should be a Windows icon file (.ico) being 16x16 or 32x32
# pixels large.
html_favicon = "img/favicon.ico"

# If true, "Created using Sphinx" is shown in the HTML footer. Default is True.
html_show_sphinx = False

# If true, "(C) Copyright ..." is shown in the HTML footer. Default is True.
html_show_copyright = True

intersphinx_mapping = {"phython3": ("https://docs.python.org/3", None)}

todo_include_todos = True
