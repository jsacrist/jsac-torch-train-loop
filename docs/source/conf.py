import inspect
import os
import sys

CURDIR = os.path.dirname(inspect.getfile(inspect.currentframe()))
CODEDIR = os.path.realpath(os.path.join(CURDIR, "../../"))
sys.path.insert(0, CODEDIR)
from jsac import torch_train_loop

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "torch_train_loop"
copyright = "2023, Jorge Sacristan"
author = "Jorge Sacristan"
release = torch_train_loop.__version__

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
html_static_path = ["_static"]
