.. _installation:

Installing Phantom
==================

This part of the documentation covers the installation of Phantom.
The first step to using any software package is getting it properly installed.


Prerequisites
-------------

The main requirements for running Phantom are a modern Python installation
(3.7 minimum) and access to the pip Python package manager.

A list of Python packages required by Phantom is given in the ``requirements.txt``
file. The packages can be installed by running the following command from the Phantom
root directory::

    make install-deps


Get the Source Code
-------------------

Phantom is actively developed at `GitHub <https://github.com/jpmorganchase/Phantom>`_.

You can clone the public repository using the command::

    $ git clone https://github.com/jpmorganchase/Phantom

Once you have a copy of the source, you can install it however you like using
setup.py. For convenience, we provide a Makefile with some helpers for
installing a local development instance (recommended)::

    $ cd phantom
    $ make dev
