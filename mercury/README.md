# Mercury

## Overview

This library contains the core core mechanisms for simulating P2P messaging
networks. It is all built on `networkx` primitives and enforces strict
observability constraints as a first-class feature.

## Installation

The main requirements for installing Mercury are a modern Python installation
(3.7 minimum) and access to the pip Python package manager.

A list of Python packages required by Mercury is given in the `requirements.txt` file.
The required packages can be installed by running:

```sh
make install_deps
```

To use this library, clone the repo, install the requirements and install the package.
For example, one can install a local development version for rapid use,

```sh
make install
```

## Development Setup

```sh
make install-dev-deps
make dev
```

