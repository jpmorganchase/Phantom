.. _install:

Installing Mercury-Core
=======================

This part of the documentation covers the installation of Requests.
The first step to using any software package is getting it properly installed.


Get the Source Code
-------------------

Mercury-Core is actively developed on CodeCommit, where the code is `available
<https://us-east-2.console.aws.amazon.com/codesuite/codecommit/repositories/mercury/browse>`_
internally to JP Morgan employees.

You can clone the public repository using the command::

    $ git clone https://<AWS-CC>/v1/repos/mercury-core

Once you have a copy of the source, you can install it however you like using
setup.py. For convenience, we provide a Makefile with some helpers for
installing a local development instance (recommended)::

    $ cd phantom
    $ make dev
