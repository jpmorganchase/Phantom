.. _quickstart:

Quickstart
==========

If you have not already installed Phantom, please see the :ref:`installation` page.

With Phantom installed you can run the provided ``Monopoly`` sample experiment
with the command:

.. code-block:: bash

    phantom-train examples/monopoly.py


Change the script for any of the other provided experiments in the examples directory.


Workflow
--------

Phantom defines an opinionated set of interfaces and tools that make it very easy
to quickly get off the ground running simulations and experiments.

Phantom provides several :ref:`Commands` that hide many of the details of
experiment setup and allow rapid development and experimentation. These are used
in conjunction with a recommended :ref:`configfile`.

The flow-chart below summarises an typical simple workflow using the provided
commands and config file layout:

.. figure:: /img/overall-flow.svg
   :width: 50%
   :figclass: align-center
