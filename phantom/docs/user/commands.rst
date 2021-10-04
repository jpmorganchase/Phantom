
.. _commands:

Commands
========

The following commands will be installed to your local path when the Phantom package is
installed.


phantom-train
-------------

Runs RL training for an experiment defined in the given Phantom experiment config file.

The config file should contain an instance of the ``TrainingParams`` class in the global
scope named ``training_params``.

Usage:

.. code-block:: bash

    phantom-train <config_file>



phantom-rollout
---------------

Performs rollouts for an already trained experiment.

The config file should contain an instance of the ``RolloutParams`` class in the global
scope named ``rollout_params``.

Usage:

.. code-block:: bash

    phantom-rollout <config_file>


phantom-profile
---------------

Runs RL training for an experiment and profiles the execution.

The config file should contain an instance of the ``TrainingParams`` class in the global
scope named ``training_params``.

Usage:

.. code-block:: bash

    phantom-profile <config_file>
