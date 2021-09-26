
.. _commands:

Commands
========

The following commands will be installed to your local path when the Phantom package is
installed.


phantom-train
-------------

Runs RL training for an experiment defined in the given Phantom experiment config file.

Usage:

.. code-block:: bash

    phantom-train <config_file>



phantom-rollout
---------------

Performs rollouts for an already trained experiment.

Usage:

.. code-block:: bash

    phantom-rollout <config_file> <results_directory>


Additional Arguments:

- ``--num-workers NUM_WORKERS`` -- How many Ray workers to use.
- ``--num-rollouts NUM_ROLLOUTS`` -- How many steps are in a training batch.
- ``--checkpoint-num CHECKPOINT_NUM`` -- The number of the checkpoint to use.
- ``--metrics-file METRICS_FILE`` -- The pickle file to save the rollout results to.


phantom-profile
---------------

Runs RL training for an experiment and profiles the execution.

Usage:

.. code-block:: bash

    phantom-profile <config_file>
