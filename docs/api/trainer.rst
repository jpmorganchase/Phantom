.. _api_trainers:

Trainers
========

Phantom provides a simple Trainer interface and class for developing and implementing
new learning algorithms. Two example implementations are provided: Q-Learning and
Proximal Policy Optimisation (PPO).

Note: This is a new feature in Phantom and is subject to change in the future.

The two implementations should be seen as examples of how to implement a Trainer rather
than fully tested and optimised trainers to use. It is recommended that RLlib other more
mature implementations are used in practice. The example implementations can be found
in the ``examples/trainers`` directory.

Base Trainer
------------

.. autoclass:: phantom.trainer.Trainer
   :inherited-members:

.. autoclass:: phantom.trainer.TrainingResults
   :inherited-members:

