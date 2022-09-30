.. _api_trainers:

Trainers
========

Phantom provides a simple Trainer interface and class for developing and implementing
new learning algorithms. Two example implementations are provided: Q-Learning and
Proximal Policy Optimisation (PPO).

Note: This is a new feature in Phantom and is subject to change in the future.

.. TODO: link to examples

Base Trainer
------------

.. autoclass:: phantom.trainers.trainer.Trainer
   :inherited-members:

.. autoclass:: phantom.trainers.trainer.TrainingResults
   :inherited-members:

