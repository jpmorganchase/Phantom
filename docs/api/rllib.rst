.. _api_rllib:

RLlib Utilities
===============

The following tools are for training and evaluating policies with RLlib. The tools take
care of a lot of boilerplate tasks such as finding the newest results directories and
checkpoints and also more Phantom specific tasks such as populating supertypes with 
Samplers and Ranges.


Training
--------

.. autofunction:: phantom.utils.rllib.train


Rollouts
--------

.. autofunction:: phantom.utils.rllib.rollout


Policy Evaluation
-----------------

.. autofunction:: phantom.utils.rllib.evaluate_policy



Rollout Trajectories & Steps
----------------------------

.. autoclass:: phantom.utils.rollout.AgentStep
   :members:

.. autoclass:: phantom.utils.rollout.Step
   :members:

.. autoclass:: phantom.utils.rollout.Rollout
   :members:
