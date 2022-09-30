.. _api_rewards:

Rewards
=======

RewardFunctions classes are used to convert observations of an agents state into a
numerical value that is used to inform the policy of the effectiveness of its actions so
as to allow the policy to learn and improve future policy decisions.

.. figure:: /img/reward-function.svg
   :figclass: align-center

RewardFunction classes are a fully optional feature of Phantom. There is no functional
difference between defining an :meth:`compute_reward()` method on an Agent and defining
a :class:`RewardFunction` (whose :meth:`reward()` method performs the same actions) and
attaching it to the Agent.


Base RewardFunction
-------------------

.. autoclass:: phantom.reward_functions.RewardFunction
   :inherited-members:


Provided Implementations
------------------------

.. autoclass:: phantom.reward_functions.Constant
   :inherited-members:
