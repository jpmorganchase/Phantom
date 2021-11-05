.. _api_agents:

Agents
======

There are three main elements that must be provided for any Agent class instance:

- Functionality that encodes observations.
- Functionality that decodes actions.
- Functionality that computes rewards.

Each of these parts can be provided in two ways:

- By setting an Encoder/Decoder/RewardFunction object to this class's
   ``self.obs_encoder``/``self.action_decoder``/``self.reward_function`` properties.
- By subclassing this class and providing a custom implementation to this class's
   ``encode_obs()``/``decode_action()``/``compute_reward()`` methods.

These two options can be mixed between the three elements.

If the ``encode_obs()`` method is provided with a custom implementation, the
``get_observation_space()`` method must also be as well.

If the ``decode_action()`` method is provided with a custom implementation, the
``get_action_space()`` method must also be as well.

.. figure:: /img/agent.svg
   :figclass: align-center


Agent
-----

.. autoclass:: phantom.agent.Agent
   :inherited-members:


ZeroIntelligenceAgent
---------------------

.. autoclass:: phantom.agent.ZeroIntelligenceAgent
   :inherited-members:
