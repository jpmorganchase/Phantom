.. _api_agents:

Agents
======

TODO: rewrite, mention differences between agent types

There are three main elements that must be provided for any Agent class instance:

- Functionality that encodes observations.
- Functionality that decodes actions.
- Functionality that computes rewards.

Each of these parts can be provided in two ways:

- By setting an Encoder/Decoder/RewardFunction object to this class's
   ``self.observation_encoder``/``self.action_decoder``/``self.reward_function`` properties.
- By subclassing this class and providing a custom implementation to this class's
   ``encode_observation()``/``decode_action()``/``compute_reward()`` methods.

These two options can be mixed between the three elements.

If the ``encode_observation()`` method is provided with a custom implementation, the
``observation_space`` property of the agent must be set.

If the ``decode_action()`` method is provided with a custom implementation, the
``action_space`` property of the agent must be set.

.. figure:: /img/agent.svg
   :figclass: align-center

Agent
-----

.. autoclass:: phantom.agents.Agent
   :inherited-members:


MessageHandlerAgent
-------------------

.. autoclass:: phantom.agents.MessageHandlerAgent
   :inherited-members:
