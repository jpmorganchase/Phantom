.. _api_agents:

Agents
======

Phantom has one base agent class, :class:`Agent`. One extra class,
:class:`MessageHandlerAgent`, builds on this by adding some additional message handling
functionality. All user implemented agent classes should be derived from one of these
two classes.

Depending on which methods are implemented / properties set, agents can have differing
levels of functionality:

The most basic type of agent is one that only responds to messages sent from other agents.
This agent does not take actions by itself and as such does not have a defined Policy or
reward function.

To implement this agent type the :meth:`handle_message` method must be implemented if
deriving from the :class:`Agent` class or at least one message handler method must be
implemented if deriving from the :class:`MessageHandlerAgent` class.

If the :meth:`generate_messages` method is implemented then the agent can now also create
messages.

Finally there is a full agent that has a Policy, reward function, observation encoder
and action decoder. The :meth:`generate_messages` should not be implemented here as the
action decoder provides the message generation functionality.

Each of the reward function, observation encoder and action decoder components can be
provided in two ways:

- By setting an Encoder/Decoder/RewardFunction object to this class's
   :attr:`observation_encoder` / :attr:`action_decoder` / :attr:`reward_function` properties.
- By subclassing this class and providing a custom implementation to this class's
   :meth:`encode_observation` / :meth:`decode_action` / :meth:`compute_reward` methods.

These two options can be mixed between the three elements.

If the :meth:`encode_observation` method is provided with a custom implementation, the
:attr:`observation_space` property of the agent must be set. Otherwise the :class:`Encoder`
will provide the observation space.

If the :meth:`decode_action` method is provided with a custom implementation, the
:attr:`action_space` property of the agent must be set. Otherwise the :class:`Decoder`
will provide the action space.

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
