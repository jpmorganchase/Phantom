.. _api_env_wrappers:

Env Wrappers
============

SingleAgentEnvAdapter
---------------------

.. figure:: /img/single-env-adapter.svg
   :figclass: align-center

In the diagram above we have 4 agents/policies. Agent "A" is the selected agent and the
:class:`SingleAgentEnvAdapter` will expose an environment as seen from the perspective
of just that agent. The other agents, with pre-defined policies, will have their actions
handled internally by the wrapper.

.. autoclass:: phantom.env_wrappers.SingleAgentEnvAdapter
   :inherited-members:

