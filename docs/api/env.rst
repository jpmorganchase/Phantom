.. _api_env:

Environment
===========

PhantomEnv
----------

This is the Phantom environment class that should be subclassed from when defining new
environments.

This class generally follows the RLlib :class:`MultiAgentEnv` class interface (However
not exactly. When using RLlib for training, a wrapper env will be used to provide full
compatibility).

.. autoclass:: phantom.PhantomEnv
   :inherited-members:


Step
----

.. autoclass:: phantom.PhantomEnv.Step
   :members:
