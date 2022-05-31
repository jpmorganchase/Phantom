.. _api_encoders:

Encoders
========

Encoders are used to translate the properties of the environment as seen through
an Agent's context view into that environment into an observation that the agent
is able to train on:

.. figure:: /img/encoder.svg
   :figclass: align-center


Encoders are a fully optional feature of Phantom. There is no functional difference
between defining an ``encode_obs()`` method on an Agent and defining an :class:`Encoder`
(whose ``encode()`` method performs the same actions) and attaching it to the Agent.

Encoders are useful when many different Agents want to encode the same properties
and code re-use is desirable. It is also possible to compose multiple encoders
(see :class:`ChainedEncoder`), allowing the construction of complex encoders from
many smaller parts.


Base Encoder
------------

.. autoclass:: phantom.encoders.Encoder
   :inherited-members:


Provided Implementations
------------------------

.. autoclass:: phantom.encoders.EmptyEncoder
   :inherited-members:

.. autoclass:: phantom.encoders.ChainedEncoder
   :inherited-members:

.. autoclass:: phantom.encoders.Constant
   :inherited-members:
