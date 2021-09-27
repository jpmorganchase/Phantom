.. _api_decoders:

Decoders
========


Decoders are used to translate actions received from the agent's policy into messages
that are sent in the network to other actors and mutators that can update the agent'same
own state:

.. figure:: /img/decoder.svg
   :figclass: align-center


Decoders are a fully optional feature of Phantom. There is no functional difference
between defining an ``decode_action()`` method on an Agent and defining an :class:`Decoder`
(whose ``decode()`` method performs the same actions) and attaching it to the Agent.

Decoders are useful when many different Agents want to decode the same properties
and code re-use is desirable. It is also possible to compose multiple decoders
(see :class:`ChainedDecoder`), allowing the construction of complex decoders from
many smaller parts.


Base Decoder
------------

.. autoclass:: phantom.decoders.Decoder
   :inherited-members:


Provided Implementations
------------------------

.. autoclass:: phantom.decoders.EmptyDecoder
   :inherited-members:

.. autoclass:: phantom.decoders.ChainedDecoder
   :inherited-members:


Packet
------

.. autoclass:: phantom.packet.Packet
   :inherited-members:
