=============================
Finite State Machine Handlers
=============================

Two stage handler classes are provided to build complex FSM environment functionality.
So that instances of these classes can be shared across actors and agents, agent/actor
state should not be stored in these classes. Instead the associated actor/agent is
passed to the hooks when called and the actor/agent internal state can be accessed.

The ``StageHandler`` class provides opt-in hooks that are called at various events in
the step process. This can be used by Actors and also Agents (when for the attached
stage the agent does not take an action). 

The ``StagePolicyHandler`` class extends the ``StageHandler`` class and provides
addtional interfaces for defining a policy for the stage. The functions required closely
mirror that of a standard Phantom Agent (ie. not an ``FSMAgent``).


Stage Handler 
=============

.. autoclass:: phantom.fsm.handlers.StageHandler
   :inherited-members:


Stage Policy Handler
====================

.. autoclass:: phantom.fsm.handlers.StagePolicyHandler
   :inherited-members:
