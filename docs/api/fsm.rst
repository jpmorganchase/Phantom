.. _api_fsm:

Finite State Machine Environment
================================

The :class:`FiniteStateMachineEnv` class maps states in a finite state machine to
functions that handle the logic of the state. At the end of each state agents take
observations and at the start of the next step the agents provide actions based on the
observations and their respective policies.

It is possible to restrict which agents take actions and compute rewards for each state
with the :attr:`acting_agents` and :attr:`rewarded_agents` properties of the
:class:`FSMStage` class.

In each handler method the user must take care to call :func:`self.network.resolve()`.
This is left to the user as to allow full flexibility on both when the messages on the
network are resolved and also, in advanced cases, which resolve method is called.

There are two methods to define the finite state machine structure. It is possible to
use a mix of both methods. The following two examples are equivalent.

The first uses the :class:`FSMStage` as a decorator directly on the state handler method:

.. code-block:: python

   class CustomEnv(ph.FiniteStateMachineEnv):
      def __init__(self):
         agents = [MinimalAgent("agent")]

         network = ph.Network(agents)

         super().__init__(num_steps=10, network=network, initial_stage="A")

      @ph.FSMStage(stage_id="A", next_stages=["A"])
      def handle(self):
         # Perform any pre-resolve tasks
         self.resolve_network()
         # Perform any post-resolve tasks


The second defines the states via a list of :class:`FSMStage` instances passed to the
:class:`FiniteStateMachineEnv` init method. This method is needed when values of
parameters passed to the :class:`FSMStage` initialisers are only known when the
environment class is initialised (eg. lists of agent IDs).

.. code-block:: python

   class CustomEnv(ph.FiniteStateMachineEnv):
      def __init__(self):
         agents = [MinimalAgent("agent")]

         network = ph.Network(agents)

         super().__init__(
               num_steps=10,
               network=network,
               initial_stage="A",
               stages=[
                  ph.FSMStage(
                     stage_id="A",
                     next_stages=["A"],
                     handler=self.handle,
                  )
               ],
         )

      def handle(self):
         # Perform any pre-resolve tasks
         self.resolve_network()
         # Perform any post-resolve tasks

Environment
===========

.. autoclass:: phantom.fsm.FiniteStateMachineEnv
   :inherited-members:


.. autoclass:: phantom.fsm.FSMEnvView
   :inherited-members:


Stages
======

.. autoclass:: phantom.fsm.FSMStage
   :inherited-members:


Errors
======

.. autoclass:: phantom.fsm.FSMValidationError
   :inherited-members:


.. autoclass:: phantom.fsm.FSMRuntimeError
   :inherited-members:
