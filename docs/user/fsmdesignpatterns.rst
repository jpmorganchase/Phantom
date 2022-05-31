###############################
FSM Environment Design Patterns
###############################


This page describes the various ways shared policies can be implemented in complex
Finite State Machine based environments with policies shared both across agents and
across stages. Included are simple code examples showing how the different combinations
should be implemented.


Standard Environment - No FSM
-----------------------------

When an FSM environment is not used shared policies across agents should be defined
using the ``policy_mapping`` parameter in the Phantom ``train`` function. This should be
a mapping of policy names to lists of agent IDs. The policy names cannot contain the
characters ``:`` and ``/``.

.. figure:: /img/policy-grid-1.svg
   :width: 70%
   :figclass: align-center

|

Implementing the environment and agents:

.. code-block:: python

    class ExampleAgent(ph.Agent):
        """Implementation details omitted for clarity"""


    class ExampleEnv(ph.PhantomEnv):
        def __init__(self):
            agents = [
                ExampleAgent("Agent1"),
                ExampleAgent("Agent2"),
                ExampleAgent("Agent3"),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(network=network, n_steps=100)

Defining the shared policies:

.. code-block:: python

    ph.train(
        ...
        policy_mapping={"shared_policy": ["Agent1", "Agent2"]},
        ...
    )


FSM Environment, No Shared Policies
-----------------------------------

The following describes an FSM based environment with 3 agents and 4 stages. All the
agents take actions in each stage with a separate policy. This gives 12 policies in
total.

.. figure:: /img/policy-grid-2.svg
   :width: 70%
   :figclass: align-center

|

In this environment the stage transitions are deterministic, so we do not need to define
an environment stage handler function (that would compute the next stage choice). For
each stage for each agent it is important that we initialise a new ``StageHandler``
object.

.. code-block:: python

    class ExampleStageHandler(ph.fsm.StagePolicyHandler):
        """Implementation details omitted for clarity"""

    class ExampleFSMAgent(ph.fsm.Agent):
        """Implementation details omitted for clarity"""

    class OddEvenFSMEnv(ph.fsm.FiniteStateMachineEnv):
        def __init__(self):
            agents = [
                ExampleFSMAgent("Agent1", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent2", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent3", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=100,
                initial_stage="Stage1",
                stages=[
                    ph.fsm.FSMStage(stage_id="Stage1", next_stages=["Stage2"]),
                    ph.fsm.FSMStage(stage_id="Stage2", next_stages=["Stage3"]),
                    ph.fsm.FSMStage(stage_id="Stage3", next_stages=["Stage4"]),
                ]
            )


FSM Environment, Policy Shared Across Stages
--------------------------------------------

In this example we have a shared stage policy across multiple stages of one agent.

.. figure:: /img/policy-grid-3.svg
   :width: 70%
   :figclass: align-center

|

For the agent with the shared policy we define one ``StageHandler`` object and reference
it multiple times in the ``stage_handlers`` parameter when initialising the agent.
Phantom will then treat these multiple references as intent to use the policy on the
stage handler as a shared stage policy across the multiple stages. It is possible to do
this as stage handlers do not locally store state.

.. code-block:: python

    class OddEvenFSMEnv(ph.fsm.FiniteStateMachineEnv):
        def __init__(self):
            shared_policy_stage_handler = ExampleStageHandler()

            agents = [
                ExampleFSMAgent("Agent1", stage_handlers={
                    "Stage1": shared_policy_stage_handler,
                    "Stage2": shared_policy_stage_handler,
                    "Stage3": shared_policy_stage_handler,
                }),
                ExampleFSMAgent("Agent2", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent3", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=100,
                initial_stage="Stage1",
                stages=[
                    ph.fsm.FSMStage(stage_id="Stage1", next_stages=["Stage2"]),
                    ph.fsm.FSMStage(stage_id="Stage2", next_stages=["Stage3"]),
                    ph.fsm.FSMStage(stage_id="Stage3", next_stages=["Stage4"]),
                ]
            )


FSM Environment, Policy Shared Across Agents
--------------------------------------------

Shared stage policies can also be defined in the same manner across multiple agents.

.. figure:: /img/policy-grid-4.svg
   :width: 70%
   :figclass: align-center

|

.. code-block:: python

    class OddEvenFSMEnv(ph.fsm.FiniteStateMachineEnv):
        def __init__(self):
            shared_policy_stage_handler = ExampleStageHandler()

            agents = [
                ExampleFSMAgent("Agent1", stage_handlers={
                    "Stage1": shared_policy_stage_handler,
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent2", stage_handlers={
                    "Stage1": shared_policy_stage_handler,
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent3", stage_handlers={
                    "Stage1": shared_policy_stage_handler,
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=100,
                initial_stage="Stage1",
                stages=[
                    ph.fsm.FSMStage(stage_id="Stage1", next_stages=["Stage2"]),
                    ph.fsm.FSMStage(stage_id="Stage2", next_stages=["Stage3"]),
                    ph.fsm.FSMStage(stage_id="Stage3", next_stages=["Stage4"]),
                ]
            )

FSM Environment, Multiple Shared Policies
-----------------------------------------

Finally, multiple shared stage policies can be defined in this manner. A combination of
multiple agents and multiple stages can be spanned by the same shared stage policy.

.. figure:: /img/policy-grid-5.svg
   :width: 70%
   :figclass: align-center

|

.. code-block:: python

    class OddEvenFSMEnv(ph.fsm.FiniteStateMachineEnv):
        def __init__(self):
            shared_policy_stage_handler_1 = ExampleStageHandler()
            shared_policy_stage_handler_2 = ExampleStageHandler()

            agents = [
                ExampleFSMAgent("Agent1", stage_handlers={
                    "Stage1": shared_policy_stage_handler_1,
                    "Stage2": ExampleStageHandler(),
                    "Stage3": ExampleStageHandler(),
                }),
                ExampleFSMAgent("Agent2", stage_handlers={
                    "Stage1": shared_policy_stage_handler_1,
                    "Stage2": shared_policy_stage_handler_2,
                    "Stage3": shared_policy_stage_handler_2,
                }),
                ExampleFSMAgent("Agent3", stage_handlers={
                    "Stage1": ExampleStageHandler(),
                    "Stage2": shared_policy_stage_handler_2,
                    "Stage3": shared_policy_stage_handler_2,
                }),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=100,
                initial_stage="Stage1",
                stages=[
                    ph.fsm.FSMStage(stage_id="Stage1", next_stages=["Stage2"]),
                    ph.fsm.FSMStage(stage_id="Stage2", next_stages=["Stage3"]),
                    ph.fsm.FSMStage(stage_id="Stage3", next_stages=["Stage4"]),
                ]
            )
