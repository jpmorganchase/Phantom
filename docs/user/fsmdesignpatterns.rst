################################################
Finite State Machine Environment Design Patterns
################################################

This page describes the various ways shared policies can be implemented in complex
Finite State Machine (FSM) based environments with policies shared both across agents
and across stages. Included are simple code examples showing how some basic different
combinations should be implemented.


Standard Environment - No FSM
-----------------------------

In both FSM and non-FSM Phantom environments policies are mapped to agents using the
:attr:`policies` parameter in :meth:`Trainer.train` functions or the
:func:`utils.rllib.train/rollout` functions. A single policy can be used by multiple
agents.

.. figure:: /img/policy-grid-1.svg
   :width: 70%
   :figclass: align-center

|

Implementing the environment and agents:

.. code-block:: python

    class ExampleAgent(ph.Agent):
        ...


    class ExampleEnv(ph.PhantomEnv):
        def __init__(self):
            agents = [
                ExampleAgent("Agent 1"),
                ExampleAgent("Agent 2"),
                ExampleAgent("Agent 3"),
            ]

            network = ph.Network(agents)

            super().__init__(num_steps=100, network=network)

Defining the left side example (no shared policies):

.. code-block:: python

    trainer.train(
        ...
        policies={
            "policy_a": ["Agent 1"],
            "policy_b": ["Agent 2"],
            "policy_c": ["Agent 3"],
        },
        ...
    )


Defining the right side example (shared policies):

.. code-block:: python

    trainer.train(
        ...
        policies={
            "shared_policy": ["Agent 1", "Agent 2"],
            "other_policy": ["Agent 3"],
        },
        ...
    )


FSM Environment, No Shared Policies
-----------------------------------

The following describes an FSM based environment with 3 agents and 3 stages. An agent
can only be assigned a single policy. In this example each agent acts in one stage each.

.. figure:: /img/policy-grid-2.svg
   :width: 70%
   :figclass: align-center

|

In this environment the stage transitions are deterministic (the stages loop around,
starting with the first stage), so we do not need to define an environment stage handler
function (that would compute the choice of next stage).

.. code-block:: python

    class ExampleAgent(ph.Agent):
        ...


    class ExampleFSMEnv1(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [
                ExampleAgent("Agent 1"),
                ExampleAgent("Agent 2"),
                ExampleAgent("Agent 3"),
            ]

            network = ph.Network(agents)

            stages = [
                ph.FSMStage(
                    stage_id="Stage 1",
                    next_stages=["Stage 2"],
                    acting_agents=["Agent 1"],
                ),
                ph.FSMStage(
                    stage_id="Stage 2",
                    next_stages=["Stage 3"],
                    acting_agents=["Agent 2"],
                ),
                ph.FSMStage(
                    stage_id="Stage 3",
                    next_stages=["Stage 1"],
                    acting_agents=["Agent 3"],
                ),
            ]

            super().__init__(
                num_steps=100,
                network=network,
                initial_stage="Stage 1",
                stages=stages,
            )


FSM Environment, Policy Shared Across Agents
--------------------------------------------

In this example we have a shared policy across two agents. This works in the same way as
a shared policy in the standard PhantomEnv. It is also possible for the shared policy to
be shared across multiple stages.

.. figure:: /img/policy-grid-3.svg
   :width: 70%
   :figclass: align-center

|

.. code-block:: python

    class ExampleFSMEnv2(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [
                ExampleFSMAgent("Agent 1"),
                ExampleFSMAgent("Agent 2"),
                ExampleFSMAgent("Agent 3"),
            ]

            network = ph.Network(agents)

            stages = [
                ph.FSMStage(
                    stage_id="Stage 1",
                    next_stages=["Stage 2"],
                    acting_agents=["Agent 1"],
                ),
                ph.FSMStage(
                    stage_id="Stage 2",
                    next_stages=["Stage 1"],
                    acting_agents=["Agent 2", "Agent 3"],
                ),
            ]

            super().__init__(
                num_steps=100,
                network=network,
                initial_stage="Stage 1",
                stages=stages,
            )

    trainer.train(
        ...
        policies={
            "shared_policy": ["Agent 2", "Agent 3"],
            "other_policy": ["Agent 1"],
        },
        ...
    )


FSM Environment, Policy Shared Across Stages
--------------------------------------------

In this example we have an agent that takes an action in multiple stages. The agent uses
the same policy (with the same observation and action spaces) in both stages.


.. figure:: /img/policy-grid-4.svg
   :width: 70%
   :figclass: align-center

|

.. code-block:: python

    class ExampleFSMEnv3(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [
                ExampleFSMAgent("Agent 1"),
                ExampleFSMAgent("Agent 2"),
                ExampleFSMAgent("Agent 3"),
            ]

            network = ph.Network(agents)

            stages = [
                ph.FSMStage(
                    stage_id="Stage 1",
                    next_stages=["Stage 2"],
                    acting_agents=["Agent 1", "Agent 2"],
                ),
                ph.FSMStage(
                    stage_id="Stage 2",
                    next_stages=["Stage 3"],
                    acting_agents=["Agent 1", "Agent 2"],
                ),
                ph.FSMStage(
                    stage_id="Stage 3",
                    next_stages=["Stage 1"],
                    acting_agents=["Agent 1", "Agent 3"],
                ),
            ]

            super().__init__(
                num_steps=100,
                network=network,
                initial_stage="Stage 1",
                stages=stages,
            )

    trainer.train(
        ...
        policies={
            "policy_1": ["Agent 1"],
            "policy_2": ["Agent 2"],
            "policy_3": ["Agent 3"],
        },
        ...
    )
