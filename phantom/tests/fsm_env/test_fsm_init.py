import mercury as me
from phantom.fsm_env import (
    FSMStage,
    FiniteStateMachineEnv,
)

from . import MinimalAgent


def test_decorator_style():
    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage="A",
            )

        @FSMStage(
            stage_id="A",
            next_stages=["A"],
        )
        def handle(self):
            return "A"

    env = Env()
    env.reset()
    env.step({})


def test_state_definition_list_style():
    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage="A",
                stages=[
                    FSMStage(
                        stage_id="A",
                        next_stages=["A"],
                        handler=self.handle,
                    )
                ],
            )

        def handle(self):
            return "A"

    env = Env()
    env.reset()
    env.step({})
