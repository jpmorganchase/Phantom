import mercury as me
from phantom.fsm_env import (
    FSMState,
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
                initial_state="A",
            )

        @FSMState(
            state_id="A",
            next_states=["A"],
        )
        def handle(self):
            pass

    Env()


def test_state_definition_list_style():
    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_state="A",
                state_definitions=[
                    FSMState(
                        state_id="A",
                        next_states=["A"],
                        handler=self.handle,
                    )
                ],
            )

        def handle(self):
            pass

    Env()
