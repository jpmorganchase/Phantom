from enum import Enum

import mercury as me
from phantom.fsm import (
    FSMStage,
    FiniteStateMachineEnv,
)

from . import MockFSMAgent, MockStageHandler


class Stages(Enum):
    A = 1


def test_decorator_style():
    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockFSMAgent("agent", {Stages.A: MockStageHandler()})]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=Stages.A,
            )

        @FSMStage(
            stage_id=Stages.A,
            next_stages=[Stages.A],
        )
        def handle(self):
            return Stages.A

    env = Env()
    env.reset()
    env.step({})


def test_state_definition_list_style():
    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockFSMAgent("agent", {Stages.A: MockStageHandler()})]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=Stages.A,
                stages=[
                    FSMStage(
                        stage_id=Stages.A,
                        next_stages=[Stages.A],
                        handler=self.handle,
                    )
                ],
            )

        def handle(self):
            return Stages.A

    env = Env()
    env.reset()
    env.step({})
