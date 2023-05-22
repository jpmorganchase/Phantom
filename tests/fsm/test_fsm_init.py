import phantom as ph

from .. import MockAgent


def test_decorator_style():
    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockAgent("A")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
            )

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["A"],
            next_stages=["StageA"],
        )
        def handle(self):
            return "StageA"

    env = Env()
    env.reset()
    env.step({})


def test_state_definition_list_style():
    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockAgent("A")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
                stages=[
                    ph.FSMStage(
                        stage_id="StageA",
                        acting_agents=["A"],
                        next_stages=["StageA"],
                        handler=self.handle,
                    )
                ],
            )

        def handle(self):
            return "StageA"

    env = Env()
    env.reset()
    env.step({})
