import phantom as ph
import pytest

from .. import MockStrategicAgent


def test_no_stages_registered():
    """
    All FSM envs must have at least one state registered using the FSMStage decorator.
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage=None,
            )

    with pytest.raises(ph.fsm.FSMValidationError):
        Env()


def test_duplicate_stages():
    """
    All FSM envs must not have more than one state registered with the same ID.
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
            )

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["agent"],
            next_stages=["StageA"],
        )
        def handle_1(self):
            pass

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["agent"],
            next_stages=["StageA"],
        )
        def handle_2(self):
            pass

    with pytest.raises(ph.fsm.FSMValidationError):
        Env()


def test_invalid_initial_stage():
    """
    All FSM envs must have an initial state that is a valid registered state.
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageB",
            )

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["agent"],
            next_stages=["StageA"],
        )
        def handle(self):
            pass

    with pytest.raises(ph.fsm.FSMValidationError):
        Env()


def test_invalid_next_state():
    """
    All next states passed into the FSMStage decorator must be valid states.
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
            )

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["agent"],
            next_stages=["StageB"],
        )
        def handle_1(self):
            pass

    with pytest.raises(ph.fsm.FSMValidationError):
        Env()


def test_invalid_no_handler_stage_next_stages():
    """
    All stages without a provided handler must have exactly one next stage
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
                stages=[ph.FSMStage("StageA", acting_agents=["agent"], next_stages=[])],
            )

    with pytest.raises(ph.fsm.FSMValidationError):
        Env()


def test_invalid_next_state_runtime():
    """
    A valid registered next state must be returned by the state handler functions.
    """

    class Env(ph.FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockStrategicAgent("agent")]

            network = ph.Network(agents)

            super().__init__(
                num_steps=1,
                network=network,
                initial_stage="StageA",
            )

        @ph.FSMStage(
            stage_id="StageA",
            acting_agents=["agent"],
            next_stages=["StageA"],
        )
        def handle_1(self):
            return "StageB"

    env = Env()
    env.reset()

    with pytest.raises(ph.fsm.FSMRuntimeError):
        env.step(actions={"agent": 0})
