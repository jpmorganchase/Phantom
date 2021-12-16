from enum import Enum


import mercury as me
import pytest
from phantom.fsm import (
    FSMStage,
    FiniteStateMachineEnv,
    FSMRuntimeError,
    FSMValidationError,
)

from . import MockFSMAgent, MockStageHandler


class Stages(Enum):
    A = 1
    B = 2


def test_no_stages_registered():
    """
    All FSM envs must have at least one state registered using the FSMStage decorator.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockFSMAgent("agent", {})]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=None,
            )

    with pytest.raises(FSMValidationError):
        Env()


def test_duplicate_stages():
    """
    All FSM envs must not have more than one state registered with the same ID.
    """

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
        def handle_1(self):
            pass

        @FSMStage(
            stage_id=Stages.A,
            next_stages=[Stages.A],
        )
        def handle_2(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_initial_stage():
    """
    All FSM envs must have an initial state that is a valid registered state.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockFSMAgent("agent", {Stages.A: MockStageHandler()})]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=Stages.B,
            )

        @FSMStage(
            stage_id=Stages.A,
            next_stages=[Stages.A],
        )
        def handle(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_next_state():
    """
    All next states passed into the FSMStage decorator must be valid states.
    """

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
            next_stages=[Stages.B],
        )
        def handle_1(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_no_handler_stage_next_stages():
    """
    All stages without a provided handler must have exactly one next stage
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MockFSMAgent("agent", {Stages.A: MockStageHandler()})]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=Stages.A,
                stages=[FSMStage(Stages.A, next_stages=[])],
            )

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_next_state_runtime():
    """
    A valid registered next state must be returned by the state handler functions.
    """

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
        def handle_1(self):
            return Stages.B

    env = Env()
    env.reset()

    with pytest.raises(FSMRuntimeError):
        env.step(actions={"agent__Stages.A": 0})
