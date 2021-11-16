from enum import Enum


import mercury as me
import pytest
from phantom.fsm_env import (
    FSMStage,
    FiniteStateMachineEnv,
    FSMRuntimeError,
    FSMValidationError,
)

from . import MinimalAgent


class States(Enum):
    A = 1
    B = 2


def test_no_stages_registered():
    """
    All FSM envs must have at least one state registered using the FSMStage decorator.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

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
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=States.A,
            )

        @FSMStage(
            stage_id=States.A,
            next_stages=[States.A],
        )
        def handle_1(self):
            pass

        @FSMStage(
            stage_id=States.A,
            next_stages=[States.A],
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
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=States.B,
            )

        @FSMStage(
            stage_id=States.A,
            next_stages=[States.A],
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
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=States.A,
            )

        @FSMStage(
            stage_id=States.A,
            next_stages=[States.B],
        )
        def handle_1(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_next_state_runtime():
    """
    A valid registered next state must be returned by the state handler functions.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_stage=States.A,
            )

        @FSMStage(
            stage_id=States.A,
            next_stages=[States.A],
        )
        def handle_1(self):
            return States.B

    env = Env()
    env.reset()

    with pytest.raises(FSMRuntimeError):
        env.step(actions={"agent": 0})
