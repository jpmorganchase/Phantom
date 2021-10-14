from enum import Enum


import gym
import mercury as me
import numpy as np
import phantom as ph
import pytest
from phantom.fsm_env import (
    fsm_state,
    FiniteStateMachineEnv,
    FSMRuntimeError,
    FSMValidationError,
)


class MinimalAgent(ph.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(agent_id=id)

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([0])

    def decode_action(self, ctx: me.Network.Context, action) -> ph.Packet:
        return ph.Packet()

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class States(Enum):
    A = 1
    B = 2


def test_no_states_registered():
    """
    All FSM envs must have at least one state registered using the fsm_state decorator.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_state=None,
            )

    with pytest.raises(FSMValidationError):
        Env()


def test_duplicate_states():
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
                initial_state=States.A,
            )

        @fsm_state(
            state_id=States.A,
            next_states=[States.A],
        )
        def handle_1(self):
            pass

        @fsm_state(
            state_id=States.A,
            next_states=[States.A],
        )
        def handle_2(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_initial_state():
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
                initial_state=States.B,
            )

        @fsm_state(
            state_id=States.A,
            next_states=[States.A],
        )
        def handle(self):
            pass

    with pytest.raises(FSMValidationError):
        Env()


def test_invalid_next_state():
    """
    All next states passed into the fsm_state decorator must be valid states.
    """

    class Env(FiniteStateMachineEnv):
        def __init__(self):
            agents = [MinimalAgent("agent")]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(
                network=network,
                n_steps=1,
                initial_state=States.A,
            )

        @fsm_state(
            state_id=States.A,
            next_states=[States.B],
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
                initial_state=States.A,
            )

        @fsm_state(
            state_id=States.A,
            next_states=[States.A],
        )
        def handle_1(self):
            return States.B

    env = Env()
    env.reset()

    with pytest.raises(FSMRuntimeError):
        env.step(actions={"agent": 0})
