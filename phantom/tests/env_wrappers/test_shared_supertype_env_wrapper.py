from dataclasses import dataclass
from textwrap import wrap

import gym
import mercury as me
import numpy as np
import phantom as ph
import pytest
import unittest


class IncrementingSampler(ph.utils.samplers.BaseSampler[int]):
    def __init__(
        self,
        start: int = 0,
        increment: int = 1,
    ) -> None:
        self.value = start
        self.increment = increment

    def sample(self) -> int:
        value, self.value = self.value, self.value + self.increment
        return value


@dataclass
class MockSupertype(ph.BaseSupertype):
    x: ph.SupertypeField[float]


class MockAgent(ph.Agent):
    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        return ph.agent.Packet()

    def is_done(self, _ctx):
        return self.id == "A"

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context):
        return np.zeros((1,))

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        return ph.Packet()

    def get_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))


@pytest.fixture
def wrapped_env():
    network = me.Network(
        resolver=me.resolvers.UnorderedResolver(),
        actors=[MockAgent("A"), MockAgent("B")],
    )

    env = ph.PhantomEnv(network, n_steps=2)

    agent_sampler = IncrementingSampler()

    agent_supertypes = {
        "A": MockSupertype(x=agent_sampler),
        "B": MockSupertype(x=agent_sampler),
    }

    env_supertype = MockSupertype(x=IncrementingSampler(start=10))

    return ph.env_wrappers.SharedSupertypeEnvWrapper(
        env, env_supertype, agent_supertypes
    )


def test_shared_supertype_env_wrapper(wrapped_env):
    assert not hasattr(wrapped_env.agents["A"], "type")
    assert not hasattr(wrapped_env.agents["B"], "type")
    assert not hasattr(wrapped_env, "env_type")

    wrapped_env.reset()

    assert wrapped_env.agents["A"].type.x == 0
    assert wrapped_env.agents["B"].type.x == 0
    assert wrapped_env.env_type.x == 10
