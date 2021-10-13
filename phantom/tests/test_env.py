import pytest
import mercury as me
import phantom as ph
import unittest
import numpy as np
from phantom.env import EnvironmentActor, PhantomEnv


class MockAgent(ph.ZeroIntelligenceAgent):
    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        return ph.agent.Packet()

    def is_done(self, _ctx):
        return self.id == "A"


@pytest.fixture
def phantom_network():
    return me.Network(
        resolver=me.resolvers.UnorderedResolver(),
        actors=[MockAgent("A"), MockAgent("B")],
    )


@pytest.fixture
def phantom_env(phantom_network):
    environment_actor = EnvironmentActor()
    return PhantomEnv(
        network=phantom_network, n_steps=2, environment_actor=environment_actor, seed=1
    )


def test_agent_ids(phantom_env):
    assert list(phantom_env.agent_ids) == ["A", "B"]


def test_n_agents(phantom_env):
    assert phantom_env.n_agents == 2


def test__get_item__(phantom_env):
    assert isinstance(phantom_env["A"], MockAgent)
    assert phantom_env["A"].id == "A"


def test_is_done(phantom_env):
    phantom_env._dones = []
    assert not phantom_env.is_done()

    phantom_env._dones = ["A"]
    assert not phantom_env.is_done()

    phantom_env._dones = ["A", "B"]
    assert phantom_env.is_done()

    phantom_env._dones = []
    phantom_env.clock._step = (
        phantom_env.clock.terminal_time // phantom_env.clock.increment
    )
    assert phantom_env.is_done()


def test_reset(phantom_env):
    phantom_env.network = unittest.mock.MagicMock()

    obs = phantom_env.reset()

    assert phantom_env.clock.elapsed == 0
    assert phantom_env.network.reset.called_once()
    assert phantom_env.network.resolve.called_once()
    assert list(obs.keys()) == ["A", "B"]


def test_step(phantom_env):
    phantom_env.network = unittest.mock.MagicMock()

    # 1st step
    current_time = phantom_env.clock.elapsed

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.clock.elapsed == current_time + phantom_env.clock.increment
    assert phantom_env.network.send_from.call_count == len(actions)
    assert phantom_env.network.resolve.called_once()

    for aid in ("A", "B"):
        assert aid in step.observations
        assert aid in step.rewards
        assert aid in step.infos
    assert step.terminals["A"]
    assert not step.terminals["__all__"]
    assert not step.terminals["B"]

    # 2nd step
    phantom_env.network.reset_mock()
    current_time = phantom_env.clock.elapsed

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.clock.elapsed == current_time + phantom_env.clock.increment
    assert phantom_env.network.send_from.call_count == len(actions)
    assert phantom_env.network.resolve.called_once()

    assert "A" not in step.observations
    assert "A" not in step.rewards
    assert "A" not in step.infos
    assert "B" in step.observations
    assert "B" in step.rewards
    assert "B" in step.infos
    assert "A" not in step.terminals
    assert step.terminals["__all__"]
    assert not step.terminals["B"]
