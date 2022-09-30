import phantom as ph
import pytest

from . import MockAgent


@pytest.fixture
def phantom_env():
    return ph.PhantomEnv(
        num_steps=2, network=ph.Network([MockAgent("A", num_steps=1), MockAgent("B")])
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
    phantom_env.current_step = phantom_env.num_steps
    assert phantom_env.is_done()


def test_reset(phantom_env):
    obs = phantom_env.reset()

    assert phantom_env.current_step == 0
    assert list(obs.keys()) == ["A", "B"]


def test_step(phantom_env):
    # 1st step
    current_time = phantom_env.current_step

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.current_step == current_time + 1

    for aid in ("A", "B"):
        assert aid in step.observations
        assert aid in step.rewards
        assert aid in step.infos
    assert step.dones["A"]
    assert not step.dones["__all__"]
    assert not step.dones["B"]

    # 2nd step
    current_time = phantom_env.current_step

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.current_step == current_time + 1

    assert "A" not in step.observations
    assert "A" not in step.rewards
    assert "A" not in step.infos
    assert "B" in step.observations
    assert "B" in step.rewards
    assert "B" in step.infos
    assert "A" not in step.dones
    assert step.dones["__all__"]
    assert not step.dones["B"]
