import numpy as np
import phantom as ph
import pytest

from . import MockAgent, MockStackedAgentEnv, MockStrategicAgent


@pytest.fixture
def phantom_env():
    return ph.PhantomEnv(
        num_steps=2,
        network=ph.Network(
            [
                MockStrategicAgent("A", num_steps=1),
                MockStrategicAgent("B"),
                MockAgent("C"),
            ]
        ),
    )


def test_n_agents(phantom_env):
    assert phantom_env.n_agents == 3


def test_agent_ids(phantom_env):
    assert phantom_env.agent_ids == ["A", "B", "C"]
    assert phantom_env.strategic_agent_ids == ["A", "B"]
    assert phantom_env.non_strategic_agent_ids == ["C"]


def test_get_agents(phantom_env):
    assert phantom_env.strategic_agents == [
        phantom_env.agents["A"],
        phantom_env.agents["B"],
    ]
    assert phantom_env.non_strategic_agents == [phantom_env.agents["C"]]


def test__get_item__(phantom_env):
    assert isinstance(phantom_env["A"], MockStrategicAgent)
    assert phantom_env["A"].id == "A"


def test_is_terminated(phantom_env):
    phantom_env._terminations = set()
    assert not phantom_env.is_terminated()

    phantom_env._terminations = set(["A"])
    assert not phantom_env.is_terminated()

    phantom_env._terminations = set(["A", "B"])
    assert phantom_env.is_terminated()


def test_is_truncated(phantom_env):
    phantom_env._truncations = set()
    assert not phantom_env.is_truncated()

    phantom_env._truncations = set(["A"])
    assert not phantom_env.is_truncated()

    phantom_env._truncations = set(["A", "B"])
    assert phantom_env.is_truncated()

    phantom_env._truncations = set()
    phantom_env._current_step = phantom_env.num_steps
    assert phantom_env.is_truncated()


def test_reset(phantom_env):
    obs, infos = phantom_env.reset()

    assert phantom_env.current_step == 0
    assert list(obs.keys()) == ["A", "B"]
    assert infos == {}


def test_step(phantom_env):
    # 1st step:
    current_time = phantom_env.current_step

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.current_step == current_time + 1

    assert list(step.observations.keys()) == ["A", "B"]
    assert list(step.rewards.keys()) == ["A", "B"]
    assert list(step.infos.keys()) == ["A", "B"]

    assert step.terminations == {"A": True, "B": False, "__all__": False}
    assert step.truncations == {"A": True, "B": False, "__all__": False}

    # 2nd step:
    current_time = phantom_env.current_step

    actions = {"A": 0, "B": 0}
    step = phantom_env.step(actions)

    assert phantom_env.current_step == current_time + 1

    assert list(step.observations.keys()) == ["B"]
    assert list(step.rewards.keys()) == ["B"]
    assert list(step.infos.keys()) == ["B"]

    assert step.terminations == {"B": False, "__all__": False}
    assert step.truncations == {"B": False, "__all__": True}


def test_stacked_agent_step():
    env = MockStackedAgentEnv()

    ids = ["a1"]
    ids += [f"__stacked__{i}__a2" for i in range(2)]
    ids += [f"__stacked__{i}__a3" for i in range(4)]

    ids = set(ids)

    obs, _ = env.reset()

    assert set(obs.keys()) == ids

    step = env.step({aid: np.array([0]) for aid in ids})

    assert set(step.observations.keys()) == ids
    assert set(step.rewards.keys()) == ids
    assert set(step.infos.keys()) == ids

    assert set(step.terminations) == set(["__all__", "a1", "a2", "a3"])
    assert set(step.truncations) == set(["__all__", "a1", "a2", "a3"])

    assert all(x == False for x in step.terminations.values())
    assert all(x == False for x in step.truncations.values())

    step = env.step({aid: np.array([0]) for aid in ids})

    assert all(step.terminations.values())
    assert all(step.truncations.values())
