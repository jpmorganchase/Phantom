import pytest
from phantom.utils.rollout_class import AgentStep, Step, Rollout


@pytest.fixture
def rollout() -> Rollout:
    return Rollout(
        rollout_id=0,
        repeat_id=0,
        env_config={},
        top_level_params={},
        env_type={},
        agent_types={},
        steps=[
            Step(
                observations={
                    "agent1": [i * 1],
                    "agent2": [i * 2],
                },
                rewards={
                    "agent1": i * 1.0,
                    "agent2": i * 2.0,
                },
                dones={
                    "agent1": False,
                    "agent2": False,
                }
                if i < 9
                else {
                    "agent1": True,
                    "agent2": True,
                },
                infos={},
                actions={
                    "agent1": i % 2,
                    "agent2": (i + 1) % 2,
                },
                messages=None,
                stage="even" if i % 2 == 0 else "odd",
            )
            for i in range(10)
        ],
        metrics={},
    )


def test_get_observations_for_agent(rollout: Rollout):
    observations = rollout.observations_for_agent("agent1")
    assert list(observations) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

    observations = rollout.observations_for_agent("agent2")
    assert list(observations) == [[0], [2], [4], [6], [8], [10], [12], [14], [16], [18]]

    observations = rollout.observations_for_agent("agent1", stages=["even"])
    assert list(observations) == [[0], [2], [4], [6], [8]]


def test_get_rewards_for_agent(rollout: Rollout):
    rewards = rollout.rewards_for_agent("agent1")
    assert list(rewards) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    rewards = rollout.rewards_for_agent("agent2")
    assert list(rewards) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

    rewards = rollout.rewards_for_agent("agent1", stages=["even"])
    assert list(rewards) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_get_dones_for_agent(rollout: Rollout):
    dones = rollout.dones_for_agent("agent1")
    assert list(dones) == [False] * 9 + [True]

    dones = rollout.dones_for_agent("agent1", stages=["even"])
    assert list(dones) == [False] * 5


def test_get_infos_for_agent(rollout: Rollout):
    infos = rollout.infos_for_agent("agent1")
    assert list(infos) == [None] * 10

    infos = rollout.infos_for_agent("agent1", stages=["even", "odd"])
    assert list(infos) == [None] * 10


def test_get_actions_for_agent(rollout: Rollout):
    actions = rollout.actions_for_agent("agent1")
    assert list(actions) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    actions = rollout.actions_for_agent("agent2")
    assert list(actions) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    actions = rollout.actions_for_agent("agent1", stages=["even"])
    assert list(actions) == [0, 0, 0, 0, 0]


def test_steps_for_agents(rollout: Rollout):
    steps = rollout.steps_for_agent(agent_id="agent1")
    steps = list(steps)
    assert steps[0] == AgentStep(
        observation=[0],
        reward=0.0,
        done=False,
        info=None,
        action=0,
        stage="even",
    )

    assert steps[1] == AgentStep(
        observation=[1],
        reward=1.0,
        done=False,
        info=None,
        action=1,
        stage="odd",
    )

    steps = rollout.steps_for_agent(agent_id="agent1", stages=["even"])
    steps = list(steps)
    assert steps[0] == AgentStep(
        observation=[0],
        reward=0.0,
        done=False,
        info=None,
        action=0,
        stage="even",
    )

    assert steps[1] == AgentStep(
        observation=[2],
        reward=2.0,
        done=False,
        info=None,
        action=0,
        stage="even",
    )


def test_count_actions(rollout: Rollout):
    action_counts = rollout.count_actions()
    assert action_counts == [(0, 10), (1, 10)]

    action_counts = rollout.count_actions(stages=["even"])
    assert action_counts == [(0, 5), (1, 5)]


def test_count_agent_actions(rollout: Rollout):
    action_counts = rollout.count_agent_actions(agent_id="agent1")
    assert action_counts == [(0, 5), (1, 5)]

    action_counts = rollout.count_agent_actions(agent_id="agent1", stages=["even"])
    assert action_counts == [(0, 5)]


def test_getitem(rollout: Rollout):
    step = rollout[1]
    assert step == Step(
        observations={
            "agent1": [1],
            "agent2": [2],
        },
        rewards={
            "agent1": 1.0,
            "agent2": 2.0,
        },
        dones={
            "agent1": False,
            "agent2": False,
        },
        infos={},
        actions={
            "agent1": 1,
            "agent2": 0,
        },
        messages=None,
        stage="odd",
    )
