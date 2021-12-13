import phantom as ph
import pytest
from phantom.utils.rollout import AgentStep, EpisodeTrajectory, Step


@pytest.fixture
def episode_trajectory() -> EpisodeTrajectory:
    return EpisodeTrajectory(
        observations=[
            {
                "agent1": [i * 1],
                "agent2": [i * 2],
            }
            for i in range(10)
        ],
        rewards=[
            {
                "agent1": i * 1.0,
                "agent2": i * 2.0,
            }
            for i in range(10)
        ],
        dones=[
            {
                "agent1": False,
                "agent2": False,
            }
            for _ in range(9)
        ]
        + [
            {
                "agent1": True,
                "agent2": True,
            }
        ],
        infos=[{} for _ in range(10)],
        actions=[
            {
                "agent1": i % 2,
                "agent2": (i + 1) % 2,
            }
            for i in range(10)
        ],
        stages=["even", "odd"] * 5,
    )


def test_get_observations_for_agent(episode_trajectory: EpisodeTrajectory):
    observations = episode_trajectory.observations_for_agent("agent1")
    assert list(observations) == [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9]]

    observations = episode_trajectory.observations_for_agent("agent2")
    assert list(observations) == [[0], [2], [4], [6], [8], [10], [12], [14], [16], [18]]

    observations = episode_trajectory.observations_for_agent("agent1", stages=["even"])
    assert list(observations) == [[0], [2], [4], [6], [8]]


def test_get_rewards_for_agent(episode_trajectory: EpisodeTrajectory):
    rewards = episode_trajectory.rewards_for_agent("agent1")
    assert list(rewards) == [0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]

    rewards = episode_trajectory.rewards_for_agent("agent2")
    assert list(rewards) == [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0, 16.0, 18.0]

    rewards = episode_trajectory.rewards_for_agent("agent1", stages=["even"])
    assert list(rewards) == [0.0, 2.0, 4.0, 6.0, 8.0]


def test_get_dones_for_agent(episode_trajectory: EpisodeTrajectory):
    dones = episode_trajectory.dones_for_agent("agent1")
    assert list(dones) == [False] * 9 + [True]

    dones = episode_trajectory.dones_for_agent("agent1", stages=["even"])
    assert list(dones) == [False] * 5


def test_get_infos_for_agent(episode_trajectory: EpisodeTrajectory):
    infos = episode_trajectory.infos_for_agent("agent1")
    assert list(infos) == [None] * 10

    infos = episode_trajectory.infos_for_agent("agent1", stages=["even", "odd"])
    assert list(infos) == [None] * 10


def test_get_actions_for_agent(episode_trajectory: EpisodeTrajectory):
    actions = episode_trajectory.actions_for_agent("agent1")
    assert list(actions) == [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]

    actions = episode_trajectory.actions_for_agent("agent2")
    assert list(actions) == [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]

    actions = episode_trajectory.actions_for_agent("agent1", stages=["even"])
    assert list(actions) == [0, 0, 0, 0, 0]


def test_steps_for_agents(episode_trajectory: EpisodeTrajectory):
    steps = episode_trajectory.steps_for_agent(agent_id="agent1")
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

    steps = episode_trajectory.steps_for_agent(agent_id="agent1", stages=["even"])
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


def test_count_actions(episode_trajectory: EpisodeTrajectory):
    action_counts = episode_trajectory.count_actions()
    assert action_counts == [(0, 10), (1, 10)]

    action_counts = episode_trajectory.count_actions(stages=["even"])
    assert action_counts == [(0, 5), (1, 5)]


def test_count_agent_actions(episode_trajectory: EpisodeTrajectory):
    action_counts = episode_trajectory.count_agent_actions(agent_id="agent1")
    assert action_counts == [(0, 5), (1, 5)]

    action_counts = episode_trajectory.count_agent_actions(
        agent_id="agent1", stages=["even"]
    )
    assert action_counts == [(0, 5)]


def test_getitem(episode_trajectory: EpisodeTrajectory):
    step = episode_trajectory[1]
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
        stage="odd",
    )
