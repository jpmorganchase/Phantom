import gym
import numpy as np
import phantom as ph
import pytest

from phantom import Context, Network, PhantomEnv, Policy, SingleAgentEnvAdapter


class MockAgent(ph.Agent):
    def decode_action(self, ctx: Context, action: np.ndarray):
        return []

    def is_done(self, ctx: Context) -> bool:
        return self.id == "B"

    def compute_reward(self, ctx: Context) -> float:
        return 0.0

    def encode_obs(self, ctx: Context):
        return 1.0

    def decode_action(self, ctx: Context, action: np.ndarray):
        print(self.id, action)
        self.last_action = action
        return []

    def get_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))


class MockPolicy(Policy):
    def compute_action(self, observation):
        return 2.0


@pytest.fixture
def gym_env():
    return SingleAgentEnvAdapter(
        env=PhantomEnv,
        agent_id="A",
        other_policies={"B": (MockPolicy, {})},
        config={"network": Network([MockAgent("A"), MockAgent("B")]), "num_steps": 2},
    )


def test_agent_ids(gym_env):
    assert list(gym_env.agent_ids) == ["A", "B"]


def test_n_agents(gym_env):
    assert gym_env.n_agents == 2


def test_reset(gym_env):
    obs = gym_env.reset()

    assert gym_env.current_step == 0
    assert obs == 1.0

    assert gym_env._observations == {
        "A": 1.0,
        "B": 1.0,
    }


def test_step(gym_env):
    gym_env.reset()

    # 1st step
    current_time = gym_env.current_step

    action = 3.0
    step = gym_env.step(action)

    assert gym_env.current_step == current_time + 1

    assert step == (1.0, 0.0, False, {})

    assert gym_env.agents["A"].last_action == 3.0
    assert gym_env.agents["B"].last_action == 2.0

    # 2nd step
    current_time = gym_env.current_step

    actions = {"A": 0, "B": 0}
    step = gym_env.step(actions)

    assert gym_env.current_step == current_time + 1


def test_bad_env():
    # Bad selected agent ID
    with pytest.raises(AssertionError):
        SingleAgentEnvAdapter(
            env=PhantomEnv,
            agent_id="X",
            other_policies={"B": (MockPolicy, {})},
            config={
                "network": Network([MockAgent("A"), MockAgent("B")]),
                "num_steps": 2,
            },
        )

    # Selected agent has other policy
    with pytest.raises(AssertionError):
        SingleAgentEnvAdapter(
            env=PhantomEnv,
            agent_id="A",
            other_policies={"A": (MockPolicy, {}), "B": (MockPolicy, {})},
            config={
                "network": Network([MockAgent("A"), MockAgent("B")]),
                "num_steps": 2,
            },
        )

    # Bad other policy agent ID
    with pytest.raises(AssertionError):
        SingleAgentEnvAdapter(
            env=PhantomEnv,
            agent_id="A",
            other_policies={"X": (MockPolicy, {})},
            config={
                "network": Network([MockAgent("A"), MockAgent("B")]),
                "num_steps": 2,
            },
        )

    # Agent B missing from other policies
    with pytest.raises(AssertionError):
        SingleAgentEnvAdapter(
            env=PhantomEnv,
            agent_id="A",
            other_policies={},
            config={
                "network": Network([MockAgent("A"), MockAgent("B")]),
                "num_steps": 2,
            },
        )
