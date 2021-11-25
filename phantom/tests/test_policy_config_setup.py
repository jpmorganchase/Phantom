from typing import List

import gym.spaces
import mercury as me
import numpy as np
import phantom as ph
from phantom.policy_wrapper import PolicyWrapper
from phantom.utils.training import create_rllib_config_dict


class MockAgent(ph.Agent):
    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([1])

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray) -> ph.Packet:
        if self.policy_class is not None:
            assert action == np.array([1])
        return ph.Packet()


class MockFSMAgent(ph.fsm.FSMAgent):
    pass


class MockStagePolicyHandler(ph.fsm.StagePolicyHandler["MockFSMAgent"]):
    @staticmethod
    def compute_reward(agent: "MockFSMAgent", ctx: me.Network.Context) -> float:
        return 0

    @staticmethod
    def encode_obs(agent: "MockFSMAgent", ctx: me.Network.Context) -> np.ndarray:
        return np.array([1])

    @staticmethod
    def decode_action(
        agent: "MockFSMAgent", ctx: me.Network.Context, action
    ) -> ph.Packet:
        return ph.Packet(messages={agent.id: ["message"]})

    @staticmethod
    def get_observation_space(agent: "MockFSMAgent") -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: "MockFSMAgent") -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class MockEnv(ph.PhantomEnv):
    def __init__(self, agents: List[ph.Agent]):
        network = me.Network(me.resolvers.UnorderedResolver(), agents)
        super().__init__(network=network, n_steps=10)


def test_single_agent():
    agent = MockAgent("a1")

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent],
        },
        alg_config={},
        policy_grouping={},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    assert policies == [
        PolicyWrapper(
            used_by=["a1"],
            trained=True,
            obs_space=agent.get_observation_space(),
            action_space=agent.get_action_space(),
        )
    ]


def test_single_agent_fixed_policy():
    class CustomPolicy(ph.FixedPolicy):
        def compute_action(self, obs):
            return np.array([1])

    agent1 = MockAgent("a1")
    agent2 = MockAgent("a2")
    agent2.policy_class = CustomPolicy

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent1, agent2],
        },
        alg_config={},
        policy_grouping={},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    assert policies == [
        PolicyWrapper(
            used_by=["a1"],
            trained=True,
            obs_space=agent1.get_observation_space(),
            action_space=agent1.get_action_space(),
        ),
        PolicyWrapper(
            used_by=["a2"],
            trained=False,
            obs_space=agent2.get_observation_space(),
            action_space=agent2.get_action_space(),
            policy_class=CustomPolicy,
        ),
    ]


def test_shared_policy():
    agent1 = MockAgent("a1")
    agent2 = MockAgent("a2")

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent1, agent2],
        },
        alg_config={},
        policy_grouping={"shared": ["a1", "a2"]},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    assert policies == [
        PolicyWrapper(
            used_by=["a1", "a2"],
            trained=True,
            obs_space=agent1.get_observation_space(),
            action_space=agent1.get_action_space(),
            shared_policy_name="shared",
        )
    ]


def test_single_agent_single_stage():
    agent = MockFSMAgent("a1", {"stage1": MockStagePolicyHandler()})

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent],
        },
        alg_config={},
        policy_grouping={},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    assert policies == [
        PolicyWrapper(
            used_by=[("a1", "stage1")],
            trained=True,
            obs_space=agent.stage_handlers[0][1].get_observation_space(agent),
            action_space=agent.stage_handlers[0][1].get_action_space(agent),
        )
    ]


def test_single_agent_multiple_stages():
    agent = MockFSMAgent("a1", {
        "stage1": MockStagePolicyHandler(),
        "stage2": MockStagePolicyHandler(),
    })

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent],
        },
        alg_config={},
        policy_grouping={},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    assert policies == [
        PolicyWrapper(
            used_by=[("a1", "stage1")],
            trained=True,
            obs_space=agent.stage_handlers[0][1].get_observation_space(agent),
            action_space=agent.stage_handlers[0][1].get_action_space(agent),
        ),
        PolicyWrapper(
            used_by=[("a1", "stage2")],
            trained=True,
            obs_space=agent.stage_handlers[0][1].get_observation_space(agent),
            action_space=agent.stage_handlers[0][1].get_action_space(agent),
        ),
    ]


def test_multiple_agents_shared_stage():
    handler = MockStagePolicyHandler()

    agent1 = MockFSMAgent("a1", {"stage1": handler})
    agent2 = MockFSMAgent("a2", {"stage1": handler})

    _, policies = create_rllib_config_dict(
        env_class=MockEnv,
        env_config={
            "agents": [agent1, agent2],
        },
        alg_config={},
        policy_grouping={"shared": ["a1__stage1", "a2__stage1"]},
        callbacks=[],
        metrics={},
        seed=0,
        num_workers=0,
    )

    for p in policies:
        print(p.__dict__)

    assert policies == [
        PolicyWrapper(
            used_by=[("a1", "stage1"), ("a2", "stage1")],
            trained=True,
            obs_space=agent1.stage_handlers[0][1].get_observation_space(agent1),
            action_space=agent1.stage_handlers[0][1].get_action_space(agent1),
            shared_policy_name="shared",
        )
    ]
