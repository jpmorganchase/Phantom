from typing import List

import gym.spaces
import mercury as me
import numpy as np
import phantom as ph
from phantom.policy_wrapper import PolicyWrapper
from phantom.utils.training import create_rllib_config_dict

ENV_NAME = "test-env"
AGENT_ID = "a1"
STEPS = 10
SEED = 0
NUM_WORKERS = 2


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


class MockEnv(ph.PhantomEnv):

    env_name: str = ENV_NAME

    def __init__(self, agents: List[ph.Agent]):
        network = me.Network(me.resolvers.UnorderedResolver(), agents)
        super().__init__(network=network, n_steps=STEPS)


def test_basic_env():
    agent = MockAgent(AGENT_ID)
    agent.policy_config = {"config": "option"}

    config, policies = create_rllib_config_dict(
        env_class=MockEnv,
        alg_config={},
        env_config={
            "agents": [agent],
        },
        env_supertype=None,
        agent_supertypes={},
        policy_grouping={},
        callbacks=[],
        metrics={},
        seed=SEED,
        num_workers=NUM_WORKERS,
    )

    assert config["env"] == ENV_NAME
    assert config["env_config"] == {"agents": [agent]}
    assert config["seed"] == SEED
    assert config["num_workers"] == NUM_WORKERS
    assert config["rollout_fragment_length"] == 10
    assert config["train_batch_size"] == 20
    assert config["sgd_minibatch_size"] == 2

    assert policies == [
        PolicyWrapper(
            used_by=[AGENT_ID],
            trained=True,
            obs_space=agent.get_observation_space(),
            action_space=agent.get_action_space(),
            policy_class=None,
            policy_config={"config": "option"},
        )
    ]
