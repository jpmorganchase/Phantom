import gym.spaces
import mercury as me
import numpy as np
import phantom as ph
import pytest


class CustomPolicy(ph.FixedPolicy):
    def compute_action(self, obs):
        return np.array([1])


class MinimalAgent(ph.agent.Agent):
    def __init__(self, id: str, policy=None) -> None:
        super().__init__(agent_id=id, policy_class=policy)

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


def test_no_trained_policies():
    class MinimalEnv(ph.PhantomEnv):

        env_name: str = "unit-testing"

        def __init__(self):
            agents = [MinimalAgent("a1", policy=CustomPolicy)]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(network=network, n_steps=3)

    # Must have at least one trained policy to train
    with pytest.raises(Exception):
        ph.train(
            experiment_name="unit-testing",
            algorithm="PPO",
            num_workers=0,
            num_episodes=1,
            env_class=MinimalEnv,
            discard_results=True,
        )


def test_fixed_policy():
    class MinimalEnv(ph.PhantomEnv):

        env_name: str = "unit-testing"

        def __init__(self):
            agents = [MinimalAgent("a1"), MinimalAgent("a2", policy=CustomPolicy)]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(network=network, n_steps=3)

    ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env_class=MinimalEnv,
        discard_results=True,
    )
