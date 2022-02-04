import gym.spaces
import mercury as me
import numpy as np
import phantom as ph
import pytest


class CustomPolicy(ph.FixedPolicy):
    def compute_action(self, obs):
        return self.action_space.sample()


class MinimalAgent(ph.agent.Agent):
    def __init__(self, id: str, action_space: gym.spaces.Space, policy=None) -> None:
        super().__init__(agent_id=id, policy_class=policy)

        self.action_space = action_space

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self) -> gym.spaces.Space:
        return self.action_space

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([1])

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray) -> ph.Packet:
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
            agents = [
                MinimalAgent("a0", gym.spaces.Box(-1.0, 1.0, (1,))),
                MinimalAgent(
                    "a1", gym.spaces.Box(-1.0, 1.0, (1,)), policy=CustomPolicy
                ),
                MinimalAgent(
                    "a2", gym.spaces.Box(-1.0, 1.0, (5,)), policy=CustomPolicy
                ),
                MinimalAgent("a3", gym.spaces.Discrete(10), policy=CustomPolicy),
                MinimalAgent(
                    "a4",
                    gym.spaces.Tuple(
                        (
                            gym.spaces.Discrete(100),
                            gym.spaces.Discrete(10),
                        )
                    ),
                    policy=CustomPolicy,
                ),
                MinimalAgent(
                    "a5",
                    gym.spaces.Dict(
                        {
                            "position": gym.spaces.Box(low=-100, high=100, shape=(3,)),
                            "velocity": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                            "front_cam": gym.spaces.Tuple(
                                (
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                )
                            ),
                            "rear_cam": gym.spaces.Box(
                                low=0, high=1, shape=(10, 10, 3)
                            ),
                        }
                    ),
                    policy=CustomPolicy,
                ),
            ]

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
