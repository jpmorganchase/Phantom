from tempfile import TemporaryDirectory
from typing import Optional

import gym.spaces
import mercury as me
import numpy as np
import phantom as ph
import pytest


class CustomPolicy(ph.FixedPolicy):
    def compute_action(self, obs):
        return self.action_space.sample()


class MinimalAgent(ph.agent.Agent):
    def __init__(
        self,
        id: str,
        action_space: Optional[gym.spaces.Space] = None,
        obs_space: Optional[gym.spaces.Space] = None,
        policy=None,
    ) -> None:
        super().__init__(agent_id=id, policy_class=policy)

        self.action_space = action_space or gym.spaces.Box(-1.0, 1.0, (1,))
        self.obs_space = obs_space or gym.spaces.Box(-1.0, 1.0, (1,))

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def get_observation_space(self) -> gym.spaces.Space:
        return self.obs_space

    def get_action_space(self) -> gym.spaces.Space:
        return self.action_space

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        return self.obs_space.sample()

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


def test_fixed_policy_action_spaces():
    class MinimalEnv(ph.PhantomEnv):

        env_name: str = "unit-testing"

        def __init__(self):
            agents = [
                MinimalAgent("a0", action_space=gym.spaces.Box(-1.0, 1.0, (1,))),
                MinimalAgent(
                    "a1",
                    action_space=gym.spaces.Box(-1.0, 1.0, (1,)),
                    policy=CustomPolicy,
                ),
                MinimalAgent(
                    "a2",
                    action_space=gym.spaces.Box(-1.0, 1.0, (5,)),
                    policy=CustomPolicy,
                ),
                MinimalAgent(
                    "a3", action_space=gym.spaces.Discrete(10), policy=CustomPolicy
                ),
                MinimalAgent(
                    "a4",
                    action_space=gym.spaces.Tuple(
                        (
                            gym.spaces.Discrete(100),
                            gym.spaces.Discrete(10),
                        )
                    ),
                    policy=CustomPolicy,
                ),
                MinimalAgent(
                    "a5",
                    action_space=gym.spaces.Dict(
                        {
                            "a": gym.spaces.Box(low=-100, high=100, shape=(3,)),
                            "b": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                            "c": gym.spaces.Tuple(
                                (
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                )
                            ),
                            "d": gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                        }
                    ),
                    policy=CustomPolicy,
                ),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(network=network, n_steps=3)

    with TemporaryDirectory() as temp_dir:
        results_dir = ph.train(
            experiment_name="unit-testing",
            algorithm="PPO",
            num_workers=0,
            num_episodes=1,
            env_class=MinimalEnv,
            results_dir=temp_dir,
        )

        ph.rollout(
            directory=results_dir,
            algorithm="PPO",
            num_workers=0,
            num_repeats=1,
        )


def test_fixed_policy_observations_spaces():
    class MinimalEnv(ph.PhantomEnv):

        env_name: str = "unit-testing"

        def __init__(self):
            agents = [
                MinimalAgent("a0", obs_space=gym.spaces.Box(-1.0, 1.0, (1,))),
                MinimalAgent(
                    "a1", obs_space=gym.spaces.Box(-1.0, 1.0, (1,)), policy=CustomPolicy
                ),
                MinimalAgent(
                    "a2", obs_space=gym.spaces.Box(-1.0, 1.0, (5,)), policy=CustomPolicy
                ),
                MinimalAgent(
                    "a3", obs_space=gym.spaces.Discrete(10), policy=CustomPolicy
                ),
                MinimalAgent(
                    "a4",
                    obs_space=gym.spaces.Tuple(
                        (
                            gym.spaces.Discrete(100),
                            gym.spaces.Discrete(10),
                        )
                    ),
                    policy=CustomPolicy,
                ),
                MinimalAgent(
                    "a5",
                    obs_space=gym.spaces.Dict(
                        {
                            "a": gym.spaces.Box(low=-100, high=100, shape=(3,)),
                            "b": gym.spaces.Box(low=-1, high=1, shape=(3,)),
                            "c": gym.spaces.Tuple(
                                (
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                    gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                                )
                            ),
                            "d": gym.spaces.Box(low=0, high=1, shape=(10, 10, 3)),
                        }
                    ),
                    policy=CustomPolicy,
                ),
            ]

            network = me.Network(me.resolvers.UnorderedResolver(), agents)

            super().__init__(network=network, n_steps=3)

    with TemporaryDirectory() as temp_dir:
        results_dir = ph.train(
            experiment_name="unit-testing",
            algorithm="PPO",
            num_workers=0,
            num_episodes=1,
            env_class=MinimalEnv,
        )

        ph.rollout(
            directory=results_dir,
            algorithm="PPO",
            num_workers=0,
            num_repeats=1,
        )
