from typing import Optional

import gym
import numpy as np
import phantom as ph


class MockAgent(ph.RLAgent):
    def __init__(self, *args, num_steps: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Discrete(1)

        self.encode_obs_count = 0
        self.decode_action_count = 0
        self.compute_reward_count = 0

        self.num_steps = num_steps

    def encode_observation(self, ctx: ph.Context):
        self.encode_obs_count += 1
        return 0

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        self.decode_action_count += 1
        return []

    def compute_reward(self, ctx: ph.Context) -> float:
        self.compute_reward_count += 1
        return 0.0

    def is_done(self, ctx: ph.Context) -> bool:
        return ctx.env_view.current_step == self.num_steps


class MockPolicy(ph.Policy):
    def compute_action(self, obs) -> int:
        return 1


class MockEnv(ph.PhantomEnv):
    def __init__(self):
        agents = [MockAgent("a1"), MockAgent("a2"), MockAgent("a3")]

        network = ph.network.Network(agents)

        network.add_connection("a1", "a2")
        network.add_connection("a2", "a3")
        network.add_connection("a3", "a1")

        super().__init__(num_steps=5, network=network)
