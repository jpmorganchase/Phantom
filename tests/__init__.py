import gym
import numpy as np
import phantom as ph


class MockAgent(ph.Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(-np.inf, np.inf, (1,))
        self.observation_space = gym.spaces.Box(-np.inf, np.inf, (1,))

        self.encode_obs_count = 0
        self.decode_action_count = 0
        self.compute_reward_count = 0

    def encode_observation(self, ctx: ph.Context):
        self.encode_obs_count += 1
        return np.zeros((1,))

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        self.decode_action_count += 1
        return []

    def compute_reward(self, ctx: ph.Context) -> float:
        self.compute_reward_count += 1
        return 0.0

    def is_done(self, ctx: ph.Context) -> bool:
        return self.id == "A"
