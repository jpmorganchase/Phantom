import gym
import mercury as me
import numpy as np
import phantom as ph


class MinimalAgent(ph.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(agent_id=id)

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([0])

    def decode_action(self, ctx: me.Network.Context, action) -> ph.Packet:
        return ph.Packet()

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()
