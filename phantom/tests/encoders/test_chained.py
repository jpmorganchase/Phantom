import gym
import mercury as me
import numpy as np
from phantom.encoders import ChainedEncoder, Encoder


class SimpleEncoder(Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


def test_chained_encoder():
    e1 = SimpleEncoder(1)
    e2 = SimpleEncoder(2)

    ce1 = ChainedEncoder([e1, e2])

    obs = ce1.encode(None)

    assert obs == (np.array([1]), np.array([2]))


def test_chained_encoder_reset():
    e1 = SimpleEncoder(1)
    e2 = SimpleEncoder(2)

    cd = ChainedEncoder([e1, e2])

    cd.reset()

    assert e1.id is None
    assert e2.id is None
