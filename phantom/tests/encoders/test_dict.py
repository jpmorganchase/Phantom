import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.encoders import DictEncoder, Encoder


class TestEncoder(Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


def test_dict_encoder():
    e1 = TestEncoder(1)
    e2 = TestEncoder(2)

    de = DictEncoder({"e1": e1, "e2": e2})

    assert de.output_space == gym.spaces.Dict(
        {
            "e1": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "e2": gym.spaces.Box(-np.inf, np.inf, (1,)),
        }
    )

    obs = de.encode(None)

    assert obs == {"e1": np.array([1]), "e2": np.array([2])}


def test_dict_encoder_reset():
    e1 = TestEncoder(1)
    e2 = TestEncoder(2)

    de = DictEncoder({"e1": e1, "e2": e2})

    de.reset()

    assert e1.id == None
    assert e2.id == None
