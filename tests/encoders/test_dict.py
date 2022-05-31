import gym
import numpy as np

from phantom import Context
from phantom.encoders import DictEncoder, Encoder


class MockEncoder(Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


def test_dict_encoder():
    e1 = MockEncoder(1)
    e2 = MockEncoder(2)

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
    e1 = MockEncoder(1)
    e2 = MockEncoder(2)

    de = DictEncoder({"e1": e1, "e2": e2})

    de.reset()

    assert e1.id is None
    assert e2.id is None
