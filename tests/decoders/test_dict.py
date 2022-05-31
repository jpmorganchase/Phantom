from typing import List, Tuple

import gym
import numpy as np

from phantom import AgentID, Context, Message
from phantom.decoders import Decoder, DictDecoder


class MockDecoder(Decoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: Context, action) -> List[Tuple[AgentID, Message]]:
        assert action == self.id

        return [("RECIPIENT", f"FROM {self.id}")]

    def reset(self):
        self.id = None


def test_dict_decoder():
    d1 = MockDecoder(1)
    d2 = MockDecoder(2)

    dd = DictDecoder({"d1": d1, "d2": d2})

    assert dd.action_space == gym.spaces.Dict(
        {
            "d1": gym.spaces.Box(-np.inf, np.inf, (1,)),
            "d2": gym.spaces.Box(-np.inf, np.inf, (1,)),
        }
    )

    messages = dd.decode(None, {"d1": 1, "d2": 2})

    assert messages == [("RECIPIENT", "FROM 1"), ("RECIPIENT", "FROM 2")]


def test_chained_decoder_reset():
    d1 = MockDecoder(1)
    d2 = MockDecoder(2)

    dd = DictDecoder({"d1": d1, "d2": d2})

    dd.reset()

    assert d1.id is None
    assert d2.id is None
