from typing import List, Tuple

import gymnasium as gym
import numpy as np

from phantom import AgentID, Context, Message
from phantom.decoders import ChainedDecoder, Decoder


class SimpleDecoder(Decoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: Context, action) -> List[Tuple[AgentID, Message]]:
        return [("RECIPIENT", f"FROM {self.id}")]

    def reset(self):
        self.id = None


def test_chained_decoder():
    d1 = SimpleDecoder(1)
    d2 = SimpleDecoder(2)

    cd1 = ChainedDecoder([d1, d2])

    messages = cd1.decode(None, [None, None])

    assert messages == [("RECIPIENT", "FROM 1"), ("RECIPIENT", "FROM 2")]

    cd2 = d1.chain(d2)

    cd2.decode(None, [None, None])

    assert messages == [("RECIPIENT", "FROM 1"), ("RECIPIENT", "FROM 2")]


def test_chained_decoder_reset():
    d1 = SimpleDecoder(1)
    d2 = SimpleDecoder(2)

    cd = ChainedDecoder([d1, d2])

    cd.reset()

    assert d1.id is None
    assert d2.id is None


def test_chained_decoder_independence():
    d1 = SimpleDecoder(1)
    d2 = SimpleDecoder(2)
    d3 = SimpleDecoder(3)

    cd1 = ChainedDecoder([d1, d2])

    cd2 = cd1.chain([d3])

    messages_cd2 = cd2.decode(None, [None, None, None])
    assert messages_cd2 == [
        ("RECIPIENT", "FROM 1"),
        ("RECIPIENT", "FROM 2"),
        ("RECIPIENT", "FROM 3"),
    ]

    messages_cd1 = cd1.decode(None, [None, None])
    assert messages_cd1 == [("RECIPIENT", "FROM 1"), ("RECIPIENT", "FROM 2")]
    assert len(messages_cd1) == 2
