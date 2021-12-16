import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.decoders import ChainedDecoder, Decoder


class SimpleDecoder(Decoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: me.Network.Context, action) -> ph.Packet:
        return ph.Packet(messages={"RECIPIENT": [f"FROM {self.id}"]})

    def reset(self):
        self.id = None


def test_chained_decoder():
    d1 = SimpleDecoder(1)
    d2 = SimpleDecoder(2)

    cd1 = ChainedDecoder([d1, d2])

    packet = cd1.decode(None, [None, None])

    assert list(packet.mutations) == []
    assert len(packet.messages) == 1
    assert list(packet.messages["RECIPIENT"]) == ["FROM 1", "FROM 2"]

    cd2 = d1.chain(d2)

    packet = cd2.decode(None, [None, None])

    assert list(packet.mutations) == []
    assert len(packet.messages) == 1
    assert list(packet.messages["RECIPIENT"]) == ["FROM 1", "FROM 2"]


def test_chained_decoder_reset():
    d1 = SimpleDecoder(1)
    d2 = SimpleDecoder(2)

    cd = ChainedDecoder([d1, d2])

    cd.reset()

    assert d1.id is None
    assert d2.id is None
