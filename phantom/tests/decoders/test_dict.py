import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.decoders import Decoder, DictDecoder


class MockDecoder(Decoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: me.Network.Context, action) -> ph.Packet:
        assert action == self.id

        return ph.Packet(messages={"RECIPIENT": [f"FROM {self.id}"]})

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

    packet = dd.decode(None, {"d1": 1, "d2": 2})

    assert list(packet.mutations) == []
    assert len(packet.messages) == 1
    assert list(packet.messages["RECIPIENT"]) == ["FROM 1", "FROM 2"]


def test_chained_decoder_reset():
    d1 = MockDecoder(1)
    d2 = MockDecoder(2)

    dd = DictDecoder({"d1": d1, "d2": d2})

    dd.reset()

    assert d1.id is None
    assert d2.id is None
