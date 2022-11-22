import phantom as ph

from . import MockAgent, MockSampler


def test_repr():
    assert str(ph.Agent("Agent")) == "[Agent Agent]"


def test_reset():
    st = MockAgent.Supertype(MockSampler(0.0))

    agent = ph.Agent("Agent", supertype=st)

    assert agent.supertype == st
    assert agent.type is None

    agent.reset()

    assert agent.type == MockAgent.Supertype(0.0)
