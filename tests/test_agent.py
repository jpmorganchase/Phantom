from dataclasses import dataclass

from . import MockAgent, MockSampler, MockSupertype


def test_repr():
    assert str(MockAgent("Agent")) == "[MockAgent Agent]"


def test_reset():
    st = MockSupertype(MockSampler())

    agent = MockAgent("Agent", supertype=st)

    assert agent.supertype == st
    assert agent.type is None

    agent.reset()

    assert agent.type == MockSupertype(0.0)
