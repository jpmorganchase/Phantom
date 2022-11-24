from dataclasses import dataclass

import pytest

import phantom as ph

from . import MockAgent, MockSampler


def test_repr():
    assert str(MockAgent("AgentID")) == "[MockAgent AgentID]"


def test_reset():
    st = MockAgent.Supertype(MockSampler(0))

    agent = MockAgent("Agent", supertype=st)

    assert agent.supertype == st
    assert agent.type is None

    agent.reset()

    assert agent.type == MockAgent.Supertype(1)

    class MockAgent2(ph.RLAgent):
        @dataclass
        class Supertype(ph.Supertype):
            type_value: float

    agent = MockAgent2("Agent", supertype=MockAgent.Supertype(0))
    agent.reset()

    agent = MockAgent2("Agent")

    with pytest.raises(Exception):
        agent.reset()
