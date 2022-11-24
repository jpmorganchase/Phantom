from dataclasses import dataclass

import pytest

import phantom as ph

from . import MockAgent, MockSampler, MockStrategicAgent


def test_repr():
    assert str(MockAgent("AgentID")) == "[MockAgent AgentID]"


def test_reset():
    st = MockStrategicAgent.Supertype(MockSampler(0))

    agent = MockStrategicAgent("Agent", supertype=st)

    assert agent.supertype == st
    assert agent.type is None

    agent.reset()

    assert agent.type == MockStrategicAgent.Supertype(1)

    class MockAgent2(ph.StrategicAgent):
        @dataclass
        class Supertype(ph.Supertype):
            type_value: float

    agent = MockAgent2("Agent", supertype=MockStrategicAgent.Supertype(0))
    agent.reset()

    agent = MockAgent2("Agent")

    with pytest.raises(Exception):
        agent.reset()
