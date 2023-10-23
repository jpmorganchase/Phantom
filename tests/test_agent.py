from dataclasses import dataclass

import pytest

import phantom as ph

from . import MockAgent, MockSampler, MockStrategicAgent


def test_repr():
    assert str(MockAgent("AgentID")) == "[MockAgent AgentID]"


def test_reset():
    st = MockStrategicAgent.Supertype(MockSampler(1))

    agent = MockStrategicAgent("Agent", supertype=st)

    assert agent.supertype == st

    agent.reset()

    assert agent.type == MockStrategicAgent.Supertype(2)

    class MockAgent2(ph.StrategicAgent):
        @dataclass
        class Supertype(ph.Supertype):
            type_value: float

    agent = MockAgent2("Agent", supertype=MockStrategicAgent.Supertype(0))
    agent.reset()


@dataclass(frozen=True)
class MockPayload1(ph.MsgPayload):
    value: float = 0.0


@dataclass(frozen=True)
class MockPayload2(ph.MsgPayload):
    value: float = 0.0


class MockAgent3(ph.StrategicAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.mock_msg_1_recv = 0

    @ph.agents.msg_handler(MockPayload1)
    def handle_mock_message_1(self, ctx, message):
        self.mock_msg_1_recv += 1


def test_message_handling():
    agent = MockAgent3("Agent")
    agent.reset()

    agent.handle_message(None, ph.Message("", "Agent", MockPayload1()))

    assert agent.mock_msg_1_recv == 1

    agent.handle_batch(None, [ph.Message("", "Agent", MockPayload1())])

    assert agent.mock_msg_1_recv == 2

    with pytest.raises(ValueError):
        agent.handle_message(None, ph.Message("", "Agent", MockPayload2()))
