from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytest

from phantom import AgentID, Context, Message, Message
from phantom.agents import msg_handler, MessageHandlerAgent
from phantom.network import BatchResolver, Network


@dataclass(frozen=True)
class MyMessage(Message):
    cash: float


class MyAgent(MessageHandlerAgent):
    def __init__(self, aid: AgentID) -> None:
        super().__init__(aid)

        self.total_cash = 0.0

    def reset(self) -> None:
        self.total_cash = 0.0

    @msg_handler(MyMessage)
    def handle_message(
        self, _: Context, sender_id: AgentID, message: MyMessage
    ) -> List[Tuple[AgentID, Message]]:
        self.total_cash += message.cash / 2.0

        return [(sender_id, MyMessage(message.cash / 2.0))]


@pytest.fixture
def net() -> Network:
    n = Network([MyAgent("mm"), MyAgent("inv")], BatchResolver(2))
    n.add_connection("mm", "inv")

    return n


def test_getters(net):
    assert "mm" in net.agent_ids
    assert "inv" in net.agent_ids

    agents = net.get_agents_where(lambda a: a.id == "mm")

    assert len(agents) == 1
    assert list(agents.keys())[0] == "mm"

    assert net.get_agents_with_type(MessageHandlerAgent) == net.agents
    assert net.get_agents_without_type(MessageHandlerAgent) == {}


def test_call_response(net):
    net.send("mm", "inv", MyMessage(100.0))
    net.resolve()

    assert net.agents["mm"].total_cash == 25.0
    assert net.agents["inv"].total_cash == 50.0


def test_send_many(net):
    net.send("mm", "inv", MyMessage(100.0))
    net.send("mm", "inv", MyMessage(100.0))
    net.resolve()

    assert net.agents["mm"].total_cash == 50.0
    assert net.agents["inv"].total_cash == 100.0


def test_context_existence(net):
    assert "inv" in net.context_for("mm")
    assert "mm" in net.context_for("inv")


def test_reset(net):
    net.send("mm", "inv", MyMessage(100.0))
    net.send("mm", "inv", MyMessage(100.0))
    net.resolve()
    net.reset()

    assert net.agents["mm"].total_cash == 0.0
    assert net.agents["inv"].total_cash == 0.0
