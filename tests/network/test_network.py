from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pytest

from phantom import AgentID, Context, Message, Message
from phantom.agents import msg_handler, MessageHandlerAgent
from phantom.message import MsgPayload
from phantom.network import BatchResolver, Network


@dataclass(frozen=True)
class MyMessage(MsgPayload):
    cash: float


class MyAgent(MessageHandlerAgent):
    def __init__(self, aid: AgentID) -> None:
        super().__init__(aid)

        self.total_cash = 0.0

    def reset(self) -> None:
        self.total_cash = 0.0

    @msg_handler(MyMessage)
    def handle_message(
        self, _: Context, message: MyMessage
    ) -> List[Tuple[AgentID, MsgPayload]]:
        self.total_cash += message.payload.cash / 2.0

        return [(message.sender_id, MyMessage(message.payload.cash / 2.0))]


@pytest.fixture
def net() -> Network:
    n = Network([MyAgent("mm"), MyAgent("inv")], BatchResolver(chain_limit=2))
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


@pytest.fixture
def net2() -> Network:
    return Network([MyAgent("a"), MyAgent("b"), MyAgent("c")])


def test_adjacency_matrix(net2):
    net2.add_connections_with_adjmat(["a", "b"], np.array([[0, 1], [1, 0]]))

    with pytest.raises(ValueError) as e:
        net2.add_connections_with_adjmat(["a", "b", "c"], np.array([[0, 1], [1, 0]]))

    assert (
        str(e.value) == "Number of agent IDs doesn't match adjacency matrix dimensions."
    )

    with pytest.raises(ValueError) as e:
        net2.add_connections_with_adjmat(["a", "b"], np.array([[0, 0, 0], [0, 0, 0]]))

    assert str(e.value) == "Adjacency matrix must be square."

    with pytest.raises(ValueError) as e:
        net2.add_connections_with_adjmat(["a", "b"], np.array([[0, 0], [1, 0]]))

    assert str(e.value) == "Adjacency matrix must be symmetric."

    with pytest.raises(ValueError) as e:
        net2.add_connections_with_adjmat(["a", "b"], np.array([[1, 1], [1, 1]]))

    assert str(e.value) == "Adjacency matrix must be hollow."
