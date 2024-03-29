from typing import List, Tuple

import numpy as np
import pytest

from phantom import AgentID, Context
from phantom.agents import msg_handler, Agent
from phantom.message import MsgPayload, msg_payload
from phantom.network import Network, NetworkError
from phantom.views import EnvView


@msg_payload()
class MockMessage:
    cash: float


class MockAgent(Agent):
    def __init__(self, aid: AgentID) -> None:
        super().__init__(aid)

        self.total_cash = 0.0

    def reset(self) -> None:
        self.total_cash = 0.0

    @msg_handler(MockMessage)
    def handle_message(
        self, _: Context, message: MockMessage
    ) -> List[Tuple[AgentID, MsgPayload]]:
        if message.payload.cash > 25:
            self.total_cash += message.payload.cash / 2.0

            return [(message.sender_id, MockMessage(message.payload.cash / 2.0))]


def test_init():
    # 1
    net = Network([MockAgent("a1"), MockAgent("a2")])
    net.add_connection("a1", "a2")

    # 2
    Network([MockAgent("a1"), MockAgent("a2")], connections=[("a1", "a2")])


def test_bad_init():
    with pytest.raises(ValueError):
        Network([MockAgent("a1"), MockAgent("a1")])

    with pytest.raises(ValueError):
        Network([MockAgent("a1")], connections=[("a1", "a2")])


@pytest.fixture
def net() -> Network:
    net = Network([MockAgent("mm"), MockAgent("inv"), MockAgent("inv2")])
    net.add_connection("mm", "inv")

    return net


def test_getters(net):
    assert "mm" in net.agent_ids
    assert "inv" in net.agent_ids

    agents = net.get_agents_where(lambda a: a.id == "mm")

    assert len(agents) == 1
    assert list(agents.keys())[0] == "mm"

    assert net.get_agents_with_type(Agent) == net.agents
    assert net.get_agents_without_type(Agent) == {}


def test_call_response(net):
    net.send("mm", "inv", MockMessage(100.0))
    net.resolve({aid: net.context_for(aid, EnvView(0, 0.0)) for aid in net.agents})

    assert net.agents["mm"].total_cash == 25.0
    assert net.agents["inv"].total_cash == 50.0


def test_send_many(net):
    net.send("mm", "inv", MockMessage(100.0))
    net.send("mm", "inv", MockMessage(100.0))
    net.resolve({aid: net.context_for(aid, EnvView(0, 0.0)) for aid in net.agents})

    assert net.agents["mm"].total_cash == 50.0
    assert net.agents["inv"].total_cash == 100.0


def test_invalid_send(net):
    with pytest.raises(NetworkError):
        net.send("mm", "inv2", MockMessage(100.0))


def test_context_existence(net):
    assert "inv" in net.context_for("mm", EnvView(0, 0.0))
    assert "mm" in net.context_for("inv", EnvView(0, 0.0))


def test_reset(net):
    net.send("mm", "inv", MockMessage(100.0))
    net.send("mm", "inv", MockMessage(100.0))
    net.resolve({aid: net.context_for(aid, EnvView(0, 0.0)) for aid in net.agents})
    net.reset()

    assert net.agents["mm"].total_cash == 0.0
    assert net.agents["inv"].total_cash == 0.0


@pytest.fixture
def net2() -> Network:
    return Network([MockAgent("a"), MockAgent("b"), MockAgent("c")])


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
