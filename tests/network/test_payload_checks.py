import pytest

import phantom as ph


class AgentA(ph.Agent):
    pass


class AgentB(ph.Agent):
    pass


@ph.msg_payload()
class Payload1:
    pass


@ph.msg_payload(AgentA, AgentB)
class Payload2:
    pass


def test_payload_checks():
    agents = [
        AgentA("A"),
        AgentB("B"),
    ]

    net = ph.Network(agents, enforce_msg_payload_checks=True)
    net.add_connection("A", "B")

    net.send("A", "B", Payload1())
    net.send("B", "A", Payload1())

    net.send("A", "B", Payload2())

    with pytest.raises(ph.network.NetworkError):
        net.send("B", "A", Payload2())

    net = ph.Network(agents, enforce_msg_payload_checks=False)
    net.add_connection("A", "B")

    net.send("A", "B", Payload2())
    net.send("B", "A", Payload2())
