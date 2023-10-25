import pytest

import phantom as ph


class AgentA(ph.Agent):
    pass


class AgentB(ph.Agent):
    pass


def test_payload_checks():
    agents = [
        AgentA("A"),
        AgentB("B"),
    ]

    net = ph.Network(agents, enforce_msg_payload_checks=True)
    net.add_connection("A", "B")

    @ph.msg_payload()
    class Payload1:
        pass

    net.send("A", "B", Payload1())
    net.send("B", "A", Payload1())

    @ph.msg_payload(AgentA, AgentB)
    class Payload2:
        pass

    net.send("A", "B", Payload2())

    @ph.msg_payload([AgentA, AgentB], [AgentA, AgentB])
    class Payload3:
        pass

    net.send("A", "B", Payload3())
    net.send("B", "A", Payload3())

    @ph.msg_payload(sender_type=AgentA, receiver_type=None)
    class Payload4:
        pass

    @ph.msg_payload(sender_type=None, receiver_type=AgentA)
    class Payload5:
        pass

    net.send("A", "B", Payload4())
    net.send("B", "A", Payload5())

    with pytest.raises(ph.network.NetworkError):
        net.send("B", "A", Payload2())

    net = ph.Network(agents, enforce_msg_payload_checks=False)
    net.add_connection("A", "B")

    net.send("A", "B", Payload2())
    net.send("B", "A", Payload2())
