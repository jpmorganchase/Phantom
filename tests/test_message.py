from dataclasses import dataclass
import warnings

import phantom as ph


def test_payload_1():
    @ph.msg_payload()
    class MockPayload:
        value: float = 0.0

    assert MockPayload._sender_types is None
    assert MockPayload._receiver_types is None


def test_payload_2():
    @ph.msg_payload(sender_type=ph.Agent, receiver_type="OtherAgent")
    class MockPayload:
        value: float = 0.0

    assert MockPayload._sender_types == ["Agent"]
    assert MockPayload._receiver_types == ["OtherAgent"]


def test_payload_3():
    @ph.msg_payload(
        sender_type=["AgentA", "AgentB"], receiver_type=["AgentC", "AgentD"]
    )
    class MockPayload:
        value: float = 0.0

    assert MockPayload._sender_types == ["AgentA", "AgentB"]
    assert MockPayload._receiver_types == ["AgentC", "AgentD"]


def test_old_payload():
    @dataclass(frozen=True)
    class MockPayload(ph.MsgPayload):
        value: float = 0.0

    net = ph.Network([ph.Agent("a"), ph.Agent("b")], connections=[("a", "b")])

    net.enforce_msg_payload_checks = True

    with warnings.catch_warnings(record=True) as w:
        net.send("a", "b", MockPayload(1.0))
        assert len(w) == 1
        assert isinstance(w[0].message, DeprecationWarning)

    # Warning only gets raised once:
    with warnings.catch_warnings(record=True) as w:
        net.send("a", "b", MockPayload(1.0))
        assert len(w) == 0

    net.enforce_msg_payload_checks = False
    net.send("a", "b", MockPayload(1.0))
