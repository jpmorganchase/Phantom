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
