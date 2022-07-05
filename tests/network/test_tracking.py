from dataclasses import dataclass
from typing import List, Optional, Tuple

from phantom import AgentID, Context, Network, Message
from phantom.agents import msg_handler, MessageHandlerAgent
from phantom.resolvers import BatchResolver, TrackedMessage


class MockEnv:
    def view(self):
        return None


@dataclass(frozen=True)
class _TestMessage(Message):
    value: int


class _TestActor(MessageHandlerAgent):
    @msg_handler(_TestMessage)
    def handle_request(
        self, _: Context, sender_id: AgentID, message: _TestMessage
    ) -> List[Tuple[AgentID, Message]]:
        if message.value > 1:
            return [(sender_id, _TestMessage(message.value // 2))]


def test_tracking():
    resolver = BatchResolver(enable_tracking=True)
    n = Network(
        [
            _TestActor("A"),
            _TestActor("B"),
            _TestActor("C"),
        ],
        resolver,
    )
    n.add_connection("A", "B")
    n.add_connection("A", "C")

    n.send("A", "B", _TestMessage(4))
    n.send("A", "C", _TestMessage(4))
    n.resolve()

    assert resolver.tracked_messages == [
        TrackedMessage("A", "B", _TestMessage(4)),
        TrackedMessage("A", "C", _TestMessage(4)),
        TrackedMessage("B", "A", _TestMessage(2)),
        TrackedMessage("C", "A", _TestMessage(2)),
        TrackedMessage("A", "B", _TestMessage(1)),
        TrackedMessage("A", "C", _TestMessage(1)),
    ]

    n.resolver.clear_tracked_messages()

    assert resolver.tracked_messages == []
