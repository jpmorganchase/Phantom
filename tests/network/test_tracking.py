from dataclasses import dataclass
from typing import List, Tuple

from phantom import AgentID, Context, Network, Message
from phantom.agents import msg_handler, MessageHandlerAgent
from phantom.message import MsgPayload
from phantom.resolvers import BatchResolver
from phantom.views import EnvView


class MockEnv:
    def view(self):
        return None


@dataclass(frozen=True)
class _TestMessage(MsgPayload):
    value: int


class _TestActor(MessageHandlerAgent):
    @msg_handler(_TestMessage)
    def handle_request(
        self, _: Context, message: _TestMessage
    ) -> List[Tuple[AgentID, Message]]:
        if message.payload.value > 1:
            return [(message.sender_id, _TestMessage(message.payload.value // 2))]


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
    n.resolve({aid: n.context_for(aid, EnvView(0)) for aid in n.agents})

    assert resolver.tracked_messages == [
        Message("A", "B", _TestMessage(4)),
        Message("A", "C", _TestMessage(4)),
        Message("B", "A", _TestMessage(2)),
        Message("C", "A", _TestMessage(2)),
        Message("A", "B", _TestMessage(1)),
        Message("A", "C", _TestMessage(1)),
    ]

    n.resolver.clear_tracked_messages()

    assert resolver.tracked_messages == []
