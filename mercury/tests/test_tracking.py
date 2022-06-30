import time
from dataclasses import dataclass

from mercury import Network, Message, Payload
from mercury.actors import SimpleSyncActor, Responses, handler
from mercury.resolvers import UnorderedResolver


@dataclass(frozen=True)
class _TestPayload(Payload):
    value: int


class _TestActor(SimpleSyncActor):
    @handler(_TestPayload)
    def handle_request(
        self, _ctx: Network.Context, msg: Message[_TestPayload]
    ) -> Responses:
        if msg.payload.value > 1:
            yield msg.sender_id, [_TestPayload(msg.payload.value // 2)]


def test_tracking():
    resolver = UnorderedResolver(enable_tracking=True)
    n = Network(
        resolver,
        [
            _TestActor("A"),
            _TestActor("B"),
            _TestActor("C"),
        ],
    )
    n.add_connection("A", "B")
    n.add_connection("A", "C")

    n.send(
        {
            "A": {"B": [_TestPayload(4)], "C": [_TestPayload(4)]},
        }
    )
    n.resolve()

    assert resolver.tracked_messages == [
        Message("A", "B", _TestPayload(4)),
        Message("A", "C", _TestPayload(4)),
        Message("B", "A", _TestPayload(2)),
        Message("C", "A", _TestPayload(2)),
        Message("A", "B", _TestPayload(1)),
        Message("A", "C", _TestPayload(1)),
    ]

    n.resolver.clear_tracked_messages()

    assert resolver.tracked_messages == []
