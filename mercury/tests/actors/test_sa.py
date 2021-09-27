"""Tests for the SyncActor."""
import pytest

from mercury import ID, Message, Batch, Network
from mercury.actors import SyncActor, SimpleSyncActor
from mercury.resolvers import UnorderedResolver


def test_message_ordering():
    # Checks that SyncActor messages are handled in a First-In-First-Out
    # (FIFO) order.
    request_messages = []
    response_messages = []

    class ExtendedSyncActor(SyncActor):
        def handle_message(self, ctx, message):
            request_messages.append(message.payload)

            yield from ()

    n = Network(UnorderedResolver(2), [SimpleSyncActor("a"), ExtendedSyncActor("b")])
    n.add_connection("a", "b")

    n.send_from("a", {"b": ["req-msg-1"]})
    n.send_from("a", {"b": ["req-msg-2"]})
    n.send_from("a", {"b": ["req-msg-3"]})

    n.resolve()

    assert request_messages == ["req-msg-1", "req-msg-2", "req-msg-3"]
