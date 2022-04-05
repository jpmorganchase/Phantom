from dataclasses import dataclass

import pytest

from mercury import ID, Network, Message, Payload
from mercury.actors import SimpleSyncActor, Responses, handler
from mercury.resolvers import UnorderedResolver


@dataclass(frozen=True)
class MessageA(Payload):
    pass


@dataclass(frozen=True)
class MessageB(Payload):
    pass


@dataclass(frozen=True)
class MessageC(Payload):
    pass


class SendingActorA(SimpleSyncActor):
    def __init__(self):
        SimpleSyncActor.__init__(self, "sA")


class SendingActorB(SimpleSyncActor):
    def __init__(self):
        SimpleSyncActor.__init__(self, "sB")


class SendingActorC(SimpleSyncActor):
    def __init__(self):
        SimpleSyncActor.__init__(self, "sC")


def test_a():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts any message from any actor
        @handler()
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")

    n.send({"sA": {"R": [MessageA()]}})
    n.send({"sB": {"R": [MessageA()]}})
    n.resolve()


def test_b():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts a single message type from any actor type
        @handler(MessageA)
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")

    n.send({"sA": {"R": [MessageA()]}})
    n.send({"sB": {"R": [MessageA()]}})
    n.resolve()

    with pytest.raises(KeyError):
        n.send({"sA": {"R": [MessageB()]}})
        n.resolve()

    with pytest.raises(KeyError):
        n.send({"sB": {"R": [MessageB()]}})
        n.resolve()


def test_c():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts any message type from a single actor type
        @handler(sending_actor_types=SendingActorA)
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")

    n.send({"sA": {"R": [MessageA()]}})
    n.send({"sA": {"R": [MessageB()]}})
    n.resolve()

    with pytest.raises(KeyError):
        n.send({"sB": {"R": [MessageA()]}})
        n.resolve()

    with pytest.raises(KeyError):
        n.send({"sB": {"R": [MessageB()]}})
        n.resolve()


def test_d():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts a single message type from a single actor type
        @handler(MessageA, SendingActorA)
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")

    n.send({"sA": {"R": [MessageA()]}})
    n.resolve()

    with pytest.raises(KeyError):
        n.send({"sA": {"R": [MessageB()]}})
        n.resolve()

    with pytest.raises(KeyError):
        n.send({"sB": {"R": [MessageA()]}})
        n.resolve()

    with pytest.raises(KeyError):
        n.send({"sB": {"R": [MessageB()]}})
        n.resolve()


def test_e():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts multiple message types from any actor type
        @handler((MessageA, MessageB))
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")

    n.send({"sA": {"R": [MessageA()]}})
    n.send({"sA": {"R": [MessageB()]}})
    n.send({"sB": {"R": [MessageA()]}})
    n.send({"sB": {"R": [MessageB()]}})
    n.resolve()

    with pytest.raises(KeyError):
        n.send({"sA": {"R": [MessageC()]}})
        n.resolve()


def test_f():
    class ReceivingActor(SimpleSyncActor):
        def __init__(self):
            SimpleSyncActor.__init__(self, "R")

        # Accepts any message type from multiple actor types
        @handler(sending_actor_types=(SendingActorA, SendingActorB))
        def handler(self, _ctx: Network.Context, msg: Message[MessageA]) -> Responses:
            yield from ()

    resolver = UnorderedResolver()
    n = Network(
        resolver,
        [
            ReceivingActor(),
            SendingActorA(),
            SendingActorB(),
            SendingActorC(),
        ],
    )
    n.add_connection("R", "sA")
    n.add_connection("R", "sB")
    n.add_connection("R", "sC")

    n.send({"sA": {"R": [MessageA()]}})
    n.send({"sA": {"R": [MessageB()]}})
    n.send({"sB": {"R": [MessageA()]}})
    n.send({"sB": {"R": [MessageB()]}})
    n.resolve()

    with pytest.raises(KeyError):
        n.send({"sC": {"R": [MessageA()]}})
        n.resolve()
