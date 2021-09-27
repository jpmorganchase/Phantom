import pytest
import typing as _t

from dataclasses import dataclass

from mercury import ID, Network, Message, Payload
from mercury.actors import SimpleSyncActor, Responses, View, handler
from mercury.resolvers import UnorderedResolver


@dataclass(frozen=True)
class MyPayload(Payload):
    cash: float


class MyActor(SimpleSyncActor):
    def __init__(self, aid: ID) -> None:
        SimpleSyncActor.__init__(self, aid)

        self.total_cash = 0.0

    def reset(self) -> None:
        self.total_cash = 0.0

    @handler(MyPayload)
    def handle_payload(
        self, ctx: Network.Context, msg: Message[MyPayload]
    ) -> Responses:
        self.total_cash += msg.payload.cash / 2.0

        yield msg.sender_id, [MyPayload(msg.payload.cash / 2.0)]


@pytest.fixture
def net() -> Network:
    n = Network(UnorderedResolver(2), [MyActor("mm"), MyActor("inv")])
    n.add_connection("mm", "inv")

    return n


def test_getters(net):
    assert "mm" in net.actor_ids
    assert "inv" in net.actor_ids

    actors = net.get_actors_where(lambda a: a.id == "mm")

    assert len(actors) == 1
    assert list(actors.keys())[0] == "mm"

    assert net.get_actors_with_type(SimpleSyncActor) == net.actors
    assert net.get_actors_without_type(SimpleSyncActor) == {}


def test_call_response(net):
    net.send({"mm": {"inv": [MyPayload(100.0)]}})
    net.resolve()

    assert net.actors["mm"].total_cash == 25.0
    assert net.actors["inv"].total_cash == 50.0


def test_send_many(net):
    net.send({"mm": {"inv": [MyPayload(100.0), MyPayload(100.0)]}})
    net.resolve()

    assert net.actors["mm"].total_cash == 50.0
    assert net.actors["inv"].total_cash == 100.0


def test_context_existence(net):
    assert "inv" in net.context_for("mm")
    assert "mm" in net.context_for("inv")


def test_reset(net):
    net.send({"mm": {"inv": [MyPayload(100.0), MyPayload(100.0)]}})
    net.resolve()
    net.reset()

    assert net.actors["mm"].total_cash == 0.0
    assert net.actors["inv"].total_cash == 0.0
