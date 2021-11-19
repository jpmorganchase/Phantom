import time
from dataclasses import dataclass

from mercury import Network, Message, Payload
from mercury.actors import SimpleSyncActor, Responses, handler
from mercury.resolvers import UnorderedResolver


@dataclass(frozen=True)
class Request(Payload):
    cash: float


@dataclass(frozen=True)
class Response(Payload):
    cash: float


class _TestActor(SimpleSyncActor):
    def __init__(self, *args, **kwargs):
        SimpleSyncActor.__init__(self, *args, **kwargs)

        self.req_time = time.time()
        self.res_time = time.time()

    @handler(Request)
    def handle_request(self, _ctx: Network.Context, msg: Message[Request]) -> Responses:
        self.req_time = time.time()

        yield msg.sender_id, [Response(msg.payload.cash / 2.0)]

    @handler(Response)
    def handle_response(
        self, _ctx: Network.Context, msg: Message[Response]
    ) -> Responses:
        self.res_time = time.time()

        yield msg.sender_id, []


def test_ordering():
    resolver = UnorderedResolver()
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
    n.add_connection("B", "C")

    n.send({"A": {"B": [Request(100.0)], "C": [Request(100.0)]}})
    n.send({"B": {"C": [Request(100.0)]}})
    n.resolve()

    assert n["A"].req_time <= n["B"].req_time
    assert n["B"].req_time <= n["C"].req_time

    assert n["C"].res_time <= n["A"].res_time
    assert n["A"].res_time <= n["B"].res_time
