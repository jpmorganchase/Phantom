import time
from dataclasses import dataclass
from typing import List, Tuple

from phantom import AgentID, Context, Network
from phantom.agents import msg_handler, Agent
from phantom.message import MsgPayload
from phantom.resolvers import BatchResolver
from phantom.views import EnvView


@dataclass(frozen=True)
class Request(MsgPayload):
    cash: float


@dataclass(frozen=True)
class Response(MsgPayload):
    cash: float


class _TestAgent(Agent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.req_time = time.time()
        self.res_time = time.time()

    @msg_handler(Request)
    def handle_request(
        self, _: Context, message: Request
    ) -> List[Tuple[AgentID, MsgPayload]]:
        self.req_time = time.time()

        return [(message.sender_id, Response(message.payload.cash / 2.0))]

    @msg_handler(Response)
    def handle_response(
        self, _: Context, message: Response
    ) -> List[Tuple[AgentID, MsgPayload]]:
        self.res_time = time.time()

        return []


def test_ordering():
    n = Network(
        [
            _TestAgent("A"),
            _TestAgent("B"),
            _TestAgent("C"),
        ],
        BatchResolver(),
    )
    n.add_connection("A", "B")
    n.add_connection("A", "C")
    n.add_connection("B", "C")

    n.send("A", "B", Request(100.0))
    n.send("A", "C", Request(100.0))
    n.send("B", "C", Request(100.0))
    n.resolve({aid: n.context_for(aid, EnvView(0)) for aid in n.agents})

    assert n["A"].req_time <= n["B"].req_time
    assert n["B"].req_time <= n["C"].req_time

    assert n["C"].res_time <= n["A"].res_time
    assert n["A"].res_time <= n["B"].res_time


def test_batch_resolver_chain_limit(caplog):
    n = Network(
        [
            _TestAgent("A"),
            _TestAgent("B"),
        ],
        BatchResolver(chain_limit=0),
    )
    n.add_connection("A", "B")

    n.send("A", "B", Request(0))

    assert len(caplog.records) == 0
    n.resolve({aid: n.context_for(aid, EnvView(0)) for aid in n.agents})
    assert len(caplog.records) == 1

    assert (
        "1 message(s) still in queue after resolver chain limit reached." in caplog.text
    )
