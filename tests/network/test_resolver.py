import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from phantom import AgentID, Context, Network, Message, Message
from phantom.agents import msg_handler, MessageHandlerAgent
from phantom.resolvers import BatchResolver


@dataclass(frozen=True)
class Request(Message):
    cash: float


@dataclass(frozen=True)
class Response(Message):
    cash: float


class _TestAgent(MessageHandlerAgent):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.req_time = time.time()
        self.res_time = time.time()

    @msg_handler(Request)
    def handle_request(
        self, _: Context, sender_id: AgentID, message: Request
    ) -> List[Tuple[AgentID, Message]]:
        self.req_time = time.time()

        return [(sender_id, Response(message.cash / 2.0))]

    @msg_handler(Response)
    def handle_response(
        self, _: Context, sender_id: AgentID, message: Response
    ) -> List[Tuple[AgentID, Message]]:
        self.res_time = time.time()

        return [(sender_id, None)]


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
    n.resolve()

    assert n["A"].req_time <= n["B"].req_time
    assert n["B"].req_time <= n["C"].req_time

    assert n["C"].res_time <= n["A"].res_time
    assert n["A"].res_time <= n["B"].res_time
