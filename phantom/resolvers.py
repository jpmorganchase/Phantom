import abc
from dataclasses import dataclass
from typing import Generic, List, Tuple, TYPE_CHECKING

from .types import AgentID
from .message import Message, MessageType

if TYPE_CHECKING:
    from .network import Network


@dataclass(frozen=True)
class TrackedMessage(Generic[MessageType]):
    """Immutable message structure."""

    sender_id: AgentID
    receiver_id: AgentID

    message: MessageType


class Resolver(abc.ABC):
    """Network message resolver.

    This type is responsible for resolution processing. That is, the order in
    which (and any special logic therein) messages are handled in a Network.

    In many cases, this type can be arbitrary since the sequence doesn't matter
    (i.e. the problem is not path dependent). In other cases, however, this is
    not the case; e.g. processing incoming market orders in an LOB.

    Implementations of this class must provide implementations of the abstract methods
    below.
    """

    def __init__(self, enable_tracking: bool = False) -> None:
        self.enable_tracking = enable_tracking
        self._tracked_messages: List[TrackedMessage] = []

    def push(self, sender_id: AgentID, receiver_id: AgentID, message: Message) -> None:
        if self.enable_tracking:
            self._tracked_messages.append(
                TrackedMessage(sender_id, receiver_id, message)
            )

        self.handle_push(sender_id, receiver_id, message)

    def clear_tracked_messages(self) -> None:
        self._tracked_messages.clear()

    @property
    def tracked_messages(self) -> List[TrackedMessage]:
        return self._tracked_messages

    @abc.abstractmethod
    def handle_push(
        self, sender_id: AgentID, receiver_id: AgentID, message: Message
    ) -> None:
        """
        Called by the resolver to handle batches of messages. Any further created
        messages (e.g. responses from agents) must be handled by being passed to the
        `push` method (not `handle_push`).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def resolve(self, network: "Network") -> None:
        """Process queues messages for a (sub) set of network contexts.

        Arguments:
            network: An instance of the Network class to resolve.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class BatchResolver(Resolver):
    def __init__(self, enable_tracking: bool = False, chain_limit: int = 2) -> None:
        super().__init__(enable_tracking)

        self.chain_limit = chain_limit

        self.queue: List[Tuple[AgentID, AgentID, Message]] = []

    def reset(self) -> None:
        self.queue.clear()

    def handle_push(
        self, sender_id: AgentID, receiver_id: AgentID, message: Message
    ) -> None:
        self.queue.append((sender_id, receiver_id, message))

    def resolve(self, network: "Network") -> None:
        for _ in range(self.chain_limit):
            if len(self.queue) == 0:
                break

            processing_queue = self.queue
            self.queue = []

            for sender_id, receiver_id, message in processing_queue:
                ctx = network.context_for(receiver_id)

                for sub_msg_receiver_id, sub_message in ctx.agent.handle_message(
                    ctx, sender_id, message
                ):
                    self.push(ctx.agent.id, sub_msg_receiver_id, sub_message)
