import abc
import logging
from collections import defaultdict
from typing import DefaultDict, List, Mapping, TYPE_CHECKING

from .context import Context
from .types import AgentID
from .message import Message

if TYPE_CHECKING:
    from .network import Network


class Resolver(abc.ABC):
    """Network message resolver.

    This type is responsible for resolution processing. That is, the order in
    which (and any special logic therein) messages are handled in a Network.

    In many cases, this type can be arbitrary since the sequence doesn't matter
    (i.e. the problem is not path dependent). In other cases, however, this is
    not the case; e.g. processing incoming market orders in an LOB.

    Implementations of this class must provide implementations of the abstract methods
    below.

    Arguments:
        enable_tracking: If True, the resolver should save all messages in an
            time-ordered list that can be accessed with :attr:`tracked_messages`.
    """

    def __init__(self, enable_tracking: bool = False) -> None:
        self.enable_tracking = enable_tracking
        self._tracked_messages: List[Message] = []

    def push(self, message: Message) -> None:
        """Called by the Network to add messages to the resolver."""
        if self.enable_tracking:
            self._tracked_messages.append(message)

        self.handle_push(message)

    def clear_tracked_messages(self) -> None:
        """Clears any stored messages.

        Useful for when incrementally processing/storing batches of tracked messages.
        """
        self._tracked_messages.clear()

    @property
    def tracked_messages(self) -> List[Message]:
        """
        Returns all messages that have passed through the resolver if tracking is enabled.
        """
        return self._tracked_messages

    @abc.abstractmethod
    def handle_push(self, message: Message) -> None:
        """
        Called by the resolver to handle batches of messages. Any further created
        messages (e.g. responses from agents) must be handled by being passed to the
        `push` method (not `handle_push`).
        """
        raise NotImplementedError

    @abc.abstractmethod
    def resolve(self, network: "Network", contexts: Mapping[AgentID, Context]) -> None:
        """Process queues messages for a (sub) set of network contexts.

        Arguments:
            network: An instance of the Network class to resolve.
            contexts: The contexts for all agents for the current step.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        """Resets the resolver and clears any potential message queues.

        Note:
            Does not clear any tracked messages.
        """
        raise NotImplementedError


class BatchResolver(Resolver):
    """
    Resolver that allows each agent to send/respond to messages before delivering the
    messages to the recipient agents.

    Arguments:
        enable_tracking: If True, the resolver should save all messages in an
            time-ordered list that can be accessed with :attr:`tracked_messages`.
        chain_limit: The maximum number of rounds of messages to resolve. If the limit
            is reached a warning will be logged.
    """

    def __init__(self, enable_tracking: bool = False, chain_limit: int = 2) -> None:
        super().__init__(enable_tracking)

        self.chain_limit = chain_limit

        self.messages: DefaultDict[AgentID, List[Message]] = defaultdict(list)

    def reset(self) -> None:
        self.messages.clear()

    def handle_push(self, message: Message) -> None:
        self.messages[message.receiver_id].append(message)

    def resolve(self, network: "Network", contexts: Mapping[AgentID, Context]) -> None:
        for _ in range(self.chain_limit):
            if len(self.messages) == 0:
                break

            processing_messages = self.messages
            self.messages = defaultdict(list)

            for receiver_id, messages in processing_messages.items():
                ctx = contexts[receiver_id]
                batch = ctx.agent.handle_batch(ctx, messages)

                if batch is not None:
                    for sub_receiver_id, sub_payload in batch:
                        self.push(Message(receiver_id, sub_receiver_id, sub_payload))

        if len(self.messages) > 0:
            logging.getLogger("BatchResolver").warning(
                "%s message(s) still in queue after resolver chain limit reached.",
                len(self.messages),
            )
