import abc
import logging
from collections import defaultdict
from typing import Callable, DefaultDict, List, TYPE_CHECKING

from .types import AgentID
from .message import Message
from .views import EnvView

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
    """

    def __init__(self, enable_tracking: bool = False) -> None:
        self.enable_tracking = enable_tracking
        self._tracked_messages: List[Message] = []

    def push(self, message: Message) -> None:
        if self.enable_tracking:
            self._tracked_messages.append(message)

        self.handle_push(message)

    def clear_tracked_messages(self) -> None:
        self._tracked_messages.clear()

    @property
    def tracked_messages(self) -> List[Message]:
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
    def resolve(self, network: "Network", env_view_fn: Callable[[], EnvView]) -> None:
        """Process queues messages for a (sub) set of network contexts.

        Arguments:
            network: An instance of the Network class to resolve.
            env_view_fn: Reference to the :meth:`env.view` method returning the public
                environment view applicable to all agents.
        """
        raise NotImplementedError

    @abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class BatchResolver(Resolver):
    def __init__(self, enable_tracking: bool = False, chain_limit: int = 2) -> None:
        super().__init__(enable_tracking)

        self.chain_limit = chain_limit

        self.messages: DefaultDict[AgentID, List[Message]] = defaultdict(list)

    def reset(self) -> None:
        self.messages.clear()

    def handle_push(self, message: Message) -> None:
        self.messages[message.receiver_id].append(message)

    def resolve(self, network: "Network", env_view_fn: Callable[[], EnvView]) -> None:
        for _ in range(self.chain_limit):
            if len(self.messages) == 0:
                break

            processing_messages = self.messages
            self.messages = defaultdict(list)

            env_view = env_view_fn()

            for receiver_id, messages in processing_messages.items():
                ctx = network.context_for(receiver_id, env_view)

                batch = ctx.agent.handle_batch(ctx, messages)

                if batch is not None:
                    for sub_receiver_id, sub_payload in batch:
                        self.push(Message(receiver_id, sub_receiver_id, sub_payload))

        if len(self.messages) > 0:
            logging.getLogger("BatchResolver").warning(
                f"{len(self.messages)} message(s) still in queue after resolver chain limit reached."
            )
