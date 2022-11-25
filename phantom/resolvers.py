import itertools
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import DefaultDict, List, Mapping, Optional, TYPE_CHECKING

import numpy as np

from .context import Context
from .types import AgentID
from .message import Message

if TYPE_CHECKING:
    from .network import Network


class Resolver(ABC):
    """Network message resolver.

    This type is responsible for resolution processing. That is, the order in which
    (and any special logic therein) messages are handled in a Network.

    In many cases, this type can be arbitrary since the sequence doesn't matter (i.e.
    the problem is not path dependent). In other cases, however, this is not the case;
    e.g. processing incoming market orders in an LOB.

    Implementations of this class must provide implementations of the abstract methods
    below.

    Arguments:
        enable_tracking: If True, the resolver will save all messages in a time-ordered
            list that can be accessed with :attr:`tracked_messages`.
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

    @abstractmethod
    def handle_push(self, message: Message) -> None:
        """
        Called by the resolver to handle batches of messages. Any further created
        messages (e.g. responses from agents) must be handled by being passed to the
        `push` method (not `handle_push`).
        """
        raise NotImplementedError

    @abstractmethod
    def resolve(self, network: "Network", contexts: Mapping[AgentID, Context]) -> None:
        """Process queues messages for a (sub) set of network contexts.

        Arguments:
            network: An instance of the Network class to resolve.
            contexts: The contexts for all agents for the current step.
        """
        raise NotImplementedError

    @abstractmethod
    def reset(self) -> None:
        """Resets the resolver and clears any potential message queues.

        Note:
            Does not clear any tracked messages.
        """
        raise NotImplementedError


class BatchResolver(Resolver):
    """
    Resolver that handles messages in multiple discrete rounds. Each round, all agents
    have the opportunity to respond to previously received messages with new messages.
    These messages are held in a queue until all agents have been processed before being
    consumed in the next round. Messages for each agent are delivered in a batch,
    allowing the recipient agent to decide how to handle the messages within the batch.

    Arguments:
        enable_tracking: If True, the resolver will save all messages in a time-ordered
            list that can be accessed with :attr:`tracked_messages`.
        round_limit: The maximum number of rounds of messages to resolve. If the limit
            is reached an exception will be thrown. By default the resolver will keep
            resolving until no more messages are sent.
        shuffle_batches: If True, the order in which messages for a particular
            recipient are sent to the recipient will be randomised.
    """

    def __init__(
        self,
        enable_tracking: bool = False,
        round_limit: Optional[int] = None,
        shuffle_batches: bool = False,
    ) -> None:
        super().__init__(enable_tracking)

        self.round_limit = round_limit
        self.shuffle_batches = shuffle_batches

        self.messages: DefaultDict[AgentID, List[Message]] = defaultdict(list)

    def reset(self) -> None:
        self.messages.clear()

    def handle_push(self, message: Message) -> None:
        self.messages[message.receiver_id].append(message)

    def resolve(self, network: "Network", contexts: Mapping[AgentID, Context]) -> None:
        iterator = (
            itertools.count() if self.round_limit is None else range(self.round_limit)
        )

        for _ in iterator:
            if len(self.messages) == 0:
                break

            processing_messages = self.messages
            self.messages = defaultdict(list)

            for receiver_id, messages in processing_messages.items():
                if receiver_id not in contexts:
                    continue

                msgs = [
                    m for m in messages if network.has_edge(m.sender_id, m.receiver_id)
                ]

                if self.shuffle_batches:
                    np.random.shuffle(msgs)

                ctx = contexts[receiver_id]
                responses = ctx.agent.handle_batch(ctx, msgs)

                if responses is not None:
                    for sub_receiver_id, sub_payload in responses:
                        network.send(receiver_id, sub_receiver_id, sub_payload)

        if len(self.messages) > 0:
            raise Exception(
                f"{len(self.messages)} message(s) still in queue after BatchResolver round limit reached."
            )
