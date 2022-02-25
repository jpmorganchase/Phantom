import abc
from typing import Dict, Iterable, List, Mapping, TYPE_CHECKING

from ..core import ID
from ..message import Batch, Message, Payload

if TYPE_CHECKING:
    from ..network import Network


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
        self.tracked_messages: List[Message] = []

    def push(self, from_id: ID, to_id: ID, payloads: Iterable[Payload]) -> None:
        if self.enable_tracking:
            for payload in payloads:
                self.tracked_messages.append(Message(from_id, to_id, payload))

        self.handle_push(from_id, to_id, payloads)

    def clear_tracked_messages(self) -> None:
        self.tracked_messages.clear()

    @abc.abstractmethod
    def handle_push(self, from_id: ID, to_id: ID, payloads: Iterable[Payload]) -> None:
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

        self._q1: Dict[ID, Batch] = dict()

        self._q2: Dict[ID, Batch] = dict()

        self._qi = 0
        self._qs = (self._q1, self._q2)

    @property
    def _cq(self) -> Dict[ID, Batch]:
        """Current queue."""
        return self._qs[self._qi]

    @property
    def _pq(self) -> Dict[ID, Batch]:
        """Populated queue."""
        return self._qs[1 - self._qi]

    def reset(self) -> None:
        self._q1.clear()
        self._q2.clear()

    def handle_push(self, from_id: ID, to_id: ID, payloads: Iterable[Payload]) -> None:
        if to_id not in self._cq:
            self._cq[to_id] = Batch(to_id)

        self._cq[to_id][from_id].extend(payloads)

    def resolve(self, network: "Network") -> None:
        self._qi = 1 - self._qi

        for i in range(self.chain_limit):
            if len(self._pq) == 0:
                break

            self.resolve_batches(network, self._pq)

            self._pq.clear()

            self._qi = 1 - self._qi

    @abc.abstractmethod
    def resolve_batches(self, network: "Network", batches: Mapping[ID, Batch]) -> None:
        raise NotImplementedError
