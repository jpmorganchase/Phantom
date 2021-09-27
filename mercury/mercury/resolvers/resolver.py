import abc as _abc
import typing as _t

import mercury as _me


class Resolver(_abc.ABC):
    """Network message resolver.

    This type is responsible for resolution processing. That is, the order in
    which (and any special logic therein) messages are handled in a Network.

    In many cases, this type can be arbitrary since the sequence doesn't matter
    (i.e. the problem is not path dependent). In other cases, however, this is
    not the case; e.g. processing incoming market orders in an LOB.
    """

    @_abc.abstractmethod
    def push(
        self, from_id: _me.ID, to_id: _me.ID, payloads: _t.Iterable[_me.Payload]
    ) -> None:
        raise NotImplementedError

    @_abc.abstractmethod
    def resolve(self, network: "_me.Network") -> None:
        """Process queues messages for a (sub) set of network contexts.

        Arguments:
            network: An instance of the Network class to resolve.
        """
        raise NotImplementedError

    @_abc.abstractmethod
    def reset(self) -> None:
        raise NotImplementedError


class BatchResolver(Resolver):
    def __init__(self, chain_limit: int = 2) -> None:
        self.chain_limit = chain_limit

        self._q1: _t.Dict[_me.ID, _me.Batch] = dict()

        self._q2: _t.Dict[_me.ID, _me.Batch] = dict()

        self._qi = 0
        self._qs = (self._q1, self._q2)

    @property
    def _cq(self) -> _t.Dict[_me.ID, _me.Batch]:
        """Current queue."""
        return self._qs[self._qi]

    @property
    def _pq(self) -> _t.Dict[_me.ID, _me.Batch]:
        """Populated queue."""
        return self._qs[1 - self._qi]

    def reset(self) -> None:
        self._q1.clear()
        self._q2.clear()

    def push(
        self, from_id: _me.ID, to_id: _me.ID, payloads: _t.Iterable[_me.Payload]
    ) -> None:
        if to_id not in self._cq:
            self._cq[to_id] = _me.message.Batch(to_id)

        self._cq[to_id][from_id].extend(payloads)

    def resolve(self, network: "_me.Network") -> None:
        self._qi = 1 - self._qi

        for i in range(self.chain_limit):
            if len(self._pq) == 0:
                break

            self.resolve_batches(network, self._pq)

            self._pq.clear()

            self._qi = 1 - self._qi

    @_abc.abstractmethod
    def resolve_batches(
        self, network: "_me.Network", batches: _t.Mapping[_me.ID, _me.message.Batch]
    ) -> None:
        raise NotImplementedError
