import random as _r
import typing as _t

import mercury as _me

from . import BatchResolver


class RandomisedActorResolver(BatchResolver):
    """Resolver that processes messages in a random (actor-level) order.

    Arguments:
        chain_limit: Optional limit on the number of chained steps that can
            propagate.
        seed: Optional seed to initialise the psuedo-random number generator.
    """

    def __init__(self, chain_limit: int = 100, seed: _t.Optional[int] = None) -> None:
        BatchResolver.__init__(self, chain_limit=chain_limit)

        self._rgen = _r.Random(seed or _r.getstate())

    def resolve_batches(
        self, network: "_me.Network", batches: _t.Mapping[_me.ID, _me.message.Batch]
    ) -> None:
        keys = list(self._pq.keys())

        self._rgen.shuffle(keys)

        for sender_id in keys:
            batch = batches[sender_id]
            ctx = network.context_for(batch.receiver_id)

            for receiver_id, payloads in ctx.actor.handle_batch(ctx, batch):
                self.push(ctx.actor.id, receiver_id, payloads)
