import random
from typing import Mapping, Optional, TYPE_CHECKING

from ..core import ID
from ..message import Batch
from . import BatchResolver

if TYPE_CHECKING:
    from ..network import Network


class RandomisedActorResolver(BatchResolver):
    """Resolver that processes messages in a random (actor-level) order.

    Arguments:
        chain_limit: Optional limit on the number of chained steps that can
            propagate.
        seed: Optional seed to initialise the psuedo-random number generator.
    """

    def __init__(
        self,
        enable_tracking: bool = False,
        chain_limit: int = 100,
        seed: Optional[int] = None,
    ) -> None:
        BatchResolver.__init__(self, enable_tracking, chain_limit)

        self._rgen = random.Random(seed or random.getstate())

    def resolve_batches(self, network: "Network", batches: Mapping[ID, Batch]) -> None:
        keys = list(self._pq.keys())

        self._rgen.shuffle(keys)

        for sender_id in keys:
            batch = batches[sender_id]
            ctx = network.context_for(batch.receiver_id)

            for receiver_id, payloads in ctx.actor.handle_batch(ctx, batch):
                self.push(ctx.actor.id, receiver_id, payloads)
