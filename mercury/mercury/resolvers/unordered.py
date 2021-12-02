from typing import Mapping, TYPE_CHECKING

from ..core import ID
from ..message import Batch
from . import BatchResolver

if TYPE_CHECKING:
    from ..network import Network


class UnorderedResolver(BatchResolver):
    """Resolver that processes operations with no preference ordering.

    Note: this resolver will process messages with respect to the order that
    the context objects have stored the queues. Namely, the order in which
    objects were added to the intenral map buffer. This is important because it
    means that there is an implicit ordering which may cause bias in some
    problems.
    """

    def resolve_batches(self, network: "Network", batches: Mapping[ID, Batch]) -> None:
        for batch in batches.values():
            ctx = network.context_for(batch.receiver_id)

            for receiver_id, payloads in ctx.actor.handle_batch(ctx, batch):
                self.push(ctx.actor.id, receiver_id, payloads)
