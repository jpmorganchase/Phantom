import typing as _t

import mercury as _me

from . import BatchResolver


class UnorderedResolver(BatchResolver):
    """Resolver that processes operations with no preference ordering.

    Note: this resolver will process messages with respect to the order that
    the context objects have stored the queues. Namely, the order in which
    objects were added to the intenral map buffer. This is important because it
    means that there is an implicit ordering which may cause bias in some
    problems.
    """

    def resolve_batches(
        self, network: "_me.Network", batches: _t.Mapping[_me.ID, _me.message.Batch]
    ) -> None:
        for batch in batches.values():
            ctx = network.context_for(batch.receiver_id)

            for receiver_id, payloads in ctx.actor.handle_batch(ctx, batch):
                self.push(ctx.actor.id, receiver_id, payloads)
