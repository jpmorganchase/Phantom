import typing as _t

from collections import defaultdict
from dataclasses import dataclass

import mercury as _m


Payloads = _t.Iterable[_m.Payload]
Responses = _t.Iterator[_t.Tuple[_m.ID, Payloads]]


@dataclass(frozen=True)
class View:
    """Immutable references to public :class:`mercury.actors.Actor` state.

    Attributes:
        actor_id: The unique :class:`mercury.ID` of the agent.
    """

    actor_id: _m.ID


ViewType = _t.TypeVar("ViewType", bound=View)


class Actor:
    """Representation of a behavioural actor in the network.

    Instances of :class:`mercury.Actor` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.
    """

    def __init__(self, actor_id: _m.ID) -> None:
        self._id = actor_id

    @property
    def id(self) -> _m.ID:
        """The unique ID of the actor."""
        return self._id

    def view(self, neighbour_id: _t.Optional[_m.ID] = None) -> View:
        """Return an immutable view to the actor's public state."""
        return View(actor_id=self._id)

    def pre_resolution(self, ctx: "_m.Network.Context") -> None:
        """Perform internal, pre-resolution updates to the actor."""

    def handle_batch(
        self, ctx: "_m.Network.Context", batch: _m.message.Batch
    ) -> Responses:
        """Handle a batch of messages from various possible senders.

        Arguments:
            ctx: The context in which the messages are being processed.
            batch: The incoming batch of payloads to handle.

        Returns:
            response: An iterator over receiver ID and payload set tuples.
        """
        raise NotImplementedError

    def post_resolution(self, ctx: "_m.Network.Context") -> None:
        """Perform internal, post-resolution updates to the actor."""

    def reset(self) -> None:
        """Reset the internal and public state of the actor."""

    def __repr__(self) -> str:
        return "[{} {}]".format(self.__class__.__name__, self.id)


class SyncActor(Actor):
    """Synchronous actor that handles messages sequentially and homogeneously.

    Synchronous actors assume separablility of messages, meaning that the
    messages do not need to be handled together, but can be split up and
    treated independently.
    """

    def handle_batch(
        self, ctx: "_m.Network.Context", batch: _m.message.Batch
    ) -> Responses:
        for sender_id in batch:
            for message in batch.messages_from(sender_id):
                yield from self.handle_message(ctx, message)

    def handle_message(
        self, ctx: "_m.Network.Context", message: _m.Message
    ) -> Responses:
        """Handle a single message from a given sender.

        Arguments:
            ctx: The context in which the messages are being processed.
            message: The incoming message to handle.

        Returns:
            response: An iterator over receiver ID and payload set tuples.
        """
        raise NotImplementedError


Handler = _t.Callable[["_m.Network.Context", _m.Message], Responses]


class SimpleSyncActor(SyncActor):
    def __init__(self, actor_id: _m.ID) -> None:
        SyncActor.__init__(self, actor_id)

        self.__handlers: _t.DefaultDict[
            _t.Type[_m.Payload], _t.List[Handler]
        ] = defaultdict(list)

        # Register handlers defined via decorator utility:
        fn_names = [attr for attr in dir(self) if callable(getattr(self, attr))]
        for fn_name in fn_names:
            fn = getattr(self, fn_name)

            if hasattr(fn, "payload_type"):
                self.register_handler(fn.payload_type, fn)

    def register_handler(
        self, payload_type: _t.Type[_m.PayloadType], handler: Handler
    ) -> None:
        self.__handlers[payload_type].append(handler)

    def handle_message(
        self, ctx: "_m.Network.Context", message: _m.Message
    ) -> Responses:
        ptype = type(message.payload)

        if ptype not in self.__handlers:
            raise KeyError(f"Unknown payload type {ptype}.")

        for bound_handler in self.__handlers[ptype]:
            yield from bound_handler(ctx, message)


def handler(payload_type: _t.Type[_m.PayloadType]) -> _t.Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        setattr(fn, "payload_type", payload_type)

        return fn

    return decorator
