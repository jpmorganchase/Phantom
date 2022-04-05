from collections import defaultdict
from dataclasses import dataclass
from typing import (
    Callable,
    DefaultDict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
    Union,
    TYPE_CHECKING,
)

from .core import ID
from .message import Batch, Message, Payload, PayloadType

if TYPE_CHECKING:
    from .network import Network

Responses = Iterator[Tuple[ID, Iterable[Payload]]]


@dataclass(frozen=True)
class View:
    """Immutable references to public :class:`mercury.actors.Actor` state.

    Attributes:
        actor_id: The unique :class:`mercury.ID` of the agent.
    """

    actor_id: ID


ViewType = TypeVar("ViewType", bound=View)


class Actor:
    """Representation of a behavioural actor in the network.

    Instances of :class:`mercury.Actor` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.
    """

    def __init__(self, actor_id: ID) -> None:
        self._id = actor_id

    @property
    def id(self) -> ID:
        """The unique ID of the actor."""
        return self._id

    def view(self, neighbour_id: Optional[ID] = None) -> View:
        """Return an immutable view to the actor's public state."""
        return View(actor_id=self._id)

    def pre_resolution(self, ctx: "Network.Context") -> None:
        """Perform internal, pre-resolution updates to the actor."""

    def handle_batch(self, ctx: "Network.Context", batch: Batch) -> Responses:
        """Handle a batch of messages from various possible senders.

        Arguments:
            ctx: The context in which the messages are being processed.
            batch: The incoming batch of payloads to handle.

        Returns:
            response: An iterator over receiver ID and payload set tuples.
        """
        raise NotImplementedError

    def post_resolution(self, ctx: "Network.Context") -> None:
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

    def handle_batch(self, ctx: "Network.Context", batch: Batch) -> Responses:
        for sender_id in batch:
            for message in batch.messages_from(sender_id):
                yield from self.handle_message(ctx, message)

    def handle_message(self, ctx: "Network.Context", message: Message) -> Responses:
        """Handle a single message from a given sender.

        Arguments:
            ctx: The context in which the messages are being processed.
            message: The incoming message to handle.

        Returns:
            response: An iterator over receiver ID and payload set tuples.
        """
        raise NotImplementedError


Handler = Callable[["Network.Context", Message], Responses]


class SimpleSyncActor(SyncActor):
    def __init__(self, actor_id: ID) -> None:
        SyncActor.__init__(self, actor_id)

        self.__handlers: DefaultDict[
            Tuple[Tuple[Type[Payload]], Tuple[Type[Actor]]], List[Handler]
        ] = defaultdict(list)

        # Register handlers defined via decorator utility:
        fn_names = [attr for attr in dir(self) if callable(getattr(self, attr))]
        for fn_name in fn_names:
            fn = getattr(self, fn_name)

            if hasattr(fn, "payload_types") and hasattr(fn, "sending_actor_types"):
                self.__handlers[(fn.payload_types, fn.sending_actor_types)].append(fn)

    def handle_message(self, ctx: "Network.Context", message: Message) -> Responses:
        has_handled = False

        for (payload_types, actor_types), handlers in self.__handlers.items():
            if payload_types == (None,) or type(message.payload) in payload_types:
                if (
                    actor_types == (None,)
                    or type(ctx._subnet[message.sender_id]) in actor_types
                ):
                    for handler in handlers:
                        yield from handler(ctx, message)
                        has_handled = True

        if not has_handled:
            raise KeyError(
                f"Payload type {type(message.payload)} in message sent from '{message.sender_id}' with type '{type(ctx._subnet[message.sender_id])}' to '{message.receiver_id}'. Agent '{message.receiver_id}' needs a message handler function capable of receiving this mesage type."
            )


def handler(
    payload_types: Union[Type[PayloadType], Tuple[Type[PayloadType]], None] = None,
    sending_actor_types: Union[Type[Actor], Tuple[Type[Actor]], None] = None,
) -> Callable[[Handler], Handler]:
    if not isinstance(payload_types, Tuple):
        payload_types = (payload_types,)

    if not isinstance(sending_actor_types, Tuple):
        sending_actor_types = (sending_actor_types,)

    def decorator(fn: Handler) -> Handler:
        setattr(fn, "payload_types", payload_types)
        setattr(fn, "sending_actor_types", sending_actor_types)

        return fn

    return decorator
