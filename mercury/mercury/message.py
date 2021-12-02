from dataclasses import dataclass
from typing import Any, DefaultDict, Generic, Iterator, List, TypeVar

from mercury import ID


@dataclass(frozen=True)
class Payload:
    """Payload structure."""


PayloadType = TypeVar("PayloadType", bound=Payload)


@dataclass(frozen=True)
class Message(Generic[PayloadType]):
    """Immutable message structure."""

    sender_id: ID
    receiver_id: ID

    payload: PayloadType


class Batch(DefaultDict[ID, List[Payload]]):
    def __init__(self, receiver_id: ID, *args: Any, **kwargs: Any) -> None:
        super().__init__(list, *args, **kwargs)

        self.receiver_id = receiver_id

    def merge(self, other: "Batch") -> None:
        for sender_id, payloads in other.items():
            self[sender_id].extend(payloads)

    def messages_from(self, sender_id: ID) -> Iterator[Message]:
        def to_message(payload: Payload) -> Message:
            return Message(sender_id, self.receiver_id, payload)

        if sender_id in self:
            yield from map(to_message, self[sender_id])

        else:
            yield from ()

    def __repr__(self) -> str:
        return f"Batch({self.receiver_id}, {dict.__repr__(self)})"
