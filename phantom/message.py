from dataclasses import dataclass
from typing import Generic, TypeVar

from .types import AgentID


@dataclass(frozen=True)
class MsgPayload:
    """Message payload structure."""


MsgPayloadType = TypeVar("MsgPayloadType", bound=MsgPayload)


@dataclass(frozen=True)
class Message(Generic[MsgPayloadType]):
    """
    Message class storing the sender agent ID, receiver agent ID and message payload.
    """

    sender_id: AgentID
    receiver_id: AgentID
    payload: MsgPayload
