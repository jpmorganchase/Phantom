from abc import ABC
from dataclasses import dataclass
from typing import Any, Generic, TypeVar

from .types import AgentID


@dataclass(frozen=True)
class MsgPayload:
    """Message payload structure."""


MsgPayloadType = TypeVar("MsgPayloadType", bound=MsgPayload)


@dataclass(frozen=True)
class Message(Generic[MsgPayloadType]):
    sender_id: AgentID
    receiver_id: AgentID
    payload: MsgPayload
