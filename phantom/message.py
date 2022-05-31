from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class Message:
    """Message structure."""


MessageType = TypeVar("MessageType", bound=Message)
