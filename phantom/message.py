from abc import ABC
from dataclasses import dataclass
from typing import TypeVar


@dataclass(frozen=True)
class Message(ABC):
    """Base dataclass for defining message payloads."""


MessageType = TypeVar("MessageType", bound=Message)
