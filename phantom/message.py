from dataclasses import dataclass
from typing import Generic, List, Type, TypeVar, Union, TYPE_CHECKING

from .types import AgentID

if TYPE_CHECKING:
    from .agents import Agent


@dataclass(frozen=True)
class MsgPayload:
    """Message payload structure."""


MsgPayloadType = TypeVar("MsgPayloadType")

AgentTypeArg = Union[Type["Agent"], str, List[Union[Type["Agent"], str]], None]


def msg_payload(sender_type: AgentTypeArg = None, receiver_type: AgentTypeArg = None):
    def wrap(message_class: Type) -> Type:
        if sender_type is None:
            s_types = None
        else:
            s_types = sender_type if isinstance(sender_type, list) else [sender_type]

            s_types = [t.__name__ if isinstance(t, type) else t for t in s_types]

        if receiver_type is None:
            r_types = None
        else:
            r_types = (
                receiver_type if isinstance(receiver_type, list) else [receiver_type]
            )

            r_types = [t.__name__ if isinstance(t, type) else t for t in r_types]

        message_class._sender_types = s_types
        message_class._receiver_types = r_types
        return dataclass(frozen=True)(message_class)

    return wrap


@dataclass(frozen=True)
class Message(Generic[MsgPayloadType]):
    """
    Message class storing the sender agent ID, receiver agent ID and message payload.
    """

    sender_id: AgentID
    receiver_id: AgentID
    payload: MsgPayload
