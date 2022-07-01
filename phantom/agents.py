from abc import ABC
from collections import defaultdict
from itertools import chain
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np

from .context import Context
from .decoders import Decoder
from .encoders import Encoder
from .message import Message, MessageType
from .reward_functions import RewardFunction
from .supertype import Supertype
from .types import AgentID
from .view import AgentView


Action = TypeVar("Action")

MessageList = List[Tuple[AgentID, Message]]


class Agent(ABC):
    """Representation of a behavioural agent in the network.

    Instances of :class:`phantom.Agent` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.
    """

    def __init__(
        self,
        agent_id: AgentID,
        observation_encoder: Optional[Encoder] = None,
        action_decoder: Optional[Decoder] = None,
        reward_function: Optional[RewardFunction] = None,
        supertype: Optional[Supertype] = None,
    ) -> None:
        self._id = agent_id

        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.reward_function = reward_function
        self.supertype = supertype

        self.type: Optional[Supertype] = None

        if action_decoder is not None:
            self.action_space = action_decoder.action_space
        elif not hasattr(self, "action_space"):
            self.action_space = None

        if observation_encoder is not None:
            self.observation_space = observation_encoder.observation_space
        elif not hasattr(self, "observation_space"):
            self.observation_space = None

    @property
    def id(self) -> AgentID:
        """The unique ID of the agent."""
        return self._id

    def view(self, neighbour_id: Optional[AgentID] = None) -> Optional[AgentView]:
        """Return an immutable view to the agent's public state."""
        return None

    def pre_message_resolution(self, ctx: Context) -> None:
        """Perform internal, pre-message resolution updates to the agent."""

    def post_message_resolution(self, ctx: Context) -> None:
        """Perform internal, post-message resolution updates to the agent."""

    def handle_message(
        self, ctx: Context, sender_id: AgentID, message: Message
    ) -> List[Tuple[AgentID, Message]]:
        """Handle a messages sent from other agents.

        Arguments:
            ctx: The context in which the messages are being processed.
            sender_id: The sender of the message.
            message: The contents of the message

        Returns:
            A list of messages (tuples of (receiver_id, message)) to send to other agents.
        """

        raise NotImplementedError(
            f"The handle_message method is not implemented for agent '{self.id}' with type {self.__class__.__name__}"
        )

    def encode_observation(self, ctx: Context) -> np.ndarray:
        """
        Encodes a local view of the environment state into a set of observations.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the local view of the environment.

        Returns:
            A numpy array encoding the observations.
        """

        if self.observation_encoder is None:
            raise NotImplementedError(
                "If the agent does not have an Encoder, a custom encode_obs method must be defined"
            )

        return self.observation_encoder.encode(ctx)

    def decode_action(
        self, ctx: Context, action: Action
    ) -> List[Tuple[AgentID, Message]]:
        """
        Decodes an action taken by the agent policy into a set of messages to be
        sent to other agents in the network.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the local view of the environment.
            action: The action taken by the agent.

        Returns:
            A Packet object containing messages to be sent to other agents.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                "If the agent does not have an Decoder, a custom decode_action method must be defined"
            )

        return self.action_decoder.decode(ctx, action)

    def compute_reward(self, ctx: Context) -> float:
        """
        Computes a reward value based on an agents current state.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the local view of the environment.

        Returns:
            A float representing the present reward value.
        """
        return (
            None if self.reward_function is None else self.reward_function.reward(ctx)
        )

    def is_done(self, ctx: Context) -> bool:
        """
        Indicates whether the agent is done for the current episode. The default
        logic is for the agent to be done only once all the timesteps have been executed.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the local view of the environment.

        Returns:
            A boolean representing the terminal status of the agent.
        """
        return False

    def collect_infos(self, ctx: Context) -> Dict[str, Any]:
        """
        Provides diagnostic information about the agent, usefult for debugging.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the local view of the environment.

        Returns:
            A dictionary containing informations about the agent
        """
        return {}

    def reset(self) -> None:
        """
        Resets the Agent.

        Can be extended by subclasses to provide additional functionality.
        """

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__} {self.id}]"


Handler = Callable[[Context, AgentID, Message], List[Tuple[AgentID, Message]]]


class MessageHandlerAgent(Agent):
    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)

        self.__handlers: DefaultDict[Type[Message], List[Handler]] = defaultdict(list)

        for attr in dir(self):
            attr = getattr(self, attr)

            if callable(attr) and hasattr(attr, "message_type"):
                self.__handlers[attr.message_type].append(attr)

    def handle_message(
        self, ctx: Context, sender_id: AgentID, message: Message
    ) -> List[Tuple[AgentID, Message]]:
        ptype = type(message)

        if ptype not in self.__handlers:
            raise KeyError(
                f"Unknown message type {ptype} in message sent from '{sender_id}' to '{self.id}'. Agent '{self.id}' needs a message handler function capable of receiving this mesage type."
            )

        return list(
            chain.from_iterable(
                bound_handler(ctx, sender_id, message)
                for bound_handler in self.__handlers[ptype]
            )
        )


def msg_handler(message_type: Type[MessageType]) -> Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        setattr(fn, "message_type", message_type)
        return fn

    return decorator
