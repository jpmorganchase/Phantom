from abc import ABC
from collections import defaultdict
from dataclasses import dataclass
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

import gym
import numpy as np

from .context import Context
from .decoders import Decoder
from .encoders import Encoder
from .message import Message, MessageType
from .reward_functions import RewardFunction
from .supertype import Supertype
from .types import AgentID


@dataclass(frozen=True)
class View:
    """Immutable references to public :class:`phantom.Agent` state.

    Attributes:
        agent_id: The unique :class:`mercury.ID` of the agent.
    """

    agent_id: AgentID


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
        obs_encoder: Optional[Encoder] = None,
        action_decoder: Optional[Decoder] = None,
        reward_function: Optional[RewardFunction] = None,
        supertype: Optional[Supertype] = None,
    ) -> None:
        self._id = agent_id

        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.reward_function = reward_function
        self.supertype = supertype

        self.type: Optional[Supertype] = None

    @property
    def id(self) -> AgentID:
        """The unique ID of the agent."""
        return self._id

    def takes_actions(self) -> bool:
        try:
            self.get_action_space()
        except NotImplementedError:
            return False
        else:
            return True

    def view(self, neighbour_id: Optional[AgentID] = None) -> View:
        """Return an immutable view to the agent's public state."""
        return View(agent_id=self._id)

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

    def encode_obs(self, ctx: Context) -> np.ndarray:
        """
        Encodes a local view of the environment state into a set of observations.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Mercury Context object representing the local view of the
                environment.

        Returns:
            A numpy array encoding the observations.
        """

        if self.obs_encoder is None:
            raise NotImplementedError(
                "If the agent does not have an Encoder, a custom encode_obs method must be defined"
            )

        return self.obs_encoder.encode(ctx)

    def decode_action(
        self, ctx: Context, action: Action
    ) -> List[Tuple[AgentID, Message]]:
        """
        Decodes an action taken by the agent policy into a set of messages to be
        sent to other agents in the network.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Mercury Context object representing the local view of the
                environment.
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
            ctx: A Mercury Context object representing the local view of the
                environment.

        Returns:
            A float representing the present reward value.
        """
        if self.reward_function is None:
            raise NotImplementedError(
                "If the agent does not have an RewardFunction, a custom compute_reward method must be defined"
            )

        return self.reward_function.reward(ctx)

    def is_done(self, ctx: Context) -> bool:
        """
        Indicates whether the agent is done for the current episode. The default
        logic is for the agent to be done only once all the timesteps have been executed.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Mercury Context object representing the local view of the
                environment.

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
            ctx: A Mercury Context object representing the local view of the
                environment.

        Returns:
            A dictionary containing informations about the agent
        """
        return {}

    def reset(self) -> None:
        """
        Resets the Agent.

        Can be extended by subclasses to provide additional functionality.
        """

    def get_observation_space(self) -> gym.Space:
        """
        Returns the gym observation space of this agent.
        """
        if self.obs_encoder is None:
            raise NotImplementedError(
                "If the agent does not have an Encoder, a custom get_observation_space method must be defined"
            )

        return self.obs_encoder.output_space

    def get_action_space(self) -> gym.Space:
        """
        Returns the gym action space of this agent.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                "If the agent does not have an Decoder, a custom get_action_space method must be defined"
            )

        return self.action_decoder.action_space

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__} {self.id}]"


Handler = Callable[[Context, AgentID, Message], List[Tuple[AgentID, Message]]]


class MessageHandlerAgent(Agent):
    def __init__(self, agent_id: AgentID) -> None:
        super().__init__(agent_id)

        self.__handlers: DefaultDict[Type[Message], List[Handler]] = defaultdict(list)

        # Register handlers defined via decorator utility:
        fn_names = [attr for attr in dir(self) if callable(getattr(self, attr))]
        for fn_name in fn_names:
            fn = getattr(self, fn_name)

            if hasattr(fn, "message_type"):
                self.__handlers[fn.message_type].append(fn)

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
