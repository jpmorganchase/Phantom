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
    Sequence,
    Tuple,
    Type,
    TypeVar,
)

import numpy as np

from .context import Context
from .decoders import Decoder
from .encoders import Encoder
from .message import Message, MsgPayload
from .reward_functions import RewardFunction
from .supertype import Supertype
from .types import AgentID
from .views import AgentView


Action = TypeVar("Action")

MessageList = List[Tuple[AgentID, Message]]


class Agent(ABC):
    """
    Representation of an agent in the network.

    Instances of :class:`phantom.Agent` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.

    Arguments:
        agent_id: Unique identifier for the agent.
        supertype: Optional :class:`Supertype` instance. When the agent's reset function
            is called the supertype will be sampled from and the values set as the
            agent's :attr:`type` property.

    Implementations can make use of the ``msg_handler`` function decorator:

    .. code-block:: python

        class SomeAgent(ph.Agent):
            ...

            @ph.agents.msg_handler(RequestMessage)
            def handle_request_msg(self, ctx: ph.Context, message: ph.Message):
                response_msgs = do_something_with_msg(message)

                return [response_msgs]
    """

    def __init__(
        self,
        agent_id: AgentID,
        supertype: Optional[Supertype] = None,
    ) -> None:
        self._id = agent_id

        self.supertype = supertype

        self.type: Optional[Supertype] = None

        self.__handlers: DefaultDict[Type[MsgPayload], List[Handler]] = defaultdict(
            list
        )

        for name, attr in self.__class__.__dict__.items():
            if callable(attr) and hasattr(attr, "_message_type"):
                self.__handlers[attr._message_type].append(getattr(self, name))

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

    def handle_batch(
        self, ctx: Context, batch: Sequence[Message]
    ) -> List[Tuple[AgentID, MsgPayload]]:
        """
        Handle a batch of messages from multiple potential senders.

        Arguments:
            ctx: A Context object representing agent's the local view of the environment.
            batch: The incoming batch of messages to handle.

        Returns:
            A list of receiver ID / message payload pairs to form into messages in
            response to further resolve.
        """
        return list(
            chain.from_iterable(
                filter(
                    lambda x: x is not None,
                    (self.handle_message(ctx, message) for message in batch),
                )
            )
        )

    def handle_message(
        self, ctx: Context, message: Message
    ) -> List[Tuple[AgentID, MsgPayload]]:
        """
        Handle a messages sent from other agents. The default implementation is the use
        the ``msg_handler`` function decorators.

        Arguments:
            ctx: A Context object representing agent's the local view of the environment.
            message: The contents of the message.

        Returns:
            A list of receiver ID / message payload pairs to form into messages in
            response to further resolve.
        """

        ptype = type(message.payload)

        if ptype not in self.__handlers:
            raise KeyError(
                f"Unknown message type {ptype} in message sent from '{message.sender_id}' to '{self.id}'. Agent '{self.id}' needs a message handler function capable of receiving this mesage type."
            )

        return list(
            chain.from_iterable(
                filter(
                    lambda x: x is not None,
                    (
                        bound_handler(ctx, message)
                        for bound_handler in self.__handlers[ptype]
                    ),
                )
            )
        )

    def generate_messages(self, ctx: Context) -> List[Tuple[AgentID, MsgPayload]]:
        return []

    def reset(self) -> None:
        """
        Resets the Agent.

        Can be extended by subclasses to provide additional functionality.
        """

        if self.supertype is not None:
            self.type = self.supertype.sample()
        elif hasattr(self, "Supertype"):
            try:
                self.type = self.Supertype().sample()
            except TypeError as e:
                raise Exception(
                    f"Tried to initialise agent {self.id}'s Supertype with default values but failed:\n\t{e}"
                )

    def __repr__(self) -> str:
        return f"[{self.__class__.__name__} {self.id}]"


class RLAgent(Agent):
    """
    Representation of a behavioural agent in the network.

    Instances of :class:`phantom.Agent` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.

    Arguments:
        agent_id: Unique identifier for the agent.
        observation_encoder: Optional :class:`Encoder` instance, otherwise define an
            :meth:`encode_observation` method on the :class:`Agent` sub-class.
        action_decoder: Optional :class:`Decoder` instance, otherwise define an
            :meth:`decode_action` method on the :class:`Agent` sub-class.
        reward_function: Optional :class:`RewardFunction` instance, otherwise define an
            :meth:`compute_reward` method on the :class:`Agent` sub-class.
        supertype: Optional :class:`Supertype` instance. When the agent's reset function
            is called the supertype will be sampled from and the values set as the
            agent's :attr:`type` property.
    """

    def __init__(
        self,
        agent_id: AgentID,
        observation_encoder: Optional[Encoder] = None,
        action_decoder: Optional[Decoder] = None,
        reward_function: Optional[RewardFunction] = None,
        supertype: Optional[Supertype] = None,
    ) -> None:
        super().__init__(agent_id, supertype)

        self.observation_encoder = observation_encoder
        self.action_decoder = action_decoder
        self.reward_function = reward_function

        if action_decoder is not None:
            self.action_space = action_decoder.action_space
        elif "action_space" not in dir(self):
            self.action_space = None

        if observation_encoder is not None:
            self.observation_space = observation_encoder.observation_space
        elif "observation_space" not in dir(self):
            self.observation_space = None

    def encode_observation(self, ctx: Context) -> np.ndarray:
        """
        Encodes a local view of the environment state into a set of observations.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing agent's the local view of the environment.

        Returns:
            A numpy array encoding the observations.
        """

        if self.observation_encoder is None:
            raise NotImplementedError(
                f"Agent '{self.id}' does not have an Encoder instance set as 'observation_encoder' or a custom 'encode_observation' method defined"
            )

        return self.observation_encoder.encode(ctx)

    def decode_action(
        self, ctx: Context, action: Action
    ) -> Optional[List[Tuple[AgentID, MsgPayload]]]:
        """
        Decodes an action taken by the agent policy into a set of messages to be
        sent to other agents in the network.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the agent's local view of the environment.
            action: The action taken by the agent.

        Returns:
            A list of receiver ID / message payload pairs to form into messages in
            response to further resolve.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                f"Agent '{self.id}' does not have an Decoder instance set as 'action_decoder' or a custom 'decode_action' method defined"
            )

        return self.action_decoder.decode(ctx, action)

    def compute_reward(self, ctx: Context) -> float:
        """
        Computes a reward value based on an agents current state.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the agent's local view of the environment.

        Returns:
            A float representing the present reward value.
        """
        if self.reward_function is None:
            raise NotImplementedError(
                f"Agent '{self.id}' does not have an RewardFunction instance set as 'reward_function' or a custom 'compute_reward' method defined"
            )

        return self.reward_function.reward(ctx)

    def is_done(self, ctx: Context) -> bool:
        """
        Indicates whether the agent is done for the current episode. The default
        logic is for the agent to be done only once all the timesteps have been executed.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Context object representing the agent's local view of the environment.

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
            ctx: A Context object representing the agent's local view of the environment.

        Returns:
            A dictionary containing informations about the agent
        """
        return {}


Handler = Callable[[Context, Message], List[Tuple[AgentID, MsgPayload]]]


def msg_handler(message_type: Type[MsgPayload]) -> Callable[[Handler], Handler]:
    def decorator(fn: Handler) -> Handler:
        setattr(fn, "_message_type", message_type)
        return fn

    return decorator
