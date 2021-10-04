from abc import ABC, abstractmethod
from typing import Any, AnyStr, Dict, Mapping, Optional, Type, TypeVar, Union

import mercury as me
import numpy as np
from gym.spaces import Box, Discrete, Space

from .decoders import Decoder
from .encoders import Encoder
from .packet import Packet, Mutation
from .rewards import RewardFunction


class AgentType(ABC):
    def to_array(self) -> np.ndarray:
        """
        Converts the parameters of the AgentType into a flattened numpy array
        for use in observation spaces.

        An exception will be raised if any of the parameters are not of the type
        int, float or numpy array.
        """

        def _to_array(field: str, obj: Any) -> np.ndarray:
            type_array = np.array([])

            if isinstance(obj, (np.ndarray, int, float)):
                type_array = np.hstack((type_array, obj))
            elif isinstance(obj, (list, tuple)):
                for i, elem in enumerate(obj):
                    type_array = np.hstack(
                        (type_array, _to_array(f"{field}[{i}]", elem))
                    )
            else:
                raise ValueError(
                    f"Can't encode field '{field}' with type '{type(obj)}' into array"
                )

            return type_array

        return np.hstack(
            [_to_array(field, obj) for field, obj in self.__dict__.items()]
        )

    def to_basic_obs_space(self, low=-np.inf, high=np.inf) -> Box:
        """
        Converts the parameters of the AgentType into a `gym.spaces.Box`
        representing the space given by the `to_array` method.

        All elements of the space span the same range given by the `low` and
        `high` arguments.

        An exception will be raised if any of the parameters are not of the type
        int, float or numpy array.

        Arguments:
            low: Optional 'low' bound for the space (default is -∞)
            high: Optional 'high' bound for the space (default is ∞)
        """
        type_array = self.to_array()

        return Box(
            low=low * np.ones_like(type_array),
            high=high * np.ones_like(type_array),
        )


A = TypeVar("A", bound=AgentType)


class Supertype(ABC):
    @abstractmethod
    def sample(self) -> A:
        """
        Base method for sampling a Type from a Supertype.

        Must be implemented by supertypes that inherit from this class.
        """
        raise NotImplementedError


class NullType(AgentType):
    """
    An implementation of AgentType that holds no values.
    """

    pass


class NullSupertype(Supertype):
    """
    An implementation of Supertype that returns a type that holds no values.
    """

    def sample(self) -> NullType:
        """
        Returns a NullType that holds no values.
        """
        return NullType()


Action = TypeVar("Action")


class Agent(me.actors.SimpleSyncActor):
    """
    The base Phantom Agent type.

    Attributes:
        agent_id: A unique string identifying the agent.
        obs_encoder: The observation encoder of the agent (optional).
        action_decoder: The action decoder of the agent (optional).
        reward_function: The reward function of the agent (optional).
        policy_type: The policy type of the agent (optional).
        policy_config: The policy config of the agent (optional).
        supertype: The supertype of the agent (optional).
    """

    def __init__(
        self,
        agent_id: me.ID,
        obs_encoder: Optional[Encoder] = None,
        action_decoder: Optional[Decoder] = None,
        reward_function: Optional[RewardFunction] = None,
        policy_type: Optional[Union[str, Type]] = None,
        policy_config: Optional[Mapping] = None,
        supertype: Optional[Supertype] = None,
    ) -> None:
        super().__init__(agent_id)

        self.obs_encoder: Optional[Encoder] = obs_encoder
        self.action_decoder: Optional[Decoder] = action_decoder
        self.reward_function: Optional[RewardFunction] = reward_function
        self.policy_type: Optional[Union[str, Type]] = policy_type
        self.policy_config: Optional[Mapping] = policy_config

        self.supertype: Supertype = (
            supertype if supertype is not None else NullSupertype()
        )

        self.reset()

    def handle_mutation(self, ctx: me.Network.Context, mutation: Mutation) -> None:
        """Handle a single mutation of the agent's internal state.

        Arguments:
            ctx: The context in which the messages are being processed.
            mutation: The incoming mutation to handle.
        """

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
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

    def decode_action(self, ctx: me.Network.Context, action: Action) -> Packet:
        """
        Decodes an action taken by the agent policy into a set of messages to be
        sent to other actors in the network.

        Note:
            This method may be extended by sub-classes to provide additional functionality.

        Arguments:
            ctx: A Mercury Context object representing the local view of the
                environment.
            action: The action taken by the agent.

        Returns:
            A Packet object containing messages to be sent to other actors.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                "If the agent does not have an Decoder, a custom decode_action method must be defined"
            )

        return self.action_decoder.decode(ctx, action)

    def compute_reward(self, ctx: me.Network.Context) -> float:
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

    def is_done(self, _ctx: me.Network.Context) -> bool:
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

    def collect_infos(self, _ctx: me.Network.Context) -> Dict[AnyStr, Any]:
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
        self.type = self.supertype.sample()

        super().reset()

    def get_observation_space(self) -> Space:
        """
        Returns the gym observation space of this agent.
        """
        if self.obs_encoder is None:
            raise NotImplementedError(
                "If the agent does not have an Encoder, a custom get_observation_space method must be defined"
            )

        return self.obs_encoder.output_space

    def get_action_space(self) -> Space:
        """
        Returns the gym action space of this agent.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                "If the agent does not have an Decoder, a custom get_action_space method must be defined"
            )

        return self.action_decoder.action_space


class ZeroIntelligenceAgent(Agent, ABC):
    """
    Boilerplate for building agents that take actions but do not learn a policy,
    do not make observations and do not compute a reward.

    This class should be subclassed and the `decode_action` method implemented.
    """

    def __init__(self, agent_id: me.ID) -> None:
        super().__init__(agent_id)

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context):
        return np.zeros((1,))

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        raise NotImplementedError

    def get_observation_space(self):
        return Box(-np.inf, np.inf, (1,))

    def get_action_space(self):
        return Discrete(1)
