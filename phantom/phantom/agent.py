from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from typing import Any, AnyStr, Dict, Mapping, Optional, Type, TypeVar, Union

import mercury as me
import numpy as np
import gym.spaces
from ray import rllib

from .decoders import Decoder
from .encoders import Encoder
from .packet import Packet, Mutation
from .rewards import RewardFunction


ObsSpaceCompatibleTypes = Union[dict, list, np.ndarray, tuple]


@dataclass
class AgentType(ABC):
    """
    Abstract base class representing Agent Types.
    """

    def to_obs_space_compatible_type(self) -> Dict[str, ObsSpaceCompatibleTypes]:
        """
        Converts the parameters of the AgentType into a dict for use in observation
        spaces.
        """

        def _to_compatible_type(field: str, obj: Any) -> ObsSpaceCompatibleTypes:
            if isinstance(obj, dict):
                return {
                    key: _to_compatible_type(key, value) for key, value in obj.items()
                }
            elif isinstance(obj, (float, int)):
                return np.array([obj])
            elif isinstance(obj, list):
                return [
                    _to_compatible_type(f"{field}[{i}]", value)
                    for i, value in enumerate(obj)
                ]
            elif isinstance(obj, tuple):
                return tuple(
                    _to_compatible_type(f"{field}[{i}]", value)
                    for i, value in enumerate(obj)
                )
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                raise ValueError(
                    f"Can't encode field '{field}' with type '{type(obj)}' into obs space compatible type"
                )

        return {
            name: _to_compatible_type(name, value)
            for name, value in asdict(self).items()
        }

    def to_obs_space(self, low=-np.inf, high=np.inf) -> gym.spaces.Space:
        """
        Converts the parameters of the AgentType into a `gym.spaces.Space` representing
        the space.

        All elements of the space span the same range given by the `low` and `high`
        arguments.

        Arguments:
            low: Optional 'low' bound for the space (default is -∞)
            high: Optional 'high' bound for the space (default is ∞)
        """

        def _to_obs_space(field: str, obj: Any) -> gym.spaces.Space:
            if isinstance(obj, dict):
                return gym.spaces.Dict(
                    {key: _to_obs_space(key, value) for key, value in obj.items()}
                )
            elif isinstance(obj, float):
                return gym.spaces.Box(low, high, (1,), np.float32)
            elif isinstance(obj, int):
                return gym.spaces.Box(low, high, (1,), np.float32)
            elif isinstance(obj, (list, tuple)):
                return gym.spaces.Tuple(
                    [
                        _to_obs_space(f"{field}[{i}]", value)
                        for i, value in enumerate(obj)
                    ]
                )
            elif isinstance(obj, np.ndarray):
                return gym.spaces.Box(low, high, obj.shape, np.float32)
            else:
                raise ValueError(
                    f"Can't encode field '{field}' with type '{type(obj)}' into gym.spaces.Space"
                )

        return gym.spaces.Dict(
            {
                field: _to_obs_space(field, value)
                for field, value in asdict(self).items()
            }
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
        policy_class: The policy type of the agent (optional).
        policy_config: The policy config of the agent (optional).
        supertype: The supertype of the agent (optional).
    """

    def __init__(
        self,
        agent_id: me.ID,
        obs_encoder: Optional[Encoder] = None,
        action_decoder: Optional[Decoder] = None,
        reward_function: Optional[RewardFunction] = None,
        policy_class: Optional[Type[rllib.Policy]] = None,
        policy_config: Optional[Mapping] = None,
        supertype: Optional[Supertype] = None,
    ) -> None:
        super().__init__(agent_id)

        self.obs_encoder = obs_encoder
        self.action_decoder = action_decoder
        self.reward_function = reward_function
        self.policy_class = policy_class
        self.policy_config = policy_config

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

    def get_observation_space(self) -> gym.spaces.Space:
        """
        Returns the gym observation space of this agent.
        """
        if self.obs_encoder is None:
            raise NotImplementedError(
                "If the agent does not have an Encoder, a custom get_observation_space method must be defined"
            )

        return self.obs_encoder.output_space

    def get_action_space(self) -> gym.spaces.Space:
        """
        Returns the gym action space of this agent.
        """
        if self.action_decoder is None:
            raise NotImplementedError(
                "If the agent does not have an Decoder, a custom get_action_space method must be defined"
            )

        return self.action_decoder.action_space
