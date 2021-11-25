from abc import abstractstaticmethod, ABC
from typing import Generic, Mapping, Optional, Type, TypeVar

import gym.spaces
import mercury as me
import numpy as np
from ray import rllib

from ..packet import Packet
from .actor import FSMActor
from .agent import FSMAgent


ActorType = TypeVar("ActorType", bound=FSMActor)
AgentType = TypeVar("AgentType", bound=FSMAgent)


class StageHandler(ABC, Generic[ActorType]):
    """
    Defines custom hooks for an actor/agent for a particular FSM stage or stages.
    """

    # TODO: implement and test
    @staticmethod
    def pre_state_hook(actor: ActorType) -> float:
        pass

    # TODO: implement and test
    @staticmethod
    def pre_msg_resolution_hook(actor: ActorType) -> float:
        pass

    # TODO: implement and test
    @staticmethod
    def post_msg_resolution_hook(actor: ActorType) -> float:
        pass

    # TODO: implement and test
    @staticmethod
    def post_state_hook(actor: ActorType) -> float:
        pass


class StagePolicyHandler(StageHandler[AgentType], ABC, Generic[AgentType]):
    """
    Defines custom hooks for an agent's policy for a particular FSM stage or stages.
    """

    def __init__(
        self,
        policy_class: Optional[Type[rllib.Policy]] = None,
        policy_config: Optional[Mapping] = None,
    ) -> None:
        self.policy_class = policy_class
        self.policy_config = policy_config or {}

    @staticmethod
    def compute_reward(agent: AgentType, ctx: me.Network.Context) -> Optional[float]:
        return None

    @abstractstaticmethod
    def encode_obs(agent: AgentType, ctx: me.Network.Context) -> np.ndarray:
        raise NotImplementedError

    @abstractstaticmethod
    def decode_action(
        agent: AgentType, ctx: me.Network.Context, action: np.ndarray
    ) -> Packet:
        raise NotImplementedError

    @abstractstaticmethod
    def get_observation_space(agent: AgentType) -> gym.spaces.Space:
        raise NotImplementedError

    @abstractstaticmethod
    def get_action_space(agent: AgentType) -> gym.spaces.Space:
        raise NotImplementedError
