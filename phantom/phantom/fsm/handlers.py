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
    def pre_stage_hook(actor: ActorType) -> None:
        """
        TODO: docstring

        Arguments:
            actor: The actor (or agent) this stage handler class belongs to.
        """

    # TODO: implement and test
    @staticmethod
    def pre_msg_resolution_hook(actor: ActorType) -> None:
        """
        TODO: docstring

        Arguments:
            actor: The actor (or agent) this stage handler class belongs to.
        """

    # TODO: implement and test
    @staticmethod
    def post_msg_resolution_hook(actor: ActorType) -> None:
        """
        TODO: docstring

        Arguments:
            actor: The actor (or agent) this stage handler class belongs to.
        """

    # TODO: implement and test
    @staticmethod
    def post_stage_hook(actor: ActorType) -> None:
        """
        TODO: docstring

        Arguments:
            actor: The actor (or agent) this stage handler class belongs to.
        """


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
        """
        See ``ph.Agent.compute_reward``.

        Arguments:
            agent: The agent this stage handler class belongs to.
            ctx: A Mercury Context object representing the local view of the
                environment.

        Returns:
            An optional float representing the present reward value.
        """
        return None

    @abstractstaticmethod
    def encode_obs(agent: AgentType, ctx: me.Network.Context) -> np.ndarray:
        """
        See ``ph.Agent.encode_obs``.

        Arguments:
            agent: The agent this stage handler class belongs to.
            ctx: A Mercury Context object representing the local view of the
                environment.

        Returns:
            A numpy array encoding the observations.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def decode_action(
        agent: AgentType, ctx: me.Network.Context, action: np.ndarray
    ) -> Packet:
        """
        See ``ph.Agent.decode_action``.

        Arguments:
            agent: The agent this stage handler class belongs to.
            ctx: A Mercury Context object representing the local view of the
                environment.
            action: The action taken by the agent.

        Returns:
            A Packet object containing messages to be sent to other actors.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def get_observation_space(agent: AgentType) -> gym.spaces.Space:
        """
        Returns the gym observation space of this stage handler.
        """
        raise NotImplementedError

    @abstractstaticmethod
    def get_action_space(agent: AgentType) -> gym.spaces.Space:
        """
        Returns the gym action space of this stage handler.
        """
        raise NotImplementedError
