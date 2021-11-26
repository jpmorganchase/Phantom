from typing import Dict, Union, TYPE_CHECKING

import mercury as me

from ..agent import Agent
from .typedefs import StageID

if TYPE_CHECKING:
    from .handlers import StageHandler, StagePolicyHandler


class FSMAgent(Agent):
    """
    FiniteStateMachine environment specialised agent class.

    Like the FSMActor class adds the ability to set per-stage hooks for tasks such as
    message resolution via the ``stage_handlers`` property.

    Beyond that, this class also adds the ability to set per-stage policies to the agent.

    See the :class:``StagePolicyHandler`` class for the full list of available hooks.

    Attributes:
        agent_id: A unique token identifying this agent.
        stage_handlers: A mapping of StageIDs to StageHandlers.
    """

    def __init__(
        self,
        agent_id: me.ID,
        stage_handlers: Dict[StageID, Union["StageHandler", "StagePolicyHandler"]],
    ):
        super().__init__(agent_id)

        self.stage_handlers = stage_handlers
