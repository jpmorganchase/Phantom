import collections
from typing import (
    Dict,
    Iterable,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import mercury as me

from ..agent import Agent
from .env import encode_stage_policy_name
from .types import PolicyID, StageID

if TYPE_CHECKING:
    from .handlers import StagePolicyHandler


class FSMAgent(Agent):
    def __init__(
        self,
        agent_id: me.ID,
        stage_handlers: Dict[Union[StageID, Iterable[StageID]], "StagePolicyHandler"],
    ):
        super().__init__(agent_id)

        self.stage_handlers: Dict[
            PolicyID, Tuple["StagePolicyHandler", Iterable[StageID]]
        ] = {}

        for stages, handler in stage_handlers.items():
            policy_name = encode_stage_policy_name(agent_id, stages)

            if isinstance(stages, str) or not isinstance(stages, collections.Iterable):
                stages = [stages]

            self.stage_handlers[policy_name] = (handler, stages)
