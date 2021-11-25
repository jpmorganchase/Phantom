import collections
from typing import (
    Dict,
    Iterable,
    List,
    Tuple,
    Union,
    TYPE_CHECKING,
)

import mercury as me

from ..agent import Agent
from .typedefs import StageID

if TYPE_CHECKING:
    from .handlers import StagePolicyHandler


class FSMAgent(Agent):
    def __init__(
        self,
        agent_id: me.ID,
        stage_handlers: Dict[Union[StageID, Iterable[StageID]], "StagePolicyHandler"],
    ):
        super().__init__(agent_id)

        self.stage_handlers: List[Tuple[List[StageID], "StagePolicyHandler"]] = []
        self.stage_handler_map: Dict[StageID, "StagePolicyHandler"] = {}

        for stage_ids, handler in stage_handlers.items():
            if isinstance(stage_ids, str) or not isinstance(
                stage_ids, collections.Iterable
            ):
                stage_ids = [stage_ids]

            self.stage_handlers.append((stage_ids, handler))

            for stage_id in stage_ids:
                self.stage_handler_map[f"{agent_id}__{stage_id}"] = handler
