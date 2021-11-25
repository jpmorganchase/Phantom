import collections
from typing import Dict, Iterable, List, Tuple, Union, TYPE_CHECKING

import mercury as me

from .types import StageID

if TYPE_CHECKING:
    from .handlers import StageHandler


class FSMActor(me.actors.SimpleSyncActor):
    def __init__(
        self,
        actor_id: me.ID,
        stage_handlers: Dict[Union[StageID, Iterable[StageID]], "StageHandler"],
    ):
        super().__init__(actor_id)

        self.stage_handlers: List[Tuple[List[StageID], "StageHandler"]] = []
        self.stage_handler_map: Dict[StageID, "StageHandler"] = {}

        for stage_ids, handler in stage_handlers.items():
            if isinstance(stage_ids, str) or not isinstance(
                stage_ids, collections.Iterable
            ):
                stage_ids = [stage_ids]

            self.stage_handlers.append((stage_ids, handler))

            for stage_id in stage_ids:
                self.stage_handler_map[stage_id] = handler
