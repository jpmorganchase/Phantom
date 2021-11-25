from typing import Dict, Iterable, Union, TYPE_CHECKING

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

        self.stage_handlers: Dict[StageID, "StageHandler"] = {}

        for stages, handler in stage_handlers.items():
            if isinstance(stages, StageID):
                self.stage_handlers[stages] = handler
            else:
                for stage in stages:
                    self.stage_handlers[stage] = handler
