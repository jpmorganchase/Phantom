from typing import Dict, TYPE_CHECKING

import mercury as me

from .typedefs import StageID

if TYPE_CHECKING:
    from .handlers import StageHandler


class FSMActor(me.actors.SimpleSyncActor):
    """
    FiniteStateMachine environment specialised actor class.

    This class adds the ability to set per-stage hooks for tasks such as message
    resolution via the ``stage_handlers`` property.

    See the :class:`StageHandler` class for the full list of available hooks.

    Attributes:
        actor_id: A unique token identifying this actor.
        stage_handlers: A mapping of StageIDs to StageHandlers.
    """

    def __init__(
        self,
        actor_id: me.ID,
        stage_handlers: Dict[StageID, "StageHandler"],
    ):
        super().__init__(actor_id)

        self.stage_handlers = stage_handlers
