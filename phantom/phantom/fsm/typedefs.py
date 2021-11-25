from typing import Callable, Hashable

StageID = Hashable
EnvStageHandler = Callable[[], StageID]
