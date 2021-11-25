from typing import Callable, Hashable


PolicyID = str
StageID = Hashable
EnvStageHandler = Callable[[], StageID]
