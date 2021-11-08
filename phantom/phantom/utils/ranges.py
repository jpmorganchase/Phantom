from abc import ABC, abstractmethod
from typing import Any, Generic, Iterable, Optional, TypeVar

import numpy as np


T = TypeVar("T")


class BaseRange(ABC, Generic[T]):
    """
    Samplers are used in Agent/Environment Types to make Supertypes.

    When training, at the start of each episode concrete values will be sampled from all
    the samplers and Types generated from the Supertypes.
    """

    @abstractmethod
    def values(self) -> Iterable[T]:
        raise NotImplementedError


class UniformRange(BaseRange[float]):
    """
    Returns an array of values even distributed between a start and end value.
    """

    def __init__(
        self,
        start: float,
        end: float,
        step: float = 1.0,
    ) -> None:
        self.start = start
        self.end = end
        self.step = step

    def values(self) -> np.ndarray:
        return np.arange(self.start, self.end, self.step)
