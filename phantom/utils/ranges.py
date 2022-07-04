from abc import ABC, abstractmethod
from typing import Generic, Iterable, Optional, TypeVar

import numpy as np


T = TypeVar("T")


class Range(ABC, Generic[T]):
    """
    Samplers are used in Agent/Environment Types to make Supertypes.

    When training, at the start of each episode concrete values will be sampled from all
    the samplers and Types generated from the Supertypes.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    @abstractmethod
    def values(self) -> Iterable[T]:
        raise NotImplementedError

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<{self.__class__.__name__} name='{self.name}'>"

        return f"<{self.__class__.__name__}>"


class UniformRange(Range[float]):
    """
    Returns an array of values spaced by a step between a start and end value.
    """

    def __init__(
        self,
        start: float,
        end: float,
        step: float = 1.0,
        name: Optional[str] = None,
    ) -> None:
        self.start = start
        self.end = end
        self.step = step

        super().__init__(name)

    def values(self) -> np.ndarray:
        return np.arange(self.start, self.end, self.step)


class LinspaceRange(Range[float]):
    """
    Returns an array of n values evenly distributed between a start and end value.
    """

    def __init__(
        self,
        start: float,
        end: float,
        n: int,
        name: Optional[str] = None,
    ) -> None:
        self.n = n
        self.start = start
        self.end = end

        super().__init__(name)

    def values(self) -> np.ndarray:
        return np.linspace(self.start, self.end, self.n)
