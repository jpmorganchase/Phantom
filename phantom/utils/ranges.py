from abc import ABC, abstractmethod
from typing import Generic, List, Optional, Sequence, TypeVar

import numpy as np


T = TypeVar("T")


class Range(ABC, Generic[T]):
    """Ranges are used in Agent/Environment Supertypes to define how they are sampled.

    Ranges are designed to be used when generating rollouts post-training and a
    non-stochastic distribution of values is required for the Supertype sampling.

    Ranges return a fixed number of total values and as such all values must be returned
    in one go with the :meth:`values` method.
    """

    def __init__(self, name: Optional[str] = None) -> None:
        self.name = name

    @abstractmethod
    def values(self) -> Sequence[T]:
        """
        Returns the complete set of values defined by the Range.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        if self.name is not None:
            return f"<{self.__class__.__name__} name='{self.name}'>"

        return f"<{self.__class__.__name__}>"


class UniformRange(Range[float]):
    """Returns an array of values spaced by a step between a start and end value.

    Uses :func:`np.arange` internally.
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
    """Returns an array of n values evenly distributed between a start and end value.

    Uses :func:`np.linspace` internally.
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


class UnitArrayUniformRange(UniformRange, Range[np.ndarray]):
    """
    Returns a list of n shape (1,) numpy arrays with values spaced by a step between a
    start and end value. Useful for encoding observation spaces with single element
    boxes.

    Uses :func:`np.arange` internally.
    """

    def values(self) -> List[np.ndarray]:
        return [np.array([x]) for x in np.arange(self.start, self.end, self.step)]


class UnitArrayLinspaceRange(LinspaceRange, Range[np.ndarray]):
    """
    Returns a list of n shape (1,) numpy arrays with values evenly distributed between a
    start and end value. Useful for encoding observation spaces with single element
    boxes.

    Uses :func:`np.linspace` internally.
    """

    def values(self) -> List[np.ndarray]:
        return [np.array([x]) for x in np.linspace(self.start, self.end, self.n)]
