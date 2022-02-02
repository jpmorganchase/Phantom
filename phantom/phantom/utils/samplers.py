import functools
from abc import ABC, abstractmethod
from typing import (
    Callable,
    Generic,
    Iterable,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import numpy as np

T = TypeVar("T")


class ComparableType(Generic[T], ABC):
    """
    Interface for Types that can be compared.
    """

    @abstractmethod
    def __lt__(self, other: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __le__(self, other: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __gt__(self, other: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __ge__(self, other: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __eq__(self, other: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __ne__(self, other: object) -> bool:
        raise NotImplementedError


ComparableT = TypeVar("ComparableT", bound=ComparableType)


class BaseSampler(ABC, Generic[T]):
    """
    Samplers are used in Agent/Environment Types to make Supertypes.

    When training, at the start of each episode concrete values will be sampled from all
    the samplers and Types generated from the Supertypes.
    """

    def __init__(self):
        self.value: Optional[T] = None

    @abstractmethod
    def sample(self) -> T:
        raise NotImplementedError


class ComparableSampler(BaseSampler[ComparableT], Generic[ComparableT]):
    """
    Extension of the `BaseSampler` for ComparableTypes in order
    to treat the `ComparableSampler` like its actual internal value.

    Example:
    >>> s = UniformSampler()
    >>> s.value = s.sample()
    >>> s <= 1.0
    # True
    >>> s == 1.5
    # False
    """

    def __lt__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        if self.value is None:
            raise ValueError("`self.value` is None")
        return self.value < other

    def __eq__(self, other: object) -> bool:
        return self.value == other

    def __ne__(self, other: object) -> bool:
        return not self.__eq__(other)

    def __le__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return self.__gt__(other) or self.__eq__(other)


class UniformSampler(ComparableSampler[float]):
    """
    Samples a single float value from a uniform distribution.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        self.low = low
        self.high = high
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> float:
        value = np.random.uniform(self.low, self.high)

        if self.clip_low is not None or self.clip_high is not None:
            value = np.clip(value, self.clip_low, self.clip_high)

        return value


class UniformArraySampler(ComparableSampler[np.ndarray]):
    """
    Samples an array of float values from a uniform distribution.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: Iterable[int] = (1,),
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        self.low = low
        self.high = high
        self.shape = shape
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> np.ndarray:
        value = np.random.uniform(self.low, self.high, self.shape)

        if self.clip_low is not None or self.clip_high is not None:
            value = np.clip(value, self.clip_low, self.clip_high)

        return value


class NormalSampler(ComparableSampler[float]):
    """
    Samples a single float value from a normal distribution.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> float:
        value = np.random.normal(self.mu, self.sigma)

        if self.clip_low is not None or self.clip_high is not None:
            value = np.clip(value, self.clip_low, self.clip_high)

        return value


class NormalArraySampler(ComparableSampler[np.ndarray]):
    """
    Samples an array of float values from a normal distribution.
    """

    def __init__(
        self,
        mu: float,
        sigma: float,
        shape: Tuple[int] = (1,),
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        self.mu = mu
        self.sigma = sigma
        self.shape = shape
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> np.ndarray:
        value = np.random.normal(self.mu, self.sigma, self.shape)

        if self.clip_low is not None or self.clip_high is not None:
            value = np.clip(value, self.clip_low, self.clip_high)

        return value


class LambdaSampler(BaseSampler[T]):
    """
    Samples using an arbitrary lambda function
    """

    def __init__(self, *args, func: Callable[..., T] = None, **kwargs):
        if func is None:
            raise ValueError("You must provide a `func`")

        self.func = func
        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def sample(self) -> T:
        return self.func(*self.args, **self.kwargs)
