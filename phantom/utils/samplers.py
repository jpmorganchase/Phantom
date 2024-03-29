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
from uuid import uuid4

import numpy as np

T = TypeVar("T")


class ComparableType(Generic[T], ABC):
    """Interface for Types that can be compared."""

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


class Sampler(ABC, Generic[T]):
    """Samplers are used in Agent/Environment Supertypes to define how they are sampled.

    Samplers are designed to be used when training policies and a stochastic
    distribution of values is required for the Supertype sampling.

    Samplers return an unbounded number of total values with one value being returned at
    a time with the :meth:`sample` method.
    """

    def __init__(self):
        self._value: Optional[T] = None
        self._id = uuid4()

    @property
    def value(self) -> Optional[T]:
        return self._value

    @abstractmethod
    def sample(self) -> T:
        """
        Returns a single value defined by the Sampler's internal distribution.

        Implementations of this function should also update the instance's
        :attr:`_value` property.
        """
        raise NotImplementedError


class ComparableSampler(Sampler[ComparableT], Generic[ComparableT]):
    """
    Extension of the :class:`Sampler` for ComparableTypes in order to treat the
    :class:`ComparableSampler` like its actual internal value.

    Example:
    >>> s = UniformFloatSampler()
    >>> s.value = s.sample()
    >>> s <= 1.0
    # True
    >>> s == 1.5
    # False
    """

    def __lt__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        if isinstance(other, ComparableSampler):
            return super().__lt__(other)
        if self.value is None:
            raise ValueError("`self.value` is None")
        return self.value < other

    def __eq__(self, other: object) -> bool:
        if isinstance(other, ComparableSampler):
            return object.__eq__(self, other)
        return self.value == other

    def __ne__(self, other: object) -> bool:
        if isinstance(other, ComparableSampler):
            return object.__ne__(self, other)
        return self.value != other

    def __le__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return self.__lt__(other) or self.__eq__(other)

    def __gt__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return not self.__le__(other)

    def __ge__(self, other: Union[ComparableT, "ComparableSampler"]) -> bool:
        return self.__gt__(other) or self.__eq__(other)


class UniformFloatSampler(ComparableSampler[float]):
    """Samples a single float value from a uniform distribution.

    Uses :func:`np.random.uniform` internally.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        assert high >= low

        self.low = low
        self.high = high
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> float:
        self._value = np.random.uniform(self.low, self.high)

        if self.clip_low is not None or self.clip_high is not None:
            self._value = np.clip(self._value, self.clip_low, self.clip_high)

        return self._value


class UniformIntSampler(ComparableSampler[int]):
    """Samples a single int value from a uniform distribution.

    Uses :func:`np.random.randint` internally.
    """

    def __init__(
        self,
        low: int = 0,
        high: int = 1,
        clip_low: Optional[int] = None,
        clip_high: Optional[int] = None,
    ) -> None:
        assert high >= low

        self.low = low
        self.high = high
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> float:
        self._value = np.random.randint(self.low, self.high)

        if self.clip_low is not None or self.clip_high is not None:
            self._value = np.clip(self._value, self.clip_low, self.clip_high)

        return self._value


class UniformArraySampler(ComparableSampler[np.ndarray]):
    """Samples an array of float values from a uniform distribution.

    Uses :func:`np.random.uniform()` internally.
    """

    def __init__(
        self,
        low: float = 0.0,
        high: float = 1.0,
        shape: Iterable[int] = (1,),
        clip_low: Optional[float] = None,
        clip_high: Optional[float] = None,
    ) -> None:
        assert high >= low

        self.low = low
        self.high = high
        self.shape = shape
        self.clip_low = clip_low
        self.clip_high = clip_high

        super().__init__()

    def sample(self) -> np.ndarray:
        self._value = np.random.uniform(self.low, self.high, self.shape)

        if self.clip_low is not None or self.clip_high is not None:
            self._value = np.clip(self._value, self.clip_low, self.clip_high)

        return self._value


class NormalSampler(ComparableSampler[float]):
    """Samples a single float value from a normal distribution.

    Uses :func:`np.random.normal()` internally.
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
        self._value = np.random.normal(self.mu, self.sigma)

        if self.clip_low is not None or self.clip_high is not None:
            self._value = np.clip(self._value, self.clip_low, self.clip_high)

        return self._value


class NormalArraySampler(ComparableSampler[np.ndarray]):
    """Samples an array of float values from a normal distribution.

    Uses :func:`np.random.normal()` internally.
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
        self._value = np.random.normal(self.mu, self.sigma, self.shape)

        if self.clip_low is not None or self.clip_high is not None:
            self._value = np.clip(self._value, self.clip_low, self.clip_high)

        return self._value


class LambdaSampler(Sampler[T]):
    """Samples using an arbitrary lambda function."""

    def __init__(self, func: Callable[..., T], *args, **kwargs):
        self.func = func
        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def sample(self) -> T:
        self._value = self.func(*self.args, **self.kwargs)
        return self._value
