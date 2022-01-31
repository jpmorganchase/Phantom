from abc import ABC, abstractmethod
from typing import Callable, Generic, Iterable, Optional, Tuple, TypeVar

import numpy as np


T = TypeVar("T")


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

    def __lt__(self, other):
        return self.value < other

    def __le__(self, other):
        return self.value <= other

    def __gt__(self, other):
        return self.value > other

    def __ge__(self, other):
        return self.value >= other

    def __eq__(self, other):
        return self.value == other

    def __ne__(self, other):
        return not self == other


class UniformSampler(BaseSampler[float]):
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


class UniformArraySampler(BaseSampler[np.ndarray]):
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


class NormalSampler(BaseSampler[float]):
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


class NormalArraySampler(BaseSampler[np.ndarray]):
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

    def __init__(
        self,
        *args,
        func: Callable[..., T] = None,
        **kwargs
    ):
        if func is None:
            raise ValueError("You must provide a `func`")

        self.func = func
        self.args = args
        self.kwargs = kwargs

        super().__init__()

    def sample(self) -> T:
        return self.func(*self.args, **self.kwargs)