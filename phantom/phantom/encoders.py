from abc import abstractmethod, abstractproperty, ABC
from typing import Any, Generic, Iterable, List, Tuple, TypeVar

import numpy as np
from gym import Space
from gym.spaces import Box, Tuple as GymTuple
from mercury import Network


Observation = TypeVar("Observation")


def flatten(xs: Iterable[Any]) -> List[Any]:
    """Recursively flatten an iterable object into a list.

    Arguments:
        xs: The iterable object.
    """
    return sum(([x] if not isinstance(x, Iterable) else flatten(x) for x in xs), [])


class Encoder(Generic[Observation], ABC):
    """A trait for types that encodes the context of an agent into an observation."""

    @abstractproperty
    def output_space(self) -> Space:
        """The output space of the encoder type."""
        pass

    @abstractmethod
    def encode(self, ctx: Network.Context) -> Observation:
        """Encode the data in a given network context into an observation.

        Arguments:
            ctx: The local network context.

        Returns:
            An observation encoding properties of the provided context.
        """
        pass

    def chain(self, others: Iterable["Encoder"]) -> "ChainedEncoder":
        """Chains this encoder together with adjoint set of encoders.

        This method returns a :class:`ChainedEncoder` instance where the output
        space reduces to a tuple with each element given by the output space
        specified in each of the encoders provided.
        """

        return ChainedEncoder(flatten([self, others]))

    def reset(self):
        """Resets the encoder."""
        pass

    def __repr__(self) -> str:
        return repr(self.output_space)

    def __str__(self) -> str:
        return str(self.output_space)


class EmptyEncoder(Encoder[np.ndarray]):
    """Generates an empty observation."""

    @property
    def output_space(self) -> Box:
        return Box(-np.inf, np.inf, (1,))

    def encode(self, _: Network.Context) -> np.ndarray:
        return np.zeros((1,))


class ChainedEncoder(Encoder[Tuple]):
    """Combines n encoders into a single encoder with a tuple action space.

    Attributes:
        encoders: An iterable collection of encoders which is flattened into a
            list.
    """

    def __init__(self, encoders: Iterable[Encoder]):
        self.encoders: List[Encoder] = flatten(encoders)

    @property
    def output_space(self) -> Space:
        return GymTuple(tuple(d.output_space for d in self.encoders))

    def encode(self, ctx: Network.Context) -> Tuple:
        return tuple(e.encode(ctx) for e in self.encoders)

    def chain(self, others: Iterable["Encoder"]) -> "ChainedEncoder":
        return ChainedEncoder(self.encoders + list(others))

    def reset(self):
        for encoder in self.encoders:
            encoder.reset()


class Constant(Encoder[np.ndarray]):
    def __init__(self, shape: Tuple[int], value: float = 0.0) -> None:
        self._shape = shape
        self._value = value

    @property
    def output_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=self._shape, dtype=np.float32)

    def encode(self, ctx: Network.Context) -> np.ndarray:
        return np.full(self._shape, self._value)


class DataEntry(Encoder[np.ndarray]):
    def __init__(self, key: Any, shape: Tuple[int]) -> None:
        self._key = key
        self._shape = shape

    @property
    def output_space(self) -> Box:
        return Box(-np.inf, np.inf, shape=self._shape, dtype=np.float32)

    def encode(self, ctx: Network.Context) -> np.ndarray:
        v = ctx.actor.data[self._key]

        if np.isscalar(v):
            return np.array([v])

        return v


class ElapsedTime(Encoder[np.ndarray]):
    """Encoder providing the fraction of time elapsed since the beginning of the episode.

    The `clock` comes from the `ActorMixin`, `MarketShareFrequencyTracker` for Market Maker
    or `TradeSideFrequencyTracker` for Investor.
    """

    @property
    def output_space(self) -> Box:
        return Box(low=-1e-12, high=1.5, shape=(1,))

    @staticmethod
    def _elapsed_time_fraction(clock):
        return (
            1.0
            if clock.is_terminal
            else 2 * np.ceil(clock.elapsed / 2) / clock.terminal_time
        )

    def encode(self, ctx: Network.Context) -> np.ndarray:
        elapsed_time = self._elapsed_time_fraction(ctx.actor.clock)
        return np.array([elapsed_time])
