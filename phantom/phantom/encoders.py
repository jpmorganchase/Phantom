from abc import abstractmethod, abstractproperty, ABC
from typing import Any, Dict, Generic, Iterable, List, Mapping, Tuple, TypeVar

import numpy as np
from gym import Space
from gym.spaces import Box, Dict as GymDict, Tuple as GymTuple
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


class DictEncoder(Encoder[Dict[str, Any]]):
    """Combines n encoders into a single encoder with a dict action space.

    Attributes:
        encoders: A mapping of encoder names to encoders.
    """

    def __init__(self, encoders: Mapping[str, Encoder]):
        self.encoders: Dict[str, Encoder] = dict(encoders)

    @property
    def output_space(self) -> Space:
        return GymDict(
            {name: encoder.output_space for name, encoder in self.encoders.items()}
        )

    def encode(self, ctx: Network.Context) -> Dict[str, Any]:
        return {name: encoder.encode(ctx) for name, encoder in self.encoders.items()}

    def reset(self):
        for encoder in self.encoders.values():
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



