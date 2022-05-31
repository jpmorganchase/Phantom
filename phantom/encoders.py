from abc import abstractmethod, ABC
from typing import Any, Dict, Generic, Iterable, List, Mapping, Tuple, TypeVar

import gym
import numpy as np

from .context import Context


Observation = TypeVar("Observation")


def flatten(xs: Iterable[Any]) -> List[Any]:
    """Recursively flatten an iterable object into a list.

    Arguments:
        xs: The iterable object.
    """
    return sum(([x] if not isinstance(x, Iterable) else flatten(x) for x in xs), [])


class Encoder(Generic[Observation], ABC):
    """A trait for types that encodes the context of an agent into an observation."""

    @property
    @abstractmethod
    def output_space(self) -> gym.Space:
        """The output space of the encoder type."""

    @abstractmethod
    def encode(self, ctx: Context) -> Observation:
        """Encode the data in a given network context into an observation.

        Arguments:
            ctx: The local network context.

        Returns:
            An observation encoding properties of the provided context.
        """

    def chain(self, others: Iterable["Encoder"]) -> "ChainedEncoder":
        """Chains this encoder together with adjoint set of encoders.

        This method returns a :class:`ChainedEncoder` instance where the output
        space reduces to a tuple with each element given by the output space
        specified in each of the encoders provided.
        """

        return ChainedEncoder(flatten([self, others]))

    def reset(self):
        """Resets the encoder."""

    def __repr__(self) -> str:
        return repr(self.output_space)

    def __str__(self) -> str:
        return str(self.output_space)


class EmptyEncoder(Encoder[np.ndarray]):
    """Generates an empty observation."""

    @property
    def output_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, _: Context) -> np.ndarray:
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
    def output_space(self) -> gym.Space:
        return gym.spaces.Tuple(tuple(d.output_space for d in self.encoders))

    def encode(self, ctx: Context) -> Tuple:
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
    def output_space(self) -> gym.Space:
        return gym.spaces.Dict(
            {name: encoder.output_space for name, encoder in self.encoders.items()}
        )

    def encode(self, ctx: Context) -> Dict[str, Any]:
        return {name: encoder.encode(ctx) for name, encoder in self.encoders.items()}

    def reset(self):
        for encoder in self.encoders.values():
            encoder.reset()


class Constant(Encoder[np.ndarray]):
    def __init__(self, shape: Tuple[int], value: float = 0.0) -> None:
        self._shape = shape
        self._value = value

    @property
    def output_space(self) -> gym.spaces.Box:
        return gym.spaces.Box(-np.inf, np.inf, shape=self._shape, dtype=np.float32)

    def encode(self, _: Context) -> np.ndarray:
        return np.full(self._shape, self._value)
