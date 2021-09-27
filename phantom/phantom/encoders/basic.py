from typing import Any, Tuple

import numpy as np
from gym.spaces import Box

from mercury import Network

from . import Encoder


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
