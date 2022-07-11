from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Union

import gym
import numpy as np

from .utils.samplers import Sampler


ObsSpaceCompatibleTypes = Union[dict, list, np.ndarray, tuple]


@dataclass
class Supertype(ABC):
    def sample(self) -> "Supertype":
        sampled_fields = {}

        for field_name in self.__dataclass_fields__:
            field = getattr(self, field_name)

            if isinstance(field, Sampler):
                if hasattr(self, "_managed"):
                    sampled_fields[field_name] = field.value
                else:
                    sampled_fields[field_name] = field.sample()
            else:
                sampled_fields[field_name] = field

        return self.__class__(**sampled_fields)

    def to_obs_space_compatible_type(self) -> Dict[str, ObsSpaceCompatibleTypes]:
        """
        Converts the parameters of the Supertype into a dict for use in observation
        spaces.
        """

        return {
            name: _to_compatible_type(name, getattr(self, name))
            for name in self.__dataclass_fields__
        }

    def to_obs_space(self, low=-np.inf, high=np.inf) -> gym.Space:
        """
        Converts the parameters of the Supertype into a `gym.Space` representing
        the space.

        All elements of the space span the same range given by the `low` and `high`
        arguments.

        Arguments:
            low: Optional 'low' bound for the space (default is -∞)
            high: Optional 'high' bound for the space (default is ∞)
        """

        return gym.spaces.Dict(
            {
                name: _to_obs_space(name, getattr(self, name), low, high)
                for name in self.__dataclass_fields__
            }
        )


def _to_compatible_type(field: str, obj: Any) -> ObsSpaceCompatibleTypes:
    """Internal function."""

    if isinstance(obj, dict):
        return {key: _to_compatible_type(key, value) for key, value in obj.items()}
    if isinstance(obj, (float, int)):
        return np.array([obj], dtype=np.float32)
    if isinstance(obj, list):
        return [
            _to_compatible_type(f"{field}[{i}]", value) for i, value in enumerate(obj)
        ]
    if isinstance(obj, tuple):
        return tuple(
            _to_compatible_type(f"{field}[{i}]", value) for i, value in enumerate(obj)
        )
    if isinstance(obj, np.ndarray):
        return obj

    raise ValueError(
        f"Can't encode field '{field}' with type '{type(obj)}' into obs space compatible type"
    )


def _to_obs_space(field: str, obj: Any, low: float, high: float) -> gym.Space:
    """Internal function."""

    if isinstance(obj, dict):
        return gym.spaces.Dict(
            {key: _to_obs_space(key, value, low, high) for key, value in obj.items()}
        )
    if isinstance(obj, float):
        return gym.spaces.Box(low, high, (1,), np.float32)
    if isinstance(obj, int):
        return gym.spaces.Box(low, high, (1,), np.float32)
    if isinstance(obj, (list, tuple)):
        return gym.spaces.Tuple(
            [
                _to_obs_space(f"{field}[{i}]", value, low, high)
                for i, value in enumerate(obj)
            ]
        )
    if isinstance(obj, np.ndarray):
        return gym.spaces.Box(low, high, obj.shape, np.float32)

    raise ValueError(
        f"Can't encode field '{field}' with type '{type(obj)}' into gym.Space"
    )
