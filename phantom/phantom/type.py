from abc import ABC
from copy import deepcopy
from dataclasses import asdict, dataclass
from typing import Any, Dict, Union

import numpy as np
import gym.spaces

from .utils.samplers import BaseSampler


ObsSpaceCompatibleTypes = Union[dict, list, np.ndarray, tuple]


@dataclass
class BaseType(ABC):
    """
    Abstract base class representing Agent Types.
    """

    def sample(self) -> "BaseType":
        agent_type = deepcopy(self)

        for field_name in self.__dataclass_fields__:
            field_value = getattr(agent_type, field_name)

            if isinstance(field_value, BaseSampler):
                setattr(agent_type, field_name, field_value.value)

        return agent_type

    def is_supertype(self) -> bool:
        """
        Checks if this type instance is a supertype by looking for any contained values
        that are subclasses of BaseSampler.
        """
        for field_name in self.__dataclass_fields__:
            field_value = getattr(self, field_name)

            if isinstance(field_value, BaseSampler):
                return True

        return False

    def to_obs_space_compatible_type(self) -> Dict[str, ObsSpaceCompatibleTypes]:
        """
        Converts the parameters of the BaseType into a dict for use in observation
        spaces.
        """

        def _to_compatible_type(field: str, obj: Any) -> ObsSpaceCompatibleTypes:
            if isinstance(obj, dict):
                return {
                    key: _to_compatible_type(key, value) for key, value in obj.items()
                }
            elif isinstance(obj, (float, int)):
                return np.array([obj], dtype=np.float32)
            elif isinstance(obj, list):
                return [
                    _to_compatible_type(f"{field}[{i}]", value)
                    for i, value in enumerate(obj)
                ]
            elif isinstance(obj, tuple):
                return tuple(
                    _to_compatible_type(f"{field}[{i}]", value)
                    for i, value in enumerate(obj)
                )
            elif isinstance(obj, np.ndarray):
                return obj
            else:
                raise ValueError(
                    f"Can't encode field '{field}' with type '{type(obj)}' into obs space compatible type"
                )

        return {
            name: _to_compatible_type(name, value)
            for name, value in asdict(self).items()
        }

    def to_obs_space(self, low=-np.inf, high=np.inf) -> gym.spaces.Space:
        """
        Converts the parameters of the BaseType into a `gym.spaces.Space` representing
        the space.

        All elements of the space span the same range given by the `low` and `high`
        arguments.

        Arguments:
            low: Optional 'low' bound for the space (default is -∞)
            high: Optional 'high' bound for the space (default is ∞)
        """

        def _to_obs_space(field: str, obj: Any) -> gym.spaces.Space:
            if isinstance(obj, dict):
                return gym.spaces.Dict(
                    {key: _to_obs_space(key, value) for key, value in obj.items()}
                )
            elif isinstance(obj, float):
                return gym.spaces.Box(low, high, (1,), np.float32)
            elif isinstance(obj, int):
                return gym.spaces.Box(low, high, (1,), np.float32)
            elif isinstance(obj, (list, tuple)):
                return gym.spaces.Tuple(
                    [
                        _to_obs_space(f"{field}[{i}]", value)
                        for i, value in enumerate(obj)
                    ]
                )
            elif isinstance(obj, np.ndarray):
                return gym.spaces.Box(low, high, obj.shape, np.float32)
            else:
                raise ValueError(
                    f"Can't encode field '{field}' with type '{type(obj)}' into gym.spaces.Space"
                )

        return gym.spaces.Dict(
            {
                field: _to_obs_space(field, value)
                for field, value in asdict(self).items()
            }
        )
