from abc import ABC
from dataclasses import asdict, dataclass, make_dataclass
from typing import Any, Dict, TypeVar, Union

import numpy as np
import gym.spaces

from .utils.ranges import BaseRange
from .utils.samplers import BaseSampler


ObsSpaceCompatibleTypes = Union[dict, list, np.ndarray, tuple]

T = TypeVar("T")
SupertypeField = Union[T, BaseSampler[T], BaseRange[T]]


@dataclass
class BaseSupertype(ABC):
    """
    Abstract base class representing Supertypes for agents and environments. For type
    system correctness the fields of any subclass should be of type SupertypeField[T]
    where T is the type of the field of the underlying Type.

    NOTE:
        'type' is used in the context of the Python type-system.
        'Type' is used in the context of an RL agent/environment.

    Example usage::

        from phantom import BaseSupertype, SupertypeField

        ExampleSupertype(BaseSupertype):
            x: SupertypeField[float]
            y: SupertypeField[float]

        s = ExampleSupertype(1.0, 2.0)

        t = s.sample()

    Supertypes provided to the ``train`` method containing values that inherit from
    ``BaseSampler`` will be automatically sampled at the start of each episode.::

        s = ExampleSupertype(
            x=UniformSampler(0.0, 1.0)
            y=NormalSampler(0.0, 2.0)
        )
        
        t = s.sample()

    Supertypes provided to the ``rollout`` method containing values that inherit from
    ``BaseRange`` will be used to construct a multidimensional space containing all
    possible combinations of the ``BaseRange`` values to perform rollouts with.::

        s = ExampleSupertype(
            x=UniformRange(0.0, 1.0, 0.1)
            y=LinspaceRange(0.0, 0.5, 11)
        )
        
        t = s.sample()
    """

    def sample(self) -> "BaseType":
        """
        Constructs a new dataclass type with the same fields as this parent supertype
        class. All field values that inherit from ``BaseSampler`` are replaced with
        values sampled from the respective sampler.
        """

        new_type = make_dataclass(
            self.__class__.__name__ + "_Type",
            [
                (
                    name,
                    # If len(value.type.__args__) == 3 then this type is likely a
                    # SupertypeField union. We can use the first member which is the
                    # generic type as the field type for the new dataclass.
                    value.type.__args__[0] if len(value.type.__args__) == 3 else value,
                )
                for name, value in self.__dataclass_fields__.items()
            ],
        )

        return new_type(
            **{
                name: getattr(self, name).value
                if isinstance(getattr(self, name), BaseSampler)
                else getattr(self, name)
                for name in self.__dataclass_fields__
            }
        )


@dataclass
class BaseType(ABC):
    """
    Abstract base class representing a Type for an agent or an environment. When using
    the phantom train/rollout functions it should not be necessary to subclass this
    type. The sample method on the BaseSupertype class is used to dynamically create
    subclasses of this class.
    """

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
