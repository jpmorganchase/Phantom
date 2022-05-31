import collections
from typing import Any, Mapping, Type

from .ranges import Range
from .samplers import Sampler


def contains_type(value: Any, type_: Type) -> bool:
    # Test object itself
    if isinstance(value, type_):
        return True
    # Special handling of strings
    if isinstance(value, str):
        return False
    # Test dataclasses
    if hasattr(value, "__dataclass_fields__"):
        for field in value.__dataclass_fields__:
            if contains_type(getattr(value, field), type_):
                return True
    # Test sub-objects in mappings, eg. dicts
    elif isinstance(value, collections.abc.Mapping):
        for v in value.values():
            if contains_type(v, type_):
                return True
    # Test sub-objects in iterables, eg. lists, sets
    elif isinstance(value, collections.abc.Iterable):
        for v in value:
            if contains_type(v, type_):
                return True

    return False


def check_env_config(env_config: Mapping[str, Any]) -> None:
    if contains_type(env_config, Range):
        raise Exception

    for name, value in env_config.items():
        if name not in ["env_supertype", "agent_supertypes"]:
            if contains_type(value, Sampler):
                raise Exception
