import collections
import inspect
import os
from typing import Any, Mapping, List, Optional, Tuple, Type, TypeVar, Union

from termcolor import colored

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


CollectedType = TypeVar("CollectedType")
ObjPath = List[Union[Tuple[bool, str], int]]


def collect_instances_of_type(
    type_: Type[CollectedType],
    obj: Any,
    collection: Optional[List[CollectedType]] = None,
) -> List[CollectedType]:
    collection = collection or []

    # Test object itself
    if isinstance(obj, type_) and obj not in collection:
        collection.append(obj)
    # Special handling of strings
    elif isinstance(obj, str):
        pass
    # Test sub-objects in mappings, eg. dicts
    elif isinstance(obj, collections.abc.Mapping):
        for val in obj.values():
            collection = collect_instances_of_type(type_, val, collection)
    # Test sub-objects in iterables, eg. lists, sets
    elif isinstance(obj, collections.abc.Iterable):
        for val in obj:
            collection = collect_instances_of_type(type_, val, collection)
    # Test dataclasses
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            collection = collect_instances_of_type(
                type_, getattr(obj, field), collection
            )

    return collection


def collect_instances_of_type_with_paths(
    type_: Type[CollectedType],
    obj: Any,
    collection: Optional[List[Tuple[CollectedType, List[ObjPath]]]] = None,
    current_path: Optional[ObjPath] = None,
) -> List[Tuple[CollectedType, List[ObjPath]]]:
    collection = collection or []
    current_path = current_path or []

    # Test object itself
    if isinstance(obj, type_):
        added = False
        for obj2, paths in collection:
            if obj == obj2:
                paths.append(current_path)
                added = True
                break
        if not added:
            collection.append((obj, [current_path]))
    # Test sub-objects in mappings, eg. dicts
    elif isinstance(obj, collections.abc.Mapping):
        for key, val in obj.items():
            collection = collect_instances_of_type_with_paths(
                type_, val, collection, current_path + [(True, key)]
            )
    # Test sub-objects in iterables, eg. lists, sets
    elif isinstance(obj, collections.abc.Iterable):
        for i, val in enumerate(obj):
            collection = collect_instances_of_type_with_paths(
                type_, val, collection, current_path + [i]
            )
    # Test dataclasses
    elif hasattr(obj, "__dataclass_fields__"):
        for field in obj.__dataclass_fields__:
            collection = collect_instances_of_type_with_paths(
                type_, getattr(obj, field), collection, current_path + [(False, field)]
            )
    # Test classes
    elif inspect.isclass(obj):
        for field in obj.__dict__:
            collection = collect_instances_of_type(
                type_, getattr(obj, field), collection
            )

    return collection


def update_val(obj: Any, path: ObjPath, new_val: Any) -> None:
    for item in path[:-1]:
        if isinstance(item, int):
            obj = obj[item]
        elif item[0] is True:
            obj = obj[item[1]]
        else:
            obj = getattr(obj, item[1])

    if isinstance(path[-1], int):
        obj[path[-1]] = new_val
    elif path[-1][0] is True:
        obj[path[-1][1]] = new_val
    else:
        obj = setattr(obj, path[-1][1], new_val)


def check_env_config(env_config: Mapping[str, Any]) -> None:
    if contains_type(env_config, Range):
        raise Exception

    for name, value in env_config.items():
        if name not in ["env_supertype", "agent_supertypes"]:
            if contains_type(value, Sampler):
                raise Exception


def show_pythonhashseed_warning() -> None:
    string = "================================================================\n"
    string += "WARNING: The $PYTHONHASHSEED environment variable is not set!\n"
    string += "Please set this before using Phantom to improve reproducibility.\n"
    string += "================================================================"

    if "PYTHONHASHSEED" not in os.environ:
        print(colored(string, "yellow"))
