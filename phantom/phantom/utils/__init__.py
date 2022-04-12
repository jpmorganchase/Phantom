import collections.abc
import inspect
import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, List, Optional, Type, TypeVar, Tuple, Union

from termcolor import colored


T = TypeVar("T")


def load_object(path: str, name: str, obj_type: Generic[T]) -> Generic[T]:
    """
    Attempts to load an object with a given type from a file.

    Arguments:
        path: A path pointing to the file containing the object.

    Returns:
        The params object
    """

    if not os.path.exists(path):
        raise Exception(f"File '{path}' does not exist!")

    module_name = path[:-3].split("/")[-1]

    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, name):
        raise Exception(f"'{name}' field not found in file.")

    obj = getattr(module, name)

    if not isinstance(obj, obj_type):
        raise Exception(f"'{name}' object is not an instance of the {obj_type} class.")

    return obj


def find_most_recent_results_dir(base_path: Union[Path, str]) -> Path:
    """
    Scans a directory containing ray experiment results and returns the path of
    the most recent experiment.

    Arguments:
        base_path: The directory to search in.
    """

    base_path = Path(os.path.expanduser(base_path))

    directories = [d for d in base_path.iterdir() if d.is_dir()]

    experiment_directories = []

    for directory in directories:
        # Not all directories will be experiment results directories. Filter by
        # attempting to parse a datetime from the directory name.
        try:
            datetime.strptime(str(directory)[-19:], "%Y-%m-%d_%H-%M-%S")
            experiment_directories.append(directory)
        except ValueError:
            pass

    if len(experiment_directories) == 0:
        raise ValueError(f"No experiment directories found in '{base_path}'")

    experiment_directories.sort(
        key=lambda d: datetime.strptime(str(d)[-19:], "%Y-%m-%d_%H-%M-%S")
    )

    return experiment_directories[-1]


def get_checkpoints(results_dir: Union[Path, str]) -> List[int]:
    """
    Scans a directory containing an experiment's results and returns a list of all the
    checkpoints in that directory.

    Arguments:
        results_dir: The directory to search in.
    """
    
    checkpoint_dirs = list(Path(results_dir).glob("checkpoint_*"))

    if len(checkpoint_dirs) == 0:
        raise FileNotFoundError(f"No checkpoints found in directory '{results_dir}'")

    return list(
        sorted(
            int(str(checkpoint_dir).split("_")[-1])
            for checkpoint_dir in checkpoint_dirs
        )
    )


def show_pythonhashseed_warning() -> None:
    string = "================================================================\n"
    string += "WARNING: The $PYTHONHASHSEED environment variable is not set!\n"
    string += "Please set this before using Phantom to improve reproducibility.\n"
    string += "================================================================"

    if "PYTHONHASHSEED" not in os.environ:
        print(colored(string, "yellow"))


def contains_type(value: Any, type_: Type) -> bool:
    # Test object itself
    if isinstance(value, type_):
        return True
    # Special handling of strings
    if isinstance(value, str):
        return False
    # Test dataclasses
    elif hasattr(value, "__dataclass_fields__"):
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


ObjPath = List[Union[Tuple[bool, str], int]]


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
