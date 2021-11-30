import importlib
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, Union

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


def show_pythonhashseed_warning() -> None:
    string = "================================================================\n"
    string += "WARNING: The $PYTHONHASHSEED environment variable is not set!\n"
    string += "Please set this before using Phantom to improve reproducibility.\n"
    string += "================================================================"

    if "PYTHONHASHSEED" not in os.environ:
        print(colored(string, "yellow"))


from . import rollout, training
