import importlib
import os
from datetime import datetime
from pathlib import Path
from typing import Generic, TypeVar, Union

T = TypeVar("T")


def load_object(path: str, name: str, type: Generic[T]) -> Generic[T]:
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

    if not isinstance(obj, type):
        raise Exception(f"'{name}' object is not an instance of the {type} class.")

    return obj


def find_most_recent_results_dir(base_path: Union[Path, str]) -> Path:
    """
    Scans a directory containing ray experiment results and returns the path of
    the most recent experiment.

    Arguments:
        base_path: The directory to search in.
    """

    base_path = Path(os.path.expanduser(base_path))

    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if len(experiment_dirs) == 0:
        raise ValueError(f"No experiment directories found in '{base_path}'")

    experiment_dirs.sort(
        key=lambda d: datetime.strptime(str(d)[-19:], "%Y-%m-%d_%H-%M-%S")
    )

    return experiment_dirs[-1]
