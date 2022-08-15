import os
from datetime import datetime
from pathlib import Path
from typing import List, Union


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


from .train import train
from .rollout import rollout
