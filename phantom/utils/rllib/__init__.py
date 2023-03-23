import os
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Tuple, Union


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
            datetime.strptime(str(directory)[-27:-8], "%Y-%m-%d_%H-%M-%S")
            experiment_directories.append(directory)
        except ValueError:
            pass

    if len(experiment_directories) == 0:
        raise ValueError(f"No experiment directories found in '{base_path}'")

    experiment_directories.sort(
        key=lambda d: datetime.strptime(str(d)[-27:-8], "%Y-%m-%d_%H-%M-%S")
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
            int(str(checkpoint_dir).rsplit("_", maxsplit=1)[-1])
            for checkpoint_dir in checkpoint_dirs
        )
    )


def construct_results_paths(
    directory: Union[str, Path], checkpoint: Optional[int] = None
) -> Tuple[Path, Path]:
    if checkpoint is not None:
        assert isinstance(checkpoint, int)

    ray_dir = os.path.expanduser("~/ray_results")

    directory = Path(directory)

    # If the user provides a path ending in '/LATEST', look for the most recent run
    # results in that directory
    if directory.stem == "LATEST":
        parent_dir = Path(os.path.expanduser(directory.parent))

        if not parent_dir.exists():
            # The user can provide a path relative to the phantom directory, if they do
            # so this will not be found when comparing to the system root so we try
            # appending it to the phantom directory path and test again.
            parent_dir = Path(ray_dir, parent_dir)

            if not parent_dir.exists():
                raise FileNotFoundError(
                    f"Base results directory '{parent_dir}' does not exist"
                )

        directory = find_most_recent_results_dir(parent_dir)
    else:
        directory = Path(os.path.expanduser(directory))

        if not directory.exists():
            directory = Path(ray_dir, directory)

            if not directory.exists():
                raise FileNotFoundError(
                    f"Results directory '{directory}' does not exist"
                )

    # If an explicit checkpoint is not given, find all checkpoints and use the newest.
    if checkpoint is None:
        checkpoint = get_checkpoints(directory)[-1]

    checkpoint_path = Path(directory, f"checkpoint_{str(checkpoint).zfill(6)}")

    return (directory, checkpoint_path)


from .policy_evaluation import evaluate_policy
from .train import train
from .rollout import rollout
