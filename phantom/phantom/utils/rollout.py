import logging
import math
import multiprocessing
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Type, Union

import cloudpickle
import mercury as me
import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
from tqdm import tqdm

from ..env import PhantomEnv
from ..logging import Logger, Metric
from . import find_most_recent_results_dir, show_pythonhashseed_warning


logger = logging.getLogger(__name__)


@dataclass
class EpisodeTrajectory:
    """
    Class describing all the actions, observations, rewards, infos and dones of
    a single episode.
    """

    observations: List[Dict[me.ID, Any]]
    rewards: List[Dict[me.ID, float]]
    dones: List[Dict[me.ID, bool]]
    infos: List[Dict[me.ID, Dict[str, Any]]]
    actions: List[Dict[me.ID, Any]]


@dataclass
class RolloutConfig:
    seed: int
    env_config: Mapping[str, Any]


@dataclass
class Rollout:
    config: RolloutConfig
    metrics: Dict[me.ID, np.ndarray]
    trajectory: Optional[EpisodeTrajectory]


def rollout(
    directory: Union[str, Path],
    algorithm: str,
    num_workers: int = 0,
    num_rollouts: int = 1,
    checkpoint: Optional[int] = None,
    env_class: Optional[Type[PhantomEnv]] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    results_file: Optional[Union[str, Path]] = "results.pkl",
    save_trajectories: bool = False,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
) -> List[Rollout]:
    """
    Performs rollouts for a previously trained Phantom experiment.

    Arguments:
        directory: Phantom results directory containing trained policies. If the default
            phantom results location is used (~/phantom-results), this part of the path
            can be ommited. If LATEST is given as the last element of the path, the
            parent directory will be scanned for the most recent run and this will be
            used.
        algorithm: RLlib algorithm to use.
        num_workers: Number of Ray rollout workers to initialise.
        num_rollouts: Number of rollouts to perform, distributed over all workers.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
        env_config: Configuration parameters to pass to the environment init method.
        metrics: Optional set of metrics to record and log.
        results_file: Name of the results file to save to, if None is given no file
            will be saved (default is "results.pkl").
        save_trajectories: If True the full set of epsiode trajectories for the
            rollouts will be saved into the results file.
        result_mapping_fn: If given, results from each rollout will be passed to this
            function, with the return values from the function aggregated and saved.
            This can be useful when wanting to cut down on the size of rollout results
            file when only a specific subset of fields are needed in further analysis.

    Returns:
        - If result_mapping_fn is None: A list of Rollout objects.
        - If result_mapping_fn is not None: A list of the outputs from the
            result_mapping_fn function.

    NOTE: It is the users responsibility to invoke rollouts via the provided ``phantom``
    command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
    the Python interpreter to run this code. Not setting this may lead to
    reproducibility issues.
    """
    show_pythonhashseed_warning()

    metrics = metrics or {}
    env_config = env_config or {}

    phantom_dir = os.path.expanduser("~/phantom-results")

    directory = Path(directory)

    # If the user provides a path ending in '/LATEST', look for the most recent run
    # results in that directory
    if directory.stem == "LATEST":
        parent_dir = Path(os.path.expanduser(directory.parent))

        if not parent_dir.exists():
            # The user can provide a path relative to the phantom directory, if they do
            # so this will not be found when comparing to the system root so we try
            # appending it to the phantom directory path and test again.
            parent_dir = Path(phantom_dir, parent_dir)

            if not parent_dir.exists():
                raise FileNotFoundError(
                    f"Base results directory '{parent_dir}' does not exist"
                )

        logger.info("Trying to find latest experiment results in '%s'", parent_dir)

        directory = find_most_recent_results_dir(parent_dir)

        logger.info("Found experiment results: '%s'", directory.stem)
    else:
        directory = Path(os.path.expanduser(directory))

        if not directory.exists():
            directory = Path(phantom_dir, directory)

            if not directory.exists():
                raise FileNotFoundError(
                    f"Results directory '{directory}' does not exist"
                )

        logger.info("Using results directory: '%s'", directory)

    # If an explicit checkpoint is not given, find all checkpoints and use the newest.
    if checkpoint is None:
        checkpoint_dirs = sorted(Path(directory).glob("checkpoint_*"))

        if len(checkpoint_dirs) == 0:
            raise FileNotFoundError(f"No checkpoints found in directory '{directory}'")

        checkpoint = int(str(checkpoint_dirs[-1]).split("_")[-1])

        logger.info("Using most recent checkpoint: %s", checkpoint)
    else:
        logger.info("Using checkpoint: %s", checkpoint)

    # Apply the number of repeats requested to each 'variation'.
    rollout_configs = [
        RolloutConfig(
            i,
            env_config,
        )
        for i in range(num_rollouts)
    ]

    # Distribute the rollouts evenly amongst the number of workers.
    rollouts_per_worker = int(math.ceil(len(rollout_configs) / max(num_workers, 1)))

    logger.info(
        "Starting %s rollout(s) using %s worker process(es)",
        len(rollout_configs),
        num_workers,
    )

    # Start the rollouts
    if num_workers == 0:
        # If num_workers is 0, run all the rollouts in this thread.
        results: List[Rollout] = _rollout_task_fn(
            directory,
            checkpoint,
            algorithm,
            rollout_configs,
            env_class,
            metrics,
            result_mapping_fn,
            None,
        )

    else:
        # Otherwise, manually create threads and give a list of rollouts to each thread.
        # multiprocessing.Pool is not used to allow the use a progressbar that shows the
        # overall state across all threads.
        result_queue = multiprocessing.Manager().Queue()

        worker_payloads = [
            (
                directory,
                checkpoint,
                algorithm,
                rollout_configs[i : i + rollouts_per_worker],
                env_class,
                metrics,
                result_mapping_fn,
                result_queue,
            )
            for i in range(0, len(rollout_configs), rollouts_per_worker)
        ]

        workers = [
            multiprocessing.Process(target=_rollout_task_fn, args=worker_payloads[i])
            for i in range(len(worker_payloads))
        ]

        for worker in workers:
            worker.start()

        results = []

        # As results asynchronously arrive in the results queue, update the progress bar
        with tqdm(total=len(rollout_configs)) as pbar:
            for item in iter(result_queue.get, None):
                results.append(item)
                pbar.update()
                if len(results) == len(rollout_configs):
                    break

        for worker in workers:
            worker.join()

    # Optionally save the results
    if results_file is not None:
        logger.info("Generating results file")

        if result_mapping_fn is None:
            results_to_save = [
                Rollout(
                    rollout.config,
                    rollout.metrics,
                    rollout.trajectory if save_trajectories else None,
                )
                for rollout in results
            ]
        else:
            results_to_save = results

        results_file = Path(directory, results_file)

        cloudpickle.dump(results_to_save, open(results_file, "wb"))

        logger.info("Saved rollout results to '%s'", results_file)

    return results


def _rollout_task_fn(
    directory: Path,
    checkpoint: int,
    algorithm: str,
    configs: List[RolloutConfig],
    env_class: Optional[Type[PhantomEnv]] = None,
    tracked_metrics: Optional[Mapping[str, Metric]] = None,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
    result_queue: Optional[multiprocessing.Queue] = None,
) -> List[Rollout]:
    """
    Internal function.
    """

    checkpoint_path = Path(
        directory,
        f"checkpoint_{str(checkpoint).zfill(6)}",
        f"checkpoint-{checkpoint}",
    )

    # Wrap ray code in a try block to ensure ray is shutdown correctly if an error occurs
    try:
        ray.init(local_mode=True, include_dashboard=False)

        # Load config from results directory.
        with open(Path(directory, "params.pkl"), "rb") as params_file:
            config = cloudpickle.load(params_file)

        # Load env class from results directory.
        with open(Path(directory, "env.pkl"), "rb") as env_file:
            env_class = cloudpickle.load(env_file)

        # Set to zero as rollout workers != training workers - if > 0 will spin up
        # unnecessary additional workers.
        config["num_workers"] = 0
        config["env_config"] = configs[0].env_config

        # Register custom environment with Ray
        register_env(env_class.env_name, lambda config: env_class(**config))

        trainer = get_trainer_class(algorithm)(env=env_class.env_name, config=config)
        trainer.restore(str(checkpoint_path))

        results = []

        iter_obj = tqdm(configs) if result_queue is None else configs

        for rollout_config in iter_obj:
            # Create environment instance from config from results directory
            env = env_class(**rollout_config.env_config)

            # Setting seed needs to come after trainer setup
            np.random.seed(rollout_config.seed)

            metric_logger = Logger(tracked_metrics)

            observation = env.reset()

            observations: List[Dict[me.ID, Any]] = [observation]
            rewards: List[Dict[me.ID, float]] = []
            dones: List[Dict[me.ID, bool]] = []
            infos: List[Dict[me.ID, Dict[str, Any]]] = []
            actions: List[Dict[me.ID, Any]] = []

            # Run rollout steps.
            for _ in range(env.clock.n_steps):
                step_actions = {}

                for agent_id, agent_obs in observation.items():
                    policy_id = config["multiagent"]["policy_mapping"][agent_id]

                    agent_action = trainer.compute_action(
                        agent_obs, policy_id=policy_id, explore=False
                    )
                    step_actions[agent_id] = agent_action

                observation, reward, done, info = env.step(step_actions)
                metric_logger.log(env)

                observations.append(observation)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)
                actions.append(step_actions)

            metrics = {k: np.array(v) for k, v in metric_logger.to_dict().items()}

            trajectory = EpisodeTrajectory(
                observations,
                rewards,
                dones,
                infos,
                actions,
            )

            result = Rollout(rollout_config, metrics, trajectory)

            if result_mapping_fn is not None:
                result = result_mapping_fn(result)

            # If in multiprocess mode, add the results to the queue, otherwise store locally
            # until all rollouts for this function call are complete.
            if result_queue is None:
                results.append(result)
            else:
                result_queue.put(result)

    except Exception as exception:
        ray.shutdown()
        raise exception

    ray.shutdown()

    # If using multi-processing this will be an empty list
    return results
