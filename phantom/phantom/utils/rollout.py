import logging
import math
import cloudpickle
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import *

import mercury as me
import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

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


def rollout(
    directory: Union[str, Path],
    num_workers: int,
    num_rollouts: int,
    algorithm: str,
    checkpoint: Optional[int] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    results_file: Optional[Union[str, Path]] = "results.pkl",
    save_trajectories: bool = False,
) -> Tuple[List[Dict[str, np.ndarray]], List[EpisodeTrajectory]]:
    """
    Performs rollouts for a previously trained Phantom experiment.

    Attributes:
        directory: Phantom results directory containing trained policies.
        num_workers: Number of Ray rollout workers to initialise.
        num_rollouts: Number of rollouts to perform, distributed over all workers.
        algorithm: RLlib algorithm to use.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_config: Configuration parameters to pass to the environment init method.
        metrics: Optional set of metrics to record and log.
        results_file: Name of the results file to save to, if None is given no file
            will be saved (default is "results.pkl").
        save_trajectories: If True the full set of epsiode trajectories for the
            rollouts will be saved into the results file.

    Returns:
        - A list of dictionaries containing recorded metrics for each rollout.
        - A list of EpisodeTrajectory's.

    NOTE: It is the users responsibility to ensure the PYTHONHASHSEED environment variable
    is set before starting the Python interpreter to run this code. Not setting this may
    lead to reproducibility issues.
    """
    show_pythonhashseed_warning()

    metrics = metrics or {}
    env_config = env_config or {}

    if directory is not None:
        directory = Path(directory)

    if directory.stem == "LATEST":
        logger.info(f"Trying to find latest experiment results in '{directory.parent}'")

        directory = find_most_recent_results_dir(directory.parent)

        logger.info(f"Found experiment results: '{directory.stem}'")
    else:
        logger.info(f"Using results directory: '{directory}'")

    if checkpoint is None:
        checkpoint_dirs = sorted(Path(directory).glob("checkpoint_*"))

        if len(checkpoint_dirs) == 0:
            logger.error(f"No checkpoints found in directory '{directory}'")
            return

        checkpoint = int(str(checkpoint_dirs[-1]).split("_")[-1])

        logger.info(f"Using most recent checkpoint: {checkpoint}")
    else:
        logger.info(f"Using checkpoint: {checkpoint}")

    num_workers = max(num_workers, 1)

    rollouts_per_worker = int(math.ceil(num_rollouts / num_workers))

    seeds = list(range(num_rollouts))

    worker_payloads = [
        ParallelFunctionArgs(
            seeds[i : i + rollouts_per_worker],
            directory,
            checkpoint,
            env_config,
            metrics,
            algorithm,
        )
        for i in range(0, len(seeds), rollouts_per_worker)
    ]

    logger.info(f"Starting {num_rollouts} rollout(s) using {num_workers} worker(s)")

    try:
        ray.init(include_dashboard=False)

        results = list(
            chain.from_iterable(
                ray.util.iter.from_items(
                    worker_payloads, num_shards=max(num_workers, 1)
                )
                .for_each(_parallel_fn)
                .gather_sync()
            )
        )

    except Exception as e:
        # ensure that Ray is properly shutdown in the instance of any error occuring
        ray.shutdown()
        raise e
    else:
        ray.shutdown()

    metrics_results, trajectories = zip(*results)

    results = {}

    if metrics != {}:
        results["metrics"] = metrics_results

    if save_trajectories:
        results["trajectories"] = trajectories

    if results_file is not None and results != {}:
        results_file = Path(directory, results_file)

        cloudpickle.dump(results, open(results_file, "wb"))

        logger.info(f"Saved rollout results to '{results_file}'")

    return metrics_results, trajectories


@dataclass
class ParallelFunctionArgs:
    """
    Internal dataclass
    """

    seeds: List[int]
    directory: Path
    checkpoint: int
    env_config: Mapping[str, Any]
    metrics: Optional[Mapping[str, Metric]]
    algorithm: str


def _parallel_fn(
    args: ParallelFunctionArgs,
) -> List[Tuple[Dict[str, np.ndarray], EpisodeTrajectory]]:
    checkpoint_path = Path(
        args.directory,
        f"checkpoint_{str(args.checkpoint).zfill(6)}",
        f"checkpoint-{args.checkpoint}",
    )

    # Load config from results directory.
    with open(Path(args.directory, "params.pkl"), "rb") as f:
        config = cloudpickle.load(f)

    # Load env class from results directory.
    with open(Path(args.directory, "env.pkl"), "rb") as f:
        env_class = cloudpickle.load(f)

    # Set to zero as rollout workers != training workers - if > 0 will spin up
    # unnecessary additional workers.
    config["num_workers"] = 0
    config["env_config"] = args.env_config

    # Register custom environment with Ray
    register_env(env_class.env_name, lambda config: env_class(**config))

    trainer = get_trainer_class(args.algorithm)(env=env_class.env_name, config=config)
    trainer.restore(str(checkpoint_path))

    results = []

    for seed in args.seeds:
        # Create environment instance from config from results directory
        env = env_class(**config["env_config"])

        # Setting seed needs to come after trainer setup
        np.random.seed(seed)

        logger = Logger(args.metrics)

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
            logger.log(env)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            actions.append(step_actions)

        metrics = {k: np.array(v) for k, v in logger.to_dict().items()}

        trajectory = EpisodeTrajectory(
            observations,
            rewards,
            dones,
            infos,
            actions,
        )

        results.append((metrics, trajectory))

    return results
