import logging
import math
import os
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain
from pathlib import Path
from typing import *

import cloudpickle
import mercury as me
import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

from ..logging import Logger, Metric
from ..type import BaseType
from .ranges import BaseRange
from .samplers import BaseSampler
from . import (
    collect_instances_of_type_with_paths,
    contains_type,
    find_most_recent_results_dir,
    show_pythonhashseed_warning,
    update_val,
)


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
    env_supertype: Optional[BaseType]
    agent_supertypes: Dict[me.ID, BaseType]


def rollout(
    directory: Union[str, Path],
    algorithm: str,
    num_workers: int = 0,
    num_repeats: int = 1,
    checkpoint: Optional[int] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    env_supertype: Optional[Mapping[str, Any]] = None,
    agent_supertypes: Optional[Mapping[me.ID, BaseType]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    results_file: Optional[Union[str, Path]] = "results.pkl",
    save_trajectories: bool = False,
) -> Tuple[List[RolloutConfig], List[Dict[str, np.ndarray]], List[EpisodeTrajectory]]:
    """
    Performs rollouts for a previously trained Phantom experiment.

    Any objects that inherit from BaseRange in the env_supertype or agent_supertypes
    parameters will be expanded out into a multidimensional space of rollouts.

    For example, if two distinct UniformRanges are used, one with a length of 10 and one
    with a length of 5, 10 * 5 = 50 rollouts will be performed.

    If num_repeats is also given, say with a value of 2, then each of the 50 rollouts
    will be repeated twice, each time with a different random seed.

    Attributes:
        directory: Phantom results directory containing trained policies.
        algorithm: RLlib algorithm to use.
        num_workers: Number of Ray rollout workers to initialise.
        num_repeats: Number of rollout repeats to perform, distributed over all workers.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_config: Configuration parameters to pass to the environment init method.
        env_supertype: Type object for the environment. Any contained objects that
            inherit from BaseRange will be sampled from and automatically applied to
            the environment (and environment actor).
        agent_supertypes: Mapping of agent IDs to Type objects for the respective agent.
            Any contained objects that inherit from BaseRange will be sampled from and
            automatically applied to the agent.
        metrics: Optional set of metrics to record and log.
        results_file: Name of the results file to save to, if None is given no file
            will be saved (default is "results.pkl").
        save_trajectories: If True the full set of epsiode trajectories for the
            rollouts will be saved into the results file.

    Returns:
        - A list of RolloutConfig objects describing the exact configuration of each rollout.
        - A list of dictionaries containing recorded metrics for each rollout.
        - A list of EpisodeTrajectory objects.

    NOTE: It is the users responsibility to invoke training via the provided ``phantom``
    command or ensure the PYTHONHASHSEED environment variable is set before starting the
    Python interpreter to run this code. Not setting this may lead to reproducibility
    issues.
    """
    show_pythonhashseed_warning()

    metrics = metrics or {}
    env_config = env_config or {}
    env_supertype = env_supertype or None
    agent_supertypes = agent_supertypes or {}

    if contains_type(env_config, BaseSampler):
        raise Exception(
            "env_config should not contain instances of classes inheriting from BaseSampler"
        )

    if contains_type(env_config, BaseRange):
        raise Exception(
            "env_config should not contain instances of classes inheriting from BaseRange"
        )

    if contains_type(env_supertype, BaseSampler):
        raise Exception(
            "env_supertype should not contain instances of classes inheriting from BaseSampler"
        )

    if contains_type(agent_supertypes, BaseSampler):
        raise Exception(
            "agent_supertypes should not contain instances of classes inheriting from BaseSampler"
        )

    phantom_dir = os.path.expanduser("~/phantom-results")

    directory = Path(directory)

    if directory.stem == "LATEST":
        parent_dir = Path(os.path.expanduser(directory.parent))

        if not parent_dir.exists():
            parent_dir = Path(phantom_dir, parent_dir)

            if not parent_dir.exists():
                logger.error(f"Base results directory '{parent_dir}' does not exist")
                return

        logger.info(f"Trying to find latest experiment results in '{parent_dir}'")

        directory = find_most_recent_results_dir(parent_dir)

        logger.info(f"Found experiment results: '{directory.stem}'")
    else:
        directory = Path(os.path.expanduser(directory))

        if not directory.exists():
            directory = Path(phantom_dir, directory)

            if not directory.exists():
                logger.error(f"Results directory '{directory}' does not exist")
                return

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

    # We find all instances of objects that inherit from BaseRange in the env supertype
    # and agent supertypes. We keep a track of where in this structure they came from
    # so we can easily replace the values at a later stage.
    # Each Range object can have multiple paths as it can exist at multiple points within
    # the data structure. Eg. shared across multiple agents.
    ranges = collect_instances_of_type_with_paths(
        BaseRange, (env_supertype, agent_supertypes)
    )

    # This 'variations' list is where we build up every combination of the expanded values
    # from the list of Ranges.
    variations = [deepcopy((env_supertype, agent_supertypes))]

    # For each iteration of this outer loop we expand another Range object.
    for range_obj, paths in reversed(ranges):
        variations2 = []
        for value in range_obj.values():
            for variation in variations:
                variation = deepcopy(variation)
                for path in paths:
                    update_val(variation, path, value)
                variations2.append(variation)

        variations = variations2

    # Apply the number of repeats requested to each 'variation'.
    rollouts = [
        RolloutConfig(
            i * num_repeats + j,
            env_config,
            env_supertype,
            agent_supertypes,
        )
        for i, (env_supertype, agent_supertypes) in enumerate(variations)
        for j in range(num_repeats)
    ]

    # Distribute the rollouts evenly amongst the number of workers.
    rollouts_per_worker = int(math.ceil(len(rollouts) / num_workers))

    worker_payloads = [
        ParallelFunctionArgs(
            directory,
            checkpoint,
            metrics,
            algorithm,
            rollouts[i : i + rollouts_per_worker],
        )
        for i in range(0, len(rollouts), rollouts_per_worker)
    ]

    logger.info(f"Starting {len(rollouts)} rollout(s) using {num_workers} worker(s)")

    # Start the rollouts
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

    # Collect the results
    metrics_results, trajectories = zip(*results)

    results = {}

    if metrics != {}:
        results["metrics"] = metrics_results

    if save_trajectories:
        results["trajectories"] = trajectories

    # Optionally save the results
    if results_file is not None and results != {}:
        results_file = Path(directory, results_file)

        cloudpickle.dump(results, open(results_file, "wb"))

        logger.info(f"Saved rollout results to '{results_file}'")

    return rollouts, metrics_results, trajectories


@dataclass
class ParallelFunctionArgs:
    """
    Internal dataclass
    """

    directory: Path
    checkpoint: int
    metrics: Optional[Mapping[str, Metric]]
    algorithm: str
    configs: List[RolloutConfig]


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
    config["env_config"] = args.configs[0].env_config

    # Register custom environment with Ray
    register_env(
        env_class.env_name, lambda config: env_class(**args.configs[0].env_config)
    )

    trainer = get_trainer_class(args.algorithm)(env=env_class.env_name, config=config)
    trainer.restore(str(checkpoint_path))

    results = []

    for rollout_config in args.configs:
        # Create environment instance from config from results directory
        env = env_class(**rollout_config.env_config)

        for agent_id, supertype in rollout_config.agent_supertypes.items():
            env.agents[agent_id].supertype = supertype

        if rollout_config.env_supertype is not None:
            env.env_type = rollout_config.env_supertype

            if "__ENV" in env.network.actor_ids:
                env.network.actors["__ENV"].env_type = env.env_type

        # Setting seed needs to come after trainer setup
        np.random.seed(rollout_config.seed)

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
