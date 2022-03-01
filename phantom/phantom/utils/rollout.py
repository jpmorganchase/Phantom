import logging
import math
import multiprocessing
import os
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

import cloudpickle
import mercury as me
import numpy as np
import ray
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
from tqdm import tqdm

from ..env import PhantomEnv
from ..fsm import FiniteStateMachineEnv
from ..logging import Logger, Metric
from ..supertype import BaseSupertype
from .rollout_class import Rollout, Step
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


def rollout(
    directory: Union[str, Path],
    algorithm: str,
    num_workers: int = 0,
    num_repeats: int = 1,
    checkpoint: Optional[int] = None,
    env_class: Optional[Type[PhantomEnv]] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    env_supertype: Optional[BaseSupertype] = None,
    agent_supertypes: Optional[Mapping[me.ID, BaseSupertype]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    results_file: Optional[Union[str, Path]] = "results.pkl",
    save_trajectories: bool = False,
    save_messages: bool = False,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
) -> Union[List[Rollout], List[Any]]:
    """
    Performs rollouts for a previously trained Phantom experiment.

    Any objects that inherit from BaseRange in the env_supertype or agent_supertypes
    parameters will be expanded out into a multidimensional space of rollouts.

    For example, if two distinct UniformRanges are used, one with a length of 10 and one
    with a length of 5, 10 * 5 = 50 rollouts will be performed.

    If num_repeats is also given, say with a value of 2, then each of the 50 rollouts
    will be repeated twice, each time with a different random seed.

    Arguments:
        directory: Phantom results directory containing trained policies. If the default
            phantom results location is used (~/phantom-results), this part of the path
            can be ommited. If LATEST is given as the last element of the path, the
            parent directory will be scanned for the most recent run and this will be
            used.
        algorithm: RLlib algorithm to use.
        num_workers: Number of Ray rollout workers to initialise.
        num_repeats: Number of rollout repeats to perform, distributed over all workers.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
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
        save_trajectories: If True the full set of episode trajectories for each of the
            rollouts will be saved into the results file.
        save_messages: If True the full list of episode messages for each of the
            rollouts will be saved into the results file. Only applies if
            `save_trajectories` is also True.
        result_mapping_fn: If given, results from each rollout will be passed to this
            function, with the return values from the function aggregated and saved.
            This can be useful when wanting to cut down on the size of rollout results
            file when only a specific subset of fields are needed in further analysis.

    Returns:
        If result_mapping_fn is None, a list of Rollout objects. If result_mapping_fn is
        not None, a list of the outputs from the result_mapping_fn function.

    NOTE: It is the users responsibility to invoke rollouts via the provided ``phantom``
    command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
    the Python interpreter to run this code. Not setting this may lead to
    reproducibility issues.
    """
    show_pythonhashseed_warning()

    metrics = metrics or {}
    env_config = env_config or {}
    env_supertype = env_supertype or None
    agent_supertypes = agent_supertypes or {}

    if contains_type(env_config, BaseSampler):
        raise TypeError(
            "env_config should not contain instances of classes inheriting from BaseSampler"
        )

    if contains_type(env_config, BaseRange):
        raise TypeError(
            "env_config should not contain instances of classes inheriting from BaseRange"
        )

    if contains_type(env_supertype, BaseSampler):
        raise TypeError(
            "env_supertype should not contain instances of classes inheriting from BaseSampler"
        )

    if contains_type(agent_supertypes, BaseSampler):
        raise TypeError(
            "agent_supertypes should not contain instances of classes inheriting from BaseSampler"
        )

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

    # We find all instances of objects that inherit from BaseRange in the env supertype
    # and agent supertypes. We keep a track of where in this structure they came from
    # so we can easily replace the values at a later stage.
    # Each Range object can have multiple paths as it can exist at multiple points within
    # the data structure. Eg. shared across multiple agents.
    ranges = collect_instances_of_type_with_paths(
        BaseRange, ({}, env_supertype, agent_supertypes)
    )

    # This 'variations' list is where we build up every combination of the expanded values
    # from the list of Ranges.
    variations = [deepcopy(({}, env_supertype, agent_supertypes))]

    unamed_range_count = 0

    # For each iteration of this outer loop we expand another Range object.
    for range_obj, paths in reversed(ranges):
        values = range_obj.values()

        name = range_obj.name
        if name is None:
            name = f"range-{unamed_range_count}"
            unamed_range_count += 1

        variations2 = []
        for value in values:
            for variation in variations:
                variation = deepcopy(variation)
                variation[0][name] = value
                for path in paths:
                    update_val(variation, path, value)
                variations2.append(variation)

        variations = variations2

    # Apply the number of repeats requested to each 'variation'.
    rollout_configs = [
        _RolloutConfig(
            i * num_repeats + j,
            j,
            env_config,
            top_level_params,
            env_supertype,
            agent_supertypes,
        )
        for i, (top_level_params, env_supertype, agent_supertypes) in enumerate(
            variations
        )
        for j in range(num_repeats)
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
            save_messages,
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
                save_messages,
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
                    rollout.rollout_id,
                    rollout.repeat_id,
                    rollout.env_config,
                    rollout.top_level_params,
                    rollout.env_type,
                    rollout.agent_types,
                    rollout.steps if save_trajectories else None,
                    rollout.metrics,
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
    configs: List["_RolloutConfig"],
    env_class: Optional[Type[PhantomEnv]] = None,
    tracked_metrics: Optional[Mapping[str, Metric]] = None,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
    result_queue: Optional[multiprocessing.Queue] = None,
    save_messages: bool = False,
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

        if env_class is None:
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

            for agent_id, supertype in rollout_config.agent_supertypes.items():
                env.agents[agent_id].supertype = supertype

            if rollout_config.env_supertype is not None:
                env.env_type = rollout_config.env_supertype

                if "__ENV" in env.network.actor_ids:
                    env.network.actors["__ENV"].env_type = env.env_type

            if save_messages:
                env.network.resolver.enable_tracking = True

            # Setting seed needs to come after trainer setup
            np.random.seed(rollout_config.rollout_id)

            metric_logger = Logger(tracked_metrics)

            steps: List[Step] = []

            observation = env.reset()

            # Run rollout steps.
            for _ in range(env.clock.n_steps):
                step_actions = {}

                for agent_id, agent_obs in observation.items():
                    policy_id = config["multiagent"]["policy_mapping"][agent_id]

                    agent_action = trainer.compute_action(
                        agent_obs, policy_id=policy_id, explore=False
                    )

                    step_actions[agent_id] = agent_action

                new_observation, reward, done, info = env.step(step_actions)
                metric_logger.log(env)

                if save_messages:
                    messages = deepcopy(env.network.resolver.tracked_messages)
                    env.network.resolver.clear_tracked_messages()
                else:
                    messages = None

                steps.append(
                    Step(
                        observation,
                        reward,
                        done,
                        info,
                        step_actions,
                        messages,
                        env.previous_stage
                        if isinstance(env, FiniteStateMachineEnv)
                        else None,
                    )
                )

                observation = new_observation

            metrics = {k: np.array(v) for k, v in metric_logger.to_dict().items()}

            result = Rollout(
                rollout_config.rollout_id,
                rollout_config.repeat_id,
                rollout_config.env_config,
                rollout_config.top_level_params,
                rollout_config.env_supertype,
                rollout_config.agent_supertypes,
                steps,
                metrics,
            )

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


@dataclass
class _RolloutConfig:
    """
    Internal class
    """

    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    top_level_params: Dict[str, Any]
    env_supertype: Optional[BaseSupertype]
    agent_supertypes: Mapping[me.ID, BaseSupertype]
