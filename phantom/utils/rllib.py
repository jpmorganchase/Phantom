import logging
import math
import multiprocessing
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from inspect import isclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
)

import cloudpickle
import gym
import numpy as np
import ray
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.rllib.agents.registry import get_trainer_class
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune.registry import register_env
from tqdm import tqdm

from ..agents import Agent
from ..env import PhantomEnv
from ..fsm import FiniteStateMachineEnv
from ..logging import Metric
from ..policy import Policy
from ..types import AgentID
from .rollout import Rollout, Step
from . import (
    check_env_config,
    collect_instances_of_type_with_paths,
    contains_type,
    show_pythonhashseed_warning,
    update_val,
    Range,
    Sampler,
)


logger = logging.getLogger(__name__)


PolicyClass = Union[Type[Policy], Type[rllib.Policy]]

PolicyMapping = Mapping[
    str,
    Union[
        Type[Agent],
        List[AgentID],
        Tuple[PolicyClass, Type[Agent]],
        Tuple[PolicyClass, Type[Agent], Mapping[str, Any]],
        Tuple[PolicyClass, List[AgentID]],
        Tuple[PolicyClass, List[AgentID], Mapping[str, Any]],
    ],
]


def train(
    algorithm: str,
    env_class: Type[PhantomEnv],
    policies: PolicyMapping,
    policies_to_train: List[str],
    num_workers: Optional[int] = 0,
    env_config: Optional[Mapping[str, Any]] = None,
    rllib_config: Optional[Mapping[str, Any]] = None,
    tune_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
):
    """
    Performs training of a Phantom experiment using the RLlib library.

    Any objects that inherit from BaseSampler in the env_supertype or agent_supertypes
    parameters will be automatically sampled from and fed back into the environment at
    the start of each episode.

    Arguments:
        algorithm: RL algorithm to use (optional - one of 'algorithm' or 'trainer' must
            be provided).
        env_class: A PhantomEnv subclass.
        policies: A mapping of policy IDs to policy configurations.
        policies_to_train: A list of policy IDs that will be trained using RLlib.
        num_workers: Number of Ray workers to initialise (defaults to NUM CPU - 1).
        env_config: Configuration parameters to pass to the environment init method.
        rllib_config: Optional algorithm parameters dictionary to pass to RLlib.
        tune_config: Optional algorithm parameters dictionary to pass to Ray Tune.
        metrics: Optional set of metrics to record and log.

    Returns:
        The Ray Tune experiment results object.

    NOTE: It is the users responsibility to invoke training via the provided ``phantom``
    command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
    the Python interpreter to run this code. Not setting this may lead to
    reproducibility issues.
    """
    show_pythonhashseed_warning()

    env_config = env_config or {}
    rllib_config = rllib_config or {}
    tune_config = tune_config or {}
    metrics = metrics or {}

    check_env_config(env_config)

    env = env_class(**env_config)
    env.reset()

    policy_specs: Dict[str, rllib.policy.policy.PolicySpec] = {}
    policy_mapping: Dict[AgentID, str] = {}

    for policy_name, params in policies.items():
        policy_class = None
        config = None

        if isinstance(params, list):
            agent_ids = params

        elif isclass(params) and issubclass(params, Agent):
            agent_ids = list(env.network.get_agents_with_type(params).keys())

        elif isinstance(params, tuple):
            if len(params) == 2:
                policy_class, agent_ids = params
            else:
                policy_class, agent_ids, config = params

            if issubclass(policy_class, Policy):
                policy_class = make_rllib_wrapped_policy_class(policy_class)

            if isclass(agent_ids) and issubclass(agent_ids, Agent):
                agent_ids = list(env.network.get_agents_with_type(agent_ids).keys())

        else:
            raise TypeError(type(params))

        policy_specs[policy_name] = rllib.policy.policy.PolicySpec(
            policy_class=policy_class,
            action_space=env.agents[agent_ids[0]].action_space,
            observation_space=env.agents[agent_ids[0]].observation_space,
            config=config,
        )

        for agent_id in agent_ids:
            policy_mapping[agent_id] = policy_name

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return policy_mapping[agent_id]

    ray.tune.registry.register_env(
        env_class.__name__, lambda config: RLlibEnvWrapper(env_class(**config))
    )

    num_workers_ = num_workers or (len(os.sched_getaffinity(0)) - 1)

    config = {
        "env": env_class.__name__,
        "env_config": env_config,
        "num_sgd_iter": 10,
        "num_workers": num_workers_,
        "rollout_fragment_length": env.num_steps,
        "train_batch_size": env.num_steps * max(1, num_workers_),
        "multiagent": {
            "policies": policy_specs,
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": policies_to_train,
        },
    }

    config.update(rllib_config)

    if algorithm == "PPO":
        config["sgd_minibatch_size"] = max(int(config["train_batch_size"] / 10), 1)

    if metrics is not None:
        config["callbacks"] = RLlibMetricLogger(metrics)

    try:
        ray.init()
        results = ray.tune.run(algorithm, config=config, **tune_config)
    except Exception as exception:
        # Ensure that Ray is properly shutdown in the instance of an error occuring
        ray.shutdown()
        raise exception
    else:
        ray.shutdown()
        return results


class RLlibMetricLogger(DefaultCallbacks):
    """
    RLlib callback that logs Phantom metrics.
    """

    def __init__(self, metrics: Mapping[str, "Metric"]) -> None:
        super().__init__()
        self.metrics = metrics

    def on_episode_start(self, *, episode, **kwargs) -> None:
        for metric_id in self.metrics.keys():
            episode.user_data[metric_id] = []

    def on_episode_step(self, *, base_env, episode, **kwargs) -> None:
        env = base_env.envs[0]

        for (metric_id, metric) in self.metrics.items():
            episode.user_data[metric_id].append(metric.extract(env))

    def on_episode_end(self, *, episode, **kwargs) -> None:
        for (metric_id, metric) in self.metrics.items():
            episode.custom_metrics[metric_id] = metric.reduce(
                episode.user_data[metric_id]
            )

    def __call__(self) -> "RLlibMetricLogger":
        return self


class RLlibEnvWrapper(rllib.MultiAgentEnv):
    """
    Wrapper around a :class:`PhantomEnv` that provides compatibility with the RLlib
    ``MultiAgentEnv`` interface.
    """

    def __init__(self, env: PhantomEnv) -> None:
        self.env = env

        self._agent_ids = set(
            agent.id
            for agent in self.env.network.agents.values()
            if agent.action_space is not None and agent.observation_space is not None
        )

        self.action_space = gym.spaces.Dict(
            {
                agent_id: env.agents[agent_id].action_space
                for agent_id in self._agent_ids
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                agent_id: env.agents[agent_id].observation_space
                for agent_id in self._agent_ids
            }
        )

        rllib.MultiAgentEnv.__init__(self)

    def step(self, action_dict: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        return self.env.step(action_dict)

    def reset(self) -> Dict[AgentID, Any]:
        return self.env.reset()

    def is_done(self) -> bool:
        return self.env.is_done()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def __getitem__(self, agent_id: AgentID) -> AgentID:
        return self.env.__getitem__(agent_id)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"


def make_rllib_wrapped_policy_class(policy_class: Type[Policy]) -> Type[rllib.Policy]:
    class RLlibPolicyWrapper(rllib.Policy):
        # NOTE:
        # If the action space is larger than -1.0 < x < 1.0, RLlib will attempt to
        # 'unsquash' the values leading to potentially unintended results.
        # (https://github.com/ray-project/ray/pull/16531)

        def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            config: Mapping[str, Any],
        ):
            self.policy = policy_class(observation_space, action_space, **config)

            super().__init__(observation_space, action_space, config)

        def get_weights(self):
            return None

        def set_weights(self, weights):
            pass

        def learn_on_batch(self, samples):
            return {}

        def compute_single_action(
            self,
            obs: Optional[TensorStructType] = None,
            state: Optional[List[TensorType]] = None,
            *,
            prev_action: Optional[TensorStructType] = None,
            prev_reward: Optional[TensorStructType] = None,
            info: dict = None,
            input_dict: Optional[SampleBatch] = None,
            episode: Optional[Episode] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            # Kwargs placeholder for future compatibility.
            **kwargs,
        ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
            return self.policy.compute_action(obs), [], {}

        def compute_actions(
            self,
            obs_batch: Union[List[TensorStructType], TensorStructType],
            state_batches: Optional[List[TensorType]] = None,
            prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
            prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
            info_batch: Optional[Dict[str, list]] = None,
            episodes: Optional[List[MultiAgentEpisode]] = None,
            explore: Optional[bool] = None,
            timestep: Optional[int] = None,
            # Kwargs placeholder for future compatibility.
            **kwargs,
        ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
            # Workaround due to known issue in RLlib
            # https://github.com/ray-project/ray/issues/10009
            if isinstance(self.action_space, gym.spaces.Tuple):
                unbatched = [self.policy.compute_action(obs) for obs in obs_batch]

                actions = tuple(
                    np.array([unbatched[j][i] for j in range(len(unbatched))])
                    for i in range(len(unbatched[0]))
                )
            else:
                actions = [self.policy.compute_action(obs) for obs in obs_batch]

            return (actions, [], {})

    return RLlibPolicyWrapper


def rollout(
    directory: Union[str, Path],
    algorithm: str,
    env_class: Type[PhantomEnv],
    env_config: Optional[Dict[str, Any]] = None,
    num_workers: int = 0,
    num_repeats: int = 1,
    checkpoint: Optional[int] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    record_messages: bool = False,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
) -> Union[List[Rollout], List[Any]]:
    """
    Performs rollouts for a previously trained Phantom experiment.

    Any objects that inherit from the Range class in the env_config parameter will be
    expanded out into a multidimensional space of rollouts.

    For example, if two distinct UniformRanges are used, one with a length of 10 and one
    with a length of 5, 10 * 5 = 50 rollouts will be performed.

    If num_repeats is also given, say with a value of 2, then each of the 50 rollouts
    will be repeated twice, each time with a different random seed.

    Arguments:
        directory: Results directory containing trained policies. By default, this is
            located within `~/ray_results/`. If LATEST is given as the last element of
            the path, the parent directory will be scanned for the most recent run and
            this will be used.
        algorithm: RLlib algorithm to use.
        num_workers: Number of Ray rollout workers to initialise.
        num_repeats: Number of rollout repeats to perform, distributed over all workers.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
        env_config: Configuration parameters to pass to the environment init method.
        metrics: Optional set of metrics to record and log.
        record_messages: If True the full list of episode messages for each of the
            rollouts will be recorded. Only applies if `save_trajectories` is also True.
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

    if contains_type(env_config, Sampler):
        raise TypeError(
            "env_config should not contain instances of classes inheriting from BaseSampler"
        )

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

        logger.info("Trying to find latest experiment results in '%s'", parent_dir)

        directory = _find_most_recent_results_dir(parent_dir)

        logger.info("Found experiment results: '%s'", directory.stem)
    else:
        directory = Path(os.path.expanduser(directory))

        if not directory.exists():
            directory = Path(ray_dir, directory)

            if not directory.exists():
                raise FileNotFoundError(
                    f"Results directory '{directory}' does not exist"
                )

        logger.info("Using results directory: '%s'", directory)

    # If an explicit checkpoint is not given, find all checkpoints and use the newest.
    if checkpoint is None:
        checkpoint = _get_checkpoints(directory)[-1]

        logger.info("Using most recent checkpoint: %s", checkpoint)
    else:
        logger.info("Using checkpoint: %s", checkpoint)

    # We find all instances of objects that inherit from BaseRange in the env supertype
    # and agent supertypes. We keep a track of where in this structure they came from
    # so we can easily replace the values at a later stage.
    # Each Range object can have multiple paths as it can exist at multiple points within
    # the data structure. Eg. shared across multiple agents.
    ranges = collect_instances_of_type_with_paths(Range, ({}, env_config))

    # This 'variations' list is where we build up every combination of the expanded
    # values from the list of Ranges.
    variations: List[List[Dict[str, Any]]] = [deepcopy([{}, env_config])]

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
            rollout_params,
        )
        for i, (rollout_params, env_config) in enumerate(variations)
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
            record_messages,
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
                record_messages,
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

    return results


def _find_most_recent_results_dir(base_path: Union[Path, str]) -> Path:
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


def _get_checkpoints(results_dir: Union[Path, str]) -> List[int]:
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


def _rollout_task_fn(
    directory: Path,
    checkpoint: int,
    algorithm: str,
    configs: List["_RolloutConfig"],
    env_class: Type[PhantomEnv],
    tracked_metrics: Optional[Mapping[str, Metric]] = None,
    result_mapping_fn: Optional[Callable[[Rollout], Any]] = None,
    result_queue: Optional[multiprocessing.Queue] = None,
    record_messages: bool = False,
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

        # Set to zero as rollout workers != training workers - if > 0 will spin up
        # unnecessary additional workers.
        config["num_workers"] = 0
        config["env_config"] = configs[0].env_config

        # Register custom environment with Ray
        register_env(
            env_class.__name__, lambda config: RLlibEnvWrapper(env_class(**config))
        )

        trainer = get_trainer_class(algorithm)(env=env_class.__name__, config=config)
        trainer.restore(str(checkpoint_path))

        results = []

        iter_obj = tqdm(configs) if result_queue is None else configs

        for rollout_config in iter_obj:
            # Create environment instance from config from results directory
            env = env_class(**rollout_config.env_config)

            if record_messages:
                env.network.resolver.enable_tracking = True

            # Setting seed needs to come after trainer setup
            np.random.seed(rollout_config.rollout_id)

            metrics: DefaultDict[str, List[float]] = defaultdict(list)

            steps: List[Step] = []

            observation = env.reset()

            # Run rollout steps.
            for i in range(env.num_steps):
                step_actions = {}

                for agent_id, agent_obs in observation.items():
                    policy_id = config["multiagent"]["policy_mapping_fn"](
                        agent_id, rollout_config.rollout_id, 0
                    )

                    agent_action = trainer.compute_single_action(
                        agent_obs, policy_id=policy_id, explore=False
                    )

                    step_actions[agent_id] = agent_action

                new_observation, reward, done, info = env.step(step_actions)

                if tracked_metrics is not None:
                    for name, metric in tracked_metrics.items():
                        metrics[name].append(metric.extract(env))

                if record_messages:
                    messages = deepcopy(env.network.resolver.tracked_messages)
                    env.network.resolver.clear_tracked_messages()
                else:
                    messages = None

                steps.append(
                    Step(
                        i,
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

            condensed_metrics = {k: np.array(v) for k, v in metrics.items()}

            result = Rollout(
                rollout_config.rollout_id,
                rollout_config.repeat_id,
                rollout_config.env_config,
                rollout_config.rollout_params,
                steps,
                condensed_metrics,
            )

            if result_mapping_fn is not None:
                result = result_mapping_fn(result)

            # If in multiprocess mode, add the results to the queue, otherwise store
            # locally until all rollouts for this function call are complete.
            if result_queue is None:
                results.append(result)
            else:
                result_queue.put(result)

    except Exception as exception:
        ray.shutdown()
        raise exception

    else:
        ray.shutdown()

        # If using multi-processing this will be an empty list
        return results


@dataclass(frozen=True)
class _RolloutConfig:
    """
    Internal class
    """

    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    rollout_params: Mapping[str, Any]
