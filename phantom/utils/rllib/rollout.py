import logging
import math
import os
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
    Callable,
    DefaultDict,
    Dict,
    Generator,
    List,
    Mapping,
    Optional,
    Type,
    Union,
)

import cloudpickle
import numpy as np
import ray
import rich.progress
from ray.rllib.policy import Policy as RLlibPolicy
from ray.util.queue import Queue

from ...env import PhantomEnv
from ...fsm import FiniteStateMachineEnv
from ...metrics import Metric, logging_helper
from ...policy import Policy
from ...types import AgentID
from ..rollout import Rollout, Step
from .. import (
    collect_instances_of_type_with_paths,
    contains_type,
    show_pythonhashseed_warning,
    update_val,
    Range,
    Sampler,
)
from .wrapper import RLlibEnvWrapper
from . import construct_results_paths


logger = logging.getLogger(__name__)

CustomPolicyMapping = Mapping[AgentID, Type[Policy]]


def rollout(
    directory: Union[str, Path],
    env_class: Optional[Type[PhantomEnv]] = None,
    env_config: Optional[Dict[str, Any]] = None,
    custom_policy_mapping: Optional[CustomPolicyMapping] = None,
    num_repeats: int = 1,
    num_workers: Optional[int] = None,
    checkpoint: Optional[int] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    record_messages: bool = False,
    show_progress_bar: bool = True,
    vectorized_env_batch_size: int = 1,
) -> Generator[Rollout, None, None]:
    """Performs rollouts for a previously trained Phantom experiment.

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
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
        env_config: Configuration parameters to pass to the environment init method.
        custom_policy_mapping: Optionally replace agent policies with custom fixed
            policies.
        num_workers: Number of rollout worker processes to initialise
            (defaults to 'NUM CPU - 1').
        num_repeats: Number of rollout repeats to perform, distributed over all workers.
        checkpoint: Checkpoint to use (defaults to most recent).
        metrics: Optional set of metrics to record and log.
        record_messages: If True the full list of episode messages for each of the
            rollouts will be recorded. Only applies if `save_trajectories` is also True.
        show_progress_bar: If True shows a progress bar in the terminal output.
        vectorized_env_batch_size: TODO

    Returns:
        A Generator of Rollouts.

    .. note::
        It is the users responsibility to invoke rollouts via the provided ``phantom``
        command or ensure the ``PYTHONHASHSEED`` environment variable is set before
        starting the Python interpreter to run this code. Not setting this may lead to
        reproducibility issues.
    """
    assert num_repeats > 0, "num_repeats must be at least 1"

    assert vectorized_env_batch_size > 0, "vectorized_env_batch_size must be at least 1"

    if vectorized_env_batch_size > 1 and issubclass(env_class, FiniteStateMachineEnv):
        raise ValueError("Cannot use FSM env when vectorized_env_batch_size > 1")

    if num_workers is not None:
        assert num_workers >= 0, "num_workers must be at least 0"

    show_pythonhashseed_warning()

    metrics = metrics or {}
    env_config = env_config or {}
    custom_policy_mapping = custom_policy_mapping or {}

    if contains_type(env_config, Sampler):
        raise TypeError(
            "env_config should not contain instances of classes inheriting from BaseSampler"
        )

    directory, checkpoint_path = construct_results_paths(directory, checkpoint)

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

    num_workers_ = (os.cpu_count() - 1) if num_workers is None else num_workers

    logger.info(
        "Starting %s rollout(s) using %s worker process(es)",
        len(rollout_configs),
        num_workers_,
    )

    # Load configs from results directory.
    with open(Path(directory, "params.pkl"), "rb") as params_file:
        config = cloudpickle.load(params_file)

    policy_mapping_fn = config["multiagent"]["policy_mapping_fn"]

    with open(Path(directory, "phantom-training-params.pkl"), "rb") as params_file:
        ph_config = cloudpickle.load(params_file)

    if env_class is None:
        env_class = ph_config["env_class"]

    # Start the rollouts
    if num_workers_ == 0:
        # If num_workers is 0, run all the rollouts in this thread.

        rollouts = _rollout_task_fn(
            policy_mapping_fn,
            checkpoint_path,
            rollout_configs,
            env_class,
            custom_policy_mapping,
            vectorized_env_batch_size,
            metrics,
            record_messages,
        )

        if show_progress_bar:
            yield from rich.progress.track(rollouts, total=len(rollout_configs))
        else:
            yield from rollouts

    else:
        q = Queue()

        # Distribute the rollouts evenly amongst the number of workers.
        rollouts_per_worker = int(
            math.ceil(len(rollout_configs) / max(num_workers_, 1))
        )

        @ray.remote
        def remote_rollout_task_fn(*args):
            for x in _rollout_task_fn(*args):
                q.put(x)

        worker_payloads = [
            (
                policy_mapping_fn,
                checkpoint_path,
                rollout_configs[i : i + rollouts_per_worker],
                env_class,
                custom_policy_mapping,
                vectorized_env_batch_size,
                metrics,
                record_messages,
            )
            for i in range(0, len(rollout_configs), rollouts_per_worker)
        ]

        for payload in worker_payloads:
            remote_rollout_task_fn.remote(*payload)

        range_iter = range(len(rollout_configs))

        if show_progress_bar:
            range_iter = rich.progress.track(range_iter)

        for _ in range_iter:
            yield q.get()


def _rollout_task_fn(
    policy_mapping_fn: Callable[[], str],
    checkpoint_path: Path,
    all_configs: List["_RolloutConfig"],
    env_class: Type[PhantomEnv],
    custom_policy_mapping: CustomPolicyMapping,
    vectorized_env_batch_size: int,
    metric_objects: Optional[Mapping[str, Metric]] = None,
    record_messages: bool = False,
) -> Generator[Rollout, None, None]:
    """Internal function"""

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    # Lazily load checkpointed policy objects
    saved_policies: Dict[str, RLlibPolicy] = {}

    # Setting seed needs to come after algo setup
    np.random.seed(all_configs[0].rollout_id)

    for configs in chunker(all_configs, vectorized_env_batch_size):
        batch_size = len(configs)

        vec_envs = [
            env_class(**rollout_config.env_config) for rollout_config in configs
        ]

        if record_messages:
            for env in vec_envs:
                env.network.resolver.enable_tracking = True

        vec_metrics = [defaultdict(list) for _ in range(batch_size)]
        vec_all_steps = [[] for _ in range(batch_size)]

        vec_observations = [env.reset() for env in vec_envs]

        initted_policy_mapping = {
            agent_id: policy(
                vec_envs[0][agent_id].observation_space,
                vec_envs[0][agent_id].action_space,
            )
            for agent_id, policy in custom_policy_mapping.items()
        }

        # Run rollout steps.
        for i in range(vec_envs[0].num_steps):
            actions = {}

            dict_observations = {
                k: [dic[k] for dic in vec_observations] for k in vec_observations[0]
            }

            for agent_id, vec_agent_obs in dict_observations.items():
                if agent_id in initted_policy_mapping:
                    actions[agent_id] = [
                        initted_policy_mapping[agent_id].compute_action(agent_obs)
                        for agent_obs in vec_agent_obs
                    ]
                else:
                    policy_id = policy_mapping_fn(agent_id, 0, 0)

                    if policy_id not in saved_policies:
                        saved_policies[policy_id] = RLlibPolicy.from_checkpoint(
                            checkpoint_path / "policies" / policy_id
                        )

                    actions[agent_id] = saved_policies[policy_id].compute_actions(
                        vec_agent_obs, explore=False
                    )[0]

            # hack for no agent acting step in Ops
            if len(dict_observations) == 0:
                vec_actions = [{}] * batch_size
            else:
                vec_actions = [dict(zip(actions, t)) for t in zip(*actions.values())]

            vec_steps = [
                env.step(actions) for env, actions in zip(vec_envs, vec_actions)
            ]

            for j in range(batch_size):
                if metric_objects is not None:
                    logging_helper(vec_envs[j], metric_objects, vec_metrics[j])

                if record_messages:
                    messages = deepcopy(env.network.resolver.tracked_messages)
                    env.network.resolver.clear_tracked_messages()
                else:
                    messages = None

                vec_all_steps[j].append(
                    Step(
                        i,
                        vec_observations[j],
                        vec_steps[j].rewards,
                        vec_steps[j].dones,
                        vec_steps[j].infos,
                        vec_actions[j],
                        messages,
                        vec_envs[j].previous_stage
                        if isinstance(vec_envs[j], FiniteStateMachineEnv)
                        else None,
                    )
                )

            vec_observations = [step.observations for step in vec_steps]

        for j in range(batch_size):
            reduced_metrics = {
                metric_id: metric_objects[metric_id].reduce(
                    vec_metrics[j][metric_id], "evaluate"
                )
                for metric_id in metric_objects
            }

            yield Rollout(
                configs[j].rollout_id,
                configs[j].repeat_id,
                configs[j].env_config,
                configs[j].rollout_params,
                vec_all_steps[j],
                reduced_metrics,
            )


@dataclass(frozen=True)
class _RolloutConfig:
    """Internal class"""

    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    rollout_params: Mapping[str, Any]
