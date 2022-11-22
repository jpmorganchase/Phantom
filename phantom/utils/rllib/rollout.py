import math
import os
import uuid
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import (
    Any,
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
import tensorflow as tf
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy import Policy as RLlibPolicy
from ray.rllib.utils.spaces import space_utils
from ray.util.queue import Queue
from tqdm import tqdm, trange

from ...env import PhantomEnv
from ...fsm import FiniteStateMachineEnv
from ...logging import Metric
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
from . import construct_results_paths


tf.compat.v1.disable_eager_execution()

CustomPolicyMapping = Mapping[AgentID, Type[Policy]]


class RolloutGenerator:
    """A class for generating episode rollouts with previously trained policies.

    Arguments:
        directory: Results directory containing trained policies. By default, this is
            located within `~/ray_results/`. If LATEST is given as the last element of
            the path, the parent directory will be scanned for the most recent run and
            this will be used.
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
        checkpoint: Checkpoint to use (defaults to most recent).
        metrics: Optional set of metrics to record and log.

    NOTE: It is the users responsibility to invoke rollouts via the provided ``phantom``
    command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
    the Python interpreter to run this code. Not setting this may lead to
    reproducibility issues.
    """

    def __init__(
        self,
        directory: Union[str, Path],
        env_class: Optional[Type[PhantomEnv]] = None,
        checkpoint: Optional[int] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
    ) -> None:
        show_pythonhashseed_warning()

        self.metrics = metrics or {}
        self.env_class = env_class

        self.directory, self.checkpoint_path = construct_results_paths(
            directory, checkpoint
        )

        RLlibPolicy.from_checkpoint(str(self.checkpoint_path))

        # Load config from results directory.
        with open(Path(self.directory, "params.pkl"), "rb") as params_file:
            self.config = cloudpickle.load(params_file)

    def rollout(
        self,
        env_config: Optional[Dict[str, Any]] = None,
        custom_policy_mapping: Optional[CustomPolicyMapping] = None,
        num_repeats: int = 1,
        num_workers: Optional[int] = None,
        record_messages: bool = False,
        show_progress_bar: bool = True,
    ) -> Generator[Rollout, None, None]:
        """Performs rollouts for a previously trained Phantom experiment.

        Any objects that inherit from the Range class in the env_config parameter will be
        expanded out into a multidimensional space of rollouts.

        For example, if two distinct UniformRanges are used, one with a length of 10 and one
        with a length of 5, 10 * 5 = 50 rollouts will be performed.

        If num_repeats is also given, say with a value of 2, then each of the 50 rollouts
        will be repeated twice, each time with a different random seed.

        Arguments:
            env_config: Configuration parameters to pass to the environment init method.
            custom_policy_mapping: Optionally replace agent policies with custom fixed
                policies.
            num_workers: Number of rollout worker processes to initialise
                (defaults to 'NUM CPU - 1').
            num_repeats: Number of rollout repeats to perform, distributed over all workers.
            record_messages: If True the full list of episode messages for each of the
                rollouts will be recorded. Only applies if `save_trajectories` is also True.
            show_progress_bar: If True shows a progress bar in the terminal output.

        Returns:
            A Generator of Rollouts.

        NOTE: It is the users responsibility to invoke rollouts via the provided ``phantom``
        command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
        the Python interpreter to run this code. Not setting this may lead to
        reproducibility issues.
        """
        assert num_repeats > 0
        assert num_workers > -1

        env_config = env_config or {}
        custom_policy_mapping = custom_policy_mapping or {}

        if contains_type(env_config, Sampler):
            raise TypeError(
                "env_config should not contain instances of classes inheriting from BaseSampler"
            )

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

        if self.env_class is None:
            with open(
                Path(self.directory, "phantom-training-params.pkl"), "rb"
            ) as params_file:
                self.env_class = cloudpickle.load(params_file)["env_class"]

        # Start the rollouts
        if num_workers_ == 0:
            # If num_workers is 0, run all the rollouts in this thread.

            rollouts = _rollout_task_fn(
                deepcopy(self.config),
                self.checkpoint_path,
                rollout_configs,
                self.env_class,
                custom_policy_mapping,
                self.metrics,
                record_messages,
            )

            if show_progress_bar:
                yield from tqdm(rollouts, total=len(rollout_configs))
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
                    deepcopy(self.config),
                    self.checkpoint_path,
                    rollout_configs[i : i + rollouts_per_worker],
                    self.env_class,
                    custom_policy_mapping,
                    self.metrics,
                    record_messages,
                )
                for i in range(0, len(rollout_configs), rollouts_per_worker)
            ]

            for payload in worker_payloads:
                remote_rollout_task_fn.remote(*payload)

            range_ = trange if show_progress_bar else range

            for _ in range_(len(rollout_configs)):
                yield q.get()


def _rollout_task_fn(
    config: Dict,
    checkpoint_path: Path,
    configs: List["_RolloutConfig"],
    env_class: Type[PhantomEnv],
    custom_policy_mapping: CustomPolicyMapping,
    tracked_metrics: Optional[Mapping[str, Metric]] = None,
    record_messages: bool = False,
) -> Generator[Rollout, None, None]:
    """Internal function"""
    config["env_config"] = configs[0].env_config

    with tf.compat.v1.variable_scope(str(uuid.uuid1())):
        policies = RLlibPolicy.from_checkpoint(str(checkpoint_path))

    for rollout_config in configs:
        # Create environment instance from config from results directory
        env = env_class(**rollout_config.env_config)

        if record_messages:
            env.network.resolver.enable_tracking = True

        # Setting seed needs to come after algo setup
        np.random.seed(rollout_config.rollout_id)

        metrics: DefaultDict[str, List[float]] = defaultdict(list)

        steps: List[Step] = []

        observation = env.reset()

        initted_policy_mapping = {
            agent_id: policy(
                env[agent_id].observation_space, env[agent_id].action_space
            )
            for agent_id, policy in custom_policy_mapping.items()
        }

        # Run rollout steps.
        for i in range(env.num_steps):
            step_actions = {}

            for agent_id, agent_obs in observation.items():
                if agent_id in initted_policy_mapping:
                    action = initted_policy_mapping[agent_id].compute_action(agent_obs)
                else:
                    policy_id = config["multiagent"]["policy_mapping_fn"](
                        agent_id, rollout_config.rollout_id, 0
                    )

                    obs_space = env[agent_id].observation_space
                    obs = get_preprocessor(obs_space)(obs_space).transform(agent_obs)

                    squashed_action = policies[policy_id].compute_single_action(
                        obs, policy_id=policy_id, explore=False, timestep=i
                    )

                    action = space_utils.unsquash_action(
                        squashed_action[0],
                        policies[policy_id].action_space_struct,
                    )

                step_actions[agent_id] = action

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

        yield Rollout(
            rollout_config.rollout_id,
            rollout_config.repeat_id,
            rollout_config.env_config,
            rollout_config.rollout_params,
            steps,
            condensed_metrics,
        )


@dataclass(frozen=True)
class _RolloutConfig:
    """Internal class"""

    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    rollout_params: Mapping[str, Any]
