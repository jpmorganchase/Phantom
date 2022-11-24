import logging
import os
from inspect import isclass
from pathlib import Path
from typing import (
    Any,
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
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType
from ray.tune.logger import LoggerCallback

from ...agents import Agent
from ...env import PhantomEnv
from ...metrics import Metric
from ...policy import Policy
from ...types import AgentID
from .. import check_env_config, show_pythonhashseed_warning
from .wrapper import RLlibEnvWrapper


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
    num_workers: Optional[int] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    rllib_config: Optional[Mapping[str, Any]] = None,
    tune_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    local_mode: bool = False,
):
    """Performs training of a Phantom experiment using the RLlib library.

    Any objects that inherit from BaseSampler in the env_supertype or agent_supertypes
    parameters will be automatically sampled from and fed back into the environment at
    the start of each episode.

    Arguments:
        algorithm: RL algorithm to use (optional - one of 'algorithm' or 'trainer' must
            be provided).
        env_class: A PhantomEnv subclass.
        policies: A mapping of policy IDs to policy configurations.
        policies_to_train: A list of policy IDs that will be trained using RLlib.
        num_workers: Number of Ray workers to initialise (defaults to 'NUM CPU - 1').
        env_config: Configuration parameters to pass to the environment init method.
        rllib_config: Optional algorithm parameters dictionary to pass to RLlib.
        tune_config: Optional algorithm parameters dictionary to pass to Ray Tune.
        metrics: Optional set of metrics to record and log.
        local_mode: Use RLlib's local mode option for training (default is False).

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

    for policy_id in policies_to_train:
        if policy_id not in policy_specs:
            raise ValueError(
                f"Policy to train '{policy_id}' is not in the list of defined policies"
            )

    def policy_mapping_fn(agent_id, *args, **kwargs):
        return policy_mapping[agent_id]

    ray.tune.registry.register_env(
        env_class.__name__, lambda config: RLlibEnvWrapper(env_class(**config))
    )

    num_workers_ = (os.cpu_count() - 1) if num_workers is None else num_workers

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

    if "callbacks" not in tune_config:
        tune_config["callbacks"] = []

    tune_config["callbacks"].append(
        RLlibTrainingStartCallback(
            {
                "algorithm": algorithm,
                "env_class": env_class,
                "policy_specs": policy_specs,
                "policy_mapping": policy_mapping,
                "policies_to_train": policies_to_train,
                "env_config": env_config,
                "rllib_config": rllib_config,
                "tune_config": tune_config,
                "metrics": metrics,
            }
        )
    )

    if metrics is not None:
        config["callbacks"] = RLlibMetricLogger(metrics)

    try:
        ray.init(local_mode=local_mode)
        results = ray.tune.run(algorithm, config=config, **tune_config)
    except Exception as exception:
        # Ensure that Ray is properly shutdown in the instance of an error occuring
        ray.shutdown()
        raise exception
    else:
        ray.shutdown()
        return results


class RLlibMetricLogger(DefaultCallbacks):
    """RLlib callback that logs Phantom metrics."""

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


class RLlibTrainingStartCallback(LoggerCallback):
    """Saves training parameters to the results directory at the start of training."""

    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.config = config

    def on_trial_start(
        self,
        iteration: int,
        trials: List[ray.tune.tune.Trial],
        trial: ray.tune.tune.Trial,
        **info: Any,
    ) -> None:
        cloudpickle.dump(
            self.config, open(Path(trial.logdir, "phantom-training-params.pkl"), "wb")
        )

    def __call__(self) -> "RLlibTrainingStartCallback":
        return self


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
            **kwargs,
        ):
            self.policy = policy_class(observation_space, action_space, **kwargs)

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
