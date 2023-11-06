import os
import tempfile
from datetime import datetime
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
import gymnasium as gym
import numpy as np
import ray
import rich.pretty
from ray import rllib
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType

from ...agents import Agent
from ...env import PhantomEnv
from ...metrics import Metric, logging_helper
from ...policy import Policy
from ...types import AgentID
from .. import check_env_config, rich_progress, show_pythonhashseed_warning
from .wrapper import RLlibEnvWrapper


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
    iterations: int,
    checkpoint_freq: Optional[int] = None,
    num_workers: Optional[int] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    rllib_config: Optional[Mapping[str, Any]] = None,
    ray_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    results_dir: str = ray.tune.result.DEFAULT_RESULTS_DIR,
    show_training_metrics: bool = False,
) -> Algorithm:
    """Performs training of a Phantom experiment using the RLlib library.

    Any objects that inherit from BaseSampler in the env_supertype or agent_supertypes
    parameters will be automatically sampled from and fed back into the environment at
    the start of each episode.

    Arguments:
        algorithm: RL algorithm to use (optional - one of 'algorithm' or 'trainer' must
            be provided).
        env_class: A PhantomEnv subclass.
        policies: A mapping of policy IDs to policy configurations.
        iterations: Number of training iterations to perform.
        checkpoint_freq: The iteration frequency to save policy checkpoints at.
        num_workers: Number of Ray rollout workers to use (defaults to 'NUM CPU - 1').
        env_config: Configuration parameters to pass to the environment init method.
        rllib_config: Optional algorithm parameters dictionary to pass to RLlib.
        ray_config: Optional algorithm parameters dictionary to pass to ``ray.init()``.
        metrics: Optional set of metrics to record and log.
        results_dir: A custom results directory, default is ~/ray_results/
        show_training_metrics: Set to True to print training metrics every iteration.

    The ``policies`` parameter defines which agents will use which policy. This is key
    to performing shared policy learning. The function expects a mapping of
    ``{<policy_id> : <policy_setup>}``. The policy setup values can take one of the
    following forms:

    *   ``Type[Agent]``: All agents that are an instance of this class will learn the
        same RLlib policy.
    *   ``List[AgentID]``: All agents that have IDs in this list will learn the same
        RLlib policy.
    *   ``Tuple[PolicyClass, Type[Agent]]``: All agents that are an instance of this
        class will use the same fixed/learnt policy.
    *   ``Tuple[PolicyClass, Type[Agent], Mapping[str, Any]]``: All agents that are an
        instance of this class will use the same fixed/learnt policy configured with the
        given options.
    *   ``Tuple[PolicyClass, List[AgentID]]``: All agents that have IDs in this list use
        the same fixed/learnt policy.
    *   ``Tuple[PolicyClass, List[AgentID], Mapping[str, Any]]``: All agents that have
        IDs in this list use the same fixed/learnt policy configured with the given
        options.

    Returns:
        The Ray Tune experiment results object.

    .. note::
        It is the users responsibility to invoke training via the provided ``phantom``
        command or ensure the ``PYTHONHASHSEED`` environment variable is set before
        starting the Python interpreter to run this code. Not setting this may lead to
        reproducibility issues.
    """
    show_pythonhashseed_warning()

    iterations = int(iterations)

    assert iterations > 0, "'iterations' parameter must be > 0"

    if num_workers is not None:
        assert num_workers >= 0, "'num_workers' parameter must be >= 0"

    env_config = env_config or {}
    rllib_config = rllib_config or {}
    metrics = metrics or {}

    check_env_config(env_config)

    ray.init(ignore_reinit_error=True, **(ray_config or {}))

    env = env_class(**env_config)
    env.reset()

    policy_specs: Dict[str, rllib.policy.policy.PolicySpec] = {}
    policy_mapping: Dict[AgentID, str] = {}
    policies_to_train: List[str] = []

    for policy_name, params in policies.items():
        policy_class = None
        config = None

        if isinstance(params, list):
            agent_ids = params
            policies_to_train.append(policy_name)

        elif isclass(params) and issubclass(params, Agent):
            agent_ids = list(env.network.get_agents_with_type(params).keys())
            policies_to_train.append(policy_name)

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
        if agent_id.startswith("__stacked__"):
            agent_id = agent_id[11:].split("__", 1)[1]

        return policy_mapping[agent_id]

    ray.tune.registry.register_env(
        env_class.__name__, lambda config: RLlibEnvWrapper(env_class(**config))
    )

    num_workers_ = (os.cpu_count() - 1) if num_workers is None else num_workers

    timestr = datetime.today().strftime("%Y-%m-%d_%H-%M-%S")
    logdir_prefix = f"{algorithm}_{env.__class__.__name__}_{timestr}"

    results_dir = os.path.expanduser(results_dir)

    def logger_creator(config):
        os.makedirs(results_dir, exist_ok=True)
        logdir = tempfile.mkdtemp(prefix=logdir_prefix, dir=results_dir)
        return ray.tune.logger.UnifiedLogger(config, logdir, loggers=None)

    config = {
        "callbacks": RLlibMetricLogger(metrics),
        "enable_connectors": True,
        "env": env_class.__name__,
        "env_config": env_config,
        "framework": "torch",
        "logger_creator": logger_creator,
        "num_sgd_iter": 10,
        "num_rollout_workers": num_workers_,
        "rollout_fragment_length": env.num_steps,
        "seed": 0,
        "train_batch_size": env.num_steps * max(1, num_workers_),
    }

    config.update(rllib_config)

    config["multiagent"] = {
        "policies": policy_specs,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": policies_to_train,
    }
    config["multiagent"].update(rllib_config.get("multiagent", {}))

    if algorithm == "PPO":
        config["sgd_minibatch_size"] = max(int(config["train_batch_size"] / 10), 1)

    config_obj = (
        ray.tune.registry.get_trainable_cls(algorithm)
        .get_default_config()
        .from_dict(config)
    )

    # Temporary fix to allow current custom model interfaces until RLlib's RLModule
    # becomes stable.
    config_obj.rl_module(_enable_rl_module_api=False)
    config_obj.training(_enable_learner_api=False)

    algo = config_obj.build()

    with rich_progress("Training...") as progress:
        for i in progress.track(range(iterations)):
            result = algo.train()

            if show_training_metrics:
                rich.pretty.pprint(
                    {
                        "iteration": i + 1,
                        "metrics": result["custom_metrics"],
                        "rewards": {
                            "policy_reward_min": result["policy_reward_min"],
                            "policy_reward_max": result["policy_reward_max"],
                            "policy_reward_mean": result["policy_reward_mean"],
                        },
                    }
                )

            if i == 0:
                config = {
                    "algorithm": algorithm,
                    "env_class": env_class,
                    "iterations": iterations,
                    "checkpoint_freq": checkpoint_freq,
                    "policy_specs": policy_specs,
                    "policy_mapping": policy_mapping,
                    "policies_to_train": policies_to_train,
                    "env_config": env_config,
                    "rllib_config": rllib_config,
                    "metrics": metrics,
                }

                with open(Path(algo.logdir, "phantom-training-params.pkl"), "wb") as f:
                    cloudpickle.dump(config, f)

            if checkpoint_freq is not None and i % checkpoint_freq == 0:
                checkpoint_path = Path(algo.logdir, f"checkpoint_{str(i).zfill(6)}")
                algo.save(checkpoint_path)

    print(f"Logs & checkpoints saved to: {algo.logdir}")

    return algo


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

        logging_helper(env, self.metrics, episode.user_data)

    def on_episode_end(self, *, episode, **kwargs) -> None:
        for metric_id, metric in self.metrics.items():
            episode.custom_metrics[metric_id] = metric.reduce(
                episode.user_data[metric_id], mode="train"
            )

    def __call__(self) -> "RLlibMetricLogger":
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
