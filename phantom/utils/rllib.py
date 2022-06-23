import os
from inspect import isclass
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

import gym
import numpy as np
from ray import rllib
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType

from ..agents import Agent
from ..env import PhantomEnv
from ..logging import Metric, RLlibMetricLogger
from ..policy import Policy
from ..types import AgentID
from . import check_env_config


PolicyClass = Union[Type[Policy], Type[rllib.Policy]]


def train(
    algorithm: str,
    env_class: Type[PhantomEnv],
    num_iterations: int,
    policies: Mapping[
        str,
        Union[
            Type[Agent],
            List[AgentID],
            Tuple[PolicyClass, Type[Agent]],
            Tuple[PolicyClass, Type[Agent], Mapping[str, Any]],
            Tuple[PolicyClass, List[AgentID]],
            Tuple[PolicyClass, List[AgentID], Mapping[str, Any]],
        ],
    ],
    policies_to_train: List[str],
    env_config: Optional[Mapping[str, Any]] = None,
    rllib_config: Optional[Mapping[str, Any]] = None,
    tune_config: Optional[Mapping[str, Any]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    num_workers: Optional[int] = 0,
) -> None:
    import ray
    from ray import rllib

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

    # for agent in env.agents.values():
    #     if agent.action_space is not None and agent.id not in policy_mapping:
    #         policy_specs[agent.id] = rllib.policy.policy.PolicySpec(
    #             action_space=agent.action_space,
    #             observation_space=agent.observation_space,
    #         )

    #         policy_mapping[agent_id] = agent_id

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
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

    stop = {"training_iteration": num_iterations}

    try:
        ray.init()
        results = ray.tune.run(algorithm, config=config, stop=stop, **tune_config)
    except Exception as exception:
        # Ensure that Ray is properly shutdown in the instance of an error occuring
        ray.shutdown()
        raise exception
    else:
        ray.shutdown()
        return results


class RLlibEnvWrapper(rllib.MultiAgentEnv):
    def __init__(self, env: PhantomEnv) -> None:
        self.env = env

        rllib.MultiAgentEnv.__init__(self)

    def step(self, actions: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        return self.env.step(actions)

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
        # 'unsquash' the values leading to unintended results.
        # (https://github.com/ray-project/ray/pull/16531)

        def __init__(
            self,
            observation_space: gym.Space,
            action_space: gym.Space,
            config: Mapping[str, Any],
        ):
            self.policy = policy_class(observation_space, action_space, config)

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
            # Kwars placeholder for future compatibility.
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
