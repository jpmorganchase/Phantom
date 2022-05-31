import os
from inspect import isclass
from typing import (
    cast,
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    Type,
    Union,
    TYPE_CHECKING,
)

from ..agents import Agent
from ..env import PhantomEnv
from ..logging import Metric, RLlibMetricLogger
from ..types import AgentID
from . import check_env_config

if TYPE_CHECKING:
    from ray import rllib


def train(
    algorithm: str,
    env_class: Type[PhantomEnv],
    num_iterations: int,
    policies: Mapping[
        str,
        Union[
            Type[Agent],
            List[AgentID],
            Tuple[Type[Agent], "rllib.Policy"],
            Tuple[List[AgentID], "rllib.Policy"],
            Tuple[Type[Agent], "rllib.Policy", Mapping[str, Any]],
            Tuple[List[AgentID], "rllib.Policy", Mapping[str, Any]],
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
                agent_ids, policy_class = cast(
                    Tuple[List[AgentID], Type["rllib.Policy"]], params
                )
            else:
                agent_ids, policy_class, config = cast(
                    Tuple[List[AgentID], Type["rllib.Policy"], Dict[str, Any]], params
                )

            if isclass(agent_ids) and issubclass(agent_ids, Agent):
                agent_ids = list(env.network.get_agents_with_type(agent_ids).keys())

        else:
            raise TypeError(type(params))

        policy_specs[policy_name] = rllib.policy.policy.PolicySpec(
            policy_class=policy_class,
            action_space=env.agents[agent_ids[0]].get_action_space(),
            observation_space=env.agents[agent_ids[0]].get_observation_space(),
            config=config,
        )

        for agent_id in agent_ids:
            policy_mapping[agent_id] = policy_name

    # for agent in env.agents.values():
    #     if agent.takes_actions() and agent.id not in policy_mapping:
    #         policy_specs[agent.id] = rllib.policy.policy.PolicySpec(
    #             action_space=agent.get_action_space(),
    #             observation_space=agent.get_observation_space(),
    #         )

    #         policy_mapping[agent_id] = agent_id

    def policy_mapping_fn(agent_id, episode, worker, **kwargs):
        return policy_mapping[agent_id]

    ray.tune.registry.register_env(
        env_class.__name__, lambda config: env_class(**config)
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
