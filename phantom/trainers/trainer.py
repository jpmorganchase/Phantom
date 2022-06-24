from abc import ABC
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from inspect import isclass
from pathlib import Path
from typing import (
    Any,
    DefaultDict,
    Dict,
    Hashable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Type,
    Union,
)

import gym
import numpy as np
import tensorboardX as tbx
from tqdm import trange

from ..types import AgentID, PolicyID
from ..agents import Agent
from ..env import PhantomEnv
from ..logging import Metric
from ..policy import Policy
from ..utils import check_env_config


PolicyMapping = Mapping[
    PolicyID,
    Union[
        Type[Agent],
        List[AgentID],
        Tuple[Type[Policy], Type[Agent]],
        Tuple[Type[Policy], Type[Agent], Mapping[str, Any]],
        Tuple[Type[Policy], List[AgentID]],
        Tuple[Type[Policy], List[AgentID], Mapping[str, Any]],
    ],
]


@dataclass(frozen=True)
class TrainingResults:
    policies: Dict[str, Policy]


class Trainer(ABC):
    """
    Base Trainer class providing interfaces and common functions for subclassed trainers.

    Subclasses must set the ``policy_class`` class variable and implement either the
    ``train`` or ``training_step`` methods.

    Arguments:
        tensorboard_log_dir: If provided, will save metrics to the given directory
            in a format that can be viewed with tensorboard.
    """

    policy_class: Type[Policy]

    def __init__(
        self,
        tensorboard_log_dir: Optional[str] = None,
    ) -> None:
        if self.policy_class is None:
            raise ValueError

        self.tensorboard_log_dir = tensorboard_log_dir
        self.metrics: Dict[str, Metric] = {}
        self.logged_metrics: DefaultDict[str, List[float]] = defaultdict(list)
        self.logged_rewards: DefaultDict[AgentID, List[float]] = defaultdict(list)

        if tensorboard_log_dir is not None:
            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            tb_dir = Path(tensorboard_log_dir, current_time)

            self.tbx_writer = tbx.SummaryWriter(tb_dir)

    def train(
        self,
        env_class: Type[PhantomEnv],
        num_iterations: int,
        policies: PolicyMapping,
        policies_to_train: Sequence[PolicyID],
        env_config: Optional[Mapping[str, Any]] = None,
        metrics: Optional[Mapping[str, Metric]] = None,
    ) -> TrainingResults:
        env_config = env_config or {}
        self.metrics = metrics or {}

        check_env_config(env_config)

        env = env_class(**env_config)
        env.reset()

        policy_mapping, policy_instances = self.setup_policy_specs_and_mapping(
            env, policies
        )

        assert len(policies_to_train) > 0

        for policy_to_train in policies_to_train:
            assert isinstance(policy_instances[policy_to_train], self.policy_class)

        for i in trange(num_iterations):
            self.training_step(env, policy_mapping, policy_instances, policies_to_train)
            self.tbx_write_values(i)

        return TrainingResults(policy_instances)

    def training_step(
        self,
        env: PhantomEnv,
        policy_mapping: Mapping[AgentID, PolicyID],
        policies: Mapping[PolicyID, Policy],
        policies_to_train: Sequence[PolicyID],
    ) -> None:
        raise NotImplementedError

    def log_metrics(self, env: PhantomEnv) -> None:
        for name, metric in self.metrics.items():
            self.logged_metrics[name].append(metric.extract(env))

    def log_vec_metrics(self, envs: Sequence[PhantomEnv]) -> None:
        for name, metric in self.metrics.items():
            self.logged_metrics[name].append(
                np.mean([metric.extract(env) for env in envs])
            )

    def log_rewards(self, rewards: Mapping[AgentID, float]) -> None:
        for agent_id, reward in rewards.items():
            self.logged_rewards[agent_id].append(reward)

    def log_vec_rewards(self, rewards: Sequence[Mapping[AgentID, float]]) -> None:
        for sub_rewards in rewards:
            for agent_id, reward in sub_rewards.items():
                self.logged_rewards[agent_id].append(reward)

    def tbx_write_values(self, step: int) -> None:
        for name, metric in self.metrics.items():
            self.tbx_write_scalar(name, metric.reduce(self.logged_metrics[name]), step)

        group_reward_count = []

        for agent_id, rewards in self.logged_rewards.items():
            self.tbx_write_scalar(f"rewards/{agent_id}", np.mean(rewards), step)
            group_reward_count += rewards

        self.tbx_write_scalar(f"rewards/group", np.mean(group_reward_count), step)

        self.logged_metrics = defaultdict(list)
        self.logged_rewards = defaultdict(list)

    def tbx_write_scalar(self, name: str, value: float, step: int) -> None:
        if self.tensorboard_log_dir is not None:
            self.tbx_writer.add_scalar(name, value, global_step=step)

    def setup_policy_specs_and_mapping(
        self, env: PhantomEnv, policies: PolicyMapping
    ) -> Tuple[Dict[AgentID, PolicyID], Dict[PolicyID, Policy]]:
        policy_specs: Dict[PolicyID, PolicySpec] = {}
        policy_mapping: Dict[AgentID, PolicyID] = {}

        for policy_name, policy_config in policies.items():
            if isclass(policy_config) and issubclass(policy_config, Agent):
                agent_class = policy_config

                agent_ids = list(env.network.get_agents_with_type(agent_class).keys())

                policy_specs[policy_name] = PolicySpec(
                    action_space=env.agents[agent_ids[0]].action_space,
                    observation_space=env.agents[agent_ids[0]].observation_space,
                )

                for agent_id in agent_ids:
                    policy_mapping[agent_id] = policy_name

            elif isinstance(policy_config, list):
                agent_ids = policy_config

                policy_specs[policy_name] = PolicySpec(
                    action_space=env.agents[agent_ids[0]].action_space,
                    observation_space=env.agents[agent_ids[0]].observation_space,
                )

                for agent_id in agent_ids:
                    policy_mapping[agent_id] = policy_name

            elif isinstance(policy_config, tuple):
                if len(policy_config) == 2:
                    policy_class, agents_param = policy_config
                    config = None
                else:
                    policy_class, agents_param, config = policy_config

                if isclass(agents_param) and issubclass(agents_param, Agent):
                    agent_ids = list(
                        env.network.get_agents_with_type(agents_param).keys()
                    )
                elif isinstance(agents_param, list):
                    agent_ids = agents_param
                else:
                    raise ValueError

                policy_specs[policy_name] = PolicySpec(
                    policy_class=policy_class,
                    action_space=env.agents[agent_ids[0]].action_space,
                    observation_space=env.agents[agent_ids[0]].observation_space,
                    config=config,
                )

                for agent_id in agent_ids:
                    policy_mapping[agent_id] = policy_name

            else:
                raise TypeError(type(policy_config))

        policy_instances = {
            name: (
                self.policy_class if spec.policy_class is None else spec.policy_class
            )(spec.observation_space, spec.action_space, spec.config)
            for name, spec in policy_specs.items()
        }

        return (policy_mapping, policy_instances)


@dataclass(frozen=True)
class PolicySpec:
    observation_space: gym.Space
    action_space: gym.Space
    policy_class: Optional[Type[Policy]] = None
    config: Dict[str, Any] = field(default_factory=dict)
