"""
Module containing classes used to extract metrics from a :class:`phantom.PhantomEnv`
instance. These types are used primarily for logging and tracking performance and
behaviour.
"""
from abc import abstractmethod, ABC
from typing import Generic, List, TypeVar, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..env import PhantomEnv


MetricValue = TypeVar("MetricValue")


class Metric(Generic[MetricValue], ABC):
    """
    Class for extracting metrics from a :class:`phantom.PhantomEnv` instance.
    """

    @abstractmethod
    def extract(self, env: "PhantomEnv") -> MetricValue:
        """
        Extract and return the current metric value from `env`.

        Arguments:
            env: The environment instance.
        """
        raise NotImplementedError

    def reduce(self, values: List[MetricValue]) -> MetricValue:
        """
        Reduce a set of observations into a single representative value.

        The default implementation is to return the latest observation.

        Arguments:
            values: Set of observations to reduce.
        """
        return values[-1]


SimpleMetricValue = TypeVar("SimpleMetricValue", int, float)


class SimpleMetric(Metric, Generic[SimpleMetricValue], ABC):
    def __init__(self, reduce_action: str = "last") -> None:
        if reduce_action not in ["last", "mean", "sum"]:
            raise ValueError(
                "reduce_action field of SimpleMetric class must be one of: 'last', 'mean' or 'sum'."
            )

        self.reduce_action = reduce_action

        super().__init__()

    def extract(self, env: "PhantomEnv") -> SimpleMetricValue:
        return getattr(env, self.env_property)

    def reduce(self, values: List[SimpleMetricValue]) -> SimpleMetricValue:
        if self.reduce_action == "last":
            return values[-1]
        elif self.reduce_action == "mean":
            return np.mean(values)
        else:
            return np.sum(values)


class SimpleAgentMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the state of a given
    agent.

    Three options are available for summarizing the values at the end of each episode:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    Arguments:
        agent_id: The ID of the agent to record the metric for.
        agent_property: The property existing on the agent to record, must be
            accessible with `getattr`.
        reduce_action: The operation to perform on all the recorded values at the end
            of the episode ('last', 'mean' or 'sum').
    """

    def __init__(
        self, agent_id: str, agent_property: str, reduce_action: str = "last"
    ) -> None:
        self.agent_id = agent_id
        self.agent_property = agent_property

        super().__init__(reduce_action)

    def extract(self, env: "PhantomEnv") -> SimpleMetricValue:
        return getattr(env.agents[self.agent_id], self.agent_property)


class SimpleEnvMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the state of the env.

    Three options are available for summarizing the values at the end of each episode:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    Arguments:
        env_property: The property existing on the environment to record, must be
            accessible with `getattr`.
        reduce_action: The operation to perform on all the recorded values at the end
            of the episode ('last', 'mean' or 'sum').
    """

    def __init__(self, env_property: str, reduce_action: str = "last") -> None:
        self.env_property = env_property

        super().__init__(reduce_action)

    def extract(self, env: "PhantomEnv") -> SimpleMetricValue:
        return getattr(env, self.env_property)


class AggregatedAgentMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the states of a group
    of agents and performing a reduction operation on the values.

    Three options are available for reducing the values from the group of agents:

        - 'min' - takes the mean of all the per-step values
        - 'max' - takes the mean of all the per-step values
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    Three options are available for summarizing the values at the end of each episode:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    Arguments:
        agent_ids: The ID's of the agents to record the metric for.
        agent_property: The property existing on the agent to record, must be
            accessible with `getattr`.
        group_reduce_action: The operation to perform on the values gathered from the
            group of agents ('min', 'max', 'mean' or 'sum').
        reduce_action: The operation to perform on all the recorded values at the end
            of the episode ('last', 'mean' or 'sum').
    """

    def __init__(
        self,
        agent_ids: List[str],
        agent_property: str,
        group_reduce_action: str = "mean",
        reduce_action: str = "last",
    ) -> None:
        if group_reduce_action not in ["min", "max", "mean", "sum"]:
            raise ValueError(
                "group_reduce_action field of SimpleMetric class must be one of: 'min', 'max', 'mean' or 'sum'."
            )

        self.agent_ids = agent_ids
        self.agent_property = agent_property
        self.group_reduce_action = group_reduce_action

        super().__init__(reduce_action)

    def extract(self, env: "PhantomEnv") -> SimpleMetricValue:
        values = [
            getattr(env.agents[agent_id], self.agent_property)
            for agent_id in self.agent_ids
        ]

        if self.group_reduce_action == "min":
            return np.min(values)
        elif self.group_reduce_action == "max":
            return np.max(values)
        elif self.group_reduce_action == "mean":
            return np.mean(values)
        else:
            return np.sum(values)
