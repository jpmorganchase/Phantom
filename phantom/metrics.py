from abc import abstractmethod, ABC
from functools import reduce
from typing import (
    Callable,
    DefaultDict,
    Generic,
    Iterable,
    List,
    Literal,
    Mapping,
    Optional,
    Sequence,
    TypeVar,
    Union,
)

import numpy as np

from .env import PhantomEnv
from .fsm import FSMStage, FiniteStateMachineEnv


MetricValue = TypeVar("MetricValue", float, int, np.typing.ArrayLike)


class NotRecorded:
    def __new__(cls):
        if not hasattr(cls, "instance"):
            cls.instance = super(NotRecorded, cls).__new__(cls)
        return cls.instance

    def __repr__(self) -> str:
        return "<NotRecorded>"


not_recorded = NotRecorded()


class Metric(Generic[MetricValue], ABC):
    """Class for extracting metrics from a :class:`phantom.PhantomEnv` instance.

    Arguments:
        fsm_stages: Optional list of FSM stages to filter metric recording on. If None
            is given metrics will be recorded on all stages when used with an FSM Env.
            If a list of FSM stages is given, the metric will only be recorded when the
            Env is in these stages, otherwise a None value will be recorded.
        description: Optional description string for use in data exploration tools.
    """

    def __init__(
        self,
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.fsm_stages = fsm_stages
        self.description = description

    @abstractmethod
    def extract(self, env: PhantomEnv) -> MetricValue:
        """Extract and return the current metric value from `env`.

        Arguments:
            env: The environment instance.
        """
        raise NotImplementedError

    def reduce(
        self, values: Sequence[MetricValue], mode: Literal["train", "evaluate"]
    ) -> MetricValue:
        """Reduce a set of observations into a single representative value.

        The default implementation is to return the latest observation.

        Arguments:
            values: Set of observations to reduce.
            mode: Whether the metric is being recorded during training or evaluation.
        """
        return values[-1]


class LambdaMetric(Metric, Generic[MetricValue]):
    """Class for extracting metrics from a :class:`phantom.PhantomEnv` instance with a
    provided extraction function.

    Arguments:
        extract_fn: Function to extract the metric value from the environment.
        train_reduce_fn: Function to reduce a set of observations into a single
            representative value during training.
        eval_reduce_fn: Function to reduce a set of observations into a single
            representative value during evaluation.
        fsm_stages: Optional list of FSM stages to filter metric recording on. If None
            is given metrics will be recorded on all stages when used with an FSM Env.
            If a list of FSM stages is given, the metric will only be recorded when the
            Env is in these stages, otherwise a None value will be recorded.
        description: Optional description string for use in data exploration tools.
    """

    def __init__(
        self,
        extract_fn: Callable[[PhantomEnv], MetricValue],
        train_reduce_fn: Callable[[Sequence[MetricValue]], MetricValue],
        eval_reduce_fn: Callable[[Sequence[MetricValue]], MetricValue],
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.extract_fn = extract_fn
        self.train_reduce_fn = train_reduce_fn
        self.eval_reduce_fn = eval_reduce_fn
        self.fsm_stages = fsm_stages
        self.description = description

    def extract(self, env: PhantomEnv) -> MetricValue:
        """Extract and return the current metric value from `env`.

        Arguments:
            env: The environment instance.
        """
        return self.extract_fn(env)

    def reduce(
        self, values: Sequence[MetricValue], mode: Literal["train", "evaluate"]
    ) -> MetricValue:
        """Reduce a set of observations into a single representative value.

        The default implementation is to return the latest observation.

        Arguments:
            values: Set of observations to reduce.
            mode: Whether the metric is being recorded during training or evaluation.
        """
        if mode == "train":
            return self.train_reduce_fn(values)
        elif mode == "evaluate":
            return self.eval_reduce_fn(values)
        else:
            raise ValueError(f"Unknown mode: {mode}")


SimpleMetricValue = TypeVar("SimpleMetricValue", int, float, np.number)


class SimpleMetric(Metric, Generic[SimpleMetricValue], ABC):
    """Base class of a set of helper metric classes."""

    def __init__(
        self,
        train_reduce_action: Literal["last", "mean", "sum"] = "mean",
        eval_reduce_action: Literal["last", "mean", "sum", "none"] = "none",
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        if train_reduce_action not in ("last", "mean", "sum"):
            raise ValueError(
                f"train_reduce_action field of {self.__class__} metric must be one of: 'last', 'mean' or 'sum'. Got '{train_reduce_action}'."
            )

        if eval_reduce_action not in ("last", "mean", "sum", "none"):
            raise ValueError(
                f"eval_reduce_action field of {self.__class__} metric class must be one of: 'last', 'mean', 'sum' or 'none'. Got '{eval_reduce_action}'."
            )

        self.train_reduce_action = train_reduce_action
        self.eval_reduce_action = eval_reduce_action

        super().__init__(fsm_stages, description)

    def reduce(
        self, values: Sequence[SimpleMetricValue], mode: Literal["train", "evaluate"]
    ) -> Union[SimpleMetricValue, np.typing.NDArray[SimpleMetricValue]]:
        reduce_action = (
            self.train_reduce_action if mode == "train" else self.eval_reduce_action
        )

        if reduce_action == "none":
            return np.array(values)

        if self.fsm_stages is not None:
            values = [v for v in values if v is not not_recorded]

        if reduce_action == "last":
            return values[-1] if len(values) > 0 else None
        if reduce_action == "mean":
            return np.mean(values)
        if reduce_action == "sum":
            return np.sum(values)

        raise ValueError


class SimpleAgentMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the state of a given
    agent.

    Three options are available for summarizing the values at the end of each episode
    during training or evaluation:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    During evaluation there is also the option for no value summarizing by using 'none'.

    Arguments:
        agent_id: The ID of the agent to record the metric for.
        agent_property: The property existing on the agent to record, can be nested
            (e.g. ``Agent.property.sub_property``).
        train_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum').
        eval_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum', 'none').
        description: Optional description string for use in data exploration tools.
    """

    def __init__(
        self,
        agent_id: str,
        agent_property: str,
        train_reduce_action: Literal["last", "mean", "sum"] = "mean",
        eval_reduce_action: Literal["last", "mean", "sum", "none"] = "none",
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.agent_id = agent_id
        self.agent_property = agent_property

        super().__init__(
            train_reduce_action, eval_reduce_action, fsm_stages, description
        )

    def extract(self, env: PhantomEnv) -> SimpleMetricValue:
        return _rgetattr(env.agents[self.agent_id], self.agent_property)


class SimpleEnvMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the state of the env.

    Three options are available for summarizing the values at the end of each episode
    during training or evaluation:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    During evaluation there is also the option for no value summarizing by using 'none'.

    Arguments:
        env_property: The property existing on the environment to record, can be nested
            (e.g. ``Agent.property.sub_property``).
        train_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum').
        eval_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum', 'none').
        description: Optional description string for use in data exploration tools.
    """

    def __init__(
        self,
        env_property: str,
        train_reduce_action: Literal["last", "mean", "sum"] = "mean",
        eval_reduce_action: Literal["last", "mean", "sum", "none"] = "none",
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        self.env_property = env_property

        super().__init__(
            train_reduce_action, eval_reduce_action, fsm_stages, description
        )

    def extract(self, env: PhantomEnv) -> SimpleMetricValue:
        return _rgetattr(env, self.env_property)


class AggregatedAgentMetric(SimpleMetric, Generic[SimpleMetricValue]):
    """
    Simple helper class for extracting single ints or floats from the states of a group
    of agents and performing a reduction operation on the values.

    Three options are available for reducing the values from the group of agents:

        - 'min' - takes the mean of all the per-step values
        - 'max' - takes the mean of all the per-step values
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    Three options are available for summarizing the values at the end of each episode
    during training or evaluation:

        - 'last' - takes the value from the last step
        - 'mean' - takes the mean of all the per-step values
        - 'sum'  - takes the sum of all the per-step values

    During evaluation there is also the option for no value summarizing by using 'none'.

    Arguments:
        agent_ids: The ID's of the agents to record the metric for.
        agent_property: The property existing on the agent to record, can be nested
            (e.g. ``Agent.property.sub_property``).
        train_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum').
        eval_reduce_action: The operation to perform on all the per-step recorded
            values at the end of the episode ('last', 'mean' or 'sum', 'none').
        description: Optional description string for use in data exploration tools.
    """

    def __init__(
        self,
        agent_ids: Iterable[str],
        agent_property: str,
        group_reduce_action: Literal["min", "max", "mean", "sum"] = "mean",
        train_reduce_action: Literal["last", "mean", "sum"] = "mean",
        eval_reduce_action: Literal["last", "mean", "sum", "none"] = "none",
        fsm_stages: Optional[Sequence[FSMStage]] = None,
        description: Optional[str] = None,
    ) -> None:
        if group_reduce_action not in ["min", "max", "mean", "sum"]:
            raise ValueError(
                "group_reduce_action field of SimpleMetric class must be one of: 'min', 'max', 'mean' or 'sum'."
            )

        self.agent_ids = agent_ids
        self.agent_property = agent_property
        self.group_reduce_action = group_reduce_action

        super().__init__(
            train_reduce_action, eval_reduce_action, fsm_stages, description
        )

    def extract(self, env: PhantomEnv) -> SimpleMetricValue:
        values = [
            _rgetattr(env.agents[agent_id], self.agent_property)
            for agent_id in self.agent_ids
        ]

        if self.group_reduce_action == "min":
            return np.min(values)
        if self.group_reduce_action == "max":
            return np.max(values)
        if self.group_reduce_action == "mean":
            return np.mean(values)
        if self.group_reduce_action == "sum":
            return np.sum(values)

        raise ValueError


def _rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return reduce(_getattr, [obj] + attr.split("."))


def logging_helper(
    env: PhantomEnv,
    metrics: Mapping[str, Metric],
    metric_values: DefaultDict[str, List[Union[MetricValue, NotRecorded]]],
) -> None:
    for metric_id, metric in metrics.items():
        if (
            not isinstance(env, FiniteStateMachineEnv)
            or metric.fsm_stages is None
            or env.current_stage in metric.fsm_stages
        ):
            value = metric.extract(env)
        else:
            value = not_recorded

        metric_values[metric_id].append(value)
