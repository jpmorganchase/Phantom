"""
Module containing classes used to extract metrics from a
:class:`phantom.PhantomEnv` instance. These types are used primarily for logging
and tracking performance and behaviour.
"""
from abc import abstractmethod, ABC
from typing import Generic, List, TypeVar, TYPE_CHECKING

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
