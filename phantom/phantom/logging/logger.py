from collections import defaultdict as _defaultdict
from typing import Dict, Mapping, Optional, TYPE_CHECKING

import mercury as me

from .metrics import Metric

if TYPE_CHECKING:
    from ..env import PhantomEnv


class Logger:
    """
    A class for logging metrics extracted from an instance of
    :py:class:`phantom.PhantomEnv`.

    Attributes:
        metrics: A key-value store of metrics that are being tracked.
        logs: A key-value store of recorded metric histories.

    Usage::

        >>> logger = Logger({
        >>>     'mm/inventory': metrics.Inventory('mm')
        >>> })
        >>> logger.add_metric('mm/spread_pnl', metrics.SpreadPnL('mm'))
        >>> logger.log(env)
    """

    # TODO: The efficiency of this class could be drastically improved by pre-allocating
    # memory for the lists. Consider replacing lists with numpy arrays.

    def __init__(self, metrics: Optional[Mapping[str, Metric]] = None):
        self.metrics: Dict[me.ID, Metric] = dict(metrics or {})
        self.logs: _defaultdict = _defaultdict(list)

    def add_metric(self, metric_id: me.ID, metric: Metric) -> "Logger":
        """
        Add a new metric into the logger, raising an error if a metric already
        has the given :code:`metric_id`.

        Args:
            metric_id: The unique identifier for the metric.
            metric: Some concrete instance of :py:class:`Metric`.
        """
        if metric_id in self.metrics:
            raise ValueError("Metric ID ({}) already present.".format(metric_id))

        self.metrics[metric_id] = metric

        return self

    def remove_metric(self, metric_id: me.ID) -> Optional[Metric]:
        """
        Remove a (maybe) pre-existing metric in the logger, returning the
        instance if it was present.

        Args:
            metric_id: The unique identifier for the metric.
        """
        return self.metrics.pop(metric_id, None)

    def replace_metric(self, metric_id: me.ID, metric: Metric) -> Optional[Metric]:
        """
        Replace a (maybe) pre-existing metric in the logger, returning the
        previous instance if it was present.

        Args:
            metric_id: The unique identifier for the metric.
            metric: Some concrete instance of :py:class:`Metric`.
        """
        old = self.remove_metric(metric_id)

        self.metrics[metric_id] = metric

        return old

    def log(self, env: "PhantomEnv") -> None:
        """
        Extract and log metrics from the :class:`phantom.PhantomEnv` instance.

        Args:
            env: The environment instance.
        """
        for (metric_id, metric) in self.metrics.items():
            self.logs[metric_id].append(metric.extract(env))

    def reset(self) -> None:
        """
        Clear all logged values but retain the set of metrics.
        """
        for metric_id in self.metrics:
            del self.logs[metric_id][:]

    def to_dict(self) -> Dict[me.ID, Metric]:
        """
        Returns a dictionary of all logged values.
        """
        return dict(self.logs)

    def to_reduced_dict(self) -> Dict[me.ID, Metric]:
        """
        Returns a dictionary of all logged values.
        """
        return {
            metric_id: metric.reduce(self.logs[metric_id])
            for metric_id, metric in self.metrics.items()
        }
