.. _api_metrics:

Metrics
=======

Metrics are included in Phantom as a tool for recording properties of agents and
environments during the running of training or evaluation episodes, without cluttering
up the code in Agent and Environment classes.

Phantom includes three implementations of the base :class:`Metric` class:
:class:`SimpleAgentMetric`, :class:`SimpleEnvMetric` and :class:`AggregatedAgentMetric`.
These will likely cover most use-cases, if not custom metric classes can be created.

Phantom provides functionality to record and store metrics through three methods:

-  When training or evaluating policies using RLlib with the Phantom helper functions,
   a dictionary of metrics can be passed to the functions. For training the metrics will
   be logged along with RLlib training metrics to Tensorboard. For evaluation these will
   be stored in the returned :class:`Rollout` objects.
-  When debugging or evaluating environments manually by calling ``env.step()`` in a
   loop (see :ref:`env_execution`) the logging of metrics, to the terminal and/or to a
   log file can be enabled with the :class:`Telemetry` class configuration methods.
-  When using a custom Phantom :class:`Trainer` class, metrics can be provided to the
   :meth:`train()` method.


Metric
------

.. autoclass:: phantom.metrics.Metric
   :inherited-members:


Simple Agent Metric
-------------------

.. autoclass:: phantom.metrics.SimpleAgentMetric
   :inherited-members:


Simple Env Metric
-----------------

.. autoclass:: phantom.metrics.SimpleEnvMetric
   :inherited-members:


Aggregated Agent Metric
-----------------------

.. autoclass:: phantom.metrics.AggregatedAgentMetric
   :inherited-members:
