.. _api_utils:

Utilities
=========

Training
--------

.. autofunction:: phantom.utils.training.train


Rollouts
--------

.. autofunction:: phantom.utils.rollout.rollout


Samplers
--------

Interface
^^^^^^^^^

.. autoclass:: phantom.utils.samplers.BaseSampler
   :inherited-members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: phantom.utils.samplers.UniformSampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.UniformArraySampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.NormalSampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.NormalArraySampler
   :inherited-members:


Ranges
------

Interface
^^^^^^^^^

.. autoclass:: phantom.utils.ranges.BaseRange
   :inherited-members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: phantom.utils.ranges.UniformRange
   :inherited-members:

.. autoclass:: phantom.utils.ranges.LinspaceRange
   :inherited-members:
