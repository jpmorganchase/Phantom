.. _api_utils:

Utilities
=========

Training (RLlib)
----------------

.. autofunction:: phantom.utils.rllib.train


.. Rollouts
.. --------

.. .. autofunction:: phantom.utils.rollout.rollout


.. Rollout Trajectories & Steps
.. ----------------------------

.. .. autoclass:: phantom.utils.rollout_class.AgentStep
..    :members:

.. .. autoclass:: phantom.utils.rollout_class.Step
..    :members:

.. .. autoclass:: phantom.utils.rollout_class.Rollout
..    :members:


Samplers
--------

Interface
^^^^^^^^^

.. autoclass:: phantom.utils.samplers.Sampler
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

.. autoclass:: phantom.utils.ranges.Range
   :inherited-members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: phantom.utils.ranges.UniformRange
   :inherited-members:

.. autoclass:: phantom.utils.ranges.LinspaceRange
   :inherited-members:
