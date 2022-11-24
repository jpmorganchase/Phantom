.. _api_utils:

Utilities
=========

Samplers
--------

Samplers are designed to be used with supertypes. See the :doc:`/user/supertypes` page
for examples on how they are used.

Interface
^^^^^^^^^

.. autoclass:: phantom.utils.samplers.Sampler
   :inherited-members:

Implementations
^^^^^^^^^^^^^^^

.. autoclass:: phantom.utils.samplers.UniformFloatSampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.UniformIntSampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.UniformArraySampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.NormalSampler
   :inherited-members:

.. autoclass:: phantom.utils.samplers.NormalArraySampler
   :inherited-members:


Ranges
------

Ranges are designed to be used with supertypes. See the :doc:`/user/supertypes` page for
examples on how they are used.

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
