.. _api_tracker:

Tracker
=======

The :class:`Tracker` class is a utility provided for tracking the values of a
changing quantity over time.

Basic usage:

.. code-block:: python

   from phantom import Tracker

   # Creates a tracker object with a default value of 0.0
   tracker = Tracker(0.0)

   assert tracker.current == 0.0

   # Set the current value of the tracker to 1.0
   tracker.current = 1.0

   # Cache the current value. This sets the first element in the history to
   # 1.0 and resets the current value to the default value of 0.0.
   tracker.cache()

   assert tracker.current == 0.0
   assert tracker.previous == 1.0

A default previous value can also be given:

.. code-block:: python

   from phantom import Tracker

   # Creates a tracker object with a default value of 1.0 and a default previous
   # value of 0.0.
   tracker = Tracker(1.0, 0.0)

   assert tracker.current == 1.0
   assert tracker.previous == 0.0

   assert tracker.diff == 1.0


.. autoclass:: phantom.tracker.Tracker
   :inherited-members:
   :special-members:
