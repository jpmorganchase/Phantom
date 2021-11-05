.. _designpatterns:

Design Patterns
===============

This page lays out some common design patterns for setting up your Phantom environment.


Simple Environment Pattern
--------------------------

This pattern is common for simple experiments. In this pattern all agents act at every
step. Subclassing an environment from the :class:`PhantomEnv` class is sufficient for this.

.. figure:: /img/patterns-simple-env.svg
   :width: 70%
   :figclass: align-center


Finite State Machine Pattern
----------------------------

For more complex experiments with multiple steps and multiple agent groups potentially
with a subset of agents taking actions each turn Phantom provides the
:class:`FiniteStateMachineEnv` class. This formalises the environment as a free state
machine by extending the :class:`PhantomEnv` class and providing a clean interface for
managing the logic of each state and the transitions between states.

.. figure:: /img/patterns-finite-state-machine.svg
   :width: 70%
   :figclass: align-center


Alternate Turn Pattern (Stackelberg Game)
-----------------------------------------

A simple but commonly used pattern is that of a free state machine with two states that
pass from one another. This is formally known as a Stackleberg game or a
`Stackelberg competition <https://en.wikipedia.org/wiki/Stackelberg_competition>`_. This
can easily be implemented using the :class:`FiniteStateMachineEnv` class.

.. figure:: /img/patterns-alternate-turn.svg
   :width: 70%
   :figclass: align-center
