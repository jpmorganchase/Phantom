.. _design:

Phantom Design
==============

This page outlines the key design concepts of Phantom. For a full reference please see
the API pages.

A phantom experiment consists of many independent episodes. Depending on the learning
algorithm, it may be possible to perform many episodes in parallel. At the end of each
episode or group of episodes, the learning policies are updated. Within each episode,
multiple steps are performed. Episodes can consist of a fixed or variable number of
steps.


Environment
-----------

The Environment is the main element of a Phantom experiment. In the Phantom framework
the environment describes all the agents and actors that are part of the environment,
how these interact and how in each step these agents and actors progress through the
episode.

Phantom provides a ``PhantomEnv`` class that provides sensible defaults for controlling
each step and the progression of the actors and agents through the episode (In advanced
use-cases it is possible to override this). It is up to the user to define the actors
and agents and define how they are connected and how they interact.

The base environment derives directly from the RLlib ``MulitAgentEnv`` class which
itself is based off the widely used OpenAI gym Environment interface. Users of either
will therefore be familiar with the Phantom environment setup.

Episode Cycle
-------------

The following diagram details the basic flow of an episode. First the entire environment
is reset - this includes all actors, agents and supertypes. This reset provides default
observations from the agents.

The episode then enters a loop of producing actions from these observations using the
policy, acting on the actions with the ``step`` function, producing more observations
and so on.

This continues until the end of the episode which is either a fixed number of steps or
at a point when all agents have finished.

.. figure:: /img/episode-flow.svg
   :width: 53%
   :figclass: align-center

Step Cycle
----------

.. figure:: /img/step-flow.svg
   :width: 50%
   :figclass: align-center
