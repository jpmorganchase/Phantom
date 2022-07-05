.. _network:

Network
=======

.. figure:: /img/icons/network.svg
   :width: 15%
   :figclass: align-center

At the core of any Phantom environment is the network. This defines the relationships
between all actors and agents, controls who can send messages to who and handles the
way in which messages are sent and resolved.

Each actor and agent, or 'entity', in the network is identified with a unique ID.
Usually a string is used for this purpose. Any entity can be connected to any number of
other actors and agents. When two entities are connected, bi-directional communication
is allowed. Connected entities also have read-only access to each others
:class:`View`'s allowing the publishing of information to other entities without having
to send messages.

Consider a simple environment where we have some agents learn to play a card game. We
have several ``PlayerAgent`` s and a single ``DealerAgent``.


Creating the Network
--------------------

The :class:`Network` class is always initialised in the ``__init__`` method of our
environments. First we need to gather all our actors and agents into a single list:

.. code-block:: python

    players = [PlayerAgent("p1"), PlayerAgent("p2"), PlayerAgent("p3")]

    dealer = DealerAgent("d1")

    agents = players + [dealer]

Next we create our network. We must pass in a :class:`Resolver` class instance along with
our list of agents.

.. code-block:: python

    network = ph.Network(agents)

Our agents are now in the network and we must now define the connections. We want to
connect all our players to the dealer. In this environment players do not communicate to
each other. We can manually add each connection one by one:

.. code-block:: python

    network.add_connection("p1", "d1")
    network.add_connection("p2", "d1")
    network.add_connection("p3", "d1")

Or we can use one of the convenience methods of the :class:`Network` class. The following
examples all acheive the same outcome. Use whichever one works best for your situation.

.. code-block:: python

    network.add_connections_from([
        ("d1", "p1"),
        ("d1", "p2"),
        ("d1", "p3"),
    ])


.. code-block:: python

    network.add_connections_between(["d1"], ["p1", "p2", "p3"])


.. code-block:: python

    network.add_connections_with_adjmat(
        ["d1", "p1", "p2", "p3"],
        np.array([
            [0, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
            [1, 0, 0, 0],
        ])
    )

Accessing the Network
---------------------

The easiest way to retrieve a single actor/agent from the Network is to use the subscript
operator:

.. code-block:: python

    dealer = network["d1"]

The Network class also provides three methods for retrieving multiple actors/agents at
once:

.. code-block:: python

    players = network.get_actors_with_type(PlayerAgent)
    dealer = network.get_actors_without_type(PlayerAgent)

    odd_players = network.get_actors_where(lambda a: a.id in ["p1", "p3"])


TODO: add stochastic network diagram