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
have several 'PlayerAgent's and a single 'DealerAgent'.


Creating the Network
--------------------

The :class:`Network` class is always initialised in the :meth:`__init__` method of our
environments. First we need to gather all our actors and agents into a single list:

.. code-block:: python

    players = [PlayerAgent("p1"), PlayerAgent("p2"), PlayerAgent("p3")]

    dealer = DealerAgent("d1")

    agents = players + [dealer]

Next we create our network with the list of agents. By default a :class:`BatchResolver`
is used as the message resolver. Alternatively a custom :class:`Resolver` class instance
can be provided.

.. code-block:: python

    network = ph.Network(agents)

Our agents are now in the network and we must now define the connections. We want to
connect all our players to the dealer. In this environment players do not communicate to
each other. We can manually add each connection one by one:

.. code-block:: python

    network.add_connection("p1", "d1")
    network.add_connection("p2", "d1")
    network.add_connection("p3", "d1")

Or we can use one of the convenience methods of the :class:`Network` class. The
following examples all acheive the same outcome. Use whichever one works best for your
situation.

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

The easiest way to retrieve a single actor/agent from the Network is to use the
subscript operator:

.. code-block:: python

    dealer = network["d1"]

The Network class also provides three methods for retrieving multiple actors/agents at
once:

.. code-block:: python

    players = network.get_actors_with_type(PlayerAgent)
    dealer = network.get_actors_without_type(PlayerAgent)

    odd_players = network.get_actors_where(lambda a: a.id in ["p1", "p3"])


StochasticNetwork
-----------------

Phantom has a :class:`StochasticNetwork` class that implements connection sampling once
top of the standard :class:`Network` class where each connection has a strength
`0.0 <= x <= 1.0`. Every time the network's :meth:`reset` method is called connections
are created or destroyed randomly, weighted by the connection's strength.

.. figure:: /img/stochastic-network.svg
   :figclass: align-center


.. _message_resolution_ref:

Message Resolution
------------------

The process of resolving messages is configurable by the user by choosing or
implementing a :class:`Resolver` class. The default provided resolver class is the
:class:`BatchResolver` and should adequately cover most typical use-cases. It works as
follows:

1. Agents send messages and the network checks that the message is being sent along a
valid connection before passing the message to the resolver:

.. figure:: /img/batch-resolver-1.svg
   :figclass: align-center

2. The :class:`BatchResolver` gathers messages sent from all agents into batches based
on the message's recipients. The :class:`BatchResolver` can optionally shuffle the
ordering of the messages within each batch using the :attr:`shuffle_batches` argument.

.. figure:: /img/batch-resolver-2.svg
   :figclass: align-center

3. The agents receive their messages in batches via the :class:`Agent` class
:meth:`handle_batch` method. By default, these are then automatically distributed to and
handled with the :meth:`handle_message` method.

.. figure:: /img/batch-resolver-3.svg
   :figclass: align-center

4. Any further sent messages are then resolved again and delivered.

.. figure:: /img/batch-resolver-4.svg
   :figclass: align-center

Each step of collecting messages from the agents, batching the messages and then
delivering the messages is known as a 'round'. By default the :class:`BatchResolver`
will continue processing infinite rounds until no messages are left to be sent.
Alternatively this can be limited using the :attr:`round_limit` argument.
