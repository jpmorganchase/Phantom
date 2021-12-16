.. _tutorial1:

Tutorial - Part 1
=================

This tutorial will walk you through the steps of designing and running a simple Phantom
experiment. It is based on the included ``supply-chain-1.py`` example that can be found
in the ``envs`` directory in the Phantom repo.


Experiment Goals
----------------

.. figure:: /img/icons/tasks.svg
   :width: 15%
   :figclass: align-center

We want to model a very simple supply chain consisting of three types of agents:
warehouses, shops and customers. Our supply chain has one product that is available in
whole units. We do not concern ourselves with prices or profits here.

.. figure:: /img/supply-chain.svg
   :width: 80%
   :figclass: align-center


Warehouse Actor
^^^^^^^^^^^^^^^

The warehouse is an actor in the experiment. This is because, unlike customers or the
shop, the warehouse does not need to take any actions - it is purely reactive.

The shop can make unlimitec requests for stock to the warehouse. The warehouse holds
unlimited stock and can dispatch unlimited stock to the shop if requested.

.. figure:: /img/supply-chain-warehouse.svg
   :width: 60%
   :figclass: align-center



Customer Agent
^^^^^^^^^^^^^^

Customers are non-learning agents. Every step they make an order to the shop for a
variable quantity of products. We model the number of products requested with a Poisson
random distribution. Customers receive products from the shop after making an order. We
do not need to do anything with this when received.

.. figure:: /img/supply-chain-customer.svg
   :width: 55%
   :figclass: align-center



Shop Agent
^^^^^^^^^^

The shop is the only learning agent in this experiment. It can hold infinite stock and
can request infinite stock from the warehouse. It receives orders from customers and
tries to fulfil these orders as best it can.

The shop takes one action each step - the request for more stock that it sends to the
warehouse. The amount it requests is decided by the policy. The policy is informed by
one observation: the amount of stock currently held by the shop.

The goal is for the shop to learn a policy where it makes the right amount of stock
requests to the warehouse so it can fulfil all it's orders without holding onto too much
unecessary stock. This goal is implemented in the shop agent's reward function.

.. figure:: /img/supply-chain-shop.svg
   :width: 90%
   :figclass: align-center


Implementation
--------------

First we import the libraries we require and define some constants.

.. code-block:: python

    import gym
    import mercury as me
    import numpy as np
    import phantom as ph


    NUM_EPISODE_STEPS = 100

    SHOP_MAX_STOCK = 100_000
    SHOP_MAX_STOCK_REQUEST = 1000

Phantom uses the ``Mercury`` library for handling the network of agents and actors and
the message passing between them and `Ray + RLlib <https://docs.ray.io/en/master/index.html>`_
for running and scaling the RL training.

As this experiment is simple we can easily define it entirely within one file. For more
complex, larger experiments it is recommended to split the code into multiple files,
making use of the modularity of Phantom.

Next, for each of our agent/actor types we define a new Python class that encapsulates
all the functionality the given agent/actor needs:


Warehouse Actor
^^^^^^^^^^^^^^^

.. figure:: /img/icons/warehouse.svg
   :width: 15%
   :figclass: align-center

The warehouse is the simplest to implement as it does not take actions and does not
store state. We inherit from Mercury's ``SimpleSyncActor`` class. The ``SimpleSyncActor``
is an actor that handles the message it receives in a synchronous order.

.. code-block:: python

    class WarehouseActor(me.actors.SimpleSyncActor):
        def __init__(self, actor_id: str):
            super().__init__(actor_id)


The ``SimpleSyncActor`` class requires that we implement a ``handle_message`` method in
our sub-class. Here we take any stock request we receive from the shop (the ``payload``
of the message) and reflect it back to the shop as the warehouse will always fulfils
stock requests.

The ``handle_message`` method must return messages as an iterator and hence we use the
``yield`` statement instead of the usual ``return`` statement.


.. code-block:: python

        def handle_message(self, ctx: me.Network.Context, msg: me.Message):
            # The warehouse receives stock request messages from shop agents. We
            # simply reflect the amount of stock requested back to the shop as the
            # warehouse has unlimited stock.
            yield (msg.sender_id, [msg.payload])
    #

Customer Agent
^^^^^^^^^^^^^^

.. figure:: /img/icons/customer.svg
   :width: 15%
   :figclass: align-center

The implementation of the customer agent class takes more work as it stores state and
takes actions. For any agent to be able to interact with the RLlib framework we need to
define methods to decode actions, encode observations, compute reward functions. Our
customer agent takes actions according to a pre-defined policy - it does not actively
learn - and so we can use a ``FixedPolicy`` derived class to define this simple policy:

.. code-block:: python

    class CustomerPolicy(ph.FixedPolicy):
        # The size of the order made for each customer is determined by this fixed policy.
        def compute_action(self, obs) -> np.ndarray:
            return np.random.poisson(5, size=(1,))

Next we define the customer agent class. We make sure to set the policy to be our
custom fixed policy.

.. code-block:: python

    class CustomerAgent(ph.Agent):
        def __init__(self, agent_id: str, shop_id: str):
            super().__init__(agent_id, policy_class=CustomerPolicy)

            # We need to store the shop's ID so we can send order requests to it.
            self.shop_id: str = shop_id

We take the ID of the shop as an initialisation parameter and store it as local state.
It is recommended to always handle IDs this way rather than hard-coding them.

We define a custom ``decode_action`` method. This is called every step and allows the
customer agent to make orders to the shop. We use a random number generator to create
varying order sizes.

.. code-block:: python

        def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
            # At the start of each step we generate an order with a random size to
            # send to the shop.
            order_size = np.random.poisson(5)

            # We perform this action by sending a stock request message to the warehouse.
            return ph.packet.Packet(messages={self.shop_id: [order_size]})
    #

From this method we return a ``Packet`` object. This is a simple container than contains
``Mutators`` and ``Messages``. In this instance we are only filling it with messages --
mutators will be covered in a later tutorial.

The ``messages`` parameter of the ``Packet`` object is a mapping of recipient IDs to a
list of message payloads. This allows multiple messages to be send to a single
agent/actor. In our case we are sending a single message containing a numeric value
(the order size) to the shop.

As before with the warehouse actor, we have to define a ``handle_message`` method. The
customer receives messages from the shop containing the products the customer requested.
The customer does not need to take any action with these messages and so we return an
empty iterator using the ``yield from ()`` syntactic sugar.

.. code-block:: python

        def handle_message(self, ctx: me.Network.Context, msg: me.Message):
            # The customer will receive it's order from the shop but we do not need
            # to take any actions on it.
            yield from ()
    #

As our customer agent does not learn we do not need to construct a reward function but
we do need to still return a value to satisfy RLlib:

.. code-block:: python

        def compute_reward(self, ctx: me.Network.Context) -> float:
            return 0.0

        def encode_obs(self, ctx: me.Network.Context):
            return np.zeros((1,))

        def get_observation_space(self):
            return gym.spaces.Box(-np.inf, np.inf, (1,))

        def get_action_space(self):
            return gym.spaces.Box(-np.inf, np.inf, (1,))

    #


Shop Agent
^^^^^^^^^^

.. figure:: /img/icons/shop.svg
   :width: 15%
   :figclass: align-center

As the learning agent in our experiment, the shop agent is the most complex and
introduces some new features of Phantom. As seen below, we store more local state than
before.

We keep track of sales and missed sales over two time spans: for each step (to guide the
policy) and for each episode (for logging purposes).

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, warehouse_id: str):
            super().__init__(agent_id)

            # We store the ID of the warehouse so we can send stock requests to it.
            self.warehouse_id: str = warehouse_id

            # We keep track of how much stock the shop has...
            self.stock: int = 0

            # ...and how many sales have been made...
            self.step_sales: int = 0
            self.total_sales: int = 0

            # ...and how many sales per step the shop has missed due to not having enough
            # stock.
            self.step_missed_sales: int = 0
            self.total_missed_sales: int = 0


We want to keep track of how many sales and missed sales we made in the step. When
messages are sent, the shop will start taking orders. So before this happens we want to
reset our counters. We can do this by defining a ``pre_resolution`` method. This is
called directly before messages are sent across the network in each step.

.. code-block:: python

    def pre_resolution(self, ctx: me.Network.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.step_sales = 0
        self.step_missed_sales = 0
    #

The ``handle_message`` method is logically split into two parts: handling messages
received from the warehouse and handling messages received from the customer.

.. code-block:: python

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        if msg.sender_id == self.warehouse_id:
            # Messages received from the warehouse contain stock.
            self.stock += msg.payload

            # We do not need to respond to these messages.
            yield from ()
        else:
            # All other messages are from customers and contain orders.
            amount_requested = msg.payload

            if amount_requested > self.stock:
                self.step_missed_sales += amount_requested - self.stock
                self.total_missed_sales += amount_requested - self.stock
                stock_to_sell = self.stock
                self.stock = 0
            else:
                stock_to_sell = amount_requested
                self.stock -= amount_requested

            self.step_sales += stock_to_sell
            self.total_sales += stock_to_sell

            # Send the customer their order.
            yield (msg.sender_id, [stock_to_sell])
    #


The observation we send to the policy on each step is the shop's amount of stock it
currently holds. We allow this information to be sent by defining an ``encode_obs``
method:

.. code-block:: python

    def encode_obs(self, ctx: me.Network.Context):
        # We encode the shop's current stock as the observation.
        return np.array([self.stock])
    #

We define a ``decode_action`` method for taking the action from the policy and
translating it into messages to send in the environment. Here the action taken is making
requests to the warehouse for more stock. We place the messages we want to send in a
``Packet`` container.

.. code-block:: python

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from
        # the warehouse.
        stock_to_request = action[0]

        # We perform this action by sending a stock request message to the warehouse.
        return ph.packet.Packet(messages={self.warehouse_id: [stock_to_request]})
    #

Next we define a ``compute_reward`` method. Every step we calculate a reward based on
the agents current state in the environment and send it to the policy so it can learn.

.. code-block:: python

    def compute_reward(self, ctx: me.Network.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        # We give a bigger reward for making sales than the penalty for missed sales and
        # unused stock.
        return 5 * self.step_sales - self.step_missed_sales - self.stock
    #

Each episode can be thought of as a completely independent trial for the environment.
However creating a new environment each time with a new network, actors and agents could
potentially slow our simulations down a lot. Instead we can reset our objects back to an
initial state. This is done with the ``reset`` method:

.. code-block:: python

    def reset(self):
        self.stock = 0
        self.total_sales = 0
        self.total_missed_sales = 0

Finally we need to let RLlib know the sizes of our observation space and action space so
it can construct the correct neural network for the agent's policy. This is done by
defining a ``get_observation_space`` method and a ``get_action_space`` method:

.. code-block:: python

    def get_observation_space(self):
        return gym.spaces.Box(low=np.array([0.0]), high=np.array([SHOP_MAX_STOCK]))

    def get_action_space(self):
        return gym.spaces.Box(
            low=np.array([0.0]), high=np.array([SHOP_MAX_STOCK_REQUEST])
        )

Here we state that we can observe between 0 and infinite stock and we can also take an
action to get between 0 and infinite stock (see constant values defined at the start).


Environment
^^^^^^^^^^^

.. figure:: /img/icons/environment.svg
   :width: 15%
   :figclass: align-center

Now we have defined all our actors and agents and their behaviours we can describe how
they will all interact by defining our environment. Phantom provides a base
``PhantomEnv`` class that the user should create their own class and inherit from. The
``PhantomEnv`` class provides a default set of required methods such as ``step`` which
coordinates the evolution of the environment for each episodes.

Advanced users of Phantom may want to implement advanced functionality and write their
own methods, but for most simple use cases the provided methods are fine. The minimum a
user needs to do is define a custom initialisation method that defines the network and
the number of episode steps.

.. code-block:: python

    class SupplyChainEnv(ph.PhantomEnv):

        env_name: str = "supply-chain-v1"

        def __init__(self, n_customers: int = 5):

The recommended design pattern when creating your environment is to define all the actor
and agent IDs up-front and not use hard-coded values:

.. code-block:: python

            # Define actor and agent IDs
            shop_id = "SHOP"
            warehouse_id = "WAREHOUSE"
            customer_ids = [f"CUST{i+1}" for i in range(n_customers)]
        #

Next we define our agents and actors by creating instances of the classes we previously
wrote:

.. code-block:: python

            shop_agent = ShopAgent(shop_id, warehouse_id=warehouse_id)
            warehouse_actor = WarehouseActor(warehouse_id)

            customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in customer_ids]
        #

Then we accumulate all our agents and actors in one list so we can add them to the
network. We then use the IDs to create the connections between our agents:

.. code-block:: python

            actors = [shop_agent, warehouse_actor] + customer_agents

            # Define Network and create connections between Actors
            network = me.Network(me.resolvers.UnorderedResolver(), actors)

            # Connect the shop to the warehouse
            network.add_connection(shop_id, warehouse_id)

            # Connect the shop to the customers
            network.add_connections_between([shop_id], customer_ids)
        #

Finally we make sure to initialise the parent ``PhantomEnv`` class:

.. code-block:: python

            super().__init__(network=network, n_steps=NUM_EPISODE_STEPS)
        #


Training the Agents
^^^^^^^^^^^^^^^^^^^

.. figure:: /img/icons/sliders.svg
   :width: 15%
   :figclass: align-center

Training the agents is done by making use of one of RLlib's many reinforcement learning
algorithms. Phantom provides a wrapper around RLlib that hides much of the complexity.

Training in Phantom is initiated by calling the ``ph.train`` function, passing in the
parameters of the experiment. Any items given in the ``env_config`` dictionary will be
passed to the initialisation method of the environment.

The experiment name is important as this determines where the experiment results will be
stored. By default experiment results are stored in a directory named `phantom-results`
in the current user's home directory. 

There are more fields available in ``ph.train`` function than what is shown here. See
:ref:`api_utils` for full documentation.

.. code-block:: python

    ph.train(
        experiment_name="supply-chain",
        algorithm="PPO",
        num_workers=2,
        num_episodes=10000,
        env_class=SupplyChainEnv,
        env_config=dict(n_customers=5),
    )


Running The Experiment
----------------------

.. figure:: /img/icons/vial.svg
   :width: 15%
   :figclass: align-center

To run our experiment we save all of the above into a single file and run the following
command:

.. code-block:: bash

    phantom path/to/config/supply-chain-1.py

Where we substitute ``path/to/config`` for the correct path.

The ``phantom`` command is a simple wrapper around the default python interpreter but
makes sure the ``PYHTONHASHSEED`` environment variable is set which can improve
reproducibility.

In a new terminal we can monitor the progress of the experiment live with TensorBoard:

.. code-block:: bash

    tensorboard --logdir ~/phantom-results/supply-chain

Note the last element of the path matches the name we gave to our experiment in the
``ph.train`` function.

Below is a screenshot of TensorBoard. By default many plots are included providing
statistics on the experiment. You can also view the experiment progress live as it is
running in TensorBoard.

.. figure:: /img/supply-chain-1-tb.png
   :width: 100%
   :figclass: align-center

The next part of the tutorial will describe how to add your own plots to TensorBoard
through Phantom.
