.. _tutorial2:

Tutorial - Part 2
=================

Part 1 of the tutorial showed how to set up a simple Phantom experiment. This next part
covers some additional features of Phantom that will help make your experiments even
better!

The complete finished code for this tutorial can be found in the ``supply-chain-2.py``
script in the ``phantom-environments`` repository.

..
    TODO: add link to phantom-environments repository


Metrics
-------

.. figure:: /img/icons/chart.svg
   :width: 15%
   :figclass: align-center

In the last part of the previous tutorial we demonstrated the use of TensorBoard for
viewing the results of our experiment. A lot of the time we want to view additional
plots of experiment data. Phantom provides a way to do this through the use of
``Metrics``. These provide a way to extract data from the experiment and save it to the
results directory so it can be loaded by TensorBoard and other offline analysis.

In our supply chain example we want to monitor the average amount of stock the shop is
holding onto as the experiment progresses. To do this we create a new subclass of the
Phantom ``Metric`` class:

.. code-block:: python

    class StockMetric(ph.logging.Metric[float]):
        def __init__(self, agent_id: str) -> None:
            self.agent_id: str = agent_id

        def extract(self, env: ph.PhantomEnv) -> float:
            return env[self.agent_id].stock

We can see that this metric is designed to record from a single agent as we require an
``agent_id`` parameter in the initialisation method. This is a commonly used pattern.

Metrics require just one method, called ``extract``, that takes the entire environment,
takes the desired value from the environment and returns it. Here we take our agent and
get its ``stock`` property.

Metrics should not make any modification to the environment or any objects within it -
they should be completely passive.

We then register our metric using the ``metrics`` property on the ``PhantomParams``
object. The name can be whatever the user wants however it is sensible to include the
name of the agent and the property that is being measured, eg. ``stock/SHOP``.

.. code-block:: python

    phantom_params = ph.PhantomParams(
        experiment_name="supply-chain",
        algorithm="PPO",
        num_workers=15,
        num_episodes=1000,
        env=SupplyChainEnv,
        env_config={
            "n_customers": 5,
            "seed": 0,
        },
        metrics={
            "stock/SHOP": StockMetric("SHOP"),
        }
    )


If we run the experiment and go to TensorBoard we can now see our stock Metric plotted.
TensorBoard will plot the min, mean and max values for each metric.

.. figure:: /img/supply-chain-2-tb.png
   :width: 50%
   :figclass: align-center

In the full example code there is also a ``SalesMetric`` and a ``MissedSalesMetric``
included.


Clock
-----

.. figure:: /img/icons/clock.svg
   :width: 15%
   :figclass: align-center

In part 1, when we passed the ``n_steps`` parameter to the ``PhantomEnv.__init__``
method, behind the scenes the ``PhantomEnv`` class created a Clock object to keep track
of time. By default this uses whole integer steps from 0..n_steps.

In some cases we may want to use a different time step size such as a datetime or we may
want to allow agent or actor to be able to keep track of time. This is useful when an
agent or actors behaviour is a function of time.

We can do this by creating our own Clock object and then passing it to the
``PhantomEnv.__init__`` method instead of the ``n_steps`` value:

.. code-block:: python

    class SupplyChainEnv(ph.PhantomEnv):

        env_name: str = "supply-chain-v2"

        def __init__(self, n_customers: int = 5, seed: int = 0):
            ...

            clock = ph.Clock(0, NUM_EPISODE_STEPS, 1)

            super().__init__(
                network=network,
                clock=clock,
                seed=seed,
            )

Shared Policies
---------------

We will now introduce multiple competing shop agents. Our experiment structure will now
look like the following:

.. figure:: /img/supply-chain-2.svg
   :width: 80%
   :figclass: align-center

To do this we make several modifications to the code:

* We modify the ``CustomerAgent`` to accept a list of shop IDs rather than a single
  shop ID. We also change the ``decode_action`` method to pick a shop at random and
  place an order at that shop each step.

.. code-block:: python

    class CustomerAgent(ph.ZeroIntelligenceAgent):
        def __init__(self, agent_id: str, shop_ids: List[str]):
            super().__init__(agent_id)

            # We need to store the shop IDs so we can send order requests to them.
            self.shop_ids: List[str] = shop_ids

        def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
            # At the start of each step we generate an order with a random size to
            # send to a random shop.
            order_size = np.random.poisson(5)

            shop_id = np.random.choice(self.shop_ids)

            # We perform this action by sending a stock request message to the warehouse.
            return ph.packet.Packet(messages={shop_id: [order_size]})

        ...

* We modify the environment to create multiple shop agents like we did previously with
  the customer agents. We make sure all customers are connected to all shops.

  NOTE: as the shops are active learning agents, we cannot define the number to create
  via the environment initialisation method like we do with the customers. This is
  because the number of learning agents must be hardcoded so the algorithm can train the
  policy.

.. code-block:: python

    class SupplyChainEnv(ph.PhantomEnv):

        env_name: str = "supply-chain-v2"

        def __init__(self, n_customers: int = 5, seed: int = 0):
            # Define actor and agent IDs
            warehouse_id = "WAREHOUSE"
            shop_ids = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]
            customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

            shop_agents = [ShopAgent(sid, warehouse_id=warehouse_id) for sid in shop_ids]
            warehouse_actor = WarehouseActor(warehouse_id)

            customer_agents = [CustomerAgent(cid, shop_ids=shop_ids) for cid in customer_ids]

            actors = [warehouse_actor] + shop_agents + customer_agents

            # Define Network and create connections between Actors
            network = me.Network(me.resolvers.UnorderedResolver(), actors)

            # Connect the shops to the warehouse
            network.add_connections_between(shop_ids, [warehouse_id])

            # Connect the shop to the customers
            network.add_connections_between(shop_ids, customer_ids)


Now we have multiple learning shop agents, we may want them to learn a shared policy. By
default each shop will learn it's own policy. To setup a shared policy we simply pass in
a ``policy_grouping`` argument to the ``PhantomEnv.__init__`` method giving for each
shared policy the name of the policy and the IDs of the agents that will learn the
policy:

.. code-block:: python

            super().__init__(
                network=network,
                clock=clock,
                seed=seed,
                policy_grouping={
                    "shared_SHOP_policy": shop_ids,
                },
            )
    #


Modular Encoders, Decoders & Reward Functions
---------------------------------------------

.. figure:: /img/icons/th.svg
   :width: 15%
   :figclass: align-center

So far we have used the ``decode_action``, ``encode_obs`` and ``compute_reward`` methods
in our agent definitions. However Phantom also provides an alternative to this for more
advanced use cases. We can create custom ``Encoder``, ``Decoder`` and ``RewardFunction``
classes that perform the same functionality and attach them to agents.

This provides two key benefits:

* Code reuse - Functionality that is shared across multiple agent types only has to be
  implemented once.
* Composability - Using the ``ChainedEncoder`` and ``ChainedDecoder`` classes we can
  cleanly combine multiple encoders and decoders into complex objects, whilst keeping
  the individual functionality of each sub encoder separated.

Phantom agents will first check to see if a custom ``encode_obs``, ``decode_action`` or
``compute_reward`` method has been implemented on the class. If not, the agent will then
check to see if a custom ``Encoder``, ``Decoder`` or ``RewardFunction`` class has been
provided for the agent. If neither is provided for any of the three, an exception will
be raised!

Lets say we want to introduce a second type of ShopAgent, one with a different type of
reward function - this new ShopAgent may not be concerned about the amount of missed
sales it has.

One option is to copy the entire ShopAgent and edit its ``compute_reward`` method.
However a better option is to remove the ``compute_reward`` method from the ShopAgent
and create two different ``RewardFunction`` objects and initialise each type of agent
with one:

.. code-block:: python

    class ShopRewardFunction(ph.RewardFunction):
        def reward(self, ctx: me.Network.Context) -> float:
            return 5 * ctx.actor.step_sales - ctx.actor.step_missed_sales - ctx.actor.stock

    class SimpleShopRewardFunction(ph.RewardFunction):
        def reward(self, ctx: me.Network.Context) -> float:
            return 5 * ctx.actor.step_sales - ctx.actor.stock

Note that we now access the ``ShopAgent``'s state through the ``ctx.actor`` variable.

We modify our ``ShopAgent`` class so that it takes a ``RewardFunction`` object as an
initialisation parameter and passes it to the underlying Phantom ``Agent`` class.

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, warehouse_id: str, reward_function: ph.RewardFunction):
            super().__init__(agent_id, reward_function=reward_function)

            ...

Next we modify our ``SupplyChainEnv`` to allow the creation of a mix of shop types:

.. code-block:: python

    class SupplyChainEnv(ph.PhantomEnv):

        env_name: str = "supply-chain-v2"

        def __init__(
            self,
            n_customers: int = 5,
            seed: int = 0,
        ):
            ...

            shop_t1_ids = [f"SHOP_T1_{i+1}" for i in range(NUM_SHOPS_TYPE_1)]
            shop_t2_ids = [f"SHOP_T2{i+1}" for i in range(NUM_SHOPS_TYPE_2)]
            shop_ids = shop_t1_ids + shop_t2_ids

            ...

            shop_agents = [
                ShopAgent(sid, warehouse_id, ShopRewardFunction())
                for sid in shop_t1_ids
            ] + [
                ShopAgent(sid, warehouse_id, SimpleShopRewardFunction())
                for sid in shop_t2_ids
            ]

            ...


Types & Supertypes
------------------

Now let's say we want to develop a rounded policy throughout the training that works
with a range of reward functions that all slightly modify the weight of the
``missed_sales`` factor. Doing this manually would be cumbersome. Instead we can use the
Phantom supertypes feature.

For the ``ShopAgent`` we define a ``AgentType`` object that defines the type of the
agent. In our case this only contains the ``missed_sales_weight`` parameter we want to
vary.

.. code-block:: python

    @dataclass
    class ShopAgentType(ph.AgentType):
        missed_sales_weight: float

Then we define a ``Supertype`` object that has a ``sample`` method that produces an
instance of the ``ShopAgentType`` we previously defined. In this we randomly sample a
value for the ``missed_sales_weight`` parameter.

.. code-block:: python

    class ShopAgentSupertype(ph.Supertype):
        def __init__(self):
            self.missed_sales_weight_low = 0.5
            self.missed_sales_weight_high = 3.0

        def sample(self) -> ShopAgentType:
            return ShopAgentType(
                missed_sales_weight=np.random.uniform(
                    self.missed_sales_weight_low,
                    self.missed_sales_weight_high
                )
            )

We attach this supertype to our agent by passing it in as an initialisation parameter.
This then gets passed to the base ``Agent``:

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, warehouse_id: str, supertype: ph.Supertype):
            super().__init__(agent_id, supertype=supertype)

            ...

Note how we no longer need to pass in a custom ``RewardFunction`` class in here anymore.

However we need to modify our ``ShopRewardFunction`` to take the ``missed_sales_weight``
parameter:

.. code-block:: python

    class ShopRewardFunction(ph.RewardFunction):
        def __init__(self, missed_sales_weight: float):
            self.missed_sales_weight = missed_sales_weight

        def reward(self, ctx: me.Network.Context) -> float:
            return 5 * ctx.actor.step_sales - self.missed_sales_weight * \
                ctx.actor.step_missed_sales - ctx.actor.stock

The final step is to modify the ``ShopAgent``'s ``reset`` method to apply the supertype:

.. code-block:: python

    class ShopAgent(ph.Agent):

        ...

        def reset(self) -> None:
            super().reset() # self.type set here

            self.reward_function = ShopRewardFunction(
                missed_sales_weight=self.type.missed_sales_weight
            )

            ...

What is happening here is that when we call ``super().reset()``, the ``Agent`` class
generates a new type instance from the supertype we set earlier. We then make use of the
type to setup the agent. The ``reset`` method is called by the environment at the start
of every episode.


Messages & Custom Handlers
--------------------------

Up until now we have sent our messages across the network in a very basic fashion - we
have sent raw integers representing requests and responses to stock and orders. In our
simple example this is manageable, however if we scale our experiment and increase its
complexity things can get out of hand quickly!

The first step we can take to make things more manageable is to create specific payload
classes for each message type:

.. code-block:: python

    @dataclass
    class OrderRequest:
        """Customer --> Shop"""
        size: int

    @dataclass
    class OrderResponse:
        """Shop --> Customer"""
        size: int

    @dataclass
    class StockRequest:
        """Shop --> Warehouse"""
        size: int

    @dataclass
    class StockResponse:
        """Warehouse --> Shop"""
        size: int


This allows us to use the type system to increase the clarity of our code and reduce
errors.

To update our code we simply wrap the values in their new payload types, for example:

.. code-block:: python

    class WarehouseActor(me.actors.SimpleSyncActor):
        def __init__(self, actor_id: str):
            super().__init__(actor_id)

        def handle_message(self, ctx: me.Network.Context, msg: me.Message):
            # The warehouse receives stock request messages from shop agents. We
            # simply reflect the amount of stock requested back to the shop as the
            # warehouse has unlimited stock.
            yield (msg.sender_id, [StockResponse(msg.payload.size)])

Now it is clear to see exactly what is being returned by the ``WarehouseActor``.

This now allows us to use another feature of Phantom: Custom Handlers. If we have an
actor or agent that accepts many types of message, we would need to route all these
message types in our ``handle_message`` method so that we take the correct actions for
each message.

Custom Handlers does this automatically for us! Taking the very simple example above,
we can replace our ``handle_message`` method of the ``WarehouseActor`` with a new method
that is prefixed with the ``@me.actors.handler`` decorator. In this decorator we pass
the type of the message payload we want to handle:

.. code-block:: python

        @me.actors.handler(StockRequest)
        def handle_stock_request(self, ctx: me.Network.Context, msg: me.Message):
            # The warehouse receives stock request messages from shop agents. We
            # simply reflect the amount of stock requested back to the shop as the
            # warehouse has unlimited stock.
            yield (msg.sender_id, [StockResponse(msg.payload.size)])
    #

We can define as many of these handlers as we want. See the code example for a full
implementation of this.


Ray Tune Hyperparameters
------------------------

TODO
