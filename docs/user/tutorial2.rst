.. _tutorial2:

Tutorial - Part 2
=================

Part 1 of the tutorial showed how to set up a simple Phantom experiment. This next part
covers some additional features of Phantom that will help make your experiments even
better!


Metrics
-------

.. figure:: /img/icons/chart.svg
   :width: 15%
   :figclass: align-center

In the latter part of the previous tutorial setup we demonstrated the use of metrics for
recording data from the environment and agents.

Metric values are recorded at the end of every step. When performing training, a single
float/integer value must be returned for the whole episode so a reduction operation
must be performed. The default is to take the most recent metric value, i.e. that of the
last step in the episode. The values seen in tensorboard are the average of these
reduced values over each batch of episodes.

When performing rollouts, every value for every step is recorded, giving fine grained
information on each step in each episode.

In our supply chain example we want to monitor the average amount of stock the shop is
holding onto as the experiment progresses. Phantom provides a base :class:`Metric` class
but for the majority of use-cases the provided helper classes :class:`SimpleAgentMetric`
and :class:`SimpleEnvMetric` are enough.

.. code-block:: python

    metrics = {
        "SHOP/stock": SimpleAgentMetric(agent_id="SHOP", agent_property="stock", reduce_action="last"),
    }

The :class:`SimpleAgentMetric` will record the given property on the agent. Similarly
the :class:`SimpleEnvMetric` records a given property that exists on the environment
instance.

As well as the ``last`` reduction operation, there is also ``sum`` and ``mean``.

We register metrics using the :attr:`metrics` property on the
:func:`ph.utils.rllib.train` function. The name can be whatever the user wants however
it is sensible to include the name of the agent and the property that is being measured,
eg. ``SHOP/stock``.


Shared Policies
---------------

We will now introduce the feature of shared policy learning in Phantom using RLlib. To
do this we will create two shops that will compete with each other. Both shops will use
the same policy, for both learning and evaluation.

To keep it simple the customers will choose one of the shops at random each step -- the
shops are technically not truly competing here.

Our experiment structure will now look like the following:

.. figure:: /img/supply-chain-2.svg
   :width: 80%
   :figclass: align-center

To do this we make several modifications to the code:

*   We modify the :class:`CustomerAgent` class to accept a list of shop IDs rather than
    a single shop ID. The customer will then choose at random one of the shops to go to
    along with the existing random generation of the order quantity.

.. code-block:: python

    class CustomerAgent(ph.Agent):
        def __init__(self, agent_id: str, shop_ids: List[str]):
            super().__init__(agent_id)

            # We need to store the shop IDs so we can send order requests to them.
            self.shop_ids: List[str] = shop_ids

*   We modify the :meth:`generate_messages`: method to pick a shop at random and place an
    order at that shop each step.

.. code-block:: python

        def generate_messages(self, ctx: ph.Context):
            # At the start of each step we generate an order with a random size to send
            # to a randomly selected shop.
            order_size = np.random.randint(CUSTOMER_MAX_ORDER_SIZE)

            shop_id = np.random.choice(self.shop_ids)

            # We perform this action by sending a stock request message to the factory.
            return [(shop_id, OrderRequest(order_size))]

    #

*   We modify the environment to create multiple shop agents like we did previously with
    the customer agents. We make sure all customers are connected to all shops.

.. code-block:: python

    NUM_SHOPS = 2

    class SupplyChainEnv(ph.PhantomEnv):
        def __init__(self):
            # Define agent IDs
            factory_id = "WAREHOUSE"
            customer_ids = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]
            shop_ids = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]

            factory_agent = FactoryAgent(factory_id)
            customer_agents = [CustomerAgent(cid, shop_ids=shop_ids) for cid in customer_ids]
            shop_agents = [ShopAgent(sid, factory_id=factory_id) for sid in shop_ids]

            agents = [factory_agent] + shop_agents + customer_agents

            # Define Network and create connections between agents
            network = ph.Network(agents)

            # Connect the shops to the factory
            network.add_connections_between(shop_ids, [factory_id])

            # Connect the shop to the customers
            network.add_connections_between(shop_ids, customer_ids)


To use a shared policy we modify the :attr:`policies` argument to the
:func:`ph.utils.rllib.train()` function. Instead of passing the a list of IDs of the
agents we want to train with the ``shop_policy`` we can pass the :class:`ShopAgent`
class. This means any agent in the envrionment that belongs to this class will use share
the policy.

.. code-block:: python

    ph.utils.rllib.train(
        ...
        policies={"shop_policy": ShopAgent},
        ...
    )
    #


Modular Encoders, Decoders & Reward Functions
---------------------------------------------

.. figure:: /img/icons/th.svg
   :width: 15%
   :figclass: align-center

So far we have used the :meth:`decode_action()`, :meth:`encode_observation()` and
:meth:`compute_reward()` methods in our :class:`ShopAgent` definition. However Phantom
also provides an alternative set of interfaces for more advanced use cases. We can
create custom :class:`Encoder`, :class:`Decoder` and :class:`RewardFunction` classes
that perform the same functionality and attach them to agents.

This provides two key benefits:

*   Code reuse - Functionality that is shared across multiple agent types only has to be
    implemented once.
*   Composability - Using the :class:`ChainedEncoder` and :class:`ChainedDecoder`
    classes we can cleanly combine multiple encoders and decoders into complex objects,
    whilst keeping the individual functionality of each sub encoder separated.

Phantom :class:`StrategicAgent`s will first check to see if a custom
:meth:`decode_action()`, :meth:`encode_observation()` or :meth:`compute_reward()` method
has been implemented on the class. If not, the agent will then check to see if a custom
:class:`Encoder`, :class:`Decoder` or :class:`RewardFunction` class has been provided
for the agent. If neither is provided for any of the three, an exception will be raised!

Lets say we want to introduce a second type of :class:`ShopAgent`, one with a different
type of reward function -- this new :class:`ShopAgent` may not be concerned about the
amount of missed sales it has.

One option is to copy the entire :class:`ShopAgent` and edit its
:meth:`compute_reward()` method. However a better option is to remove the
:meth:`compute_reward()` method from the :class:`ShopAgent` and create two different
:class:`RewardFunction` objects and initialise each type of agent with one:

.. code-block:: python

    class ShopRewardFunction(ph.RewardFunction):
        def reward(self, ctx: ph.Context) -> float:
            return ctx.agent.sales - 0.1 * ctx.agent.stock

    class SimpleShopRewardFunction(ph.RewardFunction):
        def reward(self, ctx: ph.Context) -> float:
            return ctx.agent.sales

Note that we now access the :class:`ShopAgent`'s state through the :attr:`ctx.agent`
property.

We modify our :class:`ShopAgent` class so that it takes a :class:`RewardFunction` object
as an initialisation parameter and passes it to the underlying Phantom
:class:`StrategicAgent` class.

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, factory_id: str, reward_function: ph.RewardFunction):
            super().__init__(agent_id, reward_function=reward_function)

            ...

Next we modify our :class:`SupplyChainEnv` to allow the creation of a mix of shop types:

.. code-block:: python

    NUM_SHOPS_TYPE_1 = 1
    NUM_SHOPS_TYPE_2 = 1

    class SupplyChainEnv(ph.PhantomEnv):
        def __init__(self):
            ...

            shop_t1_ids = [f"SHOP_T1_{i+1}" for i in range(NUM_SHOPS_TYPE_1)]
            shop_t2_ids = [f"SHOP_T2{i+1}" for i in range(NUM_SHOPS_TYPE_2)]
            shop_ids = shop_t1_ids + shop_t2_ids

            ...

            shop_agents = [
                ShopAgent(sid, factory_id, ShopRewardFunction())
                for sid in shop_t1_ids
            ] + [
                ShopAgent(sid, factory_id, SimpleShopRewardFunction())
                for sid in shop_t2_ids
            ]

            ...


Types & Supertypes
------------------

Now let's say we want to develop a rounded policy throughout the training that works
with a range of reward functions that all slightly modify the weight of the
:attr:`stock` factor. Doing this manually would be cumbersome. Instead we can use the
Phantom types and supertypes feature.

For the :class:`ShopAgent` we define a class as a property of the shop named
:attr:`Supertype` that inherits from the :class:`ph.Supertype` class that defines the
supertype of the agent. In our case this only contains the :attr:`excess_stock_weight`
parameter we want to vary. When defining our supertype it is good practice to give all
fields a default value!

.. code-block:: python

    MAX_EXCESS_STOCK_WEIGHT = 0.2

    class ShopAgent(ph.Agent):

        @dataclass(frozen=True)
        class Supertype(ph.Supertype):
            excess_stock_weight: float = 0.1


We no longer need to pass in a custom :class:`RewardFunction` class to the
:class:`ShopAgent`:

.. code-block:: python

        def __init__(self, agent_id: str, factory_id: str):
            super().__init__(agent_id)

            ...

    #

As we are using the RLlib backend to train, we don't need to provide the
:class:`ShopAgent` with the new supertype, this is handled by the included training and
evaluation functions and allows the use of :class:`Sampler` s and :class:`Range` s.

In this example for the sake of simplicity we go back to using the
:meth:`compute_reward` method on the :class:`ShopAgent`. We modify it to take the
:attr:`excess_stock_weight` value from the agent's type:

.. code-block:: python

        def compute_reward(self, ctx: ph.Context) -> float:
            # We reward the agent for making sales.
            # We penalise the agent for holding onto excess stock.
            return self.sales - self.type.excess_stock_weight * self.stock
    #

We also need to modify the :class:`ShopAgent`'s observation space to include it's type
values. This is key to allowing the :class:`ShopAgent` to learn a generalised policy.

.. code-block:: python

        def __init__(self, agent_id: str, factory_id: str):
            ...

            # = [Stock, Sales, Missed Sales, Type.Excess Stock Weight]
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(4,))

            ...

        def encode_observation(self, ctx: ph.Context):
            max_sales_per_step = NUM_CUSTOMERS * CUSTOMER_MAX_ORDER_SIZE

            return np.array(
                [
                    self.stock / SHOP_MAX_STOCK,
                    self.sales / max_sales_per_step,
                    self.missed_sales / max_sales_per_step,
                    self.type.excess_stock_weight / MAX_EXCESS_STOCK_WEIGHT,
                ],
                dtype=np.float32,
            )

        @property
        def observation_space(self):
            return gym.spaces.Tuple(
                [
                    # We include the agent's type in it's observation space to allow it to learn
                    # a generalised policy.
                    self.type.to_obs_space(),
                    # We also encode the shop's current stock in the observation.
                    gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
                ]
            )
    #


To sample from a distribution of values for the supertypes whilst training we add the
:attr:`agent_supertypes` argument to the train function:

.. code-block:: python

    ph.utils.rllib.train(
        ...
        agent_supertypes={
            "SHOP1": {"excess_stock_weight": UniformFloatSampler(0.0, MAX_EXCESS_STOCK_WEIGHT)},
            "SHOP2": {"excess_stock_weight": UniformFloatSampler(0.0, MAX_EXCESS_STOCK_WEIGHT)},
        },
        ...
    )

At the start of each episode in training, each shop agent's :attr:`excess_stock_weight`
type value will be independently sampled from a random uniform distribution between 0.0
and 0.2.

The supertype system in Phantom is very powerful. To see a full guide to its features
see the :ref:`supertypes` page.

.. TODO: FSM, debugging, policy evaluation, advanced resolvers
