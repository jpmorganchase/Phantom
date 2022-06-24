.. _tutorial2:

Tutorial - Part 2
=================

Part 1 of the tutorial showed how to set up a simple Phantom experiment. This next part
covers some additional features of Phantom that will help make your experiments even
better!

The complete finished code for this tutorial can be found in the ``envs`` directory in
the Phantom repo.


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

Metric values are recorded at the end of every step. When performing training, a single
float/integer value must be returned for the whole episode so a reduction operation
must be performed. The default is to take the most recent metric value, i.e. that of the
last step in the episode. The values seen in tensorboard are the average of these
reduced values over each batch of episodes.

When performing rollouts, every value for every step is recorded, giving fine grained
information on each step in each episode.

In our supply chain example we want to monitor the average amount of stock the shop is
holding onto as the experiment progresses. Phantom provides a base ``Metric`` class but
for a lot of use-cases the provided helper classes ``SimpleAgentMetric`` and
``SimpleEnvMetric`` are enough.

.. code-block:: python

    metrics = {
        "stock/SHOP": SimpleAgentMetric(agent_id="SHOP", agent_property="stock", reduce_action="last"),
    }

The ``SimpleAgentMetric`` will record the given property on the agent. Similarly the
``SimpleEnvMetric`` records a given property that exists on the Environment instance.

As well as the 'last' reduction operation, there is also 'sum' and 'mean'.

We register metrics using the ``metrics`` property on the ``ph.train`` function.
The name can be whatever the user wants however it is sensible to include the name of
the agent and the property that is being measured, eg. ``stock/SHOP``.

.. code-block:: python

    ph.train(
        experiment_name="supply-chain",
        algorithm="PPO",
        num_workers=15,
        num_episodes=1000,
        env_class=SupplyChainEnv,
        env_config=dict(n_customers=5),
        metrics=metrics,
    )


If we run the experiment and go to TensorBoard we can now see our stock Metric plotted.
TensorBoard will plot the min, mean and max values for each metric.

.. figure:: /img/supply-chain-2-tb.png
   :width: 50%
   :figclass: align-center

In the full example code there is also a ``SalesMetric`` and a ``MissedSalesMetric``
included.


Shared Policies
---------------

We will now introduce multiple competing shop agents. Our experiment structure will now
look like the following:

.. figure:: /img/supply-chain-2.svg
   :width: 80%
   :figclass: align-center

To do this we make several modifications to the code:

* We modify the ``CustomerAgent`` to accept a list of shop IDs rather than a single
  shop ID. The policy will be expanded to also decide which shop to allocate orders to.
  The action space of the policy will now be of size 2: the order size and shop index.

.. code-block:: python

    class CustomerAgent(ph.Agent):
        def __init__(self, agent_id: str, shop_ids: List[str]):
            super().__init__(
                agent_id,
                policy_class=CustomerPolicy,
                # The CustomerPolicy needs to know how many shops there are so it can
                return a valid choice.
                policy_config=dict(n_shops=len(shop_ids)),
            )

            # We need to store the shop IDs so we can send order requests to them.
            self.shop_ids: List[str] = shop_ids

* We change the ``decode_action`` method to pick a shop at random and place an order at
  that shop each step.

.. code-block:: python

        def decode_action(self, ctx: ph.Context, action: np.ndarray):
            # At the start of each step we generate an order with a random size to
            # send to a random shop.
            order_size = action[0]
            shop_id = self.shop_ids[int(action[1])]

            # We perform this action by sending a stock request message to the factory.
            return ph.packet.Packet(messages={shop_id: [OrderRequest(order_size)]})

    #

* We modify the ``CustomerPolicy`` class to accept the list of shop ID's now given to it
  from the ``CustomerAgent`` and make a random selection on which shop to choose:

.. code-block:: python

    class CustomerPolicy(ph.FixedPolicy):
        # The size of the order made and the choice of shop to make the order to for each
        # customer is determined by this fixed policy.
        def __init__(self, obs_space, action_space, config):
            super().__init__(obs_space, action_space, config)

            self.n_shops = config["n_shops"]

        def compute_action(self, obs) -> Tuple[int, int]:
            return (np.random.poisson(5), np.random.randint(self.n_shops))

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
            factory_id = "WAREHOUSE"
            shop_ids = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]
            customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

            shop_agents = [ShopAgent(sid, factory_id=factory_id) for sid in shop_ids]
            factory_actor = FactoryActor(factory_id)

            customer_agents = [CustomerAgent(cid, shop_ids=shop_ids) for cid in customer_ids]

            actors = [factory_actor] + shop_agents + customer_agents

            # Define Network and create connections between Actors
            network = ph.Network(actors)

            # Connect the shops to the factory
            network.add_connections_between(shop_ids, [factory_id])

            # Connect the shop to the customers
            network.add_connections_between(shop_ids, customer_ids)


Now we have multiple learning shop agents, we may want them to learn a shared policy. By
default each shop will learn it's own policy. To setup a shared policy we simply pass in
a ``policy_grouping`` argument to the ``PhantomEnv.__init__`` method giving for each
shared policy the name of the policy and the IDs of the agents that will learn the
policy:

.. code-block:: python

            super().__init__(
                num_steps=NUM_EPISODE_STEPS,
                network=network,
                policy_grouping=dict(
                    shared_SHOP_policy=shop_ids
                ),
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
        def reward(self, ctx: ph.Context) -> float:
            return ctx.actor.step_sales - ctx.actor.step_missed_sales - ctx.actor.stock

    class SimpleShopRewardFunction(ph.RewardFunction):
        def reward(self, ctx: ph.Context) -> float:
            return ctx.actor.step_sales - ctx.actor.stock

Note that we now access the ``ShopAgent``'s state through the ``ctx.actor`` variable.

We modify our ``ShopAgent`` class so that it takes a ``RewardFunction`` object as an
initialisation parameter and passes it to the underlying Phantom ``Agent`` class.

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, factory_id: str, reward_function: ph.RewardFunction):
            super().__init__(agent_id, reward_function=reward_function)

            ...

Next we modify our ``SupplyChainEnv`` to allow the creation of a mix of shop types:

.. code-block:: python

    class SupplyChainEnv(ph.PhantomEnv):

        env_name: str = "supply-chain-v2"

        def __init__(self, n_customers: int = 5,):
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
``missed_sales`` factor. Doing this manually would be cumbersome. Instead we can use the
Phantom supertypes feature.

For the ``ShopAgent`` we define an object that inherits from the ``BaseSupertype`` class
that defines the type of the agent. In our case this only contains the
``missed_sales_weight`` parameter we want to vary. When defining our supertype, to
satisfy the type system, the types of all fields should be wrapped in
``ph.SupertypeField``.

.. code-block:: python

    @dataclass
    class ShopAgentSupertype(ph.BaseSupertype):
        missed_sales_weight: ph.SupertypeField[float]


We no longer need to pass in a custom ``RewardFunction`` class to the ``ShopAgent``:

.. code-block:: python

    class ShopAgent(ph.Agent):
        def __init__(self, agent_id: str, factory_id: str):
            super().__init__(agent_id)

            ...

We don't even need to provide the ``ShopAgent`` with the new supertype, this is handled
by the ``ph.train`` and ``ph.rollout`` functions.

However we do need to modify our ``ShopRewardFunction`` to take the\
``missed_sales_weight`` parameter:

.. code-block:: python

    class ShopRewardFunction(ph.RewardFunction):
        def __init__(self, missed_sales_weight: float):
            self.missed_sales_weight = missed_sales_weight

        def reward(self, ctx: ph.Context) -> float:
            return 5 * ctx.actor.step_sales - self.missed_sales_weight * \
                ctx.actor.step_missed_sales - ctx.actor.stock

We also need to modify the ``ShopAgent``'s observation space to include it's type values.
This is key to allowing the ``ShopAgent`` to learn a generalised policy.

.. code-block:: python

        def encode_observation(self, ctx: ph.Context):
            return [
                # We include the agent's type in it's observation space to allow it to learn
                # a generalised policy.
                self.type.to_obs_space_compatible_type(),
                # We also encode the shop's current stock in the observation.
                self.stock,
            ]

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
generates a new type instance from the supertype that will be assigned during training
or rollouts. We then make use of the type to setup the agent. The ``reset`` method is
called by the environment at the start of every episode.

We can then pass the following to the ``agent_supertypes`` parameter in the ``ph.train``
function;

.. code-block:: python

    agent_supertypes = {
        ShopAgentSupertype(
            missed_sales_weight=UniformSampler(0.0, 8.0)
        )
        for sid in shop_ids
    })

    ph.train(
        experiment_name="supply-chain",
        algorithm="PPO",
        num_workers=2,
        num_episodes=10000,
        env_class=SupplyChainEnv,
        env_config=dict(n_customers=5),
        agent_supertypes=agent_supertypes,
    )

At the start of each episode in training, each shop agent's missed_sales_weight type
value will be independently sampled from a random uniform distribution between 0.0 and
1.0.

The supertype system in Phantom is very powerful. To see a full guide to its features
see the :ref:`_supertypes` page.


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
        """Shop --> Factory"""
        size: int

    @dataclass
    class StockResponse:
        """Factory --> Shop"""
        size: int


This allows us to use the type system to increase the clarity of our code and reduce
errors.

To update our code we simply wrap the values in their new payload types, for example:

.. code-block:: python

    class FactoryActor(ph.Agent):
        def __init__(self, actor_id: str):
            super().__init__(actor_id)

        def handle_message(self, ctx: ph.Context, msg: ph.Message):
            # The factory receives stock request messages from shop agents. We
            # simply reflect the amount of stock requested back to the shop as the
            # factory has unlimited stock.
            yield (msg.sender_id, [StockResponse(msg.payload.size)])

Now it is clear to see exactly what is being returned by the ``FactoryActor``.

This now allows us to use another feature of Phantom: Custom Handlers. If we have an
actor or agent that accepts many types of message, we would need to route all these
message types in our ``handle_message`` method so that we take the correct actions for
each message.

Custom Handlers does this automatically for us! Taking the very simple example above,
we can replace our ``handle_message`` method of the ``FactoryActor`` with a new method
that is prefixed with the ``@ph.agents.msg_handler`` decorator. In this decorator we pass
the type of the message payload we want to handle:

.. code-block:: python

        @ph.agents.msg_handler(StockRequest)
        def handle_stock_request(self, ctx: ph.Context, msg: ph.Message):
            # The factory receives stock request messages from shop agents. We
            # simply reflect the amount of stock requested back to the shop as the
            # factory has unlimited stock.
            yield (msg.sender_id, [StockResponse(msg.payload.size)])
    #

We can define as many of these handlers as we want. See the code example for a full
implementation of this.
