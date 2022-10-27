"""
A simple logistics themed environment used for demonstrating the features of Phantom.
"""

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import phantom as ph


# TODO: find better way of doing this in Phantom


class UnitArrayLinspaceRange(
    ph.utils.ranges.LinspaceRange, ph.utils.ranges.Range[np.ndarray]
):
    def values(self) -> np.ndarray:
        return [np.array([x]) for x in np.linspace(self.start, self.end, self.n)]


class UnitArrayUniformRange(
    ph.utils.ranges.UniformRange, ph.utils.ranges.Range[np.ndarray]
):
    def values(self) -> np.ndarray:
        return [np.array([x]) for x in np.arange(self.start, self.end, self.step)]


# Define fixed parameters:
NUM_EPISODE_STEPS = 100
NUM_SHOPS = 2
NUM_CUSTOMERS = 5

CUSTOMER_MAX_ORDER_SIZE = 6
SHOP_MIN_PRICE = 0.0
SHOP_MAX_PRICE = 2.0
COST_PRICE = 1.0
SHOP_MAX_STOCK = 50
SHOP_MAX_STOCK_REQUEST = int((CUSTOMER_MAX_ORDER_SIZE - 4) * NUM_CUSTOMERS * 1.5)
MAX_INITIAL_INVENTORY = 10
ALLOW_PRICE_ACTION = True


@dataclass(frozen=True)
class OrderRequest(ph.MsgPayload):
    """Customer --> Shop"""

    size: int


@dataclass(frozen=True)
class OrderResponse(ph.MsgPayload):
    """Shop --> Customer"""

    size: int


@dataclass(frozen=True)
class StockRequest(ph.MsgPayload):
    """Shop --> Factory"""

    size: int


@dataclass(frozen=True)
class StockResponse(ph.MsgPayload):
    """Factory --> Shop"""

    size: int


class CustomerPolicy(ph.Policy):
    # The size of the order made and the choice of shop to make the order to for each
    # customer is determined by this fixed policy.
    def tie_breaker_idx_sort(self, array):
        def cmp(x, y):
            if x[1] < y[1]:
                return -1
            elif x[1] > y[1]:
                return 1
            return np.random.randint(2) * 2 - 1
        
        array = sorted(enumerate(array), key=functools.cmp_to_key(cmp))
        return [x[0] for x in array]
    
    def compute_action(self, obs: np.ndarray) -> Tuple[int, int]:
        # The size of the order made and the choice of shop to make the order to for each
        # customer is determined by this fixed policy. If both the shops have same price, customers can randomly place order to the one shop or the other
        return (np.random.randint(1, CUSTOMER_MAX_ORDER_SIZE), self.tie_breaker_idx_sort(obs)[0])
   


class CustomerAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_ids: List[ph.AgentID]):
        super().__init__(agent_id)

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_ids: List[str] = shop_ids

        self.action_space = gym.spaces.Tuple(
            (
                # The number of items to order
                gym.spaces.Discrete(CUSTOMER_MAX_ORDER_SIZE),
                # The shop to order from
                gym.spaces.Discrete(len(self.shop_ids)),
            )
        )

        # The price set by each shop
        self.observation_space = gym.spaces.Box(
            low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(len(self.shop_ids),)
        )
        
    def pre_message_resolution(self, ctx: ph.Context):
        # a counter to keep track of the shops the customer have already visited.
        self.shopsvisited = 0 

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        #to track the customers that could not get their orders fulfilled so that they can be allowed to the next shop offering the best price
        self.shopsvisited += 1
        
        stock_sold = message.payload.size
        if stock_sold == 0 and self.shopsvisited < 2:
            self.customer_notsatisfied = message.receiver_id
            self.shop_withnostock = message.sender_id
           
            if message.sender_id == 'SHOP1':
                return [('SHOP2', OrderRequest(self.order_size))]
            else:
                return [('SHOP1', OrderRequest(self.order_size))]
           
        else:
            return

    def decode_action(self, ctx: ph.Context, action: Tuple[int, int]):
        # At the start of each step we generate an order with a random size to send to a
        # random shop.
        order_size, selected_shop = action
        shop_id = self.shop_ids[selected_shop]

        # We perform this action by sending a stock request message to the factory.
        return [(shop_id, OrderRequest(order_size))]

    def encode_observation(self, ctx: ph.Context):
        return np.array(
            [ctx[shop_id].price for shop_id in self.shop_ids], dtype=np.float32
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0.0


class FactoryAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    @ph.agents.msg_handler(StockRequest)
    def handle_stock_request(self, ctx: ph.Context, message: ph.Message):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        return [(message.sender_id, StockResponse(message.payload.size))]


class ShopAgent(ph.MessageHandlerAgent):
    @dataclass
    class Supertype(ph.Supertype):
        #sale_price: float
        # The cost of holding onto one unit of inventory overnight:
        cost_of_carry: float
        # The cost of producing one unit of inventory:
        #cost_per_unit: float

    @dataclass(frozen=True)
    class View(ph.AgentView):
        # The shop broadcasts its price to customers so they can choose which shop to
        # purchase from:
        price: float

    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has:
        self.stock: int = 0

        # How many sales have been made:
        self.sales: int = 0

        # How many orders the shop has missed due to not having enough stock:
        self.missed_sales: int = 0

        # How many items have been delivered by the factory in this turn:
        self.delivered_stock: int = 0

        # How many items of stock is left after customer sales
        self.leftover_stock: int = 0

        # How many items are carried to the next day (leftover + restock from prev. day)
        self.carried_stock = 0

        # How many orders in units of product
        self.orders_received = 0

        # Reward function sub-components:
        self.pnl = 0
        self.revenue: float = 0.0
        self.costs: float = 0.0

        # We initialise the price variable here, it's value will be set when the shop
        # agent takes it's first action.
        self.price: float = SHOP_MAX_PRICE

        self.initial_inventory: Optional[int] = None

    @property
    def action_space(self):
        if ALLOW_PRICE_ACTION:
            return gym.spaces.Dict(
                {
                    # The price to set for the current step:
                    "price": gym.spaces.Box(
                        low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(1,)
                    ),
                    # The number of additional units to order from the factory:
                    "restock_qty": gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST),
                }
            )
        else:
            return gym.spaces.Dict(
                {
                    # "restock_qty": gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST),
                    "restock_qty": gym.spaces.Box(
                        low=0, high=SHOP_MAX_STOCK_REQUEST, shape=(1,)
                    ),
                }
            )

    @property
    def observation_space(self):
        return gym.spaces.Dict(
            {
                # The agent's type:
                "type": self.type.to_obs_space(),
                # The agent's current stock:
                "stock": gym.spaces.Box(low=0, high=1, shape=(1,)),
                # The number of sales made by the agent in the previous step:
                "previous_sales": gym.spaces.Box(low=0, high=1, shape=(1,)),
                # The number of orders placed by customers in the previous step:
                "previous_cusorders": gym.spaces.Box(low=0, high=1, shape=(1,)),
            }
        )
    
    def handle_batch(self, ctx, batch):
        np.random.shuffle(batch)
        return list(
            chain.from_iterable(
                filter(
                    lambda x: x is not None,
                    (self.handle_message(ctx, message) for message in batch),
                )
            )
        )

    def view(self, neighbour_id: Optional[ph.AgentID] = None) -> "View":
        """Return an immutable view to the agent's public state."""
        return self.View(self.id, self.price)

    def pre_message_resolution(self, ctx: ph.Context):
        if ctx["ENV"].stage == "sales_step":
            # At the start of each step we reset the number of missed orders to 0.
            self.sales = 0
            self.missed_sales = 0
            self.orders_received = 0

            self.carried_stock = self.stock

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the factory contain stock.

        self.delivered_stock = message.payload.size

        self.stock = min(self.stock + self.delivered_stock, SHOP_MAX_STOCK)

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(self, ctx: ph.Context, message: ph.Message):
        amount_requested = message.payload.size

        self.orders_received += amount_requested

        # If the order size is more than the amount of stock, partially fill the order.
        if amount_requested > self.stock:
            self.missed_sales += amount_requested - self.stock
            stock_to_sell = self.stock
            self.stock = 0
        # ... Otherwise completely fill the order.
        else:
            stock_to_sell = amount_requested
            self.stock -= amount_requested

        self.sales += stock_to_sell
        self.leftover_stock = self.stock

        # Send the customer their order.
        return [(message.sender_id, OrderResponse(stock_to_sell))]

    def encode_observation(self, ctx: ph.Context):
        return {
            # The shop's type is included in its observation space to allow it to learn
            # a generalised policy.
            "type": self.type.to_obs_space_compatible_type(),
            # The current stock is included so the shop can learn to efficiently manage
            # its inventory.
            "stock": np.array([self.stock]) / SHOP_MAX_STOCK,
            # The number of sales made in the previous step is included so the shop can
            # learn to maximise sales.
            "previous_sales": np.array([self.sales]) / (CUSTOMER_MAX_ORDER_SIZE - 1),
            # The number of customer orders placed in the previous step
            "previous_cusorders": np.array([self.orders_received]) / ((CUSTOMER_MAX_ORDER_SIZE - 1)*(NUM_CUSTOMERS))
        }

    def decode_action(self, ctx: ph.Context, action: Tuple[np.ndarray, int]):
        # The action the shop takes is the updated price for it's products and the
        # amount of new stock to request from the factory.

        if ALLOW_PRICE_ACTION:
            # We update the shop's price:
            self.price = action["price"][0]
        else:
            self.price = self.type.sale_price

        # And we send a stock request message to the factory:
        return [(self.factory_id, StockRequest(int(np.round(action["restock_qty"][0]))))]

    def compute_reward(self, ctx: ph.Context) -> float:
        #sales made by selling stock
        self.revenue = self.sales * self.price
        self.costs = (
            # It incurs a cost for ordering new stock:
            self.delivered_stock * COST_PRICE    
            # And for holding onto leftover stock overnight:
            + self.stock * self.type.cost_of_carry     
        )
        self.pnl = self.revenue - self.costs
        return self.pnl

    def reset(self):
        super().reset()  # sampled supertype is set as self.type here

        if not ALLOW_PRICE_ACTION:
            self.price = self.type.sale_price

        if self.initial_inventory is None:
            self.stock = np.random.randint(MAX_INITIAL_INVENTORY)
        else:
            self.stock = self.initial_inventory


# Define agent IDs:
FACTORY_ID = "FACTORY"
SHOP_IDS = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]
CUSTOMER_IDS = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]


class SupplyChainEnv(ph.FiniteStateMachineEnv):
    def __init__(self, initial_inventory: Optional[int] = None, **kwargs):
        shop_agents = [ShopAgent(id, FACTORY_ID) for id in SHOP_IDS]

        if initial_inventory is not None:
            shop_agents[0].initial_inventory = initial_inventory

        factory_agent = FactoryAgent(FACTORY_ID)

        customer_agents = [CustomerAgent(id, shop_ids=SHOP_IDS) for id in CUSTOMER_IDS]

        agents = [factory_agent] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents, resolver=ph.resolvers.BatchResolver(chain_limit=4))

        # Connect the shops to the factory
        network.add_connections_between(SHOP_IDS, [FACTORY_ID])

        # Connect the shop to the customers
        network.add_connections_between(SHOP_IDS, CUSTOMER_IDS)

        super().__init__(
            num_steps=NUM_EPISODE_STEPS,
            network=network,
            initial_stage="sales_step",
            stages=[
                ph.FSMStage(
                    stage_id="sales_step",
                    next_stages=["restock_step"],
                    acting_agents=CUSTOMER_IDS,
                    rewarded_agents=SHOP_IDS,
                ),
                ph.FSMStage(
                    stage_id="restock_step",
                    next_stages=["sales_step"],
                    acting_agents=SHOP_IDS,
                    rewarded_agents=[],
                ),
            ],
            **kwargs,
        )


metrics = {}

for id in SHOP_IDS:
    # metrics[f"{id}/initial_inventory"] = ph.logging.SimpleAgentMetric(id, "type.initial_inventory", "mean")
    metrics[f"{id}/price"] = ph.logging.SimpleAgentMetric(id, "price", "mean")
    metrics[f"{id}/stock"] = ph.logging.SimpleAgentMetric(id, "stock", "mean")
    metrics[f"{id}/sales"] = ph.logging.SimpleAgentMetric(id, "sales", "mean")
    metrics[f"{id}/orders_received"] = ph.logging.SimpleAgentMetric(
        id, "orders_received", "mean"
    )
    metrics[f"{id}/missed_sales"] = ph.logging.SimpleAgentMetric(
        id, "missed_sales", "mean"
    )
    metrics[f"{id}/delivered_stock"] = ph.logging.SimpleAgentMetric(
        id, "delivered_stock", "mean"
    )
    metrics[f"{id}/carried_stock"] = ph.logging.SimpleAgentMetric(
        id, "carried_stock", "mean"
    )
    metrics[f"{id}/leftover_stock"] = ph.logging.SimpleAgentMetric(
        id, "leftover_stock", "mean"
    )
    metrics[f"{id}/revenue"] = ph.logging.SimpleAgentMetric(id, "revenue", "mean")
    metrics[f"{id}/costs"] = ph.logging.SimpleAgentMetric(id, "costs", "mean")
    metrics[f"{id}/pnl"] = ph.logging.SimpleAgentMetric(id, "pnl", "mean")


exp_name = "all_supertypes"


if sys.argv[1] == "train":
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    cost_of_carry=ph.utils.samplers.UniformFloatSampler(
                        low=-0.1,
                        high=1.1,
                        clip_low=0.0,
                        clip_high=1.0,
                    ),
                )
                for shop_id in SHOP_IDS
            }
        },
        policies={
            "shop_policy": ShopAgent,
            "customer_policy": (CustomerPolicy, CustomerAgent),
        },
        policies_to_train=["shop_policy"],
        metrics=metrics,
        rllib_config={
            "seed": 1,
            # "model": {
            #     "fcnet_hiddens": [32, 32],
            # },
            "disable_env_checking": True,
        },
        tune_config={
            "name": supply_chain,
            "checkpoint_freq": 100,
            "stop": {
                "training_iteration": 5000,
            },
        },
    )


elif sys.argv[1] == "test":
    results = ph.utils.rllib.rollout(
        directory=f"{exp_name}/LATEST",
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    # sale_price=1.0,
                    sale_price=ph.utils.ranges.UniformRange(
               
                    cost_of_carry=ph.utils.ranges.UniformRange(
                        start=0.0,
                        end=1.0 + 0.001,
                        step=0.1,
                        name="cost_of_carry",
                    ),
                )
                for shop_id in SHOP_IDS
            }
        },
        num_repeats=50,
        metrics=metrics,
        record_messages=False,
     
    )

    # cloudpickle.save(file, list(results))

    # This is the supertype parameter we are scanning over
    varied_param = "cost_of_carry"

    # We iterate over all rollout results, taking just the values we need (the varied
    # supertype parameter and 3 metrics), and placing them into a Pandas DataFrame
    df = pd.DataFrame(
        {
            varied_param: rollout.rollout_params[varied_param],
            "avg_restock_qty": np.mean(
                [
                    x["restock_qty"]
                    for x in rollout.actions_for_agent("SHOP1", drop_nones=True)
                ]
            ),
            "avg_sales": np.mean(rollout.metrics["SHOP1/sales"]),
            "avg_reward": np.mean(rollout.rewards_for_agent("SHOP1", drop_nones=True)),
        }
        for rollout in results
    )

    # Aggregate all the results for each value of the scanned supertype parameter
    df = df.groupby(varied_param).mean()

    # Plot variables, x_axis = scanned parameter, y_axis = selected metrics
    for col_name in df.columns:
        plt.scatter(df.index, df[col_name], label=col_name)

    plt.legend()
    plt.xlabel(varied_param)
    plt.savefig(f"{exp_name}__{varied_param}.png")


elif sys.argv[1] == "policy":
    obs = {
        "type": {
            "sale_price": np.array([1.0]),
            "cost_of_carry": np.array([0.1]),
            "cost_per_unit": np.array([0.5]),
        },
        "stock": ph.utils.ranges.UnitArrayLinspaceRange(
            0.0, 1.0 - 0.01, 40, name="stock"
        ),
        "previous_sales": ph.utils.ranges.UnitArrayLinspaceRange(
            0.0, 1.0 - 0.01, 20, name="previous_sales"
        ),
    }

    results = ph.utils.rllib.evaluate_policy(
        "phevaluatetest/LATEST", "PPO", SupplyChainEnv, "shop_policy", obs
    )

    restock_actions = np.array(
        [action["restock_qty"] for _, action in results]
    ).reshape(40, 20)

    plt.imshow(restock_actions)
    plt.xlabel("prev sales")
    plt.ylabel("stock")
    plt.gca().invert_yaxis()
    cbar = plt.colorbar()
    cbar.set_label("restock qty", rotation=270)
    plt.savefig(f"{exp_name}__policy_x.png")
