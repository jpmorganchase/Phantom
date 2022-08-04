"""
A simple logistics themed environment used for demonstrating the features of Phantom.
"""

import pickle
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import numpy as np
import phantom as ph


# Define fixed parameters:
NUM_EPISODE_STEPS = 100
NUM_SHOPS = 2
NUM_CUSTOMERS = 5

CUSTOMER_MAX_ORDER_SIZE = 5
SHOP_MIN_PRICE = 0.0
SHOP_MAX_PRICE = 1.0
SHOP_MAX_STOCK = 100
SHOP_MAX_STOCK_REQUEST = 20


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
    def compute_action(self, obs: np.ndarray) -> Tuple[int, int]:
        return (np.random.randint(CUSTOMER_MAX_ORDER_SIZE), np.argmin(obs))


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

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
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
        # The cost of holding onto one unit of inventory overnight:
        cost_of_carry: float
        # The cost of producing one unit of inventory:
        cost_per_unit: float

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

        # We initialise the price variable here, it's value will be set when the shop
        # agent takes it's first action.
        self.price: float = 0.0

        self.action_space = gym.spaces.Tuple(
            (
                # The price to set for the current step:
                gym.spaces.Box(low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(1,)),
                # The number of additional units to order from the factory:
                gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST),
            )
        )

        self.observation_space = gym.spaces.Tuple(
            (
                # The agent's type:
                gym.spaces.Dict(
                    {
                        "cost_of_carry": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                        "cost_per_unit": gym.spaces.Box(low=0.0, high=1.0, shape=(1,)),
                    }
                ),
                # The agent's current stock:
                gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
                # The number of sales made by the agent in the previous step:
                gym.spaces.Discrete(
                    min(CUSTOMER_MAX_ORDER_SIZE * NUM_CUSTOMERS, SHOP_MAX_STOCK) + 1
                ),
            )
        )

    def view(self, neighbour_id: Optional[ph.AgentID] = None) -> "View":
        """Return an immutable view to the agent's public state."""
        return self.View(self.id, self.price)

    def pre_message_resolution(self, ctx: ph.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the factory contain stock.
        self.delivered_stock = message.payload.size

        self.stock = min(self.stock + message.payload.size, SHOP_MAX_STOCK)

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(self, ctx: ph.Context, message: ph.Message):
        # All other messages are from customers and contain orders.
        amount_requested = message.payload.size

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

        # Send the customer their order.
        return [(message.sender_id, OrderResponse(stock_to_sell))]

    def encode_observation(self, ctx: ph.Context):
        return (
            # The shop's type is included in its observation space to allow it to learn
            # a generalised policy.
            self.type.to_obs_space_compatible_type(),
            # The current stock is included so the shop can learn to efficiently manage
            # its inventory.
            self.stock,
            # The number of sales made in the previous step is included so the shop can
            # learn to maximise sales.
            self.sales,
        )

    def decode_action(self, ctx: ph.Context, action: Tuple[np.ndarray, int]):
        # The action the shop takes is the updated price for it's products and the
        # amount of new stock to request from the factory.

        # We update the shop's price:
        self.price = action[0][0]

        # And we send a stock request message to the factory:
        stock_to_request = action[1]

        return [(self.factory_id, StockRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return (
            # The shop makes profit from selling items at the set price:
            self.sales * self.price
            # It incurs a cost for ordering new stock:
            - self.delivered_stock * self.type.cost_per_unit
            # And for holding onto excess stock overnight:
            - self.stock * self.type.cost_of_carry
        )

    def reset(self):
        super().reset()  # sampled supertype is set as self.type here

        self.stock = 0


# Define agent IDs:
FACTORY_ID = "FACTORY"
SHOP_IDS = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]
CUSTOMER_IDS = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]


class SupplyChainEnv(ph.PhantomEnv):
    def __init__(self, **kwargs):
        shop_agents = [ShopAgent(id, FACTORY_ID) for id in SHOP_IDS]

        factory_agent = FactoryAgent(FACTORY_ID)

        customer_agents = [CustomerAgent(id, shop_ids=SHOP_IDS) for id in CUSTOMER_IDS]

        agents = [factory_agent] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shops to the factory
        network.add_connections_between(SHOP_IDS, [FACTORY_ID])

        # Connect the shop to the customers
        network.add_connections_between(SHOP_IDS, CUSTOMER_IDS)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network, **kwargs)


metrics = {}

for id in SHOP_IDS:
    metrics[f"stock/{id}"] = ph.logging.SimpleAgentMetric(id, "stock", "mean")
    metrics[f"sales/{id}"] = ph.logging.SimpleAgentMetric(id, "sales", "mean")
    metrics[f"price/{id}"] = ph.logging.SimpleAgentMetric(id, "price", "mean")
    metrics[f"missed_sales/{id}"] = ph.logging.SimpleAgentMetric(
        id, "missed_sales", "mean"
    )


if sys.argv[1] == "train":
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    cost_of_carry=ph.utils.samplers.UniformSampler(low=0.0, high=0.1),
                    cost_per_unit=ph.utils.samplers.UniformSampler(low=0.2, high=0.8),
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
            "seed": 0,
            "model": {
                "fcnet_hiddens": [64, 64],
            },
        },
        tune_config={
            "checkpoint_freq": 100,
            "stop": {
                "training_iteration": 1000,
            },
        },
    )


elif sys.argv[1] == "test":
    results = ph.utils.rllib.rollout(
        directory="PPO/LATEST",
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={
            "agent_supertypes": {
                shop_id: ShopAgent.Supertype(
                    cost_of_carry=ph.utils.ranges.UniformRange(
                        start=0.0,
                        end=0.1 + 0.001,
                        step=0.01,
                    ),
                    cost_per_unit=ph.utils.ranges.UniformRange(
                        start=0.2,
                        end=0.8 + 0.001,
                        step=0.05,
                    ),
                )
                for shop_id in SHOP_IDS
            }
        },
        num_repeats=1,
        metrics=metrics,
        record_messages=False,
    )

    pickle.dump(results, open("results.pkl", "wb"))
