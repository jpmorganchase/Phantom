from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import numpy as np
import phantom as ph
from ray import rllib


# Define fixed parameters:
NUM_EPISODE_STEPS = 100
NUM_SHOPS = 2
NUM_CUSTOMERS = 5

CUSTOMER_MAX_ORDER_SIZE = 5
SHOP_MIN_PRICE = 0.0
SHOP_MAX_PRICE = 1.0
SHOP_MAX_STOCK = 100
SHOP_MAX_STOCK_REQUEST = 20


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


class CustomerPolicy(ph.Policy):
    # The size of the order made and the choice of shop to make the order to for each
    # customer is determined by this fixed policy.
    def compute_action(self, obs: np.ndarray) -> Tuple[int, int]:
        # return (np.random.poisson(5), np.random.randint(self.config["n_shops"]))
        return (np.random.randint(CUSTOMER_MAX_ORDER_SIZE), np.argmin(obs))


class CustomerAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_ids: List[ph.AgentID]):
        super().__init__(agent_id)

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_ids: List[str] = shop_ids

        self.action_space = gym.spaces.Tuple(
            (
                gym.spaces.Discrete(CUSTOMER_MAX_ORDER_SIZE),
                gym.spaces.Discrete(len(self.shop_ids)),
            )
        )
        self.observation_space = gym.spaces.Box(
            low=SHOP_MIN_PRICE, high=SHOP_MAX_PRICE, shape=(len(self.shop_ids),)
        )

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        return []

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


class FactoryAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    @ph.agents.msg_handler(StockRequest)
    def handle_stock_request(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        return [(sender_id, StockResponse(message.size))]


class ShopAgent(ph.MessageHandlerAgent):
    @dataclass(frozen=True)
    class Supertype(ph.Supertype):
        cost_of_carry: float
        cost_per_unit: float

    @dataclass(frozen=True)
    class View(ph.AgentView):
        price: float

    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has...
        self.stock: int = 0

        # ...and how many sales have been made...
        self.sales: int = 0

        # ...and how many orders the shop has missed due to not having enough stock.
        self.missed_sales: int = 0

        # we initialise the price variable here, it's value will be set when the shop
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
                gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
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
    def handle_stock_response(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        # Messages received from the factory contain stock.
        self.stock = min(self.stock + message.size, SHOP_MAX_STOCK)

        # We do not need to respond to these messages.
        return []

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        # All other messages are from customers and contain orders.
        amount_requested = message.size

        if amount_requested > self.stock:
            self.missed_sales += amount_requested - self.stock
            stock_to_sell = self.stock
            self.stock = 0
        else:
            stock_to_sell = amount_requested
            self.stock -= amount_requested

        self.sales += stock_to_sell

        # Send the customer their order.
        return [(sender_id, OrderResponse(stock_to_sell))]

    def encode_observation(self, ctx: ph.Context):
        return (
            # The agent's type is included in its observation space to allow it to learn
            # a generalised policy.
            self.type.to_obs_space_compatible_type(),
            # The current stock is included so the agent can learn to efficiently manage
            # its inventory.
            self.stock,
            # The number of sales made in the previous step is included so the agent can
            # learn to maximise sales.
            self.sales,
        )

    def decode_action(self, ctx: ph.Context, action: Tuple[np.ndarray, int]):
        # The action the shop takes is the updated price for it's products and the
        # amount of new stock to request from the factory.

        # We update the agent's price:
        self.price = action[0][0]

        # And we send a stock request message to the factory:
        stock_to_request = action[1]

        return [(self.factory_id, StockRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return (
            self.sales * (self.price - self.type.cost_per_unit)
            - self.stock * self.type.cost_of_carry
        )

    def reset(self):
        super().reset()  # sampled supertype is set as self.type here

        self.stock = 0
        self.price = 0.0


SHOP_IDS = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]


class SupplyChainEnv2(ph.PhantomEnv, rllib.MultiAgentEnv):
    def __init__(self, n_customers: int = 5, **kwargs):
        # Define actor and agent IDs
        factory_id = "FACTORY"

        customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

        shop_agents = [ShopAgent(id, factory_id) for id in SHOP_IDS]

        factory_agent = FactoryAgent(factory_id)

        customer_agents = [CustomerAgent(id, shop_ids=SHOP_IDS) for id in customer_ids]

        agents = [factory_agent] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shops to the factory
        network.add_connections_between(SHOP_IDS, [factory_id])

        # Connect the shop to the customers
        network.add_connections_between(SHOP_IDS, customer_ids)

        rllib.MultiAgentEnv.__init__(self)
        ph.PhantomEnv.__init__(
            self, network=network, num_steps=NUM_EPISODE_STEPS, **kwargs
        )


metrics = {}

for id in SHOP_IDS:
    metrics[f"stock/{id}"] = ph.logging.SimpleAgentMetric(id, "stock", "mean")
    metrics[f"sales/{id}"] = ph.logging.SimpleAgentMetric(id, "sales", "mean")
    metrics[f"price/{id}"] = ph.logging.SimpleAgentMetric(id, "price", "mean")
    metrics[f"missed_sales/{id}"] = ph.logging.SimpleAgentMetric(
        id, "missed_sales", "mean"
    )


ph.utils.rllib.train(
    algorithm="PPO",
    env_class=SupplyChainEnv2,
    env_config={
        "agent_supertypes": {
            shop_id: ShopAgent.Supertype(
                # cost_of_carry=0.1,
                # cost_per_unit=0.5,
                cost_of_carry=ph.utils.samplers.UniformSampler(low=0.0, high=1.0),
                cost_per_unit=ph.utils.samplers.UniformSampler(low=0.0, high=1.0),
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
        # "num_workers": 0,
        "seed": 0,
    },
    tune_config={
        "checkpoint_freq": 100,
        "stop": {
            "training_iteration": 5000,
        },
    },
)


# results = ph.utils.rllib.rollout(
#     directory="PPO/LATEST",
#     algorithm="PPO",
#     env_class=SupplyChainEnv2,
#     env_config={
#         "agent_supertypes": {
#             shop_id: ShopAgent.Supertype(
#                 cost_of_carry=0.1,
#                 cost_per_unit=0.5,
#                 # cost_of_carry=ph.utils.ranges.UniformRange(
#                 #     low=0.0, high=1.0, step=0.1,
#                 # ),
#                 # cost_per_unit=ph.utils.ranges.UniformRange(
#                 #     low=0.0, high=1.0, step=0.1,
#                 # ),
#             )
#             for shop_id in SHOP_IDS
#         }
#     },
#     num_workers=4,
#     num_repeats=100,
#     metrics=metrics,
#     record_messages=True,
# )
