"""
A simplified SupplyChain environment using Stable Baselines for policy training.
"""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import gym
import numpy as np
import phantom as ph
from stable_baselines3 import PPO


# Define fixed parameters:
NUM_EPISODE_STEPS = 100
NUM_CUSTOMERS = 5

CUSTOMER_MAX_ORDER_SIZE = 5
SHOP_PRICE = 1.0
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
    # The size of the order made for each customer is determined by this fixed policy.
    def compute_action(self, obs: np.ndarray) -> int:
        return np.random.randint(CUSTOMER_MAX_ORDER_SIZE)


class CustomerAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_id: ph.AgentID):
        super().__init__(agent_id)

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_id: str = shop_id

        # The number of items to order
        self.action_space = gym.spaces.Discrete(CUSTOMER_MAX_ORDER_SIZE)

        self.observation_space = None

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        return

    def decode_action(self, ctx: ph.Context, action: int):
        # At the start of each step we generate an order with a random size to send to a
        # random shop.
        order_size = action

        # We perform this action by sending a stock request message to the factory.
        return [(self.shop_id, OrderRequest(order_size))]

    def encode_observation(self, ctx: ph.Context):
        return None


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

        # The number of additional units to order from the factory:
        self.action_space = gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST)

        # Stable baselines does not support Tuple observation space so we flatten into
        # a Box with normalised components:
        #  - The agent's current stock
        #  - The number of sales made by the agent in the previous step
        #  - The agent's type:
        #    - Cost of carry
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))

    def pre_message_resolution(self, ctx: ph.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the factory contain stock.
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
        return np.array(
            [
                # The current stock is included so the shop can learn to efficiently manage
                # its inventory.
                self.stock / SHOP_MAX_STOCK,
                # The number of sales made in the previous step is included so the shop can
                # learn to maximise sales.
                self.sales
                / min(CUSTOMER_MAX_ORDER_SIZE * NUM_CUSTOMERS, SHOP_MAX_STOCK),
                # The shop's type is included in its observation space to allow it to learn
                # a generalised policy.
                self.type.cost_of_carry,
            ]
        )

    def decode_action(self, ctx: ph.Context, action: Tuple[np.ndarray, int]):
        # The action the shop takes is the amount of new stock to request from the
        # factory.
        stock_to_request = action

        return [(self.factory_id, StockRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return self.sales * SHOP_PRICE - self.stock * self.type.cost_of_carry

    def reset(self):
        super().reset()  # sampled supertype is set as self.type here

        self.stock = 0


# Define agent IDs:
FACTORY_ID = "FACTORY"
SHOP_ID = "SHOP"
CUSTOMER_IDS = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]


class SupplyChainEnv(ph.PhantomEnv):
    def __init__(self, **kwargs):
        shop_agent = ShopAgent(SHOP_ID, FACTORY_ID)

        factory_agent = FactoryAgent(FACTORY_ID)

        customer_agents = [CustomerAgent(id, shop_id=SHOP_ID) for id in CUSTOMER_IDS]

        agents = [factory_agent, shop_agent] + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shops to the factory
        network.add_connections_between([SHOP_ID], [FACTORY_ID])

        # Connect the shop to the customers
        network.add_connections_between([SHOP_ID], CUSTOMER_IDS)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network, **kwargs)


metrics = {}

metrics[f"stock/SHOP"] = ph.logging.SimpleAgentMetric("SHOP", "stock", "mean")
metrics[f"sales/SHOP"] = ph.logging.SimpleAgentMetric("SHOP", "sales", "mean")
metrics[f"price/SHOP"] = ph.logging.SimpleAgentMetric("SHOP", "price", "mean")
metrics[f"missed_sales/SHOP"] = ph.logging.SimpleAgentMetric(
    "SHOP", "missed_sales", "mean"
)


env = ph.SingleAgentEnvAdapter(
    env_class=SupplyChainEnv,
    env_config={
        "agent_supertypes": {
            "SHOP": ShopAgent.Supertype(
                cost_of_carry=ph.utils.samplers.UniformSampler(low=0.0, high=0.1),
            )
        }
    },
    agent_id="SHOP",
    other_policies={cid: (CustomerPolicy, {}) for cid in CUSTOMER_IDS},
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=1e7)
model.save("sb_model.pkl")
# model = PPO.load(PATH, env=env) loading pre-trained model

obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
