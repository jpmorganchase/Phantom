from dataclasses import dataclass

import coloredlogs
import gym
import numpy as np
import phantom as ph
from stable_baselines3 import PPO


coloredlogs.install(
    level="INFO",
    fmt="(pid=%(process)d) %(levelname)s %(name)s %(message)s",
)

NUM_EPISODE_STEPS = 100

NUM_CUSTOMERS = 5
SHOP_MAX_INVENTORY = 100
SHOP_MAX_RESTOCK_REQUEST = 50


@dataclass
class OrderMsg:
    """Customer --> Shop"""

    order_size: int


@dataclass
class DeliveryMsg:
    """Shop --> Customer"""

    delivered_quantity: int


@dataclass
class RestockOrderMsg:
    """Shop --> Factory"""

    order_size: int


@dataclass
class RestockDeliveryMsg:
    """Factory --> Shop"""

    delivered_quantity: int


class CustomerPolicy(ph.Policy):
    # The size of the order made for each customer is determined by this fixed policy.
    def compute_action(self, obs) -> int:
        return np.random.randint(0, 5)


class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: str, shop_id: str):
        super().__init__(agent_id)

        # We need to store the shop's ID so we can send order requests to it.
        self.shop_id: str = shop_id

    def handle_message(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        # The customer will receive it's order from the shop but we do not need to take
        # any actions on it.
        return []

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # At the start of each step we generate an order with a random size (determined)
        # by the policy) to send to the shop.
        order_size = action

        # We perform this action by sending a stock request message to the factory.
        return [(self.shop_id, OrderMsg(order_size))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: ph.Context):
        return 0

    @property
    def observation_space(self):
        return gym.spaces.Discrete(1)

    @property
    def action_space(self):
        return gym.spaces.Discrete(100)


class FactoryAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def handle_message(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        return [(sender_id, RestockDeliveryMsg(message.order_size))]


class ShopAgent(ph.Agent):
    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has...
        self.current_inventory: int = 0

        self.delivered_inventory: int = 0

        # ...and how many sales have been made...
        self.sales_made: int = 0

        # ...and how many orders per step the shop has missed due to not having enough
        # stock.
        self.missed_sales: int = 0

    def pre_message_resolution(self, ctx: ph.Context):
        # At the start of each step we reset the number of sales and missed orders to 0.
        self.sales_made = 0
        self.missed_sales = 0

    def post_message_resolution(self, ctx: ph.Context):
        # At the end of each step we restock the shop with the products delivered
        # from the factory
        self.current_inventory = min(
            self.current_inventory + self.delivered_inventory, SHOP_MAX_INVENTORY
        )

    def handle_message(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        if sender_id == self.factory_id:
            # Messages received from the factory contain stock.
            self.delivered_inventory = message.delivered_quantity

            # We do not need to respond to these messages.
            return []
        else:
            # All other messages are from customers and contain orders.
            quantity_ordered = message.order_size

            if quantity_ordered > self.current_inventory:
                quantity_to_sell = self.current_inventory
                self.current_inventory = 0
                self.sales_missed = quantity_ordered - quantity_to_sell
            else:
                quantity_to_sell = quantity_ordered
                self.current_inventory -= quantity_ordered

            self.sales_made += quantity_to_sell

            # Send the customer their order.
            return [(sender_id, DeliveryMsg(quantity_to_sell))]

    def encode_obs(self, ctx: ph.Context):
        # We encode the shop's current stock as the observation.
        return self.current_inventory

    def decode_action(self, ctx: ph.Context, action: int):
        # The action the shop takes is the amount of new stock to request from the
        # factory.
        stock_to_request = action

        # We perform this action by sending a stock request message to the factory.
        return [(self.factory_id, RestockOrderMsg(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        # We give a bigger reward for making sales than the penalty for missed sales and
        # unused stock.
        return self.sales_made - self.current_inventory * 0.25

    def reset(self):
        super().reset()

        self.current_inventory = 0
        self.delivered_inventory = 0
        self.sales_made = 0

    @property
    def observation_space(self):
        return gym.spaces.Discrete(SHOP_MAX_INVENTORY + 1)

    @property
    def action_space(self):
        return gym.spaces.Discrete(SHOP_MAX_RESTOCK_REQUEST)


CUSTOMER_IDS = [f"CUST{i+1}" for i in range(5)]


class SupplyChainEnv1(ph.PhantomEnv):
    def __init__(self, **kwargs):
        # Define actor and agent IDs
        shop_id = "SHOP"
        factory_id = "FACTORY"

        shop_agent = ShopAgent(shop_id, factory_id=factory_id)
        factory_agent = FactoryAgent(factory_id)

        customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in CUSTOMER_IDS]

        agents = [shop_agent, factory_agent] + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shop to the factory
        network.add_connection(shop_id, factory_id)

        # Connect the shop to the customers
        network.add_connections_between([shop_id], CUSTOMER_IDS)

        ph.PhantomEnv.__init__(
            self, network=network, num_steps=NUM_EPISODE_STEPS, **kwargs
        )


env = ph.SingleAgentEnvAdapter(
    SupplyChainEnv1, "SHOP", {cid: (CustomerPolicy, {}) for cid in CUSTOMER_IDS}
)

model = PPO("MlpPolicy", env, verbose=1, tensorboard_log="./tensorboard/")
model.learn(total_timesteps=1e7)

obs = env.reset()
for i in range(100):
    action, _state = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
        obs = env.reset()
