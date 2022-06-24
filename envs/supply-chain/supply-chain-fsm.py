from dataclasses import dataclass

import coloredlogs
import gym
import numpy as np
import phantom as ph


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


from typing import Any, Mapping


class SupplyChainEnvFSM(ph.PhantomFSMEnv):
    def __init__(self, n_customers: int = 5, **kwargs):
        # Define actor and agent IDs
        shop_id = "SHOP"
        factory_id = "FACTORY"
        customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

        shop_agent = ShopAgent(shop_id, factory_id=factory_id)
        factory_agent = FactoryAgent(factory_id)

        customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in customer_ids]

        agents = [shop_agent, factory_agent] + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shop to the factory
        network.add_connection(shop_id, factory_id)

        # Connect the shop to the customers
        network.add_connections_between([shop_id], customer_ids)

        ph.PhantomFSMEnv.__init__(
            self, network=network, num_steps=NUM_EPISODE_STEPS, **kwargs
        )

        self.stage = "A"

    def step(self, actions: Mapping[ph.AgentID, Any]) -> ph.PhantomEnv.Step:

        if self.stage == "A":
            self.fsm_step()


metrics = {
    "SHOP/inventory": ph.logging.SimpleAgentMetric("SHOP", "current_inventory", "mean"),
    "SHOP/delivered": ph.logging.SimpleAgentMetric(
        "SHOP", "delivered_inventory", "mean"
    ),
    "SHOP/sales": ph.logging.SimpleAgentMetric("SHOP", "sales_made", "mean"),
}


# We can easily swap trainer classes:
trainer = ph.trainers.PPOTrainer(tensorboard_log_dir="../logs/ppo")
# trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir="../logs/qlearn")

results = trainer.train(
    env_class=SupplyChainEnv1,
    num_iterations=1000,
    policies={
        # We don't specify a policy here so the trainer's default is used.
        "shop_policy": ShopAgent,
        # Here we tell all instances of CustomerAgent to use the CustomerPolicy policy.
        "customer_policy": (CustomerPolicy, CustomerAgent),
    },
    # We tell the trainer to train the shop_policy, this must be compatible with the trainer.
    # (currently only single policy training is implemented)
    policies_to_train=["shop_policy"],
    metrics=metrics,
)

# Or we can use the RLlib trainers (with a thin Phantom wrapper):

ph.utils.rllib.train(
    algorithm="PPO",
    env_class=SupplyChainEnv1,
    num_iterations=1000,
    policies={
        "shop_policy": ShopAgent,
        "customer_policy": (CustomerPolicy, CustomerAgent),
    },
    policies_to_train=["shop_policy"],
    metrics=metrics,
)

# # if len(sys.argv) == 1 or sys.argv[1].lower() == "train":
# #     ph.train(
# #         experiment_name="supply-chain-1",
# #         algorithm="PPO",
# #         num_workers=8,
# #         num_episodes=5000,
# #         env_class=SupplyChainEnv,
# #         env_config=dict(n_customers=NUM_CUSTOMERS),
# #         metrics=metrics,
# #     )

# # elif sys.argv[1].lower() == "rollout":
# #     results = ph.rollout(
# #         directory="supply-chain-1/LATEST",
# #         algorithm="PPO",
# #         num_workers=0,
# #         num_repeats=100,
# #         env_config=dict(n_customers=NUM_CUSTOMERS),
# #         metrics=metrics,
# #         save_trajectories=True,
# #         save_messages=True,
# #     )
