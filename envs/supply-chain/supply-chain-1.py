import sys

import coloredlogs
import gym
import mercury as me
import numpy as np
import phantom as ph


coloredlogs.install(
    level="INFO",
    fmt="(pid=%(process)d) %(levelname)s %(name)s %(message)s",
)

NUM_EPISODE_STEPS = 100

NUM_CUSTOMERS = 5
SHOP_MAX_STOCK = 1000
SHOP_MAX_STOCK_REQUEST = 100


class CustomerPolicy(ph.FixedPolicy):
    # The size of the order made for each customer is determined by this fixed policy.
    def compute_action(self, obs) -> int:
        return np.random.poisson(5)


class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: str, shop_id: str):
        super().__init__(agent_id, policy_class=CustomerPolicy)

        # We need to store the shop's ID so we can send order requests to it.
        self.shop_id: str = shop_id

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        # The customer will receive it's order from the shop but we do not need to take
        # any actions on it.
        yield from ()

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # At the start of each step we generate an order with a random size (determined)
        # by the policy) to send to the shop.
        order_size = action

        # We perform this action by sending a stock request message to the factory.
        return ph.packet.Packet(messages={self.shop_id: [order_size]})

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context):
        return 0

    def get_observation_space(self):
        return gym.spaces.Discrete(1)

    def get_action_space(self):
        return gym.spaces.Discrete(100)


class FactoryActor(me.actors.SimpleSyncActor):
    def __init__(self, actor_id: str):
        super().__init__(actor_id)

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        yield (msg.sender_id, [msg.payload])


class ShopAgent(ph.Agent):
    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has...
        self.stock: int = 0

        # ...and how many sales have been made...
        self.sales: int = 0

        # ...and how many orders per step the shop has missed due to not having enough
        # stock.
        self.missed_sales: int = 0

    def pre_resolution(self, ctx: me.Network.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        if msg.sender_id == self.factory_id:
            # Messages received from the factory contain stock.
            self.stock = min(self.stock + msg.payload, SHOP_MAX_STOCK)

            # We do not need to respond to these messages.
            yield from ()
        else:
            # All other messages are from customers and contain orders.
            amount_requested = msg.payload

            if amount_requested > self.stock:
                self.missed_sales += amount_requested - self.stock
                stock_to_sell = self.stock
                self.stock = 0
            else:
                stock_to_sell = amount_requested
                self.stock -= amount_requested

            self.sales += stock_to_sell

            # Send the customer their order.
            yield (msg.sender_id, [stock_to_sell])

    def encode_obs(self, ctx: me.Network.Context):
        # We encode the shop's current stock as the observation.
        return self.stock

    def decode_action(self, ctx: me.Network.Context, action: int):
        # The action the shop takes is the amount of new stock to request from the
        # factory.
        stock_to_request = action

        # We perform this action by sending a stock request message to the factory.
        return ph.packet.Packet(messages={self.factory_id: [stock_to_request]})

    def compute_reward(self, ctx: me.Network.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        # We give a bigger reward for making sales than the penalty for missed sales and
        # unused stock.
        return self.sales - self.missed_sales - self.stock * 5

    def reset(self):
        self.stock = 0

    def get_observation_space(self):
        return gym.spaces.Discrete(SHOP_MAX_STOCK + 1)

    def get_action_space(self):
        return gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST)


class SupplyChainEnv(ph.PhantomEnv):

    env_name: str = "supply-chain-v1"

    def __init__(self, n_customers: int = 5):
        # Define actor and agent IDs
        shop_id = "SHOP"
        factory_id = "FACTORY"
        customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

        shop_agent = ShopAgent(shop_id, factory_id=factory_id)
        factory_actor = FactoryActor(factory_id)

        customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in customer_ids]

        actors = [shop_agent, factory_actor] + customer_agents

        # Define Network and create connections between Actors
        network = me.Network(me.resolvers.UnorderedResolver(), actors)

        # Connect the shop to the factory
        network.add_connection(shop_id, factory_id)

        # Connect the shop to the customers
        network.add_connections_between([shop_id], customer_ids)

        super().__init__(network=network, n_steps=NUM_EPISODE_STEPS)


metrics = {}
metrics["SHOP/stock"] = ph.logging.SimpleAgentMetric("SHOP", "stock", "mean")
metrics["SHOP/sales"] = ph.logging.SimpleAgentMetric("SHOP", "sales", "mean")
metrics["SHOP/missed_sales"] = ph.logging.SimpleAgentMetric(
    "SHOP", "missed_sales", "mean"
)


if len(sys.argv) == 1 or sys.argv[1].lower() == "train":
    ph.train(
        experiment_name="supply-chain-1",
        algorithm="PPO",
        num_workers=8,
        num_episodes=5000,
        env_class=SupplyChainEnv,
        env_config=dict(n_customers=NUM_CUSTOMERS),
        metrics=metrics,
    )

elif sys.argv[1].lower() == "rollout":
    ph.rollout(
        directory="supply-chain-1/LATEST",
        algorithm="PPO",
        num_workers=1,
        num_repeats=10,
        env_config=dict(n_customers=NUM_CUSTOMERS),
        save_trajectories=True,
        save_messages=True,
        metrics=metrics,
    )
