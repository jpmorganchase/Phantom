import sys

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import phantom as ph


NUM_EPISODE_STEPS = 100

NUM_CUSTOMERS = 5
CUSTOMER_MAX_ORDER_SIZE = 5
SHOP_MAX_STOCK = 100


@ph.msg_payload("CustomerAgent", "ShopAgent")
class OrderRequest:
    size: int


@ph.msg_payload("ShopAgent", "CustomerAgent")
class OrderResponse:
    size: int


@ph.msg_payload("ShopAgent", "FactoryAgent")
class StockRequest:
    size: int


@ph.msg_payload("FactoryAgent", "ShopAgent")
class StockResponse:
    size: int


class FactoryAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    @ph.agents.msg_handler(StockRequest)
    def handle_stock_request(self, ctx: ph.Context, message: ph.Message):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        return [(message.sender_id, StockResponse(message.payload.size))]


class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: ph.AgentID, shop_id: ph.AgentID):
        super().__init__(agent_id)

        # We need to store the shop's ID so we know who to send order requests to.
        self.shop_id: str = shop_id

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        # The customer will receive it's order from the shop but we do not need to take
        # any actions on it.
        return

    def generate_messages(self, ctx: ph.Context):
        # At the start of each step we generate an order with a random size to send to
        # the shop.
        order_size = np.random.randint(CUSTOMER_MAX_ORDER_SIZE)

        # We perform this action by sending a stock request message to the factory.
        return [(self.shop_id, OrderRequest(order_size))]


class ShopAgent(ph.StrategicAgent):
    def __init__(self, agent_id: str, factory_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.factory_id: str = factory_id

        # We keep track of how much stock the shop has...
        self.stock: int = 0

        # ...and how many sales have been made...
        self.sales: int = 0

        # ...and how many sales per step the shop has missed due to not having enough
        # stock.
        self.missed_sales: int = 0

        # = [Stock, Sales, Missed Sales]
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(3,))

        # = [Restock Quantity]
        self.action_space = gym.spaces.Box(low=0.0, high=SHOP_MAX_STOCK, shape=(1,))

    def pre_message_resolution(self, ctx: ph.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    @ph.agents.msg_handler(StockResponse)
    def handle_stock_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the factory contain stock.
        self.delivered_stock = message.payload.size

        self.stock = min(self.stock + self.delivered_stock, SHOP_MAX_STOCK)

    @ph.agents.msg_handler(OrderRequest)
    def handle_order_request(self, ctx: ph.Context, message: ph.Message):
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
        max_sales_per_step = NUM_CUSTOMERS * CUSTOMER_MAX_ORDER_SIZE

        return np.array(
            [
                self.stock / SHOP_MAX_STOCK,
                self.sales / max_sales_per_step,
                self.missed_sales / max_sales_per_step,
            ],
            dtype=np.float32,
        )

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from
        # the factory, clipped so the shop never requests more stock than it can hold.
        stock_to_request = min(int(round(action[0])), SHOP_MAX_STOCK - self.stock)

        # We perform this action by sending a stock request message to the factory.
        return [(self.factory_id, StockRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto excess stock.
        return self.sales - 0.1 * self.stock

    def reset(self):
        self.stock = 0


class SupplyChainEnv(ph.PhantomEnv):
    def __init__(self):
        # Define agent IDs
        factory_id = "WAREHOUSE"
        customer_ids = [f"CUST{i+1}" for i in range(NUM_CUSTOMERS)]
        shop_id = "SHOP"

        factory_agent = FactoryAgent(factory_id)
        customer_agents = [CustomerAgent(cid, shop_id=shop_id) for cid in customer_ids]
        shop_agent = ShopAgent(shop_id, factory_id=factory_id)

        agents = [shop_agent, factory_agent] + customer_agents

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the shop to the factory
        network.add_connection(shop_id, factory_id)

        # Connect the shop to the customers
        network.add_connections_between([shop_id], customer_ids)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network)


metrics = {
    "SHOP/stock": ph.metrics.SimpleAgentMetric("SHOP", "stock", "mean"),
    "SHOP/sales": ph.metrics.SimpleAgentMetric("SHOP", "sales", "mean"),
    "SHOP/missed_sales": ph.metrics.SimpleAgentMetric("SHOP", "missed_sales", "mean"),
}


if sys.argv[1] == "train":
    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={},
        iterations=500,
        checkpoint_freq=50,
        policies={"shop_policy": ["SHOP"]},
        metrics=metrics,
        results_dir="~/ray_results/supply_chain",
    )

elif sys.argv[1] == "rollout":
    results = ph.utils.rllib.rollout(
        directory="~/ray_results/supply_chain/LATEST",
        num_repeats=100,
        num_workers=1,
        metrics=metrics,
    )

    results = list(results)

    shop_actions = []
    shop_stock = []
    shop_sales = []
    shop_missed_sales = []

    for rollout in results:
        shop_actions += list(
            int(round(x[0])) for x in rollout.actions_for_agent("SHOP")
        )
        shop_stock += list(rollout.metrics["SHOP/stock"])
        shop_sales += list(rollout.metrics["SHOP/sales"])
        shop_missed_sales += list(rollout.metrics["SHOP/missed_sales"])

    # Plot distribution of shop action (stock request) per step for all rollouts
    plt.hist(shop_actions, bins=20)
    plt.title("Distribution of Shop Action Values (Stock Requested Per Step)")
    plt.xlabel("Shop Action (Stock Requested Per Step)")
    plt.ylabel("Frequency")
    plt.savefig("supply_chain_shop_action_values.png")
    plt.close()

    plt.hist(shop_stock, bins=20)
    plt.title("Distribution of Shop Held Stock")
    plt.xlabel("Shop Held Stock (Per Step)")
    plt.ylabel("Frequency")
    plt.savefig("supply_chain_shop_held_stock.png")
    plt.close()

    plt.hist(shop_sales, bins=20)
    plt.axvline(np.mean(shop_sales), c="k")
    plt.title("Distribution of Shop Sales Made")
    plt.xlabel("Shop Sales Made (Per Step)")
    plt.ylabel("Frequency")
    plt.savefig("supply_chain_shop_sales_made.png")
    plt.close()

    plt.hist(shop_missed_sales, bins=20)
    plt.title("Distribution of Shop Missed Sales")
    plt.xlabel("Shop Missed Sales (Per Step)")
    plt.ylabel("Frequency")
    plt.savefig("supply_chain_shop_missed_sales.png")
    plt.close()
