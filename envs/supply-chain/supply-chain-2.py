import sys
from dataclasses import dataclass
from typing import List

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
NUM_SHOPS = 2
NUM_CUSTOMERS = 5

SHOP_MAX_STOCK = 100_000
SHOP_MAX_STOCK_REQUEST = 1000


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
    """Shop --> Warehouse"""

    size: int


@dataclass
class StockResponse:
    """Warehouse --> Shop"""

    size: int


class CustomerPolicy(ph.FixedPolicy):
    # The size of the order made and the choice of shop to make the order to for each
    # customer is determined by this fixed policy.
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)

        self.n_shops = config["n_shops"]

    def compute_action(self, obs) -> np.ndarray:
        return np.array([np.random.poisson(5), np.random.randint(self.n_shops)])


class CustomerAgent(ph.Agent):
    def __init__(self, agent_id: str, shop_ids: List[str]):
        super().__init__(
            agent_id,
            policy_class=CustomerPolicy,
            # The CustomerPolicy needs to know how many shops there are so it can return
            # a valid choice.
            policy_config=dict(n_shops=len(shop_ids)),
        )

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_ids: List[str] = shop_ids

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        # The customer will receive it's order from the shop but we do not need
        # to take any actions on it.
        yield from ()

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # At the start of each step we generate an order with a random size to
        # send to a random shop.
        order_size = action[0]
        shop_id = self.shop_ids[int(action[1])]

        # We perform this action by sending a stock request message to the warehouse.
        return ph.packet.Packet(messages={shop_id: [OrderRequest(order_size)]})

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context):
        return np.zeros((1,))

    def get_observation_space(self):
        return gym.spaces.Box(low=-np.inf, high=np.inf, shape=(1,))

    def get_action_space(self):
        return gym.spaces.Box(low=0, high=np.inf, shape=(2,))


class WarehouseActor(me.actors.SimpleSyncActor):
    def __init__(self, actor_id: str):
        super().__init__(actor_id)

    @me.actors.handler(StockRequest)
    def handle_stock_request(self, ctx: me.Network.Context, msg: me.Message):
        # The warehouse receives stock request messages from shop agents. We
        # simply reflect the amount of stock requested back to the shop as the
        # warehouse has unlimited stock.
        yield (msg.sender_id, [StockResponse(msg.payload.size)])


class ShopRewardFunction(ph.RewardFunction):
    def __init__(self, missed_sales_weight: float):
        self.missed_sales_weight = missed_sales_weight

    def reward(self, ctx: me.Network.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        # We give a bigger reward for making sales than the penalty for missed sales and
        # unused stock.
        return (
            5 * ctx.actor.step_sales
            - self.missed_sales_weight * ctx.actor.step_missed_sales
            - ctx.actor.stock
        )


@dataclass
class ShopAgentType(ph.AgentType):
    missed_sales_weight: float


class ShopAgentSupertype(ph.Supertype):
    def __init__(self):
        self.missed_sales_weight_low = 0.5
        self.missed_sales_weight_high = 3.0

    def sample(self) -> ShopAgentType:
        return ShopAgentType(
            missed_sales_weight=np.random.uniform(
                self.missed_sales_weight_low, self.missed_sales_weight_high
            )
        )


class ShopAgent(ph.Agent):
    def __init__(self, agent_id: str, warehouse_id: str, supertype: ph.Supertype):
        super().__init__(agent_id, supertype=supertype)

        # We store the ID of the warehouse so we can send stock requests to it.
        self.warehouse_id: str = warehouse_id

        # We keep track of how much stock the shop has...
        self.stock: int = 0

        # ...and how many sales have been made...
        self.step_sales: int = 0
        self.total_sales: int = 0

        # ...and how many orders per step the shop has missed due to not having enough
        # stock.
        self.step_missed_sales: int = 0
        self.total_missed_sales: int = 0

    def pre_resolution(self, ctx: me.Network.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.step_sales = 0
        self.step_missed_sales = 0

    @me.actors.handler(StockResponse)
    def handle_stock_response(self, ctx: me.Network.Context, msg: me.Message):
        # Messages received from the warehouse contain stock.
        self.stock += msg.payload.size

        # We do not need to respond to these messages.
        yield from ()

    @me.actors.handler(OrderRequest)
    def handle_order_request(self, ctx: me.Network.Context, msg: me.Message):
        # All other messages are from customers and contain orders.
        amount_requested = msg.payload.size

        if amount_requested > self.stock:
            self.step_missed_sales += amount_requested - self.stock
            self.total_missed_sales += amount_requested - self.stock
            stock_to_sell = self.stock
            self.stock = 0
        else:
            stock_to_sell = amount_requested
            self.stock -= amount_requested

        self.step_sales += stock_to_sell
        self.total_sales += stock_to_sell

        # Send the customer their order.
        yield (msg.sender_id, [OrderResponse(stock_to_sell)])

    def encode_obs(self, ctx: me.Network.Context):
        # We encode the shop's current stock as the observation.
        return np.array([self.stock])

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from
        # the warehouse.
        stock_to_request = action[0]

        # We perform this action by sending a stock request message to the warehouse.
        return ph.packet.Packet(
            messages={self.warehouse_id: [StockRequest(stock_to_request)]}
        )

    def reset(self):
        super().reset()  # self.type set here

        self.reward_function = ShopRewardFunction(
            missed_sales_weight=self.type.missed_sales_weight
        )

        self.stock = 0
        self.total_sales = 0
        self.total_missed_sales = 0

    def get_observation_space(self):
        return gym.spaces.Box(low=0.0, high=SHOP_MAX_STOCK, shape=(1,))

    def get_action_space(self):
        return gym.spaces.Box(low=0.0, high=SHOP_MAX_STOCK_REQUEST, shape=(1,))


shop_ids = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]


class SupplyChainEnv(ph.PhantomEnv):

    env_name: str = "supply-chain-v2"

    def __init__(self, n_customers: int = 5):
        # Define actor and agent IDs
        warehouse_id = "WAREHOUSE"

        customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

        shop_agents = [
            ShopAgent(sid, warehouse_id, ShopAgentSupertype()) for sid in shop_ids
        ]

        warehouse_actor = WarehouseActor(warehouse_id)

        customer_agents = [
            CustomerAgent(cid, shop_ids=shop_ids) for cid in customer_ids
        ]

        actors = [warehouse_actor] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = me.Network(me.resolvers.UnorderedResolver(), actors)

        # Connect the shops to the warehouse
        network.add_connections_between(shop_ids, [warehouse_id])

        # Connect the shop to the customers
        network.add_connections_between(shop_ids, customer_ids)

        clock = ph.Clock(0, NUM_EPISODE_STEPS, 1)

        super().__init__(network=network, clock=clock)


class StockMetric(ph.logging.Metric[float]):
    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id

    def extract(self, env: ph.PhantomEnv) -> float:
        return env[self.agent_id].stock


class SalesMetric(ph.logging.Metric[float]):
    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id

    def extract(self, env: ph.PhantomEnv) -> float:
        return env[self.agent_id].total_sales / NUM_EPISODE_STEPS


class MissedSalesMetric(ph.logging.Metric[float]):
    def __init__(self, agent_id: str) -> None:
        self.agent_id: str = agent_id

    def extract(self, env: ph.PhantomEnv) -> float:
        return env[self.agent_id].total_missed_sales / NUM_EPISODE_STEPS


metrics = {}

metrics.update(
    {f"stock/SHOP{i+1}": StockMetric(f"SHOP{i+1}") for i in range(NUM_SHOPS)}
)

metrics.update(
    {f"sales/SHOP{i+1}": SalesMetric(f"SHOP{i+1}") for i in range(NUM_SHOPS)}
)

metrics.update(
    {
        f"missed_sales/SHOP{i+1}": MissedSalesMetric(f"SHOP{i+1}")
        for i in range(NUM_SHOPS)
    }
)


if sys.argv[1].lower() == "train":
    ph.train(
        experiment_name="supply-chain-2",
        algorithm="PPO",
        num_workers=4,
        num_episodes=100,
        env=SupplyChainEnv,
        env_config={"n_customers": NUM_CUSTOMERS},
        metrics=metrics,
        policy_grouping={"shared_SHOP_policy": shop_ids},
    )

elif sys.argv[1].lower() == "rollout":
    ph.rollout(
        directory="supply-chain-2/LATEST",
        algorithm="PPO",
        num_workers=1,
        num_rollouts=10,
        env_config={"n_customers": NUM_CUSTOMERS},
    )
