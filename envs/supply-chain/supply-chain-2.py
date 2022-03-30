import sys
from dataclasses import dataclass
from typing import List, Tuple

import coloredlogs
import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.logging import SimpleAgentMetric
from phantom.utils.ranges import UniformRange
from phantom.utils.samplers import UniformSampler

coloredlogs.install(
    level="INFO",
    fmt="(pid=%(process)d) %(levelname)s %(name)s %(message)s",
)

NUM_EPISODE_STEPS = 100
NUM_SHOPS = 2
NUM_CUSTOMERS = 5

SHOP_MAX_STOCK = 1000
SHOP_MAX_STOCK_REQUEST = 100


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


class CustomerPolicy(ph.FixedPolicy):
    # The size of the order made and the choice of shop to make the order to for each
    # customer is determined by this fixed policy.
    def __init__(self, obs_space, action_space, config):
        super().__init__(obs_space, action_space, config)

        self.n_shops = config["n_shops"]

    def compute_action(self, obs) -> Tuple[int, int]:
        return (np.random.poisson(5), np.random.randint(self.n_shops))


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
        # The customer will receive it's order from the shop but we do not need to take
        # any actions on it.
        yield from ()

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # At the start of each step we generate an order with a random size to send to a
        # random shop.
        order_size = action[0]
        shop_id = self.shop_ids[int(action[1])]

        # We perform this action by sending a stock request message to the factory.
        return ph.packet.Packet(messages={shop_id: [OrderRequest(order_size)]})

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0.0

    def encode_obs(self, ctx: me.Network.Context):
        return 0

    def get_observation_space(self):
        return gym.spaces.Discrete(1)

    def get_action_space(self):
        return gym.spaces.Tuple(
            (
                gym.spaces.Discrete(100),
                gym.spaces.Discrete(len(self.shop_ids)),
            )
        )


class FactoryActor(me.actors.SimpleSyncActor):
    def __init__(self, actor_id: str):
        super().__init__(actor_id)

    @me.actors.handler(StockRequest)
    def handle_stock_request(self, ctx: me.Network.Context, msg: me.Message):
        # The factory receives stock request messages from shop agents. We simply
        # reflect the amount of stock requested back to the shop as the factory can
        # produce unlimited stock.
        yield (msg.sender_id, [StockResponse(msg.payload.size)])


class ShopRewardFunction(ph.RewardFunction):
    def __init__(self, missed_sales_weight: float):
        self.missed_sales_weight = missed_sales_weight

    def reward(self, ctx: me.Network.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        return (
            ctx.actor.sales
            - self.missed_sales_weight * ctx.actor.missed_sales
            - ctx.actor.stock
        )


@dataclass
class ShopAgentSupertype(ph.BaseSupertype):
    missed_sales_weight: ph.SupertypeField[float]


class ShopAgent(ph.Agent):
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

    def pre_resolution(self, ctx: me.Network.Context):
        # At the start of each step we reset the number of missed orders to 0.
        self.sales = 0
        self.missed_sales = 0

    @me.actors.handler(StockResponse)
    def handle_stock_response(self, ctx: me.Network.Context, msg: me.Message):
        # Messages received from the factory contain stock.
        self.stock = min(self.stock + msg.payload.size, SHOP_MAX_STOCK)

        # We do not need to respond to these messages.
        yield from ()

    @me.actors.handler(OrderRequest)
    def handle_order_request(self, ctx: me.Network.Context, msg: me.Message):
        # All other messages are from customers and contain orders.
        amount_requested = msg.payload.size

        if amount_requested > self.stock:
            self.missed_sales += amount_requested - self.stock
            stock_to_sell = self.stock
            self.stock = 0
        else:
            stock_to_sell = amount_requested
            self.stock -= amount_requested

        self.sales += stock_to_sell

        # Send the customer their order.
        yield (msg.sender_id, [OrderResponse(stock_to_sell)])

    def encode_obs(self, ctx: me.Network.Context):
        return [
            # We include the agent's type in it's observation space to allow it to learn
            # a generalised policy.
            self.type.to_obs_space_compatible_type(),
            # We also encode the shop's current stock in the observation.
            self.stock,
        ]

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from the
        # factory.
        stock_to_request = action

        # We perform this action by sending a stock request message to the factory.
        return ph.packet.Packet(
            messages={self.factory_id: [StockRequest(stock_to_request)]}
        )

    def reset(self):
        super().reset()  # self.type set here

        self.reward_function = ShopRewardFunction(
            missed_sales_weight=self.type.missed_sales_weight
        )

        self.stock = 0

    def get_observation_space(self):
        return gym.spaces.Tuple(
            [
                # We include the agent's type in it's observation space to allow it to
                # learn a generalised policy.
                self.type.to_obs_space(),
                # We also encode the shop's current stock in the observation.
                gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
            ]
        )

    def get_action_space(self):
        return gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST)


SHOP_IDS = [f"SHOP{i+1}" for i in range(NUM_SHOPS)]


class SupplyChainEnv(ph.PhantomEnv):

    env_name: str = "supply-chain-v2"

    def __init__(self, n_customers: int = 5):
        # Define actor and agent IDs
        factory_id = "FACTORY"

        customer_ids = [f"CUST{i+1}" for i in range(n_customers)]

        shop_agents = [ShopAgent(id, factory_id) for id in SHOP_IDS]

        factory_actor = FactoryActor(factory_id)

        customer_agents = [CustomerAgent(id, shop_ids=SHOP_IDS) for id in customer_ids]

        actors = [factory_actor] + shop_agents + customer_agents

        # Define Network and create connections between Actors
        network = me.Network(me.resolvers.UnorderedResolver(), actors)

        # Connect the shops to the factory
        network.add_connections_between(SHOP_IDS, [factory_id])

        # Connect the shop to the customers
        network.add_connections_between(SHOP_IDS, customer_ids)

        clock = ph.Clock(0, NUM_EPISODE_STEPS, 1)

        super().__init__(network=network, clock=clock)


metrics = {}

for id in SHOP_IDS:
    metrics[f"stock/{id}"] = SimpleAgentMetric(id, "stock", "mean")
    metrics[f"sales/{id}"] = SimpleAgentMetric(id, "sales", "mean")
    metrics[f"missed_sales/{id}"] = SimpleAgentMetric(id, "missed_sales", "mean")


if len(sys.argv) == 1 or sys.argv[1].lower() == "train":
    ph.train(
        experiment_name="supply-chain-2",
        algorithm="PPO",
        num_workers=8,
        num_episodes=5000,
        env_class=SupplyChainEnv,
        env_config=dict(
            n_customers=NUM_CUSTOMERS,
        ),
        agent_supertypes={
            id: ShopAgentSupertype(
                missed_sales_weight=UniformSampler(low=0.0, high=8.0)
            )
            for id in SHOP_IDS
        },
        metrics=metrics,
        policy_grouping={"shared_SHOP_policy": SHOP_IDS},
    )

elif sys.argv[1].lower() == "rollout":
    ph.rollout(
        directory="supply-chain-2/LATEST",
        algorithm="PPO",
        num_workers=8,
        num_repeats=20,
        env_class=SupplyChainEnv,
        env_config=dict(
            n_customers=NUM_CUSTOMERS,
        ),
        agent_supertypes={
            id: ShopAgentSupertype(
                missed_sales_weight=UniformRange(
                    start=0.0, end=8.0, step=1.0, name=f"{id} Missed Sales Weight"
                )
            )
            for id in SHOP_IDS
        },
        metrics=metrics,
        save_messages=True,
        save_trajectories=True,
    )
