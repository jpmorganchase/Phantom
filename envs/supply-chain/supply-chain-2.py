from dataclasses import dataclass
from typing import List, Tuple

import coloredlogs
import gym
import numpy as np
import phantom as ph
from ray import rllib


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


class CustomerPolicy(ph.RLlibFixedPolicy):
    # The size of the order made and the choice of shop to make the order to for each
    # customer is determined by this fixed policy.
    def compute_action(self, obs) -> Tuple[int, int]:
        return (np.random.poisson(5), np.random.randint(self.config["n_shops"]))


class CustomerAgent(ph.MessageHandlerAgent):
    def __init__(self, agent_id: ph.AgentID, shop_ids: List[ph.AgentID]):
        super().__init__(agent_id)

        # We need to store the shop IDs so we can send order requests to them.
        self.shop_ids: List[str] = shop_ids

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(
        self, ctx: ph.Context, sender_id: ph.AgentID, message: ph.Message
    ):
        return []

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # At the start of each step we generate an order with a random size to send to a
        # random shop.
        order_size = action[0]
        shop_id = self.shop_ids[int(action[1])]

        # We perform this action by sending a stock request message to the factory.
        return [(shop_id, OrderRequest(order_size))]

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0.0

    def encode_observation(self, ctx: ph.Context):
        return 0

    @property
    def observation_space(self):
        return gym.spaces.Discrete(1)

    @property
    def action_space(self):
        return gym.spaces.Tuple(
            (
                gym.spaces.Discrete(100),
                gym.spaces.Discrete(len(self.shop_ids)),
            )
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


class ShopRewardFunction(ph.RewardFunction):
    def __init__(self, missed_sales_weight: float):
        self.missed_sales_weight = missed_sales_weight

    def reward(self, ctx: ph.Context) -> float:
        # We reward the agent for making sales.
        # We penalise the agent for holding onto stock and for missing orders.
        return (
            ctx.agent.sales
            - self.missed_sales_weight * ctx.agent.missed_sales
            - ctx.agent.stock
        )


class ShopAgent(ph.MessageHandlerAgent):
    @dataclass(frozen=True)
    class Supertype(ph.Supertype):
        missed_sales_weight: float

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

    def encode_obs(self, ctx: ph.Context):
        return (
            # We include the agent's type in it's observation space to allow it to learn
            # a generalised policy.
            self.type.to_obs_space_compatible_type(),
            # We also encode the shop's current stock in the observation.
            self.stock,
        )

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # The action the shop takes is the amount of new stock to request from the
        # factory.
        stock_to_request = action

        # We perform this action by sending a stock request message to the factory.
        return [(self.factory_id, StockRequest(stock_to_request))]

    def reset(self):
        super().reset()  # self.type set here

        self.reward_function = ShopRewardFunction(
            missed_sales_weight=self.type.missed_sales_weight
        )

        self.stock = 0

    @property
    def observation_space(self):
        return gym.spaces.Tuple(
            [
                # We include the agent's type in it's observation space to allow it to
                # learn a generalised policy.
                self.type.to_obs_space(),
                # We also encode the shop's current stock in the observation.
                gym.spaces.Discrete(SHOP_MAX_STOCK + 1),
            ]
        )

    @property
    def action_space(self):
        return gym.spaces.Discrete(SHOP_MAX_STOCK_REQUEST)


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
    metrics[f"missed_sales/{id}"] = ph.logging.SimpleAgentMetric(
        id, "missed_sales", "mean"
    )


ph.utils.rllib.train(
    algorithm="PPO",
    env_class=SupplyChainEnv2,
    env_config={
        "agent_supertypes": {
            shop_id: {
                "missed_sales_weight": ph.utils.samplers.UniformSampler(
                    low=0.0, high=8.0
                ),
            }
            for shop_id in SHOP_IDS
        }
    },
    num_iterations=2,
    policies={
        "shop_policy": ShopAgent,
        "customer_policy": (CustomerPolicy, CustomerAgent, {"n_shops": NUM_SHOPS}),
    },
    policies_to_train=["shop_policy"],
    metrics=metrics,
)


# if len(sys.argv) == 1 or sys.argv[1].lower() == "train":
#     ph.train(
#         experiment_name="supply-chain-2",
#         algorithm="PPO",
#         num_workers=8,
#         num_episodes=5000,
#         env_class=SupplyChainEnv,
#         env_config=dict(
#             n_customers=NUM_CUSTOMERS,
#         ),
#         agent_supertypes={
#             id: ShopAgentSupertype(
#                 missed_sales_weight=UniformSampler(low=0.0, high=8.0)
#             )
#             for id in SHOP_IDS
#         },
#         metrics=metrics,
#         policy_grouping={"shared_SHOP_policy": SHOP_IDS},
#     )

# elif sys.argv[1].lower() == "rollout":
#     ph.rollout(
#         directory="supply-chain-2/LATEST",
#         algorithm="PPO",
#         num_workers=8,
#         num_repeats=20,
#         env_class=SupplyChainEnv,
#         env_config=dict(
#             n_customers=NUM_CUSTOMERS,
#         ),
#         agent_supertypes={
#             id: ShopAgentSupertype(
#                 missed_sales_weight=UniformRange(
#                     start=0.0, end=8.0, step=1.0, name=f"{id} Missed Sales Weight"
#                 )
#             )
#             for id in SHOP_IDS
#         },
#         metrics=metrics,
#         save_messages=True,
#         save_trajectories=True,
#     )
