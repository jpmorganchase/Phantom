from dataclasses import dataclass
import random

import phantom as ph
import numpy as np
from gym.spaces import Box, Discrete

# Message Payloads
##############################################################

# Could extend to include price + vol / price curve
@dataclass(frozen=True)
class Price:
    price: float


@dataclass(frozen=True)
class Order:
    vol: int


# Buyer Agent
###############################################################

# Buyer type = buyer's intrinsic value for the good
@dataclass
class Value(ph.AgentType):
    value: float


class BuyerSupertype(ph.Supertype):
    def __init__(self, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val

    def sample(self) -> Value:
        return Value(np.random.uniform(self.min_val, self.max_val))


class BuyerAgent(ph.Agent):
    # Buyer agent decides whether to buy/not buy and from whom
    # Demand (if it exists) is always 1 unit

    def __init__(self, agent_id, demand_prob, supertype):
        super().__init__(agent_id, supertype=supertype)
        self.seller_prices = dict()
        self.demand_prob = demand_prob
        self.current_reward = 0

    def decode_action(
        self, ctx, action
    ):  # TODO: could do this with an action decoder if we used a view-implementation of a seller
        msgs = dict()
        min_price = min(self.seller_prices.values())

        if action:
            # Get best price seller
            min_sellers = [k for k, v in self.seller_prices.items() if v == min_price]
            seller = random.choice(min_sellers)

            # Send an order to the seller
            msgs[seller] = [Order(action)]
            self.current_reward += -action * min_price + self.type.value

        return ph.packet.Packet([], msgs)

    @property
    def action_space(self):
        # buy (1) or no-buy (0)
        return Discrete(2)

    def encode_obs(self, ctx):
        min_price = min(self.seller_prices.values())
        demand = np.random.binomial(1, self.demand_prob)
        return np.array([min_price, demand, self.type.value])

    @property
    def observation_space(self):
        return Box(low=0, high=1, shape=(3,))

    def compute_reward(self, ctx):
        reward = self.current_reward
        self.current_reward = 0
        return reward

    def handle_message(self, ctx, message):
        if isinstance(message.payload, Price):
            self.seller_prices[message.sender_id] = message.payload.price
        yield from ()

    def reset(self):
        super().reset()
        self.seller_prices = dict()
        self.current_reward = 0


# Seller Agent
####################################################################################################


class SellerAgent(ph.Agent):
    # Infinite supply
    # No inventory costs
    # TODO: could publish a view containing its price - alternate implementation

    def __init__(self, agent_id: ph.AgentID):
        super().__init__(agent_id)
        self.current_price = 0
        self.current_revenue = 0
        self.current_tx = 0  # volume transacted

    def decode_action(self, ctx, action):
        # simple decoder - send price to all connected agents
        self.current_price = action
        msgs = dict()
        for nid in ctx.neighbour_ids:
            msgs[nid] = [Price(action)]
        return ph.packet.Packet([], msgs)

    @property
    def action_space(self):
        # price in [0, 1]
        return Box(low=0, high=1, shape=(1,))

    def encode_obs(self, ctx):
        obs = np.array([self.current_tx, ctx.views["__ENV"].avg_price])
        self.current_tx = 0
        return obs

    @property
    def observation_space(self):
        # [ total transaction vol, Avg market price]
        return Box(np.array([0, 0]), np.array([np.Inf, 1]))

    def compute_reward(self, ctx):
        reward = self.current_revenue
        self.current_revenue = 0
        return reward

    def reset(self):
        self.current_price = self.action_space.sample()
        self.current_revenue = 0
        self.current_tx = 0

    def handle_message(self, ctx, message):
        if isinstance(message.payload, Order):
            self.current_revenue += self.current_price * message.payload.vol
            self.current_tx += message.payload.vol

        yield from ()


# Environment Actor - Not an agent ; a utility for computing average price
#########################################################################################


class SimpleMktEnvActor(ph.env.EnvironmentActor):
    @dataclass(frozen=True)
    class View(ph.agents.View):
        avg_price: float

    def __init__(self):
        super().__init__()
        self.seller_prices = dict()
        self.avg_price = 0.0

    def handle_message(self, ctx, message):
        if isinstance(message.payload, Price):
            self.seller_prices[message.sender_id] = message.payload.price
        yield from ()

    def post_resolution(self, ctx):
        self.avg_price = np.mean(np.array(list(self.seller_prices.values())))

    def view(self, neighbour_id=None) -> "SimpleMktEnvActor.View":
        return self.View(
            actor_id=self._id,
            avg_price=self.avg_price,
        )

    def reset(self):
        super().reset()
        self.seller_prices = dict()
        self.avg_price = 0.0
