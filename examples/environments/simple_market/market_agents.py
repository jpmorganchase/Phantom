from dataclasses import dataclass
import random

import phantom as ph
import numpy as np
from gymnasium.spaces import Box, Discrete

# Message Payloads
##############################################################


# Could extend to include price + vol / price curve
@dataclass(frozen=True)
class Price(ph.MsgPayload):
    price: float


@dataclass(frozen=True)
class Order(ph.MsgPayload):
    vol: int


# Buyer Agent
###############################################################


# Buyer type = buyer's intrinsic value for the good
@dataclass
class BuyerSupertype(ph.Supertype):
    value: float


class BuyerAgent(ph.StrategicAgent):
    # Buyer agent decides whether to buy/not buy and from whom
    # Demand (if it exists) is always 1 unit

    def __init__(self, agent_id, demand_prob, supertype):
        super().__init__(agent_id, supertype=supertype)
        self.seller_prices = {}
        self.demand_prob = demand_prob
        self.current_reward = 0

        # buy (1) or no-buy (0)
        self.action_space = Discrete(2)

        self.observation_space = Box(low=0, high=1, shape=(3,))

    def decode_action(self, ctx, action):
        # TODO: could do this with an action decoder if we used a view-implementation of a seller
        msgs = []
        min_price = min(self.seller_prices.values())

        if action:
            # Get best price seller
            min_sellers = [k for k, v in self.seller_prices.items() if v == min_price]
            seller = random.choice(min_sellers)

            # Send an order to the seller
            msgs.append((seller, Order(action)))
            self.current_reward += -action * min_price + self.type.value

        return msgs

    def encode_observation(self, ctx):
        min_price = min(self.seller_prices.values())
        demand = np.random.binomial(1, self.demand_prob)
        return np.array([min_price, demand, self.type.value])

    def compute_reward(self, ctx):
        reward = self.current_reward
        self.current_reward = 0
        return reward

    @ph.agents.msg_handler(Price)
    def handle_price_message(self, ctx, message):
        self.seller_prices[message.sender_id] = message.payload.price

    def reset(self):
        super().reset()
        self.seller_prices = {}
        self.current_reward = 0


# Seller Agent
####################################################################################################


class SellerAgent(ph.StrategicAgent):
    # Infinite supply
    # No inventory costs
    # TODO: could publish a view containing its price - alternate implementation

    def __init__(self, agent_id: ph.AgentID):
        super().__init__(agent_id)
        self.current_price = 0
        self.current_revenue = 0
        self.current_tx = 0  # volume transacted

        # price in [0, 1]
        self.action_space = Box(low=0, high=1, shape=(1,))

        # [total transaction vol, avg market price]
        self.observation_space = Box(np.array([0, 0]), np.array([np.inf, 1]))

    def decode_action(self, ctx, action):
        # simple decoder - send price to all connected agents
        self.current_price = action

        return [(nid, Price(action)) for nid in ctx.neighbour_ids]

    def encode_observation(self, ctx):
        obs = np.array([self.current_tx, ctx.env_view.avg_price])
        self.current_tx = 0
        return obs

    def compute_reward(self, ctx):
        reward = self.current_revenue
        self.current_revenue = 0
        return reward

    def reset(self):
        self.current_price = self.action_space.sample()
        self.current_revenue = 0
        self.current_tx = 0

    @ph.agents.msg_handler(Order)
    def handle_order_message(self, ctx, message):
        self.current_revenue += self.current_price * message.payload.vol
        self.current_tx += message.payload.vol
