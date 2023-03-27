from dataclasses import dataclass
from market_agents import Price, Order, BuyerAgent, SellerAgent
import numpy as np
import phantom as ph
from gymnasium.spaces import Box


# Message Payloads
##############################################


@dataclass(frozen=True)
class Leak:
    victim_id: str
    price: float


# Agents
############################################


class MaybeSneakySeller(SellerAgent):
    # Acts as a regular seller agent
    # Except when victim_id is set - then it acts as an adversary

    def __init__(self, agent_id: ph.AgentID, victim_id=None):
        super().__init__(agent_id)
        self.victim_id = victim_id  # None if benign seller
        self.victims_price = 0  # how to represent NA

        # [total transaction vol, Avg market price, (maybe) victim's price]
        self.observation_space = Box(np.array([0, 0, 0]), np.array([np.Inf, 1, 1]))

    def encode_observation(self, ctx):
        obs = np.array([self.current_tx, ctx.env_view.avg_price, self.victims_price])
        self.current_tx = 0
        return obs

    @ph.agents.msg_handler(Order)
    def handle_order_message(self, ctx, message):
        self.current_revenue += self.current_price * message.payload.vol
        self.current_tx += message.payload.vol

    @ph.agents.msg_handler(Leak)
    def handle_leak_message(self, ctx, message):
        self.victims_price = message.payload.price
        print("received leak message")
        print(message)


class MaybeLeakyBuyer(BuyerAgent):
    def __init__(self, agent_id, demand_prob, supertype, victim_id=None, adv_id=None):
        super().__init__(agent_id, demand_prob, supertype)
        self.victim_id = victim_id  # if victim and adv id are set -- the leaky buyer will leak info about victim to adv
        self.adv_id = adv_id  # TODO: check if it works as expected if leaky buyer is not connected to victim

    @ph.agents.msg_handler(Price)
    def handle_price_message(self, ctx, message):
        self.seller_prices[message.sender_id] = message.payload.price
        # Leaks info if victim and adv ids are set
        if message.sender_id == self.victim_id:
            responses = [
                (
                    self.adv_id,
                    Leak(victim_id=self.victim_id, price=message.payload.price),
                )
            ]
            print(responses)
            return responses
