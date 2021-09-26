from dataclasses import dataclass
from market_agents import Price, Order, BuyerAgent, SellerAgent
import numpy as np
import mercury as me
from gym.spaces import Box


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

    def __init__(self, agent_id: me.ID, victim_id=None):
        super().__init__(agent_id)
        self.victim_id = victim_id  # None if benign seller
        self.victims_price = 0  # how to represent NA

    def encode_obs(self, ctx):
        obs = np.array(
            [self.current_tx, ctx.views["__ENV"].avg_price, self.victims_price]
        )
        self.current_tx = 0
        return obs

    def get_observation_space(self):
        # [ total transaction vol, Avg market price, (maybe) victim's price]
        return Box(np.array([0, 0, 0]), np.array([np.Inf, 1, 1]))

    def handle_message(self, ctx, message):
        if isinstance(message.payload, Order):
            self.current_revenue += self.current_price * message.payload.vol
            self.current_tx += message.payload.vol

        if isinstance(message.payload, Leak):
            self.victims_price = message.payload.price
            print("received leak message")
            print(message)

        yield from ()


class MaybeLeakyBuyer(BuyerAgent):
    def __init__(self, agent_id, demand_prob, supertype, victim_id=None, adv_id=None):
        super().__init__(agent_id, demand_prob, supertype)
        self.victim_id = victim_id  # if victim and adv id are set -- the leaky buyer will leak info about victim to adv
        self.adv_id = adv_id  # TODO: check if it works as expected if leaky buyer is not connected to victim

    def handle_message(self, ctx, message):
        responses = ()
        if isinstance(message.payload, Price):
            self.seller_prices[message.sender_id] = message.payload.price
            if (
                message.sender_id == self.victim_id
            ):  # Leaks info if victim and adv ids are set
                responses = [
                    (
                        self.adv_id,
                        [Leak(victim_id=self.victim_id, price=message.payload.price)],
                    )
                ]
                print(responses)

        yield from responses
