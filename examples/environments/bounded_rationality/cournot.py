import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from typing import Sequence
import pandas as pd
import sys

import phantom as ph
from phantom.utils.samplers import NormalSampler

from ray.rllib.models import ModelCatalog
from custom import BoundedStrategicAgent, TorchOrdinalCategorical, BoundedPolicies


NUM_EPISODE_STEPS = 200
NUM_FIRMS = 2  # Oligopoly size, e.g., 2 for duopoly, 3 for triopoloy
MIN_QUANTITY = 8
MAX_QUANTITY = 32

# For output files
names = {
    2: "duopoly",
    3: "triopoly",
}

A = 2.4
B = 0.04


PROMINENT_PRIORS = True


@dataclass(frozen=True)
class QuantityProduced(ph.MsgPayload):
    """Firm --> Market"""
    quantity: int


@dataclass(frozen=True)
class MarketResponse(ph.MsgPayload):
    """Market --> Firm"""
    profit: float
    other_firm_total_quantities: int


class MarketAgent(ph.Agent):
    """
    The Market determines the price of the goods and profit for each firm based on the overall quantities produced.
    """
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def profit_for_quantity(self, ctx: ph.Context, message: ph.Message, price: float, total_quantity_produced: int):
        quantity_produced = message.payload.quantity

        # Based on overall quantities produced by all firms
        profit = quantity_produced * price

        # Quantitites from other firms
        other_firm_total_quantities = total_quantity_produced - quantity_produced

        return [(message.sender_id, MarketResponse(profit, other_firm_total_quantities))]

    def handle_batch(
            self, ctx: ph.Context, batch: Sequence[ph.Message]
    ):
        # The total quantity of goods produced by all firms
        all_quantities = [message.payload.quantity for message in batch]
        total_quantity_produced = sum(all_quantities)

        # Market price dependent on the quantity produced
        price = A - B * total_quantity_produced

        all_responses = []
        for message in batch:
            response = self.profit_for_quantity(ctx, message, price=price,
                                                total_quantity_produced=total_quantity_produced)
            if response:
                all_responses += response

        return all_responses


class FirmAgent(BoundedStrategicAgent):
    def __init__(self, agent_id: str, market_id: str):
        super().__init__(agent_id)

        # We store the ID of the market so we can send produce to it
        self.market_id: str = market_id

        # Can generate quantities between min and max quantity (inclusive of both)
        number_possible_quantities = MAX_QUANTITY - MIN_QUANTITY + 1

        # Note: in decode_action, we add min_quantity to the result. Using the start param here doesnt work here
        # as this is newer than rllib supports
        self.action_space = gym.spaces.Discrete(number_possible_quantities)

        # Uniform prior
        self.prior_beliefs = [1./number_possible_quantities] * number_possible_quantities

        # "Spiky" prior
        if PROMINENT_PRIORS:
            self.prior_beliefs = [1.] * number_possible_quantities
            # 0 = 8, 1=9, 2=10, etc...
            self.prior_beliefs[2] = 10.
            self.prior_beliefs[7] = 10.
            self.prior_beliefs[12] = 10.
            self.prior_beliefs[17] = 10.
            self.prior_beliefs[22] = 10.

        self.prior_beliefs = np.asarray(self.prior_beliefs)
        self.prior_beliefs /= np.sum(self.prior_beliefs)

        # = [Firm ID, Sum other contributions,  Mean historical sum other contributions, Penalty, Priors]
        self.observation_space = gym.spaces.Box(low=np.array([1, MIN_QUANTITY * (NUM_FIRMS - 1), 0, 0] + [0.0] * len(self.prior_beliefs)),
                                                high=np.array([NUM_FIRMS] + [MAX_QUANTITY * (NUM_FIRMS - 1)] * 2 + [np.inf] + [1.0] * len(self.prior_beliefs)),
                                                shape=(4 + len(self.prior_beliefs),))

        self.other_firm_total_quantities = random.randint(MIN_QUANTITY * (NUM_FIRMS - 1), MAX_QUANTITY * (NUM_FIRMS - 1))
        self.historical_total_quantities = []

    @ph.agents.msg_handler(MarketResponse)
    def handle_market_profit(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the market contain the profit for the agent
        profit = message.payload.profit

        # Scaled as done in the experiment
        self.profit = 0.01 * profit

        # The total amount produced from the other firms
        self.other_firm_total_quantities = message.payload.other_firm_total_quantities

        self.historical_total_quantities.append(self.other_firm_total_quantities)

    def _safe_mean(self, values):
        # Protect against taking mean of empty list
        return np.mean(values) if len(values) > 0 else 0

    def encode_observation(self, ctx: ph.Context):
        integer_id = int(self._id.split("FIRM")[1])  # ID is in form FIRMX
        return np.array([integer_id, self.other_firm_total_quantities, self._safe_mean(self.historical_total_quantities),
                         self.type.penalty] + list(self.prior_beliefs), dtype=np.float32)

    def decode_action(self, ctx: ph.Context, action: int):
        # Actions start at 0 but should start at min_quantity
        action = action + MIN_QUANTITY

        # The action the firm takes is the amount of goods to produce. Send this to the market
        return [(self.market_id, QuantityProduced(action))]

    def compute_reward(self, ctx: ph.Context):
        return self.profit

    def reset(self):
        super().reset()
        self.profit = 0
        self.other_firm_total_quantities = random.randint(MIN_QUANTITY * (NUM_FIRMS - 1), MAX_QUANTITY * (NUM_FIRMS - 1))
        self.historical_total_quantities = []

class CournotEnv(ph.PhantomEnv):
    def __init__(self):
        # Define agent IDs
        market_ID = "MARKET"
        firm_IDs = [f"FIRM{i+1}" for i in range(NUM_FIRMS)]

        # Create agents
        market_agent = MarketAgent(market_ID)
        firm_agents = [FirmAgent(cid, market_id=market_ID) for cid in firm_IDs]
        agents = [market_agent] + firm_agents

        agent_supertypes = {
            # We clip values to ensure >0
            cid: {"penalty": NormalSampler(mu=MU, sigma=SIGMA, clip_low=1e-05)}
            for cid in firm_IDs
        }

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the firms to the market
        network.add_connections_between([market_ID], firm_IDs)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network, agent_supertypes=agent_supertypes)


def train():
    ModelCatalog.register_custom_action_dist("ordinal_categorical_dist", TorchOrdinalCategorical)

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=CournotEnv,
        env_config={},
        rllib_config={"model": {"fcnet_hiddens": [64, 64], "custom_action_dist": "ordinal_categorical_dist"}},
        iterations=500,
        checkpoint_freq=None,
        custom_callbacks=[BoundedPolicies],
        policies={"firm_policy": FirmAgent},  # Each firm shares policy
        metrics=metrics,
        results_dir=folder_name,
        show_training_metrics=True,
    )


def rollout():
    results = ph.utils.rllib.rollout(
        directory=folder_name + "/LATEST",
        num_repeats=100,
        num_workers=1,
        metrics=metrics,
        env_class=CournotEnv,
    )

    results = list(results)

    firms = [f"FIRM{i+1}" for i in range(NUM_FIRMS)]

    all_requests = []

    for firm in firms:

        firm_quantities = []

        for rollout in results:
            firm_quantities += [action + MIN_QUANTITY for action in rollout.actions_for_agent(firm)]

        all_requests.append(firm_quantities)

    all_contributions = pd.DataFrame(all_requests)

    if PROMINENT_PRIORS:
        all_contributions.to_csv(f"out/cournot-spiky/{names[NUM_FIRMS]}/{MU}-{SIGMA}_firm_quantities.csv")
    else:
        all_contributions.to_csv(f"out/cournot/{names[NUM_FIRMS]}/{MU}-{SIGMA}_firm_quantities.csv")


if __name__ == '__main__':
    metrics = {}

    MU = float(sys.argv[2])
    SIGMA = MU * float(sys.argv[3])

    if PROMINENT_PRIORS:
        folder_name = f"~/ray_results/cournot-spiky/{names[NUM_FIRMS]}/{MU}-{SIGMA}"
    else:
        folder_name = f"~/ray_results/cournot/{names[NUM_FIRMS]}/{MU}-{SIGMA}"

    if sys.argv[1] == "all":
        train()
        rollout()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "rollout":
        rollout()
