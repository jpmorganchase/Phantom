import random
from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from typing import Sequence
import pandas as pd

# Use local version of phantom
import sys
PATH = "INSERT_FOLDER_PATH_HERE"
sys.path.insert(0, PATH + 'Bounded MARL/')
import phantom as ph
print("Phantom version", ph.__version__)
from bounded_games.bounded_phantom import BoundedStrategicAgent
from phantom.utils.samplers import NormalSampler

NUM_EPISODE_STEPS = 50
NUM_PARTICIPANTS = 6
LAMBDA = 2
A = 13.8
B = 1.5

# 0..10
MAX_PRICE = 10.0


@dataclass(frozen=True)
class PricePrediction(ph.MsgPayload):
    """Producer --> Market"""
    price: float


@dataclass(frozen=True)
class MarketResponse(ph.MsgPayload):
    """Market --> Producer"""
    payoff: float
    realized_market_price: float


class MarketAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def supply(self, price):
        return np.tanh(LAMBDA * (price - NUM_PARTICIPANTS)) + 1

    def realized_equilibrium_price(self, predictions):
        noise = np.random.normal(0, 0.5)
        return (A - np.sum(self.supply(predictions))) / B + noise

    def payoff(self, predicted_price, realized_market_price):
        # Normalised to 0..1
        return max(0, 1300 - 260 * (realized_market_price - predicted_price) ** 2) / 1300

    def handle_order_request(self, ctx: ph.Context, message: ph.Message,
                             realized_market_price: float):

        prediction = message.payload.price
        payoff = self.payoff(prediction, realized_market_price)

        return [(message.sender_id, MarketResponse(payoff, realized_market_price))]

    def handle_batch(
            self, ctx: ph.Context, batch: Sequence[ph.Message]
    ):
        # Predicted price from each producer
        predictions = np.asarray([message.payload.price for message in batch])
        realized_market_price = self.realized_equilibrium_price(predictions)

        # Generate individual messages based on the realized price
        all_responses = []
        for message in batch:
            response = self.handle_order_request(ctx, message, realized_market_price=realized_market_price)

            if response:
                all_responses += response

        return all_responses


class ProducerAgent(BoundedStrategicAgent):
    def __init__(self, agent_id: str, market_id: str):
        super().__init__(agent_id)

        # We store the ID of the factory so we can send stock requests to it.
        self.market_id: str = market_id

        # = Price prediction. Discretized, we divide by MAX_PRICE in decode_action
        self.action_space = gym.spaces.Discrete(100)
        self.prior_beliefs = [1./100] * 100

        # = [Producer ID, Realized price, Mean Historical Realized price, Mean Historical Forecasted price, Penalty, Priors]
        self.observation_space = gym.spaces.Box(low=np.array([1, 0.0, 0.0, 0.0, 0.0] + [0.0] * len(self.prior_beliefs)),
                                                high=np.array([NUM_PARTICIPANTS, MAX_PRICE+1, MAX_PRICE+1, MAX_PRICE+1, np.inf] + [1.0] * len(self.prior_beliefs)),
                                                shape=(5 + len(self.prior_beliefs),))

        # Random to begin with, gets updated by market
        self.realized_market_price = random.randint(0, 100) / MAX_PRICE
        self.payoff = random.random()
        self.historical_forecasts = []
        self.historical_realized_price = []

    @ph.agents.msg_handler(MarketResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the market contain the payoff
        self.payoff = message.payload.payoff

        # And the realized market price
        self.realized_market_price = message.payload.realized_market_price
        self.historical_realized_price.append(self.realized_market_price)

    def _safe_mean(self, values):
        # Protect against taking mean of empty list
        return np.mean(values) if len(values) > 0 else 0

    def encode_observation(self, ctx: ph.Context):
        integer_id = int(self._id.split("PRODUCER")[1])  # ID is in form PRODUCERX
        return np.array([integer_id,
                         self.realized_market_price, self._safe_mean(self.historical_realized_price), self._safe_mean(self.historical_forecasts),
                         self.type.penalty] + list(self.prior_beliefs))

    def decode_action(self, ctx: ph.Context, action: int):
        # We have discrete actions 0..100, but convert to range 0..10
        action = action / MAX_PRICE

        # Track all decisions we made for passing into observation space
        self.historical_forecasts.append(action)

        # The action the producer takes is the predicted price
        return [(self.market_id, PricePrediction(action))]

    def compute_reward(self, ctx: ph.Context):
        # Received from the market
        return self.payoff

    def reset(self):
        super().reset()
        self.realized_market_price = random.randint(0, 100) / MAX_PRICE
        self.payoff = random.random()
        self.historical_forecasts = []
        self.historical_realized_price = []


class CobwebEnv(ph.PhantomEnv):
    def __init__(self):
        # Configure agent IDs
        market_ID = "MARKET"
        producer_IDs = [f"PRODUCER{i+1}" for i in range(NUM_PARTICIPANTS)]

        # Create agents
        market_agent = MarketAgent(market_ID)
        producer_agents = [ProducerAgent(cid, market_id=market_ID) for cid in producer_IDs]
        agents = [market_agent] + producer_agents

        # Processing penalties based on MU and sigma
        agent_supertypes = {
            # We clip values to ensure >0
            cid: {"penalty": NormalSampler(mu=MU, sigma=SIGMA, clip_low=1e-05)}
            for cid in producer_IDs
        }

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the producers to the market
        network.add_connections_between([market_ID], producer_IDs)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network, agent_supertypes=agent_supertypes)


def train():
    ph.utils.rllib.train(
        algorithm=ALGORITHM,
        env_class=CobwebEnv,
        env_config={},
        rllib_config={"model": {"fcnet_hiddens": [64, 64], "custom_action_dist": "ordinal_categorical_dist"}},
        iterations=100,
        checkpoint_freq=None,
        policies={"producer_policy": ProducerAgent},  # Each producer shares policy
        metrics=metrics,
        results_dir=f"~/ray_results/{ALGORITHM}/cobweb/{MU}-{SIGMA}",
        show_training_metrics=True,
    )


def rollout():
    results = ph.utils.rllib.rollout(
        directory=f"~/ray_results/{ALGORITHM}/cobweb/{MU}-{SIGMA}/LATEST",
        num_repeats=100,
        num_workers=1,
        metrics=metrics,
        env_class=CobwebEnv,
    )

    results = list(results)

    producers = [f"PRODUCER{i+1}" for i in range(NUM_PARTICIPANTS)]

    all_requests = []

    for producer in producers:

        producer_requests = []

        for rollout in results:
            producer_requests += [action / MAX_PRICE for action in rollout.actions_for_agent(producer)]

        all_requests.append(producer_requests)

    all_contributions = pd.DataFrame(all_requests)
    all_contributions.to_csv(f"out/cobweb/{MU}-{SIGMA}_preds.csv")


if __name__ == '__main__':

    metrics = {}
    ALGORITHM = "BoundedPPO"  # BoundedPPO for proposed, PPO for comparison

    MU = float(sys.argv[2])
    SIGMA = MU * float(sys.argv[3])

    if sys.argv[1] == "all":
        train()
        rollout()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "rollout":
        rollout()