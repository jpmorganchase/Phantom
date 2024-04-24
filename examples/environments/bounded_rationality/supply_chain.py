from dataclasses import dataclass
import gymnasium as gym
import numpy as np
from typing import Sequence
import pandas as pd
import sys

import phantom as ph
from phantom.utils.samplers import NormalSampler
from custom import BoundedStrategicAgent, TorchOrdinalCategorical, BoundedPolicies
from ray.rllib.models import ModelCatalog

NUM_EPISODE_STEPS = 200
NUM_RETAILERS = 2
RETAILER_MAX_ORDER_SIZE = 100
SUPPLIER_CAPACITY = 90

DEMAND = 50
WASTAGE_COST = 2
SHORTAGE_COST = 5

# We only have data on the bins, so group our actions into these bins instead
ACTION_BINS = {
    0: 20,
    1: 40,
    2: 50,
    3: 60,
    4: 70,
    5: 80,
    6: 90,
    7: 100,
}

# For indexing
ACTION_BIN_VALUES = list(ACTION_BINS.values())


@dataclass(frozen=True)
class OrderRequest(ph.MsgPayload):
    """Retailer --> Supplier"""
    size: int


@dataclass(frozen=True)
class OrderResponse(ph.MsgPayload):
    """Supplier --> Retailer"""
    size: int


class SupplierAgent(ph.Agent):
    def __init__(self, agent_id: str):
        super().__init__(agent_id)

    def handle_order_request(self, ctx: ph.Context, message: ph.Message,
                             total_request_size: int):
        request_size = message.payload.size

        # Grants proportionally based on the supply and overall requests (and protects against divide by 0)
        granted_size = SUPPLIER_CAPACITY * (request_size / total_request_size) if total_request_size > 0 else 0

        return [(message.sender_id, OrderResponse(granted_size))]

    def handle_batch(
            self, ctx: ph.Context, batch: Sequence[ph.Message]
    ):
        # For proportionally allocating, we take the total amount of requests
        total_request_size = sum(message.payload.size for message in batch)

        all_responses = []

        for message in batch:
            response = self.handle_order_request(ctx, message, total_request_size=total_request_size)

            if response:
                all_responses += response

        return all_responses


class RetailerAgent(BoundedStrategicAgent):
    def __init__(self, agent_id: str, supplier_id: str):
        super().__init__(agent_id)

        # We store the ID of the supplier so we can send stock requests to it
        self.supplier_id: str = supplier_id

        # Request size (binned)
        self.action_space = gym.spaces.Discrete(len(ACTION_BINS))
        self.prior_beliefs = [1./len(ACTION_BINS)] * len(ACTION_BINS)

        # = [Retailer ID, Received stock, Penalty, Priors]
        self.observation_space = gym.spaces.Box(low=np.array([1, 0.0, 0.0] + [0.0] * len(self.prior_beliefs)),
                                                high=np.array([NUM_RETAILERS, RETAILER_MAX_ORDER_SIZE, np.inf] + [1.0] * len(self.prior_beliefs)),
                                                shape=(3 + len(self.prior_beliefs),))

        self.received_stock = 0

    @ph.agents.msg_handler(OrderResponse)
    def handle_order_response(self, ctx: ph.Context, message: ph.Message):
        # Messages received from the supplier contain stock.
        self.received_stock = message.payload.size

    def encode_observation(self, ctx: ph.Context):
        integer_id = int(self._id.split("RETAILER")[1])  # ID is in form RETAILERX
        obs = np.array([integer_id, self.received_stock, self.type.penalty] + list(self.prior_beliefs),
                       dtype=np.float32)
        return obs

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        # The action the retailer takes is the amount of stock to request from the supplier
        stock_to_request = ACTION_BINS[action]

        self.requested_bin = action

        # We perform this action by sending a stock request message to the factory.
        return [(self.supplier_id, OrderRequest(stock_to_request))]

    def compute_reward(self, ctx: ph.Context):
        received_stock = self.received_stock

        wastage = max(received_stock - DEMAND, 0)
        shortage = max(DEMAND - received_stock, 0)
        reward = (DEMAND * SHORTAGE_COST) - WASTAGE_COST * wastage - SHORTAGE_COST * shortage

        # Scale reward
        max_reward = DEMAND * SHORTAGE_COST
        return reward / max_reward

    def reset(self):
        super().reset()
        self.received_stock = 0
        self.prior_beliefs = [1./len(ACTION_BINS)] * len(ACTION_BINS)


class SupplyChainEnv(ph.PhantomEnv):
    def __init__(self):
        # Define agent IDs
        supplier_ID = "SUPPLIER"
        retailer_IDs = [f"RETAILER{i+1}" for i in range(NUM_RETAILERS)]

        # Create agents
        supplier_agent = SupplierAgent(supplier_ID)
        retailer_agents = [RetailerAgent(cid, supplier_id=supplier_ID) for cid in retailer_IDs]
        agents = [supplier_agent] + retailer_agents

        agent_supertypes = {
            # We clip values to ensure >0
            cid: {"penalty": NormalSampler(mu=MU, sigma=SIGMA, clip_low=1e-05)}
            for cid in retailer_IDs
        }

        # Define Network and create connections between Actors
        network = ph.Network(agents)

        # Connect the retailers to the supplier
        network.add_connections_between([supplier_ID], retailer_IDs)

        super().__init__(num_steps=NUM_EPISODE_STEPS, network=network, agent_supertypes=agent_supertypes)


def train():
    ModelCatalog.register_custom_action_dist("ordinal_categorical_dist", TorchOrdinalCategorical)

    ph.utils.rllib.train(
        algorithm="PPO",
        env_class=SupplyChainEnv,
        env_config={},
        rllib_config={"model": {"fcnet_hiddens": [64, 64], "custom_action_dist": "ordinal_categorical_dist"}},
        iterations=500,
        checkpoint_freq=None,
        policies={"retailer_policy": RetailerAgent},  # Each retailer shares policy
        custom_callbacks=[BoundedPolicies],
        metrics=metrics,
        results_dir=dir,
        show_training_metrics=True,
    )


def rollout():
    results = ph.utils.rllib.rollout(
        directory=dir + "/LATEST",
        num_repeats=100,
        num_workers=1,
        metrics=metrics,
        env_class=SupplyChainEnv,
    )

    results = list(results)

    retailers = [f"RETAILER{i+1}" for i in range(NUM_RETAILERS)]

    all_requests = []

    for retailer in retailers:

        retailer_requests = []

        for rollout in results:
            retailer_requests += rollout.actions_for_agent(retailer)

        all_requests.append(retailer_requests)

    all_contributions = pd.DataFrame(all_requests)

    all_contributions.to_csv(f"out/supply_chain/{MU}-{SIGMA}_requests.csv")


if __name__ == '__main__':

    metrics = {}

    MU = float(sys.argv[2])
    SIGMA = MU * float(sys.argv[3])

    dir = f"~/ray_results/supply_chain/{MU}-{SIGMA}"

    if sys.argv[1] == "all":
        train()
        rollout()
    elif sys.argv[1] == "train":
        train()
    elif sys.argv[1] == "rollout":
        rollout()