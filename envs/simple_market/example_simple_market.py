import phantom as ph
from phantom.utils.rollout import Step
from phantom.utils.samplers import UniformSampler

from base_policy import BuyerPolicy, SellerPolicy
from market_agents import BuyerAgent, BuyerSupertype, SellerAgent
from simple_mkt_env import SimpleMarketEnv


# Agents
b1 = BuyerAgent("b1", 0.2, supertype=BuyerSupertype(UniformSampler(0.2, 0.2)))
b2 = BuyerAgent("b2", 0.9, supertype=BuyerSupertype(UniformSampler(1, 1)))
b3 = BuyerAgent("b3", 0.9, supertype=BuyerSupertype(UniformSampler(0.5, 0.5)))
s1 = SellerAgent("s1")
s2 = SellerAgent("s2")
buyer_agents = [b1, b2, b3]
seller_agents = [s1, s2]
all_agents = buyer_agents + seller_agents

# Network definition
network = ph.Network(all_agents)

# Add connections in the network
network.add_connections_between(["b1", "b2", "b3"], ["s1", "s2"])

# Setup env
env = SimpleMarketEnv(num_steps=10, network=network)

# Run
observations = env.reset()
rewards = {}
infos = {}

while env.current_step < env.num_steps:
    print(env.current_step)

    print("\nobservations:")
    print(observations)
    print("\nrewards:")
    print(rewards)
    print("\ninfos:")
    print(infos)

    # Only the agents that got observations can act
    actions = {}
    for aid, obs in observations.items():
        agent = env.agents[aid]
        if isinstance(agent, BuyerAgent):
            actions[aid] = BuyerPolicy(obs)
        elif isinstance(agent, SellerAgent):
            actions[aid] = SellerPolicy(obs)

    print("\nactions:")
    print(actions)

    observations, rewards, _, infos = env.step(actions)
