import phantom as ph
from market_agents import BuyerAgent, BuyerSupertype, SellerAgent
from base_policy import BuyerPolicy, SellerPolicy
from simple_mkt_env import SimpleMarketEnv

# Agents
b1 = BuyerAgent("b1", 0.2, supertype=BuyerSupertype(0.2, 0.2))
b2 = BuyerAgent("b2", 0.9, supertype=BuyerSupertype(1, 1))
b3 = BuyerAgent("b3", 0.9, supertype=BuyerSupertype(0.5, 0.5))
s1 = SellerAgent("s1")
s2 = SellerAgent("s2")
buyer_agents = [b1, b2, b3]
seller_agents = [s1, s2]
all_agents = buyer_agents + seller_agents

# Network definition
network = ph.Network(ph.resolvers.BatchResolver(chain_limit=2), actors=all_agents)

# Add connections in the network
network.add_connections_between(["b1", "b2", "b3"], ["s1", "s2"])

# Setup env
env = SimpleMarketEnv(num_steps=10, network=network)

# Run
out = env.reset()
while env.current_step < env.num_steps:
    print(env.current_step)

    print("obs")
    print(out.observations)
    print("rewards")
    print(out.rewards)
    print("info")
    print(out.infos)

    # Only the agents that got observations can act
    actions = dict()
    for aid, obs in out.observations.items():
        agent = env.agents[aid]
        if isinstance(agent, BuyerAgent):
            actions[aid] = BuyerPolicy(obs)
        elif isinstance(agent, SellerAgent):
            actions[aid] = SellerPolicy(obs)

    print("actions")
    print(actions)

    out = env.step(actions)
