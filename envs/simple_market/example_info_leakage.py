import mercury as me
import phantom as ph
from market_agents import BuyerSupertype, BuyerAgent, SellerAgent
from base_policy import BuyerPolicy, SellerPolicy
from info_leakage_agents import MaybeLeakyBuyer, MaybeSneakySeller
from info_leakage_env import LeakySimpleMarketEnv
from mercury.resolvers import UnorderedResolver


def rollout(env, clock):
    out = env.reset()

    while not clock.is_terminal:
        print(clock.elapsed)

        print("obs")
        print(out.observations)
        print("rewards")
        print(out.rewards)

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


if __name__ == "__main__":
    # Setup some benign Agents; fixed types
    b1 = MaybeLeakyBuyer("b1", 0.2, supertype=BuyerSupertype(0.2, 0.2))
    b2 = MaybeLeakyBuyer("b2", 0.9, supertype=BuyerSupertype(1, 1))
    b3 = MaybeLeakyBuyer("b3", 0.9, supertype=BuyerSupertype(0.5, 0.5))
    s1 = MaybeSneakySeller("s1")
    s2 = MaybeSneakySeller("s2")
    buyer_agents = [b1, b2, b3]
    seller_agents = [s1, s2]
    all_agents = buyer_agents + seller_agents

    # Network definition
    network = me.Network(UnorderedResolver(chain_limit=4), actors=all_agents)

    # Add connections in the network
    network.add_connections_between(["b1", "b2", "b3"], ["s1", "s2"])

    # Setup env without any leaks - should work just like the vanilla simple market env
    print("====================================")
    print("NO INFO LEAKAGE")
    print("====================================")
    clock = ph.Clock(0, 5, 1)
    env = LeakySimpleMarketEnv(network=network, clock=clock)
    rollout(env, clock)

    # Shift to an adversarial setup
    print("====================================")
    print("WITH INFO LEAKAGE")
    print("====================================")
    env.adversarial_setup(leaky_buyer="b1", adv_seller="s1", victim_seller="s2")

    print("leaky buyer")
    print(env.leaky_buyer)
    print("adv seller")
    print(env.adv_seller)
    print("victim seller")
    print(env.victim_seller)

    rollout(env, clock)
