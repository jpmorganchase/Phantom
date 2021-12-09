import phantom as ph
import mercury as me
from mercury.resolvers import UnorderedResolver

from market_agents import BuyerAgent, SellerAgent, SimpleMktEnvActor


class SimpleMarketEnv(ph.PhantomEnv):
    def __init__(self, network, clock):
        self.network = network
        self.clock = clock
        self.agents = {
            aid: actor
            for aid, actor in network.actors.items()
            if isinstance(actor, ph.Agent)
        }

        self._add_env_actor(self.network)

        # TODO: replace with agent groups from mercury
        self.agent_groups = list()
        self.agent_groups.append(
            [
                aid
                for aid, actor in network.actors.items()
                if isinstance(actor, SellerAgent)
            ]
        )
        self.agent_groups.append(
            [
                aid
                for aid, actor in network.actors.items()
                if isinstance(actor, BuyerAgent)
            ]
        )
        print(self.agent_groups)

        self.turn = 0
        self.num_groups = 2

    def _add_env_actor(self, network: me.Network) -> None:
        """Add an omniscient actor to the network."""
        env_actor = SimpleMktEnvActor()
        network.add_actor(env_actor)
        self.network.add_connections_between([env_actor.id], list(network.actor_ids))

    def step(self, actions):
        self.clock.tick()

        print("messages sent")

        # Decode actions and send messages
        for aid, action in actions.items():
            ctx = self.network.context_for(aid)
            packet = self.agents[aid].decode_action(ctx, action)
            self.network.send_from(aid, packet.messages)
            print(packet.messages)

        self.network.resolve()

        # Pass the turn
        self.turn = (self.turn + 1) % self.num_groups

        # Return observations for agents with the turn
        # so they can act in the next step
        obs = dict()
        rewards = dict()
        info = dict()
        for aid in self.agent_groups[self.turn]:
            agent = self.agents[aid]
            ctx = self.network.context_for(aid)
            obs[aid] = agent.encode_obs(ctx)
            rewards[aid] = agent.compute_reward(ctx)
            info[aid] = {"turn": True}

        return self.Step(
            observations=obs,
            rewards=rewards,
            terminals={"__all__": self.clock.is_terminal},
            infos=info,
        )

    def reset(self):
        self.clock.reset()
        self.network.reset()  # will call reset on the agents/actors
        self.turn = 0

        obs = dict()
        info = dict()
        for aid in self.agent_groups[self.turn]:
            obs[aid] = self.agents[aid].get_observation_space().sample()
            info[aid] = {"turn": True}

        return self.Step(
            observations=obs,
            rewards={},
            terminals={"__all__": self.clock.is_terminal},
            infos=info,
        )
