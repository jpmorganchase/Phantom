from dataclasses import dataclass

import numpy as np
import phantom as ph

from market_agents import BuyerAgent, SellerAgent


class SimpleMarketEnv(ph.FiniteStateMachineEnv):
    @dataclass(frozen=True)
    class View(ph.fsm.FSMEnvView):
        avg_price: float

    def __init__(self, num_steps, network):
        buyers = [
            aid
            for aid, agent in network.agents.items()
            if isinstance(agent, BuyerAgent)
        ]

        sellers = [
            aid
            for aid, agent in network.agents.items()
            if isinstance(agent, SellerAgent)
        ]

        stages = [
            ph.FSMStage(
                stage_id="Buyers",
                next_stages=["Sellers"],
                acting_agents=buyers,
                rewarded_agents=buyers,
            ),
            ph.FSMStage(
                stage_id="Sellers",
                next_stages=["Buyers"],
                acting_agents=sellers,
                rewarded_agents=sellers,
            ),
        ]

        self.avg_price = 0.0

        super().__init__(num_steps, network, stages=stages, initial_stage="Sellers")

    def view(self, neighbour_id=None) -> "SimpleMarketEnv.View":
        return self.View(avg_price=self.avg_price, **super().view({}).__dict__)

    def post_message_resolution(self):
        super().post_message_resolution()

        seller_prices = [
            agent.current_price
            for agent in self.agents.values()
            if isinstance(agent, SellerAgent)
        ]

        self.avg_price = np.mean(seller_prices)

    # def step(self, actions):
    #     self.current_step += 1

    #     print("messages sent")

    #     # Decode actions and send messages
    #     for aid, action in actions.items():
    #         ctx = self.network.context_for(aid)
    #         messages = self.agents[aid].decode_action(ctx, action)
    #         self.network.send_from(aid, packet.messages)
    #         print(packet.messages)

    #     self.resolve_network()

    #     seller_prices = [
    #         agent.current_price
    #         for agent in self.agents.values()
    #         if isinstance(agent, SellerAgent)
    #     ]

    #     self.avg_price = np.mean(seller_prices)

    #     # Pass the turn
    #     self.turn = (self.turn + 1) % self.num_groups

    #     # Return observations for agents with the turn
    #     # so they can act in the next step
    #     obs = dict()
    #     rewards = dict()
    #     info = dict()
    #     for aid in self.agent_groups[self.turn]:
    #         agent = self.agents[aid]
    #         ctx = self.network.context_for(aid)
    #         obs[aid] = agent.encode_observation(ctx)
    #         rewards[aid] = agent.compute_reward(ctx)
    #         info[aid] = {"turn": True}

    #     return self.Step(
    #         observations=obs,
    #         rewards=rewards,
    #         terminals={"__all__": self.current_step == self.num_steps},
    #         infos=info,
    #     )

    # def reset(self):
    #     self.current_step = 0
    #     self.network.reset()  # will call reset on the agents/actors
    #     self.turn = 0

    #     obs = {}
    #     info = {}
    #     for aid in self.agent_groups[self.turn]:
    #         obs[aid] = self.agents[aid].observation_space.sample()
    #         info[aid] = {"turn": True}

    #     return self.Step(
    #         observations=obs,
    #         rewards={},
    #         terminals={"__all__": self.current_step == self.num_steps},
    #         infos=info,
    #     )
