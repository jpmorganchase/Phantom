from dataclasses import dataclass
from simple_mkt_env import SimpleMarketEnv
import mercury as me


@dataclass
class AdversarialSetup:
    adv_seller: me.ID
    victim_seller: me.ID
    adv_seller: me.ID


# Version of the simple market env that allows for an adversarial setup with
# information leakage
###########################


class LeakySimpleMarketEnv(SimpleMarketEnv):
    def __init__(self, network, clock, adv_setup=None):
        super().__init__(network, clock)
        # Information leakage / adversarial setup
        self.leaky = False
        if adv_setup:
            self.adversarial_setup(
                adv_setup.leaky_buyer, adv_setup.adv_seller, adv_setup.victim_seller
            )

    def adversarial_setup(
        self,
        leaky_buyer,
        adv_seller,
        victim_seller,
        victim_reward_coeff=1.0,
        adv_reward_coeff=1.0,
    ):
        self.leaky = True
        self.leaky_buyer = leaky_buyer
        self.adv_seller = adv_seller
        self.victim_seller = victim_seller
        self.agents[leaky_buyer].victim_id = victim_seller
        self.agents[leaky_buyer].adv_id = adv_seller
        self.agents[adv_seller].victim_id = victim_seller  # TODO: Is this needed?
        self.victim_coeff = victim_reward_coeff
        self.adv_coeff = adv_reward_coeff

    def compute_adv_reward(self, attacker_reward, victim_reward):
        """
        Computing the adversarial rewards, which is a combination of
        the penalized reward and the original agent reward
        """
        return -self.victim_coeff * victim_reward + self.adv_coeff * attacker_reward

    def step(self, actions, verbose=False):
        self.clock.tick()
        if verbose:
            print("messages sent")

        # Decode actions and send messages
        for aid, action in actions.items():
            ctx = self.network.context_for(aid)
            packet = self.agents[aid].decode_action(ctx, action)
            self.network.send_from(aid, packet.messages)
            if verbose:
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

        # Modify reward for adv seller
        # TODO - pass in the adv reward function
        if self.leaky and self.turn == 0:
            # rewards[self.adv_seller] = -1*rewards[self.victim_seller]
            rewards[self.adv_seller] = self.compute_adv_reward(
                rewards[self.adv_seller], rewards[self.victim_seller]
            )

        return self.Step(
            observations=obs,
            rewards=rewards,
            terminals={"__all__": self.clock.is_terminal},
            infos=info,
        )
