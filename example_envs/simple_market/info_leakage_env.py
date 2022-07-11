from dataclasses import dataclass

import phantom as ph

from simple_mkt_env import SimpleMarketEnv


@dataclass(frozen=True)
class AdversarialSetup:
    leaky_buyer: ph.AgentID
    victim_seller: ph.AgentID
    adv_seller: ph.AgentID


# Version of the simple market env that allows for an adversarial setup with
# information leakage
###########################


class LeakySimpleMarketEnv(SimpleMarketEnv):
    def __init__(self, num_steps, network, adv_setup=None):
        super().__init__(num_steps, network)
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
        step = super().step(actions)

        # Modify reward for adv seller
        # TODO - pass in the adv reward function
        if self.leaky and self.current_stage == "Sellers":
            # rewards[self.adv_seller] = -1*rewards[self.victim_seller]
            step.rewards[self.adv_seller] = self.compute_adv_reward(
                step.rewards[self.adv_seller], step.rewards[self.victim_seller]
            )

        return step
