from ray.rllib.models.torch.torch_action_dist import TorchCategorical
import torch

from typing import (
    Dict,
    Tuple,
)
from dataclasses import dataclass

import numpy as np
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.evaluation import Episode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import PolicyID

from phantom.policy import Policy
from phantom.types import AgentID
from phantom.agents import StrategicAgent, Supertype


class BoundedStrategicAgent(StrategicAgent):
    """
    Representation of an informationally-constrained behavioural agent in the network.

    NOTE: This implementation only currently supports discrete action spaces.

    Instances of :class:`phantom.Agent` occupy the nodes on the network graph.
    They are resonsible for storing and monitoring internal state, constructing
    :class:`View` instances and handling messages.

    Arguments:
        agent_id: Unique identifier for the agent.
        observation_encoder: Optional :class:`Encoder` instance, otherwise define an
            :meth:`encode_observation` method on the :class:`Agent` sub-class.
        action_decoder: Optional :class:`Decoder` instance, otherwise define an
            :meth:`decode_action` method on the :class:`Agent` sub-class.
        reward_function: Optional :class:`RewardFunction` instance, otherwise define an
            :meth:`compute_reward` method on the :class:`Agent` sub-class.
        supertype: Optional :class:`Supertype` instance. When the agent's reset function
            is called the supertype will be sampled from and the values set as the
            agent's :attr:`type` property.
    """

    @dataclass
    class Supertype(Supertype):
        penalty: float = 0.0  # Default to having inf resources (no penalty)

    def collect_infos(self, ctx):
        return {
            "penalty": self.type.penalty,
            "prior_beliefs": self.prior_beliefs,
        }


class BoundedPolicies(DefaultCallbacks):
    """ RLLib Callback to model bounded rational reasoning by
     modifying the reward function to penalise divergences from prior beliefs.

     NOTE: Only currently supports discrete action spaces
     """

    def on_postprocess_trajectory(
        self,
        *,
        worker: "RolloutWorker",
        episode: Episode,
        agent_id: AgentID,
        policy_id: PolicyID,
        policies: Dict[PolicyID, Policy],
        postprocessed_batch: SampleBatch,
        original_batches: Dict[AgentID, Tuple[Policy, SampleBatch]],
        **kwargs,
    ) -> None:
        # Info provided by each agent (in collect_info)
        agent_infos = postprocessed_batch[SampleBatch.INFOS]

        # Processing resources for each agent
        penalties = np.asarray([info.get("penalty", 0.) for info in agent_infos])

        # Convert the network outputs (logits) to action probabilities
        logits = postprocessed_batch[SampleBatch.ACTION_DIST_INPUTS]
        policy = policies[policy_id]
        probabilities = policy.dist_class(logits, policy.model).dist.probs.numpy()

        # Each agent has prior beliefs. If they don't, set to uniform priors
        num_steps, num_actions = probabilities.shape
        uniform_prior = [1. / num_actions] * num_actions
        agent_priors = [info.get("prior_beliefs", uniform_prior) for info in agent_infos]

        # For tracking individual components (before adding penalty)
        postprocessed_batch["environment_rewards"] = postprocessed_batch[SampleBatch.REWARDS]

        # Chosen actions at each timestep
        actions = postprocessed_batch[SampleBatch.ACTIONS]

        # Calculate divergences at the chosen action points
        agent_priors = np.asarray(agent_priors) + 1E-5  # Just to be safe add small value to prevent divide by 0

        # 2d arrays - First dimension is the steps, so we index there first, and then second dimension the specific actions taken
        q = agent_priors[range(num_steps), actions]
        p = probabilities[range(num_steps), actions]

        # Approximate KL divergence (of just the actions chosen)
        divergence = p * np.log(p / q)

        # For tracking/plotting purposes only
        postprocessed_batch["divergences"] = divergence

        # Update rewards by adding penalty based on information costs
        postprocessed_batch[SampleBatch.REWARDS] -= penalties * divergence


class TorchOrdinalCategorical(TorchCategorical):
    """
    An ordinal categorical distribution.
    Following: Discretizing Continuous Action Space for On-Policy Optimization
    """
    def __init__(
            self,
            inputs,
            model,
            temperature: float = 1.0,
    ):
        super().__init__(inputs, model, temperature)
        # Take the sigmoid of each
        s = torch.sigmoid(self.inputs) + 1e-8  # Apply sigmoid to each. Add small vale to prevent taking log 0 below
        ordinal_logits = torch.cumsum(torch.log(s), dim=-1) + TorchOrdinalCategorical._after_cumsum(torch.log(1-s))
        self.dist = torch.distributions.categorical.Categorical(logits=ordinal_logits)

    @staticmethod
    def _after_cumsum(x):
        """
        Reversed cumsum for the remainder of the array (not including current elem),
        e.g. [1,2,3] = [5, 3, 0]
        """
        # Reverse the array
        flipped = torch.flip(x, dims=(-1,))
        # Take cumsum
        cumsum = torch.cumsum(flipped, dim=-1)
        # We do not want to include the current elem in the sum, so subtract this item
        cumsum_exclude_elem = cumsum - flipped
        # Reverse back to the original ordering
        return torch.flip(cumsum_exclude_elem, dims=(-1,))


