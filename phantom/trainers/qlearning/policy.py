from typing import Any, Mapping

import gym
import numpy as np
from ...policies import Policy


class QLearningPolicy(Policy):
    """
    Simple QLearning policy implementation.

    Arguments:
        observation_space: Observation space of the policy.
        action_space: Action space of the policy.
    """

    def __init__(
        self,
        observation_space: gym.spaces.Discrete,
        action_space: gym.spaces.Discrete,
    ) -> None:
        super().__init__(observation_space, action_space, {})

        self.q_table = np.zeros([observation_space.n, action_space.n])

    def compute_action(self, observation: int) -> Any:
        """
        Arguments:
            observation: A single observation for the policy to act on.

        Returns:
            The action taken by the policy based on the given observation.
        """
        return np.argmax(self.q_table[observation])
