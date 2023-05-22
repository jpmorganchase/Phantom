from abc import abstractmethod, ABC
from typing import Any

import gymnasium as gym


class Policy(ABC):
    """
    Base Policy class for defining custom policies.

    Arguments:
        observation_space: Observation space of the policy.
        action_space: Action space of the policy.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def compute_action(self, observation: Any) -> Any:
        """
        Arguments:
            observation: A single observation for the policy to act on.

        Returns:
            The action taken by the policy based on the given observation.
        """
        raise NotImplementedError
