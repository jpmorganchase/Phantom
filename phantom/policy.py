from abc import abstractmethod, ABC
from typing import Any, Mapping, Optional

import gym


class Policy(ABC):
    """
    Base Policy class for defining custom policies.

    Arguments:
        observation_space: Observation space of the policy.
        action_space: Action space of the policy.
        config: Optional configuration parameters.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Optional[Mapping[str, Any]],
    ) -> None:
        self.observation_space = observation_space
        self.action_space = action_space
        self.config = config

    @abstractmethod
    def compute_action(self, observation: Any) -> Any:
        """
        Arguments:
            observation: A single observation for the policy to act on.

        Returns:
            The action taken by the policy based on the given observation.
        """
        raise NotImplementedError
