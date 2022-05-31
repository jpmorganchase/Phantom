from abc import abstractmethod, ABC
from typing import Any, Dict, List, Optional, Mapping, Tuple, Union

import gym
import numpy as np
from ray import rllib
from ray.rllib.evaluation import Episode, MultiAgentEpisode
from ray.rllib.policy.sample_batch import SampleBatch
from ray.rllib.utils.typing import TensorStructType, TensorType


class Policy(ABC):
    """
    Arguments:
        observation_space: Observation space of the policy.
        action_space: Action space of the policy.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Mapping[str, Any],
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


class FixedPolicy(Policy, ABC):
    """
    TODO: docstring
    """


class RLlibFixedPolicy(FixedPolicy, rllib.Policy, ABC):
    """
    Wrapper around the ``rllib.Policy`` class for implementing simple fixed policies in
    Phantom. For more advanced use-cases it may be better to subclass from the
    ``rllib.Policy`` class directly.

    Arguments:
        observation_space: Observation space of the policy.
        action_space: Action space of the policy.

    NOTE:
        If the action space is larger than -1.0 < x < 1.0, RLlib will attempt to
        'unsquash' the values leading to unintended results.
        (https://github.com/ray-project/ray/pull/16531)
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        config: Mapping[str, Any],
    ):
        FixedPolicy.__init__(self, observation_space, action_space, config)
        rllib.Policy.__init__(self, observation_space, action_space, dict(config))

    def get_weights(self):
        return None

    def set_weights(self, weights):
        pass

    def learn_on_batch(self, samples):
        return {}

    # The following methods are used to interface with RLlib and do not need to be
    # modified. If the features of the following methods are needed the RLlib Policy
    # class should be subclassed directly.

    def compute_single_action(
        self,
        obs: Optional[TensorStructType] = None,
        state: Optional[List[TensorType]] = None,
        *,
        prev_action: Optional[TensorStructType] = None,
        prev_reward: Optional[TensorStructType] = None,
        info: dict = None,
        input_dict: Optional[SampleBatch] = None,
        episode: Optional[Episode] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        # Kwars placeholder for future compatibility.
        **kwargs
    ) -> Tuple[TensorStructType, List[TensorType], Dict[str, TensorType]]:
        return self.compute_action(obs), [], {}

    def compute_actions(
        self,
        obs_batch: Union[List[TensorStructType], TensorStructType],
        state_batches: Optional[List[TensorType]] = None,
        prev_action_batch: Union[List[TensorStructType], TensorStructType] = None,
        prev_reward_batch: Union[List[TensorStructType], TensorStructType] = None,
        info_batch: Optional[Dict[str, list]] = None,
        episodes: Optional[List[MultiAgentEpisode]] = None,
        explore: Optional[bool] = None,
        timestep: Optional[int] = None,
        **kwargs
    ) -> Tuple[TensorType, List[TensorType], Dict[str, TensorType]]:
        # Workaround due to known issue in RLlib
        # https://github.com/ray-project/ray/issues/10009
        if isinstance(self.action_space, gym.spaces.Tuple):
            unbatched = [self.compute_action(obs) for obs in obs_batch]

            actions = tuple(
                np.array([unbatched[j][i] for j in range(len(unbatched))])
                for i in range(len(unbatched[0]))
            )
        else:
            actions = [self.compute_action(obs) for obs in obs_batch]

        return (actions, [], {})
