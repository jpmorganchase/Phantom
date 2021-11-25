from typing import Any, List, Mapping, Optional, Sequence, Tuple, Type, Union

import gym.spaces
import mercury as me
from ray import rllib

# TODO: upgrade to latest Ray and use PolicySpec
# from ray.rllib.policy.policy import PolicySpec

from .fsm.types import StageID


class PolicyWrapper:
    """
    Internal class.
    """

    def __init__(
        self,
        used_by: Sequence[Union[me.ID, Tuple[me.ID, StageID]]],
        trained: bool,
        obs_space: gym.spaces.Space,
        action_space: gym.spaces.Space,
        policy_class: Optional[Type[rllib.policy.Policy]] = None,
        policy_config: Optional[Mapping[Any, Any]] = None,
        shared_policy_name: Optional[str] = None,
    ) -> None:
        if len(set(used_by)) > len(used_by):
            raise ValueError("Duplicate stages found in PolicyWrapper")

        self.used_by = used_by
        self.trained = trained
        self.policy_class = policy_class
        self.policy_config = policy_config or {}
        self.obs_space = obs_space
        self.action_space = action_space
        self.shared_policy_name = shared_policy_name

    def get_spec(
        self,
    ) -> Tuple[
        Optional[Type[rllib.policy.Policy]],
        gym.spaces.Space,
        gym.spaces.Space,
        Mapping[Any, Any],
    ]:
        # TODO: upgrade to latest Ray and use PolicySpec
        return (
            self.policy_class,
            self.obs_space,
            self.action_space,
            self.policy_config,
        )

    def get_name(self) -> str:
        if self.shared_policy_name is not None:
            return self.shared_policy_name

        stages = [x for x in self.used_by if isinstance(x, Tuple)]

        if len(stages) > 0:
            stages_str = "+".join(str(stage_id) for _, stage_id in stages)

            return f"{stages[0][0]}__{stages_str}"

        agent_ids = [x for x in self.used_by if not isinstance(x, Tuple)]

        if len(agent_ids) > 0:
            return str(agent_ids[0])

        raise Exception
