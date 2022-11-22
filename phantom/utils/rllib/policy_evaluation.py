from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy import Policy as RLlibPolicy
from ray.rllib.utils.spaces import space_utils

from .. import (
    collect_instances_of_type_with_paths,
    update_val,
    Range,
)
from . import construct_results_paths


def evaluate_policy(
    directory: Union[str, Path],
    policy_id: str,
    obs: Any,
    obs_space: gym.spaces.Space,
    checkpoint: Optional[int] = None,
) -> List[Tuple[Dict[str, Any], Any]]:
    """
    Evaluates a given pre-trained RLlib policy over a one of more dimensional
    observation space.

    Arguments:
        directory: Results directory containing trained policies. By default, this is
            located within `~/ray_results/`. If LATEST is given as the last element of
            the path, the parent directory will be scanned for the most recent run and
            this will be used.
        policy_id: The ID of the trained policy to evaluate.
        obs: The observation space to evaluate the policy with, of which can include
            :class:`Range` class instances to evaluate the policy over multiple
            dimensions in a similar fashion to the :func:`ph.utils.rllib.rollout`
            function.
        obs_space: The observation space of the policy.
        checkpoint: Checkpoint to use (defaults to most recent).

    Returns:
        A list of tuples of the form (observation, action).
    """
    directory, checkpoint_path = construct_results_paths(directory, checkpoint)

    policy = RLlibPolicy.from_checkpoint(str(checkpoint_path / "policies" / policy_id))

    pp = get_preprocessor(obs_space)(obs_space).transform

    ranges = collect_instances_of_type_with_paths(Range, ({}, obs))

    # This 'variations' list is where we build up every combination of the expanded
    # values from the list of Ranges.
    variations: List[List[Dict[str, Any]]] = [[{}, deepcopy(obs)]]

    unamed_range_count = 0

    # For each iteration of this outer loop we expand another Range object.
    for range_obj, paths in reversed(ranges):
        values = range_obj.values()

        name = range_obj.name
        if name is None:
            name = f"range-{unamed_range_count}"
            unamed_range_count += 1

        variations2 = []
        for value in values:
            for variation in variations:
                variation = deepcopy(variation)
                variation[0][name] = value
                for path in paths:
                    update_val(variation, path, value)
                variations2.append(variation)

        variations = variations2

    return [
        (
            params,
            space_utils.unsquash_action(
                policy.compute_single_action(pp(obs), explore=False)[0],
                policy.action_space_struct,
            ),
        )
        for (params, obs) in variations
    ]
