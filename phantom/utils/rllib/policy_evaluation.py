from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple, Union

import cloudpickle
import rich.progress
from ray.rllib.models.preprocessors import get_preprocessor
from ray.rllib.policy import Policy
from ray.rllib.utils.spaces.space_utils import unsquash_action

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
    explore: bool,
    batch_size: int = 100,
    checkpoint: Optional[int] = None,
    show_progress_bar: bool = True,
) -> Generator[Tuple[Dict[str, Any], Any, Any], None, None]:
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
        explore: Parameter passed to the policy.
        batch_size: Number of observations to evaluate at a time.
        checkpoint: Checkpoint to use (defaults to most recent).
        show_progress_bar: If True shows a progress bar in the terminal output.

    Returns:
        A generator of tuples of the form (params, obs, action).
    """
    directory, checkpoint_path = construct_results_paths(directory, checkpoint)

    policy = Policy.from_checkpoint(checkpoint_path / "policies" / policy_id)

    with open(Path(directory, "phantom-training-params.pkl"), "rb") as params_file:
        ph_config = cloudpickle.load(params_file)

    policy_specs = ph_config["policy_specs"]

    obs_s = policy_specs[policy_id].observation_space
    preprocessor = get_preprocessor(obs_s)(obs_s)

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

    def chunker(seq, size):
        return (seq[pos : pos + size] for pos in range(0, len(seq), size))

    batched_variations = chunker(variations, batch_size)

    if show_progress_bar:
        batched_variations = rich.progress.track(batched_variations)

    for variation_batch in batched_variations:
        params, obs = zip(*variation_batch)

        processed_obs = [preprocessor.transform(ob) for ob in obs]

        squashed_actions = policy.compute_actions(processed_obs, explore=explore)[0]

        actions = [
            unsquash_action(action, policy.action_space_struct)
            for action in squashed_actions
        ]

        for p, o, a in zip(params, obs, actions):
            yield (p, o, a)
