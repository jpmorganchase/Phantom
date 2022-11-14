from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Type, Union

import cloudpickle
from ray.rllib.algorithms.registry import get_algorithm_class
from ray.tune.registry import register_env


from ...env import PhantomEnv
from .. import (
    collect_instances_of_type_with_paths,
    update_val,
    Range,
)
from .wrapper import RLlibEnvWrapper
from . import construct_results_paths


def evaluate_policy(
    directory: Union[str, Path],
    algorithm: str,
    env_class: Type[PhantomEnv],
    policy_id: str,
    obs: Any,
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
        algorithm: RLlib algorithm to use.
        env_class: Optionally pass the Environment class to use. If not give will
            fallback to the copy of the environment class saved during training.
        policy_id: The ID of the trained policy to evaluate.
        obs: The observation space to evaluate the policy with, of which can include
            :class:`Range` class instances to evaluate the policy over multiple
            dimensions in a similar fashion to the :func:`ph.utils.rllib.rollout`
            function.
        checkpoint: Checkpoint to use (defaults to most recent).

    Returns:
        A list of tuples of the form (observation, action).
    """
    directory, checkpoint_path = construct_results_paths(directory, checkpoint)

    # Load config from results directory.
    with open(Path(directory, "params.pkl"), "rb") as params_file:
        config = cloudpickle.load(params_file)

    # Set to zero as rollout workers != training workers - if > 0 will spin up
    # unnecessary additional workers.
    config["num_workers"] = 0

    if isinstance(env_class, RLlibEnvWrapper):
        register_env(env_class.__name__, lambda config: env_class(**config))
    else:
        register_env(
            env_class.__name__, lambda config: RLlibEnvWrapper(env_class(**config))
        )

    algo = get_algorithm_class(algorithm)(env=env_class.__name__, config=config)
    algo.restore(str(checkpoint_path))

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
        (params, algo.compute_single_action(obs, policy_id=policy_id, explore=False))
        for (params, obs) in variations
    ]
