import math
import cloudpickle
from dataclasses import dataclass
from itertools import chain
from logging import error, info
from pathlib import Path
from typing import *

import mercury as me
import numpy as np
import ray
from phantom.params import RolloutParams
from phantom.logging import Logger
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env

from . import find_most_recent_results_dir


@dataclass
class EpisodeTrajectory:
    """
    Class describing all the actions, observations, rewards, infos and dones of
    a single episode.
    """

    observations: List[Dict[me.ID, Any]]
    rewards: List[Dict[me.ID, float]]
    dones: List[Dict[me.ID, bool]]
    infos: List[Dict[me.ID, Dict[str, Any]]]
    actions: List[Dict[me.ID, Any]]


def run_rollouts(
    params: RolloutParams,
) -> Tuple[List[Dict[str, np.ndarray]], List[EpisodeTrajectory]]:
    if params.directory is not None:
        params.directory = Path(params.directory)

    if params.directory.stem == "LATEST":
        info(f"Trying to find latest experiment results in '{params.directory.parent}'")

        params.directory = find_most_recent_results_dir(params.directory.parent)

        info(f"Found experiment results: '{params.directory.stem}'")
    else:
        info(f"Using results directory: '{params.directory}'")

    if params.checkpoint is None:
        checkpoint_dirs = sorted(Path(params.directory).glob("checkpoint_*"))

        if len(checkpoint_dirs) == 0:
            error(f"No checkpoints found in directory '{params.directory}'")
            return

        params.checkpoint = int(str(checkpoint_dirs[-1]).split("_")[-1])

        info(f"Using most recent checkpoint: {params.checkpoint}")
    else:
        info(f"Using checkpoint: {params.checkpoint}")

    params.num_workers = max(params.num_workers, 1)

    rollouts_per_worker = int(math.ceil(params.num_rollouts / params.num_workers))

    seeds = list(range(params.num_rollouts))

    worker_payloads = [
        (params, seeds[i : i + rollouts_per_worker])
        for i in range(0, len(seeds), rollouts_per_worker)
    ]

    info(
        f"Starting {params.num_rollouts} rollout(s) using {params.num_workers} worker(s)"
    )

    try:
        ray.init(include_dashboard=False)

        results = list(
            chain.from_iterable(
                ray.util.iter.from_items(
                    worker_payloads, num_shards=max(params.num_workers, 1)
                )
                .for_each(_parallel_fn)
                .gather_sync()
            )
        )

    except Exception as e:
        # ensure that Ray is properly shutdown in the instance of any error occuring
        ray.shutdown()
        raise e
    else:
        ray.shutdown()

    metrics, trajectories = zip(*results)

    results = {}

    if params.metrics != {}:
        results["metrics"] = metrics

    if params.save_trajectories:
        results["trajectories"] = trajectories

    if params.results_file is not None and results != {}:
        results_file = Path(params.directory, params.results_file)

        cloudpickle.dump(results, open(results_file, "wb"))

        info(f"Saved rollout results to '{results_file}'")

    return metrics, trajectories


def _parallel_fn(
    args: Tuple[RolloutParams, List[int]]
) -> List[Tuple[Dict[str, np.ndarray], EpisodeTrajectory]]:
    params, seeds = args

    checkpoint_path = Path(
        params.directory,
        f"checkpoint_{str(params.checkpoint).zfill(6)}",
        f"checkpoint-{params.checkpoint}",
    )

    # Load config from results directory.
    with open(Path(params.directory, "params.pkl"), "rb") as f:
        config = cloudpickle.load(f)

    # Load env class from results directory.
    with open(Path(params.directory, "env.pkl"), "rb") as f:
        env_class = cloudpickle.load(f)

    # Set to zero as rollout workers != training workers - if > 0 will spin up
    # unnecessary additional workers.
    config["num_workers"] = 0
    config["env_config"] = params.env_config

    # Register custom environment with Ray
    register_env(env_class.env_name, lambda config: env_class(**config))

    trainer = get_trainer_class("PPO")(env=env_class.env_name, config=config)
    trainer.restore(str(checkpoint_path))

    results = []

    for seed in seeds:
        # Create environment instance from config from results directory
        env = env_class(**config["env_config"])

        shared_policy_mapping = {}

        # Construct mapping of agent_id --> shared_policy_id
        if env.policy_grouping is not None:
            for policy_id, agent_ids in env.policy_grouping.items():
                for agent_id in agent_ids:
                    shared_policy_mapping[agent_id] = policy_id

        # Setting seed needs to come after trainer setup
        np.random.seed(seed)

        logger = Logger(params.metrics)

        observation = env.reset()

        observations: List[Dict[me.ID, Any]] = [observation]
        rewards: List[Dict[me.ID, float]] = []
        dones: List[Dict[me.ID, bool]] = []
        infos: List[Dict[me.ID, Dict[str, Any]]] = []
        actions: List[Dict[me.ID, Any]] = []

        # Run rollout steps.
        for _ in range(env.clock.n_steps):
            step_actions = {}

            for agent_id, agent_obs in observation.items():
                policy_id = shared_policy_mapping.get(agent_id, agent_id)

                agent_action = trainer.compute_action(
                    agent_obs, policy_id=policy_id, explore=False
                )
                step_actions[agent_id] = agent_action

            observation, reward, done, info = env.step(step_actions)
            logger.log(env)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            actions.append(step_actions)

        metrics = {k: np.array(v) for k, v in logger.to_dict().items()}

        trajectory = EpisodeTrajectory(
            observations,
            rewards,
            dones,
            infos,
            actions,
        )

        results.append((metrics, trajectory))

    return results
