import math
import pickle
from dataclasses import dataclass
from itertools import chain
from logging import info
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
class EpisodeReplay:
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
) -> Tuple[List[Dict[str, np.ndarray]], List[EpisodeReplay]]:
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
        params.checkpoint = int(str(checkpoint_dirs[-1]).split("_")[-1])

    rollouts_per_worker = int(math.ceil(params.num_rollouts / params.num_workers))

    seeds = list(range(params.num_rollouts))

    worker_payloads = [
        (params, seeds[i : i + rollouts_per_worker])
        for i in range(0, len(seeds), rollouts_per_worker)
    ]

    ray.init()

    results = list(
        chain.from_iterable(
            ray.util.iter.from_items(
                worker_payloads, num_shards=max(params.num_workers, 1)
            )
            .for_each(_parallel_fn)
            .gather_sync()
        )
    )

    metrics, replays = zip(*results)

    if params.metrics_file is not None:
        pickle.dump(
            list(metrics), open(Path(params.directory, params.metrics_file), "wb")
        )

    if params.replays_file is not None:
        pickle.dump(
            list(replays), open(Path(params.directory, params.replays_file), "wb")
        )

    ray.shutdown()

    return metrics, replays


def _parallel_fn(args: Tuple[RolloutParams, List[int]]) -> List[Dict[str, Any]]:
    params, seeds = args

    checkpoint_path = Path(
        params.directory,
        f"checkpoint_{str(params.checkpoint).zfill(6)}",
        f"checkpoint-{params.checkpoint}",
    )

    # Load config from results directory.
    with open(Path(params.directory, "params.pkl"), "rb") as f:
        config = pickle.load(f)

    # Set to zero as rollout workers != training workers - if > 0 will spin up
    # unnecessary additional workers.
    config["num_workers"] = 0
    config["env_config"] = params.env_config

    # Register custom environment with Ray
    register_env(params.env.env_name, lambda config: params.env(**config))

    trainer = get_trainer_class("PPO")(env=params.env.env_name, config=config)
    trainer.restore(str(checkpoint_path))

    results = []

    for seed in seeds:
        # Create environment instance from config from results directory
        env = params.env(**config["env_config"])

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

        replay = EpisodeReplay(
            observations,
            rewards,
            dones,
            infos,
            actions,
        )

        results.append((metrics, replay))

    return results
