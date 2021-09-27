import importlib
import logging as log
import pickle
import os
import shutil
import tempfile
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import gym
import mercury as me
import numpy as np
import ray
from ray import tune
from ray.rllib.agents.registry import get_trainer_class
from ray.tune.registry import register_env
from tabulate import tabulate
from termcolor import colored

from .agent import ZeroIntelligenceAgent
from .logging import Logger, MetricsLoggerCallbacks, MultiCallbacks
from .logging.callbacks import TBXExtendedLoggerCallback
from .params import PhantomParams
from .rollout import RolloutReplay


def train_from_config_path(
    config_path: Union[str, Path], local_mode: bool = False, print_info: bool = True
) -> Optional[Path]:
    """
    Performs training of a Phantom experiment.

    Arguments:
        config_path: The filesystem path pointing to the config Python file.
        local_mode: If true will force Ray to run in one process (useful for
            profiling & debugging).
        print_info: If true will print a summary of the configuration before
            running.

    NOTE: this method and the other train* methods do not ensure that PYTHONHASHSEED
    is set. In most cases, the phantom-train command should be used instead.
    """

    _, params = load_config(config_path)

    results_dir = train_from_params_object(params, local_mode, print_info, config_path)

    if params.discard_results == False and len(params.copy_files_to_results_dir) > 0:
        source_code_dir = results_dir.joinpath("source_code")
        os.mkdir(source_code_dir)

        base_dir = Path(config_path).parent

        for file in params.copy_files_to_results_dir:
            old_path = Path(base_dir, file)
            new_path = Path(source_code_dir, file)

            if old_path.exists():
                shutil.copy(old_path, new_path)
            else:
                print(
                    colored(
                        f"Could not find file '{old_path}' to copy to results directory",
                        "yellow",
                    )
                )

    return results_dir


def train_from_params_object(
    params: PhantomParams,
    local_mode: bool = False,
    print_info: bool = True,
    config_path: Union[str, Path, None] = None,
) -> Optional[Path]:
    """
    Performs training of a Phantom experiment.

    Arguments:
        params: A populated PhantomParams object.
        local_mode: If true will force Ray to run in one process (useful for
            profiling & debugging).
        print_info: If true will print a summary of the configuration before
            running.
        config_path: The filesystem path pointing to the config Python file
            (optional - used for display purposes only).

    NOTE: this method and the other train* methods do not ensure that PYTHONHASHSEED
    is set. In most cases, the phantom-train command should be used instead.
    """

    config = create_rllib_config_dict(params)

    if print_info:
        print_experiment_info(params, config, config_path)

    register_env(params.env.env_name, lambda config: params.env(**config))

    ray.init(local_mode=local_mode)

    training_it = (
        int(params.num_episodes / params.num_workers)
        if params.num_workers > 0
        else params.num_episodes
    )

    if params.discard_results:
        results_dir = tempfile.mkdtemp()
    else:
        results_dir = params.results_dir

    _ = tune.run(
        params.algorithm,
        name=params.experiment_name,
        local_dir=results_dir,
        checkpoint_freq=params.checkpoint_freq,
        checkpoint_at_end=True,
        stop={"training_iteration": training_it},
        config=config,
        callbacks=[TBXExtendedLoggerCallback()],
    )

    ray.shutdown()

    if params.discard_results:
        return None
    else:
        return find_most_recent_results_dir(Path(results_dir, params.experiment_name))


def rollout(
    params: PhantomParams,
    results_dir: Union[str, Path],
    checkpoint_num: int,
    num_rollouts: int,
    num_workers: int,
    metrics_file: Optional[str] = "metrics.pkl",
    replays_file: Optional[str] = "replays.pkl",
) -> Tuple[List[Dict[str, np.ndarray]], List[RolloutReplay]]:
    """
    Performs rollout of a previously trained Phantom experiment.

    Arguments:
        params: The populated PhantomParams object that was used in training.
        results_dir: The directory that the experiment results reside in.
        checkpoint_num: The checkpoint to use.
        num_rollouts: The number of rollouts to perform.
        num_workers: The number of rollout workers to use.
        metrics_file: Filename relative to the results directory to save rollout
        metrics (None = no file saved).
        replays_file: Filename relative to the results directory to save rollout
        replays (None = no file saved).
    """

    checkpoint_path = Path(
        results_dir,
        f"checkpoint_{str(checkpoint_num).zfill(6)}",
        f"checkpoint-{checkpoint_num}",
    )

    if not os.path.exists(checkpoint_path):
        print(colored(f"Checkpoint {checkpoint_num} not found!", "red"))
        return

    def parallel_fn(args) -> Tuple[RolloutReplay, Dict[str, np.ndarray]]:
        env, i = args

        log.info(f"Running rollout {i+1}/{num_rollouts}")
        np.random.seed(i)

        # Load config from results directory.
        with open(Path(results_dir, "params.pkl"), "rb") as f:
            config = pickle.load(f)

        # Set to zero as rollout workers != training workers - if > 0 will spin up
        # unnecessary additional workers.
        config["num_workers"] = 0

        trainer = get_trainer_class(params.algorithm)(env=env.env_name, config=config)

        trainer.restore(str(checkpoint_path))

        # Create environment instance from config from results directory.
        env = env(**config["env_config"])

        logger2 = deepcopy(logger)

        shared_policy_mapping = {}

        # Construct mapping of agent_id --> shared_policy_id
        if env.policy_grouping is not None:
            for policy_id, agent_ids in env.policy_grouping.items():
                for agent_id in agent_ids:
                    shared_policy_mapping[agent_id] = policy_id

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
            logger2.log(env)

            observations.append(observation)
            rewards.append(reward)
            dones.append(done)
            infos.append(info)
            actions.append(step_actions)

        metrics = {k: np.array(v) for k, v in logger2.to_dict().items()}

        replay = RolloutReplay(
            observations,
            rewards,
            dones,
            infos,
            actions,
        )

        return metrics, replay

    ray.init()

    logger = Logger(params.metrics)

    # Register custom environment with Ray
    register_env(params.env.env_name, lambda config: params.env(**config))

    parallel_fn_args = [(params.env, i) for i in range(num_rollouts)]

    results = list(
        ray.util.iter.from_items(parallel_fn_args, num_shards=max(num_workers, 1))
        .for_each(parallel_fn)
        .gather_sync()
    )

    metrics, replays = list(zip(*results))

    ray.shutdown()

    if replays_file is not None:
        pickle.dump(replays, open(os.path.join(results_dir, replays_file), "wb"))

    if metrics_file is not None:
        pickle.dump(metrics, open(os.path.join(results_dir, metrics_file), "wb"))

    return replays, metrics


def load_config(config_path: str) -> Tuple[str, PhantomParams]:
    """
    Attempts to load a PhantomParams object from a config file.

    Arguments:
        config_path: The filesystem path pointing to the config Python file.

    Returns:
        A tuple containing the config name (taken from the filename) and the
            PhantomParams object found in the file (under the name 'phantom_params').
    """

    if not os.path.exists(config_path):
        raise Exception(f"Config file '{config_path}' does not exist!")

    module_name = config_path[:-3].split("/")[-1]

    spec = importlib.util.spec_from_file_location(module_name, config_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    if not hasattr(module, "phantom_params"):
        raise Exception("'phantom_params' field not found in config file.")

    if not isinstance(module.phantom_params, PhantomParams):
        raise Exception(
            "'phantom_params' object is not an instance of the PhantomParams class."
        )

    return (module_name, module.phantom_params)


def create_rllib_config_dict(params: PhantomParams) -> dict:
    """
    Converts a PhantomParams object into a config dictionary compatible with
    Ray/RLlib.
    """

    # Users are able to use ray.tune hyperparameter space objects (e.g. GridSearch)
    # in the env_config. When running the actual experiments ray will convert
    # these to real values. However here we do not do that and we cannot pass
    # these objects into the environment init. Instead we attempt to create an
    # environment and if this fails we try and create an environment with only
    # the default parameters.
    try:
        env = params.env(**params.env_config)
    except:
        env = params.env()

    mas = {}

    if env.policy_grouping is not None:
        custom_policies = {}
        custom_policies_to_train = []
        mapping = {}

        for pid, aids in env.policy_grouping.items():
            custom_policies[pid] = (
                None,
                env.agents[aids[0]].get_observation_space(),
                env.agents[aids[0]].get_action_space(),
                env.agents[aids[0]].policy_config or dict(),
            )

            for aid in aids:
                mapping[aid] = pid

            custom_policies_to_train.append(pid)

        for aid, agent in env.agents.items():
            if aid not in mapping:
                custom_policies[aid] = (
                    agent.policy_type,
                    agent.get_observation_space(),
                    agent.get_action_space(),
                    agent.policy_config or dict(),
                )

                mapping[aid] = aid

                if agent.policy_type is None and not isinstance(
                    agent, ZeroIntelligenceAgent
                ):
                    custom_policies_to_train.append(aid)

        mas["policies"] = custom_policies
        mas["policy_mapping"] = mapping
        mas["policy_mapping_fn"] = lambda agent_id, episode=None, **kwargs: mapping[
            agent_id
        ]
        mas["policies_to_train"] = custom_policies_to_train

    else:
        mas["policies"] = {
            aid: (
                agent.policy_type,
                agent.get_observation_space(),
                agent.get_action_space(),
                agent.policy_config or dict(),
            )
            for aid, agent in env.agents.items()
        }

        mas["policy_mapping"] = {aid: aid for aid in env.agents.keys()}
        mas["policy_mapping_fn"] = lambda agent_id, episode=None, **kwargs: agent_id

        mas["policies_to_train"] = [
            agent_id
            for agent_id, policy_spec in mas["policies"].items()
            if policy_spec[0] is None
            and not isinstance(env.agents[agent_id], ZeroIntelligenceAgent)
        ]

    config = {}

    config["env"] = params.env.env_name
    config["env_config"] = params.env_config
    config["seed"] = params.seed
    config["multiagent"] = mas
    config["num_workers"] = params.num_workers
    config["rollout_fragment_length"] = env.clock.n_steps

    config["train_batch_size"] = int(
        (config["rollout_fragment_length"] * params.num_workers)
        if params.num_workers > 0
        else config["rollout_fragment_length"]
    )

    config["sgd_minibatch_size"] = max(int(config["train_batch_size"] / 10), 1)

    if params.callbacks is not None:
        callbacks = params.callbacks
        if not isinstance(params.callbacks, (list, tuple)):
            callbacks = [callbacks]
        config["callbacks"] = MultiCallbacks(callbacks)

    if params.metrics:
        if "callbacks" in config:
            config["callbacks"] = MultiCallbacks(
                [config["callbacks"], MetricsLoggerCallbacks("phantom", params.metrics)]
            )
        else:
            config["callbacks"] = MetricsLoggerCallbacks("phantom", params.metrics)

    config.update(**params.alg_config)

    return config


def find_most_recent_results_dir(base_path: Union[Path, str]) -> Path:
    """
    Scans a directory containing ray experiment results and returns the path of
    the most recent experiment.

    Arguments:
        base_path: The directory to search in.
    """

    base_path = Path(os.path.expanduser(base_path))

    experiment_dirs = [d for d in base_path.iterdir() if d.is_dir()]

    if len(experiment_dirs) == 0:
        raise ValueError(f"No experiment directories found in '{base_path}'")

    experiment_dirs.sort(
        key=lambda d: datetime.strptime(str(d)[-19:], "%Y-%m-%d_%H-%M-%S")
    )

    return experiment_dirs[-1]


def print_experiment_info(
    phantom_params: PhantomParams, config: Dict, config_path: Optional[str] = None
):
    def get_space_size(space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Box):
            return sum(space.shape)
        elif isinstance(space, gym.spaces.Discrete):
            return 1
        elif isinstance(space, gym.spaces.Tuple):
            return sum(get_space_size(elem) for elem in space)
        else:
            raise NotImplementedError

    print()
    print("General Parameters")
    print("==================")
    if config_path is not None:
        print(f"Config file      : {config_path}")
    print(f"Experiment name  : {phantom_params.experiment_name}")
    print(f"Environment name : {phantom_params.env.env_name}")
    print(f"Num workers      : {phantom_params.num_workers}")
    print(f"Num episodes     : {phantom_params.num_episodes}")
    print(f"Algorithm        : {phantom_params.algorithm}")
    print(f"Num steps        : {config['rollout_fragment_length']}")
    print(f"Checkpoint freq. : {phantom_params.checkpoint_freq}")
    print()

    trained_policy_data = []
    untrained_policy_data = []

    for policy_name, (_, obs_size, act_size, _) in config["multiagent"][
        "policies"
    ].items():
        used_by = ",".join(
            [
                aid
                for aid, pid in config["multiagent"]["policy_mapping"].items()
                if pid == policy_name
            ]
        )

        data = (
            policy_name,
            get_space_size(obs_size),
            get_space_size(act_size),
            used_by,
        )

        if policy_name in config["multiagent"]["policies_to_train"]:
            trained_policy_data.append(data)
        else:
            untrained_policy_data.append(data)

    print("Trained Policies")
    print("================")
    if len(untrained_policy_data) > 0:
        print(
            tabulate(
                trained_policy_data,
                headers=["Policy", "Observation Size", "Action Size", "Used By"],
                tablefmt="pretty",
            )
        )
    else:
        print("None")
    print()

    print("Untrained Policies")
    print("==================")
    if len(untrained_policy_data) > 0:
        print(
            tabulate(
                untrained_policy_data,
                headers=["Policy", "Observation Size", "Action Size", "Used By"],
                tablefmt="pretty",
            )
        )
    else:
        print("None")
    print()
