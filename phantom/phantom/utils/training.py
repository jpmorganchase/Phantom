import os
import shutil
import tempfile
from pathlib import Path
from typing import Dict, Optional, Union

import gym
import ray
from ray import tune
from ray.tune.registry import register_env
from tabulate import tabulate
from termcolor import colored

from ..agent import ZeroIntelligenceAgent
from ..logging import MetricsLoggerCallbacks, MultiCallbacks
from ..logging.callbacks import TBXExtendedLoggerCallback
from ..params import TrainingParams
from . import find_most_recent_results_dir, load_object


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

    params = load_object(config_path, "training_params", TrainingParams)

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
    params: TrainingParams,
    local_mode: bool = False,
    print_info: bool = True,
    config_path: Union[str, Path, None] = None,
) -> Optional[Path]:
    """
    Performs training of a Phantom experiment.

    Arguments:
        params: A populated TrainingParams object.
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


def create_rllib_config_dict(params: TrainingParams) -> dict:
    """
    Converts a TrainingParams object into a config dictionary compatible with
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


def print_experiment_info(
    phantom_params: TrainingParams, config: Dict, config_path: Optional[str] = None
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
