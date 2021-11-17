import __main__
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Type, Union

import cloudpickle
import gym
import ray
from ray import tune

# Enable with Ray 1.7.0:
# from ray.rllib.policy.policy import PolicySpec

from ray.rllib.agents.callbacks import DefaultCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune.registry import register_env
from tabulate import tabulate

from ..env import PhantomEnv
from ..logging import Metric, MetricsLoggerCallbacks, MultiCallbacks
from ..logging.callbacks import TBXExtendedLoggerCallback
from ..policy import FixedPolicy
from . import find_most_recent_results_dir, show_pythonhashseed_warning


logger = logging.getLogger(__name__)


def train(
    experiment_name: str,
    env: Type[PhantomEnv],
    num_workers: int,
    num_episodes: int,
    algorithm: str,
    seed: int = 0,
    checkpoint_freq: Optional[int] = None,
    env_config: Optional[Mapping[str, Any]] = None,
    alg_config: Optional[Mapping[str, Any]] = None,
    policy_grouping: Optional[Mapping[str, List[str]]] = None,
    metrics: Optional[Mapping[str, Metric]] = None,
    callbacks: Optional[Iterable[DefaultCallbacks]] = None,
    discard_results: bool = False,
    results_dir: Union[str, Path] = "~/phantom_results",
    copy_files_to_results_dir: Optional[Iterable[Union[str, Path]]] = None,
    local_mode: bool = False,
    print_info: bool = True,
) -> Optional[Path]:
    """
    Performs training of a Phantom experiment.

    Arguments:
        experiment_name: Experiment name used for tensorboard logging.
        env: A PhantomEnv subclass.
        num_workers: Number of Ray workers to initialise.
        num_episodes: Number of episodes to train for, distributed over all workers.
        algorithm: RL algorithm to use.
        seed: Optional seed to pass to environment.
        checkpoint_freq: Episodic frequency at which to save checkpoints.
        env_config: Configuration parameters to pass to the environment init method.
        alg_config: Optional algorithm parameters dictionary to pass to RLlib.
        policy_grouping: A mapping between custom policy names and lists of agents
            sharing the same policy.
        metrics: Optional set of metrics to record and log.
        callbacks: Optional Ray Callbacks for custom metrics.
            (https://docs.ray.io/en/master/rllib-training.html#callbacks-and-custom-metrics)
        discard_results: If True, all results are discarded (useful for unit testing & development).
        results_dir: Directory where training results will be saved (defaults to "~/phantom_results").
        copy_files_to_results_dir: Any files given here will be copied to a
            "copied_files" sub-directory in the experiment results directory. Paths
            should be given relative to the main experiment entry point script.
        local_mode: If true will force Ray to run in one process (useful for
            profiling & debugging).
        print_info: If true will print a summary of the configuration before
            running.

    Returns:
        The results directory of the experiment if results are saved and the experiment
        was successful.

    NOTE: It is the users responsibility to ensure the PYTHONHASHSEED environment variable
    is set before starting the Python interpreter to run this code. Not setting this may
    lead to reproducibility issues.
    """
    show_pythonhashseed_warning()

    env_config = env_config or {}
    alg_config = alg_config or {}
    policy_grouping = policy_grouping
    metrics = metrics or {}
    callbacks = callbacks or []
    copy_files_to_results_dir = copy_files_to_results_dir or []

    local_files_to_copy = []
    local_dir = Path(__main__.__file__).parent

    if discard_results == False and len(copy_files_to_results_dir) > 0:
        # Check that files in the copy_files_to_results_dir list exist
        for file in copy_files_to_results_dir:
            path = Path(local_dir, file)

            if path.exists():
                local_files_to_copy.append(file)
            else:
                logger.warning(
                    f"Could not find file '{path}' to copy to results directory",
                )

    config = create_rllib_config_dict(
        env,
        env_config,
        alg_config,
        policy_grouping,
        callbacks,
        metrics,
        seed,
        num_workers,
    )

    if print_info:
        print_experiment_info(
            config,
            experiment_name,
            env.env_name,
            num_workers,
            num_episodes,
            algorithm,
            checkpoint_freq,
        )

    register_env(env.env_name, lambda config: env(**config))

    training_it = int(num_episodes / num_workers) if num_workers > 0 else num_episodes

    results_dir = tempfile.mkdtemp() if discard_results else results_dir

    ray.init(local_mode=local_mode)

    try:
        tune.run(
            algorithm,
            name=experiment_name,
            local_dir=results_dir,
            checkpoint_freq=checkpoint_freq,
            checkpoint_at_end=True,
            stop={"training_iteration": training_it},
            config=config,
            callbacks=[
                TBXExtendedLoggerCallback(),
                TrialStartTasksCallback(env, local_dir, local_files_to_copy),
            ],
        )

    except Exception as e:
        # Ensure that Ray is properly shutdown in the instance of an error occuring
        ray.shutdown()
        raise e
    else:
        ray.shutdown()

    if discard_results:
        return None
    else:
        return find_most_recent_results_dir(Path(results_dir, experiment_name))


def create_rllib_config_dict(
    env: PhantomEnv,
    env_config: Mapping[str, Any],
    alg_config: Mapping[str, Any],
    policy_grouping: Mapping[str, Any],
    callbacks: Iterable[DefaultCallbacks],
    metrics: Mapping[str, Metric],
    seed: int,
    num_workers: int,
) -> Dict[str, Any]:
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
        env = env(**env_config)
    except:
        env = env()

    ma_config = {}

    if policy_grouping is not None:
        custom_policies = {}
        custom_policies_to_train = []
        mapping = {}

        for pid, aids in policy_grouping.items():
            # Enable with Ray 1.7.0:
            # custom_policies[pid] = PolicySpec(
            #     policy_class=None,
            #     observation_space=env.agents[aids[0]].get_observation_space(),
            #     action_space=env.agents[aids[0]].get_action_space(),
            #     config=env.agents[aids[0]].policy_config or dict(),
            # )
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
                # Enable with Ray 1.7.0:
                # custom_policies[aid] = PolicySpec(
                #     policy_class=agent.policy_class,
                #     observation_space=agent.get_observation_space(),
                #     action_space=agent.get_action_space(),
                #     config=agent.policy_config or dict(),
                # )
                custom_policies[aid] = (
                    agent.policy_class,
                    agent.get_observation_space(),
                    agent.get_action_space(),
                    agent.policy_config or dict(),
                )

                mapping[aid] = aid

                if (
                    agent.policy_class is None
                    or FixedPolicy not in agent.policy_class.__mro__
                ):
                    custom_policies_to_train.append(aid)

        ma_config["policies"] = custom_policies
        ma_config["policy_mapping"] = mapping
        ma_config[
            "policy_mapping_fn"
        ] = lambda agent_id, episode=None, **kwargs: mapping[agent_id]
        ma_config["policies_to_train"] = custom_policies_to_train

    else:
        ma_config["policies"] = {
            # Enable with Ray 1.7.0:
            # aid: PolicySpec(
            #     policy_class=agent.policy_class,
            #     observation_space=agent.get_observation_space(),
            #     action_space=agent.get_action_space(),
            #     config=agent.policy_config or dict(),
            # )
            aid: (
                agent.policy_class,
                agent.get_observation_space(),
                agent.get_action_space(),
                agent.policy_config or dict(),
            )
            for aid, agent in env.agents.items()
        }

        ma_config["policy_mapping"] = {aid: aid for aid in env.agents.keys()}
        ma_config[
            "policy_mapping_fn"
        ] = lambda agent_id, episode=None, **kwargs: agent_id

        ma_config["policies_to_train"] = [
            agent_id
            for agent_id, policy_spec in ma_config["policies"].items()
            if policy_spec[0] is None or FixedPolicy not in policy_spec[0].__mro__
        ]

    if len(ma_config["policies_to_train"]) == 0:
        raise Exception("Must have at least one trained policy to perform training.")

    config = {}

    config["env"] = env.env_name
    config["env_config"] = env_config
    config["seed"] = seed
    config["multiagent"] = ma_config
    config["num_workers"] = num_workers
    config["rollout_fragment_length"] = env.clock.n_steps

    config["train_batch_size"] = int(
        (config["rollout_fragment_length"] * num_workers)
        if num_workers > 0
        else config["rollout_fragment_length"]
    )

    config["sgd_minibatch_size"] = max(int(config["train_batch_size"] / 10), 1)

    if callbacks is not None:
        callbacks = callbacks
        if not isinstance(callbacks, (list, tuple)):
            callbacks = [callbacks]
        config["callbacks"] = MultiCallbacks(callbacks)

    if metrics:
        if "callbacks" in config:
            config["callbacks"] = MultiCallbacks(
                [config["callbacks"], MetricsLoggerCallbacks("phantom", metrics)]
            )
        else:
            config["callbacks"] = MetricsLoggerCallbacks("phantom", metrics)

    config.update(**alg_config)

    return config


def print_experiment_info(
    config: Dict[str, Any],
    experiment_name: str,
    env_name: str,
    num_workers: int,
    num_episodes: int,
    algorithm: str,
    checkpoint_freq: int,
) -> None:
    def get_space_size(space: gym.Space) -> int:
        if isinstance(space, gym.spaces.Box):
            return sum(space.shape)
        elif isinstance(space, gym.spaces.Discrete):
            return 1
        elif isinstance(space, gym.spaces.Tuple):
            return sum(get_space_size(elem) for elem in space)
        elif isinstance(space, gym.spaces.Dict):
            return sum(get_space_size(elem) for elem in space.spaces.values())
        else:
            raise NotImplementedError(type(space))

    print()
    print("General Parameters")
    print("==================")
    print(f"Experiment name  : {experiment_name}")
    print(f"Environment name : {env_name}")
    print(f"Num workers      : {num_workers}")
    print(f"Num episodes     : {num_episodes}")
    print(f"Algorithm        : {algorithm}")
    print(f"Num steps        : {config['rollout_fragment_length']}")
    print(f"Checkpoint freq. : {checkpoint_freq}")
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
    if len(trained_policy_data) > 0:
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


class TrialStartTasksCallback(LoggerCallback):
    """
    Internal Callback for performing tasks at the start of each trial such as copying
    files to the results directory.
    """

    def __init__(
        self,
        env: Type[PhantomEnv],
        local_dir: Optional[Path],
        files: List[Union[str, Path]],
    ) -> None:
        self.env = env
        self.local_dir = local_dir
        self.files = files

    def log_trial_start(self, trial: tune.trial.Trial) -> None:
        # Save environment for use by rollout script
        cloudpickle.dump(self.env, open(Path(trial.logdir, "env.pkl"), "wb"))

        # Copy any files provided in the copy_files_to_results_dir field
        if self.local_dir is not None:
            source_code_dir = Path(trial.logdir).joinpath("copied_files")
            os.mkdir(source_code_dir)

            for file in self.files:
                old_path = Path(self.local_dir, file)
                new_path = Path(source_code_dir, file)

                shutil.copy(old_path, new_path)

    def __call__(self) -> "TrialStartTasksCallback":
        return self
