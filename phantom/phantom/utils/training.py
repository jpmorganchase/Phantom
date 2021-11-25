import __main__
import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Type, Union

import cloudpickle
import gym
import mercury as me
import ray
from ray import tune
from ray import rllib
from ray.rllib.agents.callbacks import DefaultCallbacks, MultiCallbacks
from ray.tune.logger import LoggerCallback
from ray.tune.registry import register_env
from tabulate import tabulate

from ..env import PhantomEnv
from ..fsm import FSMAgent, StagePolicyHandler
from ..logging import Metric, MetricsLoggerCallbacks
from ..logging.callbacks import TBXExtendedLoggerCallback
from ..policy import FixedPolicy
from ..policy_wrapper import PolicyWrapper
from ..typedefs import PolicyID
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

    NOTE: It is the users responsibility to invoke training via the provided ``phantom``
    command or ensure the ``PYTHONHASHSEED`` environment variable is set before starting
    the Python interpreter to run this code. Not setting this may lead to
    reproducibility issues.
    """
    show_pythonhashseed_warning()

    env_config = env_config or {}
    alg_config = alg_config or {}
    policy_grouping = policy_grouping or {}
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

    config, policies = create_rllib_config_dict(
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
            policies,
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
    env_class: Type[PhantomEnv],
    env_config: Mapping[str, Any],
    alg_config: Mapping[str, Any],
    policy_grouping: Mapping[str, List[str]],
    callbacks: Iterable[DefaultCallbacks],
    metrics: Mapping[str, Metric],
    seed: int,
    num_workers: int,
) -> Tuple[Dict[str, Any], List[PolicyWrapper]]:
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
        env = env_class(**env_config)
    except:
        env = env_class()

    def is_trained(policy_class=Optional[Type[rllib.Policy]]) -> bool:
        # To find out if policy_class is an subclass of FixedPolicy normally
        # would use isinstance() but since policy_class is a class and not
        # an instance this doesn't work.
        return policy_class is None or FixedPolicy not in policy_class.__mro__

    ma_config: Dict[str, Any] = {}

    policies: List[PolicyWrapper] = []
    policies_to_train: List[PolicyID] = []
    policy_mapping: Dict[PolicyID, Union[PolicyID, me.ID]] = {}

    if policy_grouping != {}:
        for shared_policy_name, ids in policy_grouping.items():
            if len(ids) == 0:
                raise ValueError(
                    f"Shared policy grouping '{shared_policy_name}' must have at least one agent using it"
                )

            policy_classes = []
            policy_configs = []
            obs_spaces = []
            action_spaces = []

            used_by = []

            for agent_or_stage_id in ids:
                if agent_or_stage_id in env.agents:
                    agent_id = agent_or_stage_id
                    used_by.append(agent_id)
                    agent = env.agents[agent_id]

                    policy_classes.append(agent.policy_class)
                    policy_configs.append(agent.policy_config)
                    obs_spaces.append(agent.get_observation_space())
                    action_spaces.append(agent.get_action_space())
                else:
                    elems = agent_or_stage_id.split("__")
                    if len(elems) != 2:
                        raise Exception

                    agent_id, stage_id = elems
                    agent = env.agents[agent_id]
                    used_by.append((agent_id, stage_id))
                    handler = agent.stage_handler_map[agent_or_stage_id]

                    policy_classes.append(handler.policy_class)
                    policy_configs.append(handler.policy_config)
                    obs_spaces.append(handler.get_observation_space(agent))
                    action_spaces.append(handler.get_action_space(agent))

            if not all(x == policy_classes[0] for x in policy_classes):
                raise ValueError(
                    f"All agents in shared policy grouping '{shared_policy_name}' must have same policy class (got '{policy_classes[0]}' and '{policy_classes[1]}')"
                )

            if not all(x == policy_configs[0] for x in policy_configs):
                raise ValueError(
                    f"All agents in shared policy grouping '{shared_policy_name}' must have same policy config (got '{policy_configs[0]}' and '{policy_configs[1]}')"
                )

            if not all(x == obs_spaces[0] for x in obs_spaces):
                raise ValueError(
                    f"All agents in shared policy grouping '{shared_policy_name}' must have same observation space (got '{obs_spaces[0]}' and '{obs_spaces[1]}')"
                )

            if not all(x == action_spaces[0] for x in action_spaces):
                raise ValueError(
                    f"All agents in shared policy grouping '{shared_policy_name}' must have same action space (got '{action_spaces[0]}' and '{action_spaces[1]}')"
                )

            policy_wrapper = PolicyWrapper(
                used_by=used_by,
                trained=is_trained(policy_classes[0]),
                obs_space=obs_spaces[0],
                action_space=action_spaces[0],
                policy_class=policy_classes[0],
                policy_config=policy_configs[0],
                shared_policy_name=shared_policy_name,
            )

            policies.append(policy_wrapper)

            policy_id = policy_wrapper.get_name()

            for id in ids:
                policy_mapping[id] = policy_id

            policies_to_train.append(policy_id)

    print(policy_mapping)

    for agent_id, agent in env.agents.items():
        if agent_id in map(str, policy_mapping.values()):
            raise ValueError(
                f"Can't add agent '{agent_id}' policy, policy with same name already exists"
            )

        if isinstance(agent, FSMAgent):
            for stage_ids, stage_handler in agent.stage_handlers:
                if isinstance(stage_handler, StagePolicyHandler):
                    policy_name = (
                        agent_id
                        + "__"
                        + "+".join(str(stage_id) for stage_id in stage_ids)
                    )

                    # Ignore this agent if it has already been included via a shared policy
                    if policy_name in policy_mapping:
                        continue

                    policy_wrapper = PolicyWrapper(
                        used_by=[(agent_id, stage_id) for stage_id in stage_ids],
                        trained=is_trained(agent.policy_class),
                        obs_space=stage_handler.get_observation_space(agent),
                        action_space=stage_handler.get_action_space(agent),
                        policy_class=stage_handler.policy_class,
                        policy_config=stage_handler.policy_config,
                    )

                    policies.append(policy_wrapper)

                    policy_id = policy_wrapper.get_name()

                    policy_mapping[policy_id] = policy_id

        else:
            # Ignore this agent if it has already been included via a shared policy
            if agent_id in policy_mapping:
                continue

            policy_wrapper = PolicyWrapper(
                used_by=[agent_id],
                trained=is_trained(agent.policy_class),
                obs_space=agent.get_observation_space(),
                action_space=agent.get_action_space(),
                policy_class=agent.policy_class,
                policy_config=agent.policy_config,
            )

            policies.append(policy_wrapper)

            policy_id = policy_wrapper.get_name()

            policy_mapping[agent_id] = policy_id

    ma_config["policies"] = {
        policy.get_name(): policy.get_spec() for policy in policies
    }
    ma_config["policies_to_train"] = [
        policy.get_name() for policy in policies if policy.trained
    ]
    ma_config["policy_mapping"] = policy_mapping
    ma_config["policy_mapping_fn"] = lambda id, **kwargs: str(policy_mapping[id])

    if len(ma_config["policies_to_train"]) == 0:
        raise Exception("Must have at least one trained policy to perform training.")

    config: Dict[str, Any] = {}

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
        config["callbacks"] = MultiCallbacks(callbacks)

    if metrics:
        if "callbacks" in config:
            config["callbacks"] = MultiCallbacks(
                [config["callbacks"], MetricsLoggerCallbacks("phantom", metrics)]
            )
        else:
            config["callbacks"] = MetricsLoggerCallbacks("phantom", metrics)

    config.update(**alg_config)

    return config, policies


def print_experiment_info(
    config: Dict[str, Any],
    policies: List[PolicyWrapper],
    experiment_name: str,
    env_name: str,
    num_workers: int,
    num_episodes: int,
    algorithm: str,
    checkpoint_freq: Optional[int],
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

    for policy in policies:
        used_by = []
        for x in policy.used_by:
            used_by.append(f"{x[0]}[{x[1]}]" if isinstance(x, Tuple) else x)

        data = (
            policy.get_name(),
            get_space_size(policy.obs_space),
            get_space_size(policy.action_space),
            ", ".join(used_by),
        )

        if str(policy.get_name()) in config["multiagent"]["policies_to_train"]:
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
