from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, List, Mapping, Optional, Type, Union

from ray.rllib.agents.callbacks import DefaultCallbacks

from .logging import Metric
from .env import PhantomEnv


@dataclass
class TrainingParams:
    """
    Class containing configuration parameters for running a Phantom experiment.

    Attributes:
        experiment_name: Experiment name used for tensorboard logging.
        env: A PhantomEnv subclass.
        num_workers: Number of Ray workers to initialise.
        num_episodes: Number of episodes to train for, distributed over all workers.
        algorithm: RL algorithm to use, defaults to PPO.
        seed: Optional seed to pass to environment.
        checkpoint_freq: Episodic frequency at which to save checkpoints.
        env_config: Configuration parameters to pass to the environment init method.
        alg_config: Optional algorithm parameters dictionary to pass to RLlib.
        metrics: Optional set of metrics to record and log.
        callbacks: Optional Ray Callbacks for custom metrics.
            (https://docs.ray.io/en/master/rllib-training.html#callbacks-and-custom-metrics)
        discard_results: If True, all results are discarded (useful for unit testing & development).
        results_dir: Directory where training results will be saved (defaults to "~/phantom_results").
        copy_files_to_results_dir: Any files given here will be copied to a
            "source_code" sub-directory in the experiment results directory. Paths
            should be given relative to the main experiment entry point script.
            NOTE: currently only functional when using phantom-train command.
    """

    experiment_name: str
    env: Type[PhantomEnv]
    num_workers: int
    num_episodes: int
    algorithm: str = "PPO"
    seed: int = 0
    checkpoint_freq: Optional[int] = None
    env_config: Mapping[str, Any] = field(default_factory=dict)
    alg_config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    callbacks: Optional[Iterable[DefaultCallbacks]] = None
    discard_results: bool = False
    results_dir: Union[str, Path] = "~/phantom_results"
    copy_files_to_results_dir: List[str] = field(default_factory=list)


@dataclass
class RolloutParams:
    """
    Class containing configuration parameters for running rollouts for a previously
    trained Phantom experiment.

    Attributes:
        directory: Phantom results directory containing trained policies.
        num_workers: Number of Ray rollout workers to initialise.
        num_rollouts: Number of rollouts to perform, distributed over all workers.
        algorithm: RL algorithm to use, defaults to PPO.
        checkpoint: Checkpoint to use (defaults to most recent).
        env_config: Configuration parameters to pass to the environment init method.
        metrics: Optional set of metrics to record and log.
        callbacks: Optional Ray Callbacks for custom metrics.
            (https://docs.ray.io/en/master/rllib-training.html#callbacks-and-custom-metrics)
        metrics_file: Name of the metrics file to save to, if None is given no file
            will be saved (default is "metrics.pkl").
        trajectories_file: Name of the trajectories file to save to, if None is
            given no file will be saved (default is "trajectories.pkl").
    """

    directory: Union[str, Path]
    num_workers: int
    num_rollouts: int
    algorithm: str = "PPO"
    checkpoint: Optional[int] = None
    env_config: Mapping[str, Any] = field(default_factory=dict)
    metrics: Mapping[str, Metric] = field(default_factory=dict)
    callbacks: Optional[Iterable[DefaultCallbacks]] = None
    metrics_file: Optional[Union[str, Path]] = "metrics.pkl"
    trajectories_file: Optional[Union[str, Path]] = "trajectories.pkl"
