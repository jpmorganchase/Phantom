from dataclasses import dataclass, field
from typing import Any, Iterable, List, Mapping, Optional, Type

from ray.rllib.agents.callbacks import DefaultCallbacks

from .logging import Metric
from .env import PhantomEnv


@dataclass
class PhantomParams:
    """
    Class containing configuration parameters for running a Phantom experiment.

    Attributes:
        experiment_name: Experiment name used for tensorboard logging.
        env: A PhantomEnv subclass.
        num_workers: Number of Ray workers to initialise.
        num_episodes: umber of episodes to train for, distributed over all workers.
        algorithm: RL algorithm to use, defaults to PPO.
        seed: Optional seed to pass to environment.
        checkpoint_freq: Episodic frequency at which to save checkpoints.
        env_config: Configuration parameters to pass to the environment init method.
        alg_config: Optional algorithm parameters dictionary to pass to RLlib.
        params: ptional set of metrics to record and log.
    """

    # Experiment name used for tensorboard logging.
    experiment_name: str

    # A PhantomEnv subclass.
    env: Type[PhantomEnv]

    # Number of Ray workers to initialise.
    num_workers: int

    # Number of episodes to train for, distributed over all workers.
    num_episodes: int

    # RL algorithm to use, defaults to PPO.
    algorithm: str = "PPO"

    # Seed used by Ray.
    seed: int = 0

    # Episodic frequency at which to save checkpoints.
    checkpoint_freq: Optional[int] = None

    # Configuration parameters to pass to the environment init method
    env_config: Mapping[str, Any] = field(default_factory=dict)

    # Optional algorithm parameters dictionary to pass to RLlib.
    alg_config: Mapping[str, Any] = field(default_factory=dict)

    # Optional set of metrics to record and log.
    metrics: Mapping[str, Metric] = field(default_factory=dict)

    # Optional Ray Callbacks for custom metrics.
    # https://docs.ray.io/en/master/rllib-training.html#callbacks-and-custom-metrics
    callbacks: Optional[Iterable[DefaultCallbacks]] = None

    # If True, all results are discarded (useful for unit testing & development).
    discard_results: bool = False

    # The directory where results will be saved by Ray.
    results_dir: str = "~/phantom_results"

    # Any files given here will be copied to a "source_code" sub-directory in the
    # experiment results directory. Paths should be given relative to the main
    # experiment entry point script.
    #
    # NOTE: currently only functional when using phantom-train command.
    copy_files_to_results_dir: List[str] = field(default_factory=list)
