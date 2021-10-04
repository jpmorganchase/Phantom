import shutil
import os
from pathlib import Path

import cloudpickle
import gym
import mercury as me
import numpy as np
import phantom as ph


class MinimalAgent(ph.agent.ZeroIntelligenceAgent):
    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        return ph.packet.Packet()


class MinimalEnv(ph.PhantomEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("ZI1")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(network=network, n_steps=3)


def test_experiment():
    training_params = ph.TrainingParams(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=3,
        env=MinimalEnv,
        env_config={},
    )

    results_dir = ph.utils.training.train_from_params_object(training_params)

    assert os.path.exists(results_dir)

    rollout_params = ph.RolloutParams(
        directory=results_dir,
        algorithm="PPO",
        num_workers=1,
        num_rollouts=5,
        env_config={},
        save_trajectories=True,
    )

    metrics, trajectories = ph.utils.rollout.run_rollouts(rollout_params)

    assert len(metrics) == 5
    assert len(trajectories) == 5
    assert type(metrics[0]) == dict
    assert type(trajectories[0]) == ph.utils.rollout.EpisodeTrajectory

    results = cloudpickle.load(
        open(Path(results_dir, rollout_params.results_file), "rb")
    )

    assert type(results) == dict
    assert "metrics" not in results
    assert "trajectories" in results

    assert len(results["trajectories"]) == 5
    assert type(results["trajectories"][0]) == ph.utils.rollout.EpisodeTrajectory

    shutil.rmtree(results_dir)
