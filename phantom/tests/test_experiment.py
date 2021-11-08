import os
import shutil
from pathlib import Path

import cloudpickle
import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.decoders import Decoder, DictDecoder
from phantom.encoders import DictEncoder, Encoder


class TestEncoder(Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


class TestDecoder(Decoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: me.Network.Context, action) -> ph.Packet:
        return ph.Packet()

    def reset(self):
        self.id = None


class MinimalAgent(ph.agent.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id,
            obs_encoder=DictEncoder({"e1": TestEncoder(1), "e2": TestEncoder(2)}),
            action_decoder=DictDecoder({"d1": TestDecoder(1), "d2": TestDecoder(2)}),
        )

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0


class MinimalEnv(ph.PhantomEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("a1"), MinimalAgent("a2"), MinimalAgent("a3")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(network=network, n_steps=3)


def test_experiment():
    results_dir = ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=3,
        env=MinimalEnv,
        env_config={},
        policy_grouping={"shared_policy": ["a2", "a3"]},
    )

    assert os.path.exists(results_dir)

    metrics, trajectories = ph.rollout(
        directory=results_dir,
        algorithm="PPO",
        num_workers=1,
        num_repeats=5,
        env_config={},
        save_trajectories=True,
    )

    assert len(metrics) == 5
    assert len(trajectories) == 5
    assert type(metrics[0]) == dict
    assert type(trajectories[0]) == ph.utils.rollout.EpisodeTrajectory

    results = cloudpickle.load(open(Path(results_dir, "results.pkl"), "rb"))

    assert type(results) == dict
    assert "metrics" not in results
    assert "trajectories" in results

    assert len(results["trajectories"]) == 5
    assert type(results["trajectories"][0]) == ph.utils.rollout.EpisodeTrajectory

    shutil.rmtree(results_dir)
