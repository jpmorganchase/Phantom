import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import cloudpickle
import gym
import numpy as np
import phantom as ph


class MinimalEncoder(ph.Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: ph.Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


class MinimalDecoder(ph.Decoder):
    def __init__(self, aid: str):
        self.aid = aid

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: ph.Context, action) -> List[Tuple[ph.AgentID, ph.Message]]:
        if self.aid == "a1":
            return [("a2", 10)]
        if self.aid == "a2":
            return [("a1", 10)]

    def reset(self):
        self.id = None


class MinimalAgent(ph.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id,
            obs_encoder=ph.encoders.DictEncoder(
                {"e1": MinimalEncoder(1), "e2": MinimalEncoder(2)}
            ),
            action_decoder=ph.encoders.DictDecoder({"d1": MinimalDecoder(id)}),
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0

    def handle_message(
        self, ctx: ph.Context, sender_id: ph.AgentID, msg: ph.Message
    ) -> List[Tuple[ph.AgentID, ph.Message]]:
        return []


class MinimalEnv(ph.PhantomEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("a1"), MinimalAgent("a2")]

        network = ph.network.StochasticNetwork(agents)

        network.add_connection("a1", "a2")

        super().__init__(num_steps=3, network=network)


def test_experiment():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Perform training

        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        results = trainer.train(
            env_class=MinimalEnv,
            num_iterations=5,
            policies={
                "shared_policy": MinimalAgent,
            },
            policies_to_train=["shared_policy"],
        )

        ################################################################################

        # Perform rollouts without EnvSupertype

        # rollouts_1 = ph.rollout(
        #     directory=results_dir,
        #     algorithm="PPO",
        #     num_workers=0,
        #     num_repeats=1,
        #     save_trajectories=True,
        #     save_messages=True,
        # )

        # rollouts_2 = cloudpickle.load(open(Path(results_dir, "results.pkl"), "rb"))

        # for rollouts in [rollouts_1, rollouts_2]:
        #     assert type(rollouts) == list
        #     assert len(rollouts) == 1
        #     assert type(rollouts[0]) == ph.utils.rollout.Rollout
        #     assert rollouts[0].rollout_params == {}
        #     assert len(rollouts[0].steps) == 3
        #     assert rollouts[0].steps[0].messages == [
        #         me.Message(sender_id="a1", receiver_id="a2", payload=10),
        #         me.Message(sender_id="a2", receiver_id="a1", payload=20),
        #     ]
        #     assert rollouts[0].steps[1].messages == [
        #         me.Message(sender_id="a1", receiver_id="a2", payload=10),
        #         me.Message(sender_id="a2", receiver_id="a1", payload=20),
        #     ]

        ################################################################################

        # Perform rollouts with EnvSupertype (to test Ranges, rollout_params, etc.)

        # @dataclass
        # class EnvSupertype(ph.BaseSupertype):
        #     weight: ph.SupertypeField[float]

        # rollouts_1 = ph.rollout(
        #     directory=results_dir,
        #     algorithm="PPO",
        #     num_workers=0,
        #     num_repeats=1,
        #     save_trajectories=True,
        #     save_messages=True,
        #     env_supertype=EnvSupertype(
        #         weight=ph.utils.ranges.LinspaceRange(0.0, 1.0, 2, name="EnvRange")
        #     ),
        # )

        # rollouts_2 = cloudpickle.load(open(Path(results_dir, "results.pkl"), "rb"))

        # for rollouts in [rollouts_1, rollouts_2]:
        #     assert type(rollouts) == list
        #     assert len(rollouts) == 2
        #     assert type(rollouts[0]) == ph.utils.rollout.Rollout
        #     assert rollouts[0].rollout_params == {"EnvRange": 0.0}
        #     assert rollouts[1].rollout_params == {"EnvRange": 1.0}
