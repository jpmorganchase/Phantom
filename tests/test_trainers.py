import tempfile
from typing import List, Tuple

import gym
import numpy as np
import phantom as ph


class MockPolicy(ph.Policy):
    def compute_action(self, obs) -> int:
        return np.random.randint(0, 5)


class MockEncoder(ph.Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def observation_space(self) -> gym.Space:
        return gym.spaces.Discrete(5)

    def encode(self, ctx: ph.Context) -> np.ndarray:
        return self.id

    def reset(self):
        self.id = None


class MockDecoder(ph.Decoder):
    def __init__(self, aid: str):
        self.aid = aid

    @property
    def action_space(self) -> gym.Space:
        return gym.spaces.Discrete(1)

    def decode(self, ctx: ph.Context, action) -> List[Tuple[ph.AgentID, ph.Message]]:
        if self.aid == "a1":
            return [("a2", 10)]
        if self.aid == "a2":
            return [("a1", 10)]

    def reset(self):
        self.id = None


class MockAgent(ph.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id,
            # TODO: implement dict + tuple decoders for PPO trainer
            # observation_encoder=ph.encoders.DictEncoder(
            #     {"e1": MockEncoder(1), "e2": MockEncoder(2)}
            # ),
            # action_decoder=ph.decoders.DictDecoder({"d1": MockDecoder(id)}),
            observation_encoder=MockEncoder(1),
            action_decoder=MockDecoder(id),
        )

    def compute_reward(self, ctx: ph.Context) -> float:
        return 0

    def handle_message(
        self, ctx: ph.Context, sender_id: ph.AgentID, msg: ph.Message
    ) -> List[Tuple[ph.AgentID, ph.Message]]:
        return []


class MockEnv(ph.PhantomEnv):

    def __init__(self):
        agents = [MockAgent("a1"), MockAgent("a2")]

        network = ph.network.StochasticNetwork(agents)

        network.add_connection("a1", "a2")

        super().__init__(num_steps=10, network=network)


def test_experiment_1():
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Perform training

        trainer = ph.trainers.PPOTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "ppo_policy": MockAgent,
            },
            policies_to_train=["ppo_policy"],
        )

def test_experiment_2():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy": MockAgent,
            },
            policies_to_train=["q_policy"],
        )

def test_experiment_3():
    with tempfile.TemporaryDirectory() as tmp_dir:
        trainer = ph.trainers.QLearningTrainer(tensorboard_log_dir=tmp_dir)

        trainer.train(
            env_class=MockEnv,
            num_iterations=5,
            policies={
                "q_policy": ["a1"],
                "fixed_policy": (MockPolicy, ["a2"]),
            },
            policies_to_train=["q_policy"],
        )


def test_experiment_4():
    with tempfile.TemporaryDirectory() as tmp_dir:
        ph.utils.rllib.train(
            algorithm="PPO",
            env_class=MockEnv,
            num_iterations=1,
            policies={
                "ppo_policy": MockAgent,
            },
            policies_to_train=["ppo_policy"],
            tune_config={
                "local_dir": tmp_dir,
            }
        )
