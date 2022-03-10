import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.decoders import Decoder, DictDecoder
from phantom.encoders import DictEncoder, Encoder
from ray.rllib.agents.ppo.ppo_tf_policy import PPOTFPolicy
from ray.rllib.agents.trainer_template import build_trainer
from ray.rllib.evaluation.worker_set import WorkerSet
from ray.rllib.execution.metric_ops import StandardMetricsReporting
from ray.rllib.execution.rollout_ops import (
    ParallelRollouts,
    ConcatBatches,
    StandardizeFields,
    SelectExperiences,
)
from ray.rllib.execution.train_ops import TrainOneStep


class MinimalEncoder(Encoder):
    def __init__(self, id: int):
        self.id = id

    @property
    def output_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def encode(self, ctx: me.Network.Context) -> np.ndarray:
        return np.array([self.id])

    def reset(self):
        self.id = None


class MinimalDecoder(Decoder):
    def __init__(self, aid: str):
        self.aid = aid

    @property
    def action_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def decode(self, ctx: me.Network.Context, action) -> ph.Packet:
        if self.aid == "a1":
            return ph.packet.Packet(
                messages={
                    "a2": [
                        10,
                    ]
                }
            )
        if self.aid == "a2":
            return ph.packet.Packet(
                messages={
                    "a1": [
                        20,
                    ]
                }
            )

    def reset(self):
        self.id = None


class MinimalAgent(ph.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id,
            obs_encoder=DictEncoder({"e1": MinimalEncoder(1), "e2": MinimalEncoder(2)}),
            action_decoder=DictDecoder({"d1": MinimalDecoder(id)}),
        )

    def compute_reward(self, ctx: me.Network.Context) -> float:
        return 0

    def handle_message(self, ctx: me.Network.Context, msg: me.Message):
        yield from ()


class MinimalEnv(ph.PhantomEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("a1"), MinimalAgent("a2")]

        network = me.StochasticNetwork(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("a1", "a2")

        super().__init__(network=network, n_steps=3)


def test_experiment():
    def custom_training_workflow(workers: WorkerSet, config: dict):
        rollouts = ParallelRollouts(workers, mode="bulk_sync")

        ppo_train_op = (
            rollouts.for_each(SelectExperiences(["shared_policy"]))
            .combine(ConcatBatches(min_batch_size=200, count_steps_by="env_steps"))
            .for_each(StandardizeFields(["advantages"]))
            .for_each(
                TrainOneStep(
                    workers,
                    policies=["shared_policy"],
                    num_sgd_iter=10,
                    sgd_minibatch_size=128,
                )
            )
        )

        return StandardMetricsReporting(ppo_train_op, workers, config)

    trainer = build_trainer(
        name="PPO_MultiAgent",
        default_policy=PPOTFPolicy,
        execution_plan=custom_training_workflow,
    )

    ph.train(
        experiment_name="unit-testing",
        trainer=trainer,
        num_workers=0,
        num_episodes=1,
        env_class=MinimalEnv,
        policy_grouping={"shared_policy": ["a1", "a2"]},
        discard_results=True,
    )
