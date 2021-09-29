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
    params = ph.TrainingParams(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=3,
        env=MinimalEnv,
        env_config={},
        discard_results=True,
    )

    ph.utils.training.train_from_params_object(params)
