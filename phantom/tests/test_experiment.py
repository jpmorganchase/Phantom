from dataclasses import dataclass

import gym
import mercury as me
import numpy as np
import phantom as ph


@dataclass
class MinimalAgentType(ph.Type):
    weight: int


class MinimalAgentSupertype(ph.Supertype):
    def sample(self) -> MinimalAgentType:
        return MinimalAgentType(weight=1)


class MinimalAgent(ph.agent.ZeroIntelligenceAgent):
    def __init__(
        self,
        agent_id: str,
        supertype: ph.Supertype,
    ) -> None:
        super().__init__(agent_id, supertype=supertype)

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        assert self.type.weight == 1
        assert self.env_type.weight == 2

        return ph.packet.Packet()


@dataclass
class MinimalEnvType(ph.Type):
    weight: int


class MinimalEnvSupertype(ph.Supertype):
    def sample(self) -> MinimalEnvType:
        return MinimalEnvType(weight=2)


class MinimalEnv(ph.PhantomEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("ZI1", MinimalAgentSupertype())]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(network=network, n_steps=3, supertype=MinimalEnvSupertype())


def test_experiment():
    phantom_params = ph.PhantomParams(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=3,
        env=MinimalEnv,
        env_config={},
        discard_results=True,
    )

    ph.cmd_utils.train_from_params_object(phantom_params, local_mode=True)
