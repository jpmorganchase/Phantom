"""
Test a very simple FSM Env with two stages and one agents.

The agent uses a different policy based on the stage.

The episode duration is two steps.
"""

import logging
from enum import Enum

import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.fsm import FSMStage, FiniteStateMachineEnv


logging.basicConfig(level=logging.INFO)


class Stages(Enum):
    UNIT = 1


class MockAgent(ph.Agent):
    def __init__(self, agent_id: me.ID) -> None:
        super().__init__(agent_id)

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    def decode_action(self, ctx: me.Network.Context, action: np.ndarray):
        self.decode_action_count += 1
        return ph.agent.Packet()

    def encode_obs(self, ctx: me.Network.Context):
        self.encode_obs_count += 1
        return np.zeros((1,))

    def compute_reward(self, ctx: me.Network.Context) -> float:
        self.compute_reward_count += 1
        return 0.0

    def get_observation_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))

    def get_action_space(self):
        return gym.spaces.Box(-np.inf, np.inf, (1,))


class MockFSMEnv(FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MockAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.UNIT,
            stages=[
                FSMStage(
                    stage_id=Stages.UNIT,
                    next_stages=[Stages.UNIT],
                )
            ],
        )


def test_with_standard_agent():
    env = MockFSMEnv()

    assert env.reset() == {"agent": np.array([0])}

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == Stages.UNIT
    assert step.observations == {"agent": np.array([0])}
    assert step.rewards == {"agent": 0.0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1
