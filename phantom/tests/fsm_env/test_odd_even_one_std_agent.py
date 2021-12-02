"""
Test a very simple FSM Env with two stages and one agents.

The agent uses a different policy based on the stage.

The episode duration is two steps.
"""

import logging
from enum import Enum

import mercury as me
import numpy as np
from phantom.fsm import FSMStage, FiniteStateMachineEnv

from . import MockAgent


logging.basicConfig(level=logging.INFO)


class Stages(Enum):
    ODD = 1
    EVEN = 2


class MockFSMEnv(FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MockAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.ODD,
            stages=[
                FSMStage(
                    stage_id=Stages.ODD,
                    next_stages=[Stages.EVEN],
                ),
                FSMStage(
                    stage_id=Stages.EVEN,
                    next_stages=[Stages.ODD],
                ),
            ],
        )


def test_odd_even_one_std_agent():
    env = MockFSMEnv()

    assert env.reset() == {"agent": np.array([0])}

    assert env.current_stage == Stages.ODD
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == Stages.EVEN
    assert step.observations == {"agent": np.array([0])}
    assert step.rewards == {"agent": 0.0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1