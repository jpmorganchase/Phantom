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
        agents = [MockAgent("odd_agent"), MockAgent("even_agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.ODD,
            stages=[
                FSMStage(
                    stage_id=Stages.ODD,
                    next_stages=[Stages.EVEN],
                    acting_agents=["odd_agent"],
                    rewarded_agents=["odd_agent"],
                ),
                FSMStage(
                    stage_id=Stages.EVEN,
                    next_stages=[Stages.ODD],
                    acting_agents=["even_agent"],
                    rewarded_agents=["even_agent"],
                ),
            ],
        )


def test_odd_even_two_std_agents():
    env = MockFSMEnv()

    assert env.reset() == {"odd_agent": np.array([0])}

    assert env.current_stage == Stages.ODD
    assert env.agents["odd_agent"].compute_reward_count == 0
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 0

    assert env.agents["even_agent"].compute_reward_count == 0
    assert env.agents["even_agent"].encode_obs_count == 0
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"odd_agent": np.array([1])})

    assert env.current_stage == Stages.EVEN
    assert step.observations == {"even_agent": np.array([0])}
    assert step.rewards == {"even_agent": None}
    assert step.terminals == {"__all__": False, "even_agent": False, "odd_agent": False}
    assert step.infos == {"even_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1

    assert env.agents["even_agent"].compute_reward_count == 0
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"even_agent": np.array([0])})

    assert env.current_stage == Stages.ODD
    assert step.observations == {"odd_agent": np.array([0])}
    assert step.rewards == {"odd_agent": 0.0}
    assert step.terminals == {"__all__": False, "odd_agent": False, "even_agent": False}
    assert step.infos == {"odd_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["odd_agent"].decode_action_count == 1

    assert env.agents["even_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].decode_action_count == 1
