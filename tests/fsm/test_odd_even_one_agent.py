"""
Test a very simple FSM Env with two stages and one agents.

The agent uses a different policy based on the stage.

The episode duration is two steps.
"""

import numpy as np
import phantom as ph

from .. import MockStrategicAgent


class MockFSMEnv(ph.fsm.FiniteStateMachineEnv):
    def __init__(self):
        agents = [MockStrategicAgent("agent")]

        network = ph.Network(agents)

        super().__init__(
            num_steps=3,
            network=network,
            initial_stage="ODD",
            stages=[
                ph.fsm.FSMStage(
                    stage_id="ODD",
                    next_stages=["EVEN"],
                ),
                ph.fsm.FSMStage(
                    stage_id="EVEN",
                    next_stages=["ODD"],
                ),
            ],
        )


def test_odd_even_one_agent():
    env = MockFSMEnv()

    assert env.reset() == {"agent": np.array([0])}

    assert env.current_stage == "ODD"
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == "EVEN"
    assert step.observations == {"agent": np.array([1.0 / 3.0])}
    assert step.rewards == {"agent": 0.0}
    assert step.dones == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1
