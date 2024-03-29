"""
Test a very simple FSM Env with two stages and one agents.

The agent uses a different policy based on the stage.

The episode duration is two steps.
"""

import numpy as np
import phantom as ph

from .. import MockStrategicAgent


class MockFSMEnv(ph.FiniteStateMachineEnv):
    def __init__(self):
        agents = [MockStrategicAgent("odd_agent"), MockStrategicAgent("even_agent")]

        network = ph.Network(agents)

        super().__init__(
            num_steps=3,
            network=network,
            initial_stage="ODD",
            stages=[
                ph.FSMStage(
                    stage_id="ODD",
                    next_stages=["EVEN"],
                    acting_agents=["odd_agent"],
                    rewarded_agents=["odd_agent"],
                ),
                ph.FSMStage(
                    stage_id="EVEN",
                    next_stages=["ODD"],
                    acting_agents=["even_agent"],
                    rewarded_agents=["even_agent"],
                ),
            ],
        )


def test_odd_even_two_agents():
    env = MockFSMEnv()

    assert env.reset() == ({"odd_agent": np.array([0])}, {})

    assert env.current_stage == "ODD"
    assert env.agents["odd_agent"].compute_reward_count == 0
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 0

    assert env.agents["even_agent"].compute_reward_count == 0
    assert env.agents["even_agent"].encode_obs_count == 0
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"odd_agent": np.array([1])})

    assert env.current_stage == "EVEN"
    assert step.observations == {"even_agent": np.array([1.0 / 3.0])}
    assert step.rewards == {"even_agent": None}
    assert step.terminations == {
        "even_agent": False,
        "odd_agent": False,
        "__all__": False,
    }
    assert step.truncations == {
        "even_agent": False,
        "odd_agent": False,
        "__all__": False,
    }
    assert step.infos == {"even_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1

    assert env.agents["even_agent"].compute_reward_count == 0
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"even_agent": np.array([0])})

    assert env.current_stage == "ODD"
    assert step.observations == {"odd_agent": np.array([2.0 / 3.0])}
    assert step.rewards == {"odd_agent": 0.0}
    assert step.terminations == {
        "even_agent": False,
        "odd_agent": False,
        "__all__": False,
    }
    assert step.truncations == {
        "even_agent": False,
        "odd_agent": False,
        "__all__": False,
    }
    assert step.infos == {"odd_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["odd_agent"].decode_action_count == 1

    assert env.agents["even_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].decode_action_count == 1
