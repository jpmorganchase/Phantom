"""
Test a very simple FSM Env with one state and one agent.

The agent takes actions and returns rewards on each step.

The episode duration is two steps.
"""

import tempfile

import numpy as np
import phantom as ph

from .. import MockStrategicAgent


class OneStateFSMEnvWithHandler(ph.fsm.FiniteStateMachineEnv):
    def __init__(self):
        agents = [MockStrategicAgent("agent")]

        network = ph.Network(agents)

        network.add_connection("agent", "agent")

        super().__init__(
            num_steps=2,
            network=network,
            initial_stage="UNIT",
        )

    @ph.fsm.FSMStage(stage_id="UNIT", next_stages=["UNIT"])
    def handle(self):
        return "UNIT"


def test_one_state_with_handler():
    env = OneStateFSMEnvWithHandler()

    assert env.reset() == {"agent": np.array([0.0])}

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    assert step.observations == {"agent": np.array([0.5])}
    assert step.rewards == {"agent": 0}
    assert step.dones == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    assert step.observations == {"agent": np.array([1.0])}
    assert step.rewards == {"agent": 0}
    assert step.dones == {"__all__": True, "agent": False}
    assert step.infos == {"agent": {}}


# TODO: replace with Phantom trainer
# def test_one_state_with_handler_with_ray():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         ph.utils.rllib.train(
#             algorithm="PPO",
#             num_workers=0,
#             num_iterations=1,
#             env_class=OneStateFSMEnvWithHandler,
#             env_config={},
#             policies={
#                 "agent_policy": ["agent"],
#             },
#             policies_to_train=["agent_policy"],
#             tune_config={
#                 "local_dir": tmpdir,
#             }
#         )


class OneStateFSMEnvWithoutHandler(ph.fsm.FiniteStateMachineEnv):
    def __init__(self):
        agents = [MockStrategicAgent("agent")]

        network = ph.Network(agents)

        network.add_connection("agent", "agent")

        super().__init__(
            num_steps=2,
            network=network,
            initial_stage="UNIT",
            stages=[
                ph.fsm.FSMStage(stage_id="UNIT", next_stages=["UNIT"], handler=None)
            ],
        )


def test_one_state_without_handler():
    env = OneStateFSMEnvWithoutHandler()

    assert env.reset() == {"agent": np.array([0.0])}

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    assert step.observations == {"agent": np.array([0.5])}
    assert step.rewards == {"agent": 0}
    assert step.dones == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    step = env.step({"agent": np.array([0])})

    assert env.current_stage == "UNIT"
    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    assert step.observations == {"agent": np.array([1.0])}
    assert step.rewards == {"agent": 0}
    assert step.dones == {"__all__": True, "agent": False}
    assert step.infos == {"agent": {}}


# TODO: replace with Phantom trainer
# def test_one_state_without_handler_with_ray():
#     with tempfile.TemporaryDirectory() as tmpdir:
#         ph.utils.rllib.train(
#             algorithm="PPO",
#             num_workers=0,
#             num_iterations=1,
#             env_class=OneStateFSMEnvWithoutHandler,
#             env_config={},
#             policies={
#                 "agent_policy": ["agent"],
#             },
#             policies_to_train=["agent_policy"],
#             tune_config={
#                 "local_dir": tmpdir,
#             }
#         )
