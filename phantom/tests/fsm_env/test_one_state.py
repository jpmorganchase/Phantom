"""
Test a very simple FSM Env with one state and one agent.

The agent takes actions and returns rewards on each step.

The episode duration is two steps.
"""

from enum import Enum

import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.fsm import FSMAgent, FSMStage, FiniteStateMachineEnv, StagePolicyHandler


class Stages(Enum):
    UNIT = 1


class OneStageAgentHandler(StagePolicyHandler["OneStageAgent"]):
    @staticmethod
    def compute_reward(
        agent: "OneStageAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> float:
        agent.compute_reward_count += 1
        return 0

    @staticmethod
    def encode_obs(
        agent: "OneStageAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> np.ndarray:
        agent.encode_obs_count += 1
        return np.array([1])

    @staticmethod
    def decode_action(
        agent: "OneStageAgent", stage: ph.fsm.StageID, ctx: me.Network.Context, action
    ) -> ph.Packet:
        agent.decode_action_count += 1
        return ph.Packet(messages={agent.id: ["message"]})

    @staticmethod
    def get_observation_space(agent: "OneStageAgent") -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: "OneStageAgent") -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class OneStageAgent(FSMAgent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id, stage_handlers={Stages.UNIT: OneStageAgentHandler()}
        )

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class OneStateFSMEnvWithHandler(FiniteStateMachineEnv):
    def __init__(self):
        agents = [OneStageAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("agent", "agent")

        super().__init__(
            network=network,
            n_steps=2,
            initial_stage=Stages.UNIT,
        )

    @FSMStage(stage_id=Stages.UNIT, next_stages=[Stages.UNIT])
    def handle(self):
        print(self.network.resolver._cq)
        assert self.network.resolver._cq == {"agent": {"agent": ["message"]}}

        self.network.resolve()

        return Stages.UNIT


def test_one_state_with_handler():
    env = OneStateFSMEnvWithHandler()

    assert env.reset() == {"agent__Stages.UNIT": np.array([1])}

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent__Stages.UNIT": np.array([0])})

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    assert step.observations == {"agent__Stages.UNIT": np.array([1])}
    assert step.rewards == {"agent__Stages.UNIT": 0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent__Stages.UNIT": {}}

    step = env.step({"agent__Stages.UNIT": np.array([0])})

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    assert step.observations == {"agent__Stages.UNIT": np.array([1])}
    assert step.rewards == {"agent__Stages.UNIT": 0}
    assert step.terminals == {"__all__": True, "agent": False}
    assert step.infos == {"agent__Stages.UNIT": {}}


def test_one_state_with_handler_with_ray():
    ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env=OneStateFSMEnvWithHandler,
        env_config={},
        discard_results=True,
    )


class OneStateFSMEnvWithoutHandler(FiniteStateMachineEnv):
    def __init__(self):
        agents = [OneStageAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("agent", "agent")

        super().__init__(
            network=network,
            n_steps=2,
            initial_stage=Stages.UNIT,
            stages=[
                FSMStage(stage_id=Stages.UNIT, next_stages=[Stages.UNIT], handler=None)
            ],
        )


def test_one_state_without_handler():
    env = OneStateFSMEnvWithoutHandler()

    assert env.reset() == {"agent__Stages.UNIT": np.array([1])}

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent__Stages.UNIT": np.array([0])})

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    assert step.observations == {"agent__Stages.UNIT": np.array([1])}
    assert step.rewards == {"agent__Stages.UNIT": 0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent__Stages.UNIT": {}}

    step = env.step({"agent__Stages.UNIT": np.array([0])})

    assert env.current_stage == Stages.UNIT
    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    assert step.observations == {"agent__Stages.UNIT": np.array([1])}
    assert step.rewards == {"agent__Stages.UNIT": 0}
    assert step.terminals == {"__all__": True, "agent": False}
    assert step.infos == {"agent__Stages.UNIT": {}}


def test_one_state_without_handler_with_ray():
    ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env=OneStateFSMEnvWithoutHandler,
        env_config={},
        discard_results=True,
    )
