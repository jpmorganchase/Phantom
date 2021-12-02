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
from phantom.fsm import (
    FSMAgent,
    FSMStage,
    FiniteStateMachineEnv,
    StageID,
    StagePolicyHandler,
)


logging.basicConfig(level=logging.INFO)


class Stages(Enum):
    ODD = 1
    EVEN = 2


class OddEvenAgentHandler(StagePolicyHandler["OddEvenAgent"]):
    @staticmethod
    def compute_reward(
        agent: "OddEvenAgent", stage: StageID, ctx: me.Network.Context
    ) -> float:
        agent.compute_reward_count += 1
        return 0

    @staticmethod
    def encode_obs(
        agent: "OddEvenAgent", stage: StageID, ctx: me.Network.Context
    ) -> np.ndarray:
        agent.encode_obs_count += 1
        return np.array([1])

    @staticmethod
    def decode_action(
        agent: "OddEvenAgent", stage: StageID, ctx: me.Network.Context, action
    ) -> ph.Packet:
        agent.decode_action_count += 1
        return ph.Packet(messages={agent.id: ["message"]})

    @staticmethod
    def get_observation_space(agent: "OddEvenAgent") -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    @staticmethod
    def get_action_space(agent: "OddEvenAgent") -> gym.spaces.Space:
        return gym.spaces.Discrete(1)


class OddEvenAgent(FSMAgent):
    def __init__(self, id: str) -> None:
        super().__init__(
            agent_id=id,
            stage_handlers={
                Stages.ODD: OddEvenAgentHandler(),
                Stages.EVEN: OddEvenAgentHandler(),
            },
        )

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class OddEvenFSMEnv(FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [OddEvenAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("agent", "agent")

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.ODD,
        )

    @FSMStage(stage_id=Stages.ODD, next_stages=[Stages.EVEN])
    def handle_odd(self):
        assert self.network.resolver._cq == {"agent": {"agent": ["message"]}}

        self.network.resolve()

        return Stages.EVEN

    @FSMStage(stage_id=Stages.EVEN, next_stages=[Stages.ODD])
    def handle_even(self):
        assert self.network.resolver._cq == {"agent": {"agent": ["message"]}}

        self.network.resolve()

        return Stages.ODD


def test_odd_even_one_fsm_agent():
    env = OddEvenFSMEnv()

    # Start in ODD stage, ODD agent provides initial observations
    assert env.reset() == {"agent__Stages.ODD": np.array([1])}

    assert env.current_stage == Stages.ODD
    assert env.agents["agent"].compute_reward_count == 0
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent__Stages.ODD": np.array([1])})

    assert env.current_stage == Stages.EVEN
    assert step.observations == {"agent__Stages.EVEN": np.array([1])}
    assert step.rewards == {"agent__Stages.EVEN": None}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent__Stages.EVEN": {}}

    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    step = env.step({"agent__Stages.EVEN": np.array([1])})

    assert env.current_stage == Stages.ODD
    assert step.observations == {"agent__Stages.ODD": np.array([1])}
    assert step.rewards == {"agent__Stages.ODD": 0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent__Stages.ODD": {}}

    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    step = env.step({"agent__Stages.ODD": np.array([1])})

    assert env.current_stage == Stages.EVEN
    assert step.observations == {
        "agent__Stages.EVEN": np.array([1]),
        "agent__Stages.ODD": np.array([1]),
    }
    assert step.rewards == {"agent__Stages.EVEN": 0, "agent__Stages.ODD": 0}
    assert step.terminals == {"__all__": True, "agent": False}
    assert step.infos == {"agent__Stages.EVEN": {}, "agent__Stages.ODD": {}}

    assert env.agents["agent"].compute_reward_count == 3
    assert env.agents["agent"].encode_obs_count == 4
    assert env.agents["agent"].decode_action_count == 3
