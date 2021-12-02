"""
Test a very simple FSM Env with two stages and two agents.

The agents take turns to take actions and return rewards.

The episode duration is two steps.
"""

import logging
from enum import Enum

import gym
import mercury as me
import numpy as np
import phantom as ph
from phantom.fsm import FSMAgent, FSMStage, FiniteStateMachineEnv, StagePolicyHandler


logging.basicConfig(level=logging.INFO)


class Stages(Enum):
    ODD = 1
    EVEN = 2


class OddEvenAgentHandler(StagePolicyHandler["OddEvenAgent"]):
    @staticmethod
    def compute_reward(
        agent: "OddEvenAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> float:
        agent.compute_reward_count += 1
        return 0

    @staticmethod
    def encode_obs(
        agent: "OddEvenAgent", stage: ph.fsm.StageID, ctx: me.Network.Context
    ) -> np.ndarray:
        agent.encode_obs_count += 1
        return np.array([1]) if agent.id == "odd_agent" else np.array([0])

    @staticmethod
    def decode_action(
        agent: "OddEvenAgent", stage: ph.fsm.StageID, ctx: me.Network.Context, action
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
    def __init__(self, id: str, stage: Stages) -> None:
        super().__init__(agent_id=id, stage_handlers={stage: OddEvenAgentHandler()})

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class OddEvenFSMEnv(FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [
            OddEvenAgent("odd_agent", Stages.ODD),
            OddEvenAgent("even_agent", Stages.EVEN),
        ]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("odd_agent", "odd_agent")
        network.add_connection("even_agent", "even_agent")

        super().__init__(
            network=network,
            n_steps=3,
            initial_stage=Stages.ODD,
        )

    @FSMStage(stage_id=Stages.ODD, next_stages=[Stages.EVEN])
    def handle_odd(self):
        assert self.network.resolver._cq == {"odd_agent": {"odd_agent": ["message"]}}

        self.network.resolve()

        return Stages.EVEN

    @FSMStage(stage_id=Stages.EVEN, next_stages=[Stages.ODD])
    def handle_even(self):
        assert self.network.resolver._cq == {"even_agent": {"even_agent": ["message"]}}

        self.network.resolve()

        return Stages.ODD


def test_odd_even():
    env = OddEvenFSMEnv()

    # Start in ODD stage, ODD agent provides initial observations
    assert env.reset() == {"odd_agent__Stages.ODD": np.array([1])}

    assert env.current_stage == Stages.ODD
    # ODD agent has not taken an action, so does not compute reward
    assert env.agents["odd_agent"].compute_reward_count == 0
    assert env.agents["even_agent"].compute_reward_count == 0
    # ODD agent encodes it's observations during the reset() call
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].encode_obs_count == 0
    assert env.agents["odd_agent"].decode_action_count == 0
    assert env.agents["even_agent"].decode_action_count == 0

    # Policy returns action for odd agent as observations were provided
    step = env.step({"odd_agent__Stages.ODD": np.array([1])})

    # In this step:
    # - decode actions for odd agent
    # - handle messages from odd agent
    # - compute reward for odd agent's actions (do not return as odd agent not in obs)
    # - encode observations from even agent

    assert env.current_stage == Stages.EVEN
    assert step.observations == {"even_agent__Stages.EVEN": np.array([0])}
    assert step.rewards == {"even_agent__Stages.EVEN": None}
    assert step.terminals == {"__all__": False, "even_agent": False, "odd_agent": False}
    assert step.infos == {"even_agent__Stages.EVEN": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].compute_reward_count == 0
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"even_agent__Stages.EVEN": np.array([0])})

    assert env.current_stage == Stages.ODD
    assert step.observations == {"odd_agent__Stages.ODD": np.array([1])}
    assert step.rewards == {"odd_agent__Stages.ODD": 0}
    assert step.terminals == {"__all__": False, "even_agent": False, "odd_agent": False}
    assert step.infos == {"odd_agent__Stages.ODD": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1
    assert env.agents["even_agent"].decode_action_count == 1

    step = env.step({"odd_agent__Stages.ODD": np.array([0])})

    assert env.current_stage == Stages.EVEN
    assert step.observations == {
        "even_agent__Stages.EVEN": np.array([0]),
        "odd_agent__Stages.ODD": np.array([1]),
    }
    assert step.rewards == {"even_agent__Stages.EVEN": 0, "odd_agent__Stages.ODD": 0}
    assert step.terminals == {"__all__": True, "even_agent": False, "odd_agent": False}
    assert step.infos == {"even_agent__Stages.EVEN": {}, "odd_agent__Stages.ODD": {}}

    assert env.agents["odd_agent"].compute_reward_count == 2
    assert env.agents["even_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["even_agent"].encode_obs_count == 2
    assert env.agents["odd_agent"].decode_action_count == 2
    assert env.agents["even_agent"].decode_action_count == 1


def test_odd_even_two_agents_with_ray():
    ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env_class=OddEvenFSMEnv,
        env_config={},
        discard_results=True,
    )
