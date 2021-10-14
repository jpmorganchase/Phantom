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
from phantom.fsm_env import fsm_state, FiniteStateMachineEnv


class MinimalAgent(ph.agent.Agent):
    def __init__(self, id: str) -> None:
        super().__init__(agent_id=id)

        self.compute_reward_count = 0
        self.encode_obs_count = 0
        self.decode_action_count = 0

    def compute_reward(self, ctx: me.Network.Context) -> float:
        self.compute_reward_count += 1
        return 0

    def encode_obs(self, ctx: me.Network.Context) -> np.ndarray:
        self.encode_obs_count += 1
        return np.array([1])

    def decode_action(self, ctx: me.Network.Context, action) -> ph.Packet:
        self.decode_action_count += 1
        return ph.Packet(messages={self.id: ["message"]})

    def get_observation_space(self) -> gym.spaces.Space:
        return gym.spaces.Box(shape=(1,), low=-np.inf, high=np.inf)

    def get_action_space(self) -> gym.spaces.Space:
        return gym.spaces.Discrete(1)

    @me.actors.handler(str)
    def handle_message(self, ctx, msg):
        yield from ()


class States(Enum):
    UNIT = 1


class OneStateFSMEnv(FiniteStateMachineEnv):
    def __init__(self):
        agents = [MinimalAgent("agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("agent", "agent")

        super().__init__(
            network=network,
            n_steps=2,
            initial_state=States.UNIT,
        )

    @fsm_state(state_id=States.UNIT, next_states=[States.UNIT])
    def handle(self):
        assert self.network.resolver._cq == {"agent": {"agent": ["message"]}}

        self.network.resolve()

        return States.UNIT


def test_one_state():
    env = OneStateFSMEnv()

    assert env.reset() == {"agent": 1}

    assert env.current_state == States.UNIT
    assert env.agents["agent"].compute_reward_count == 1
    assert env.agents["agent"].encode_obs_count == 1
    assert env.agents["agent"].decode_action_count == 0

    step = env.step({"agent": 0})

    assert env.current_state == States.UNIT
    assert env.agents["agent"].compute_reward_count == 2
    assert env.agents["agent"].encode_obs_count == 2
    assert env.agents["agent"].decode_action_count == 1

    assert step.observations == {"agent": 1}
    assert step.rewards == {"agent": 0}
    assert step.terminals == {"__all__": False, "agent": False}
    assert step.infos == {"agent": {}}

    step = env.step({"agent": 0})

    assert env.current_state == States.UNIT
    assert env.agents["agent"].compute_reward_count == 3
    assert env.agents["agent"].encode_obs_count == 3
    assert env.agents["agent"].decode_action_count == 2

    assert step.observations == {"agent": 1}
    assert step.rewards == {"agent": 0}
    assert step.terminals == {"__all__": True, "agent": False}
    assert step.infos == {"agent": {}}


def test_one_state_with_ray():
    ph.train(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env=OneStateFSMEnv,
        env_config={},
        discard_results=True,
    )
