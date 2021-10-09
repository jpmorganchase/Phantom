"""
Test a very simple FSM Env with two states and two agents.

The agents take turns to take actions and return rewards.

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
        return np.array([1]) if self.id == "odd_agent" else np.array([0])

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
    ODD = 1
    EVEN = 2


class OddEvenFSMEnv(FiniteStateMachineEnv):

    env_name: str = "unit-testing"

    def __init__(self):
        agents = [MinimalAgent("odd_agent"), MinimalAgent("even_agent")]

        network = me.Network(me.resolvers.UnorderedResolver(), agents)

        network.add_connection("odd_agent", "odd_agent")
        network.add_connection("even_agent", "even_agent")

        super().__init__(
            network=network,
            n_steps=3,
            initial_state=States.ODD,
        )

    @fsm_state(
        state_id=States.ODD,
        next_states=[States.EVEN],
        enforced_actions=["odd_agent"],
        enforced_rewards=["odd_agent"],
    )
    def handle_odd(self):
        assert self.network.resolver._cq == {"odd_agent": {"odd_agent": ["message"]}}

        self.network.resolve()

        return States.EVEN

    @fsm_state(
        state_id=States.EVEN,
        next_states=[States.ODD],
        enforced_actions=["even_agent"],
        enforced_rewards=["even_agent"],
    )
    def handle_even(self):
        assert self.network.resolver._cq == {"even_agent": {"even_agent": ["message"]}}

        self.network.resolve()

        return States.ODD


def test_odd_even():
    env = OddEvenFSMEnv()

    assert env.reset() == {"odd_agent": 1}

    assert env.current_state == States.ODD
    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].compute_reward_count == 1
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].encode_obs_count == 0
    assert env.agents["odd_agent"].decode_action_count == 0
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"odd_agent": 0})

    assert env.current_state == States.EVEN
    assert step.observations == {"even_agent": 0}
    assert step.rewards == {"even_agent": 0}
    assert step.terminals == {"__all__": False, "even_agent": False, "odd_agent": False}
    assert step.infos == {"even_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 1
    assert env.agents["even_agent"].compute_reward_count == 2
    assert env.agents["odd_agent"].encode_obs_count == 1
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1
    assert env.agents["even_agent"].decode_action_count == 0

    step = env.step({"even_agent": 0})

    assert env.current_state == States.ODD
    assert step.observations == {"odd_agent": 1}
    assert step.rewards == {"odd_agent": 0}
    assert step.terminals == {"__all__": False, "even_agent": False, "odd_agent": False}
    assert step.infos == {"odd_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 2
    assert env.agents["even_agent"].compute_reward_count == 2
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["even_agent"].encode_obs_count == 1
    assert env.agents["odd_agent"].decode_action_count == 1
    assert env.agents["even_agent"].decode_action_count == 1

    step = env.step({"odd_agent": 0})

    assert env.current_state == States.EVEN
    assert step.observations == {"even_agent": 0, "odd_agent": 1}
    assert step.rewards == {"even_agent": 0, "odd_agent": 0}
    assert step.terminals == {"__all__": True, "even_agent": False, "odd_agent": False}
    assert step.infos == {"even_agent": {}, "odd_agent": {}}

    assert env.agents["odd_agent"].compute_reward_count == 2
    assert env.agents["even_agent"].compute_reward_count == 3
    assert env.agents["odd_agent"].encode_obs_count == 2
    assert env.agents["even_agent"].encode_obs_count == 2
    assert env.agents["odd_agent"].decode_action_count == 2
    assert env.agents["even_agent"].decode_action_count == 1


def test_odd_event_with_ray():
    training_params = ph.TrainingParams(
        experiment_name="unit-testing",
        algorithm="PPO",
        num_workers=0,
        num_episodes=1,
        env=OddEvenFSMEnv,
        env_config={},
        discard_results=True,
    )

    ph.utils.training.train_from_params_object(training_params)
