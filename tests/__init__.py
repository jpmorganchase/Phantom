from dataclasses import dataclass
from typing import Optional

import gymnasium as gym
import numpy as np
import phantom as ph


class MockSampler(ph.utils.samplers.Sampler[float]):
    def __init__(self, value: float) -> None:
        self._value = value

    def sample(self) -> float:
        self._value += 1
        return self._value


class MockComparableSampler(ph.utils.samplers.ComparableSampler[float]):
    def __init__(self, value: float) -> None:
        self._value = value

    def sample(self) -> float:
        self._value += 1
        return self._value


class MockAgent(ph.Agent):
    def __init__(self, *args, num_steps: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.num_steps = num_steps


class MockStrategicAgent(ph.StrategicAgent):
    @dataclass
    class Supertype(ph.Supertype):
        type_value: float = 0.0

    def __init__(self, *args, num_steps: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(0, 1, (1,))
        self.observation_space = gym.spaces.Box(0, 1, (1,))

        self.encode_obs_count = 0
        self.decode_action_count = 0
        self.compute_reward_count = 0

        self.num_steps = num_steps

    def encode_observation(self, ctx: ph.Context):
        self.encode_obs_count += 1
        return np.array([ctx.env_view.proportion_time_elapsed])

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        self.decode_action_count += 1
        return []

    def compute_reward(self, ctx: ph.Context) -> float:
        self.compute_reward_count += 1
        return 0.0

    def is_terminated(self, ctx: ph.Context) -> bool:
        return ctx.env_view.current_step == self.num_steps

    def is_truncated(self, ctx: ph.Context) -> bool:
        return ctx.env_view.current_step == self.num_steps


class MockStackedStrategicAgent(ph.StrategicAgent):
    @dataclass
    class Supertype(ph.Supertype):
        type_value: float = 0.0

    def __init__(self, *args, stack_size: int, **kwargs):
        super().__init__(*args, **kwargs)

        self.action_space = gym.spaces.Box(0, 1, (1,))
        self.observation_space = gym.spaces.Box(0, 1, (1,))

        self.stack_size = stack_size

    def encode_observation(self, ctx: ph.Context):
        return [np.array([0.0], dtype="float32") for _ in range(self.stack_size)]

    def decode_action(self, ctx: ph.Context, action: np.ndarray):
        if self.stack_size == 1:
            assert not isinstance(action, list)
        else:
            assert isinstance(action, list)
            assert len(action) == self.stack_size

        return []

    def compute_reward(self, ctx: ph.Context):
        return [0.0 for _ in range(self.stack_size)]

    def is_terminated(self, ctx: ph.Context) -> bool:
        return ctx.env_view.current_step == 2

    def is_truncated(self, ctx: ph.Context) -> bool:
        return ctx.env_view.current_step == 2


class MockPolicy(ph.Policy):
    def compute_action(self, obs) -> int:
        return 1


class MockEnv(ph.PhantomEnv):
    @dataclass
    class Supertype(ph.Supertype):
        type_value: float

    def __init__(self, env_supertype=None):
        agents = [MockStrategicAgent("a1"), MockStrategicAgent("a2"), MockAgent("a3")]

        network = ph.network.Network(agents)

        network.add_connection("a1", "a2")
        network.add_connection("a2", "a3")
        network.add_connection("a3", "a1")

        super().__init__(num_steps=5, network=network, env_supertype=env_supertype)


class MockStackedAgentEnv(ph.PhantomEnv):
    @dataclass
    class Supertype(ph.Supertype):
        type_value: float

    def __init__(self, env_supertype=None):
        agents = [
            MockStackedStrategicAgent("a1", stack_size=1),
            MockStackedStrategicAgent("a2", stack_size=2),
            MockStackedStrategicAgent("a3", stack_size=4),
        ]

        network = ph.network.Network(agents)

        super().__init__(num_steps=5, network=network, env_supertype=env_supertype)
