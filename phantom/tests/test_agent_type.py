from dataclasses import dataclass, field

import numpy as np
import phantom as ph
import pytest
from gym.spaces import Box


def test_agent_type_1():
    @dataclass
    class Type(ph.AgentType):
        a: int = 1
        b: float = 2.0
        c: np.ndarray = np.array([3, 4, 5])

    t = Type()

    assert (t.to_array() == np.array([1, 2, 3, 4, 5])).all()
    assert t.to_basic_obs_space() == Box(-np.inf, np.inf, (5,), np.float32)
    assert t.to_basic_obs_space(low=0, high=1) == Box(0, 1, (5,), np.float32)


def test_agent_type_2():
    @dataclass
    class Type(ph.AgentType):
        s: str = "s"

    t = Type()

    with pytest.raises(ValueError):
        t.to_array()

    with pytest.raises(ValueError):
        t.to_basic_obs_space()


def test_agent_type_3():
    @dataclass
    class Type(ph.AgentType):
        d: dict = field(default_factory=dict)

    t = Type()

    with pytest.raises(ValueError):
        t.to_array()

    with pytest.raises(ValueError):
        t.to_basic_obs_space()
