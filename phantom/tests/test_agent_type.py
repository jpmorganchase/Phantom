from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import phantom as ph
import pytest
from gym.spaces import Box


def test_agent_type_simple():
    # 3 basic accepted types are int, float and np.ndarray
    @dataclass
    class Type(ph.AgentType):
        a: int = 1
        b: float = 2.0
        c: np.ndarray = np.array([3, 4, 5])

    t = Type()

    assert (t.to_array() == np.array([1, 2, 3, 4, 5])).all()
    assert t.to_basic_obs_space() == Box(-np.inf, np.inf, (5,), np.float32)
    assert t.to_basic_obs_space(low=0, high=1) == Box(0, 1, (5,), np.float32)


def test_agent_type_complex():
    # 3 basic accepted types are int, float and np.ndarray
    @dataclass
    class Type(ph.AgentType):
        a: List[int]
        b: Tuple[int, int]

    t = Type([1, 2, 3], (4, 5))

    assert (t.to_array() == np.array([1, 2, 3, 4, 5])).all()
    assert t.to_basic_obs_space() == Box(-np.inf, np.inf, (5,), np.float32)
    assert t.to_basic_obs_space(low=0, high=1) == Box(0, 1, (5,), np.float32)

    # Test nested types
    @dataclass
    class Type(ph.AgentType):
        a: List[List[int]]

    t = Type([[1, 2, 3], [4, 5]])

    assert (t.to_array() == np.array([1, 2, 3, 4, 5])).all()
    assert t.to_basic_obs_space() == Box(-np.inf, np.inf, (5,), np.float32)
    assert t.to_basic_obs_space(low=0, high=1) == Box(0, 1, (5,), np.float32)


def test_agent_type_errors():
    # String is not currently a supported type
    @dataclass
    class Type(ph.AgentType):
        s: str = "s"

    t = Type()

    with pytest.raises(ValueError):
        t.to_array()

    with pytest.raises(ValueError):
        t.to_basic_obs_space()

    # Dict is currently not a supported type
    @dataclass
    class Type(ph.AgentType):
        d: dict = field(default_factory=dict)

    t = Type()

    with pytest.raises(ValueError):
        t.to_array()

    with pytest.raises(ValueError):
        t.to_basic_obs_space()
