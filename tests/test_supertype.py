from dataclasses import dataclass
from typing import Dict, List, Tuple

import gym
import numpy as np
import pytest

from phantom import Supertype

from . import MockSampler

def test_base_supertype_sample():
    @dataclass
    class TestSupertype(Supertype):
        a: float
        b: float

    s1 = TestSupertype(1.0, "string")
    t1 = s1.sample()

    assert isinstance(t1, TestSupertype)
    assert t1.__dict__ == {
        "a": 1.0,
        "b": "string",
    }

    s2 = TestSupertype(MockSampler(1.0), "string")
    t2 = s2.sample()

    assert t2.__dict__ == {
        "a": 1.0,
        "b": "string",
    }


def test_base_type_utilities():
    @dataclass
    class Type(Supertype):
        a: int
        b: float
        c: List[int]
        d: Tuple[int]
        e: np.ndarray
        f: Dict[str, int]

    t = Type(
        a=1,
        b=2.0,
        c=[6, 7, 8],
        d=(9, 10, 11),
        e=np.array([15, 16, 17], dtype=np.float32),
        f={"x": 12, "y": 13, "z": 14},
    )

    t_compat = t.to_obs_space_compatible_type()

    assert len(t_compat) == 6
    assert t_compat["a"] == t.a
    assert t_compat["b"] == t.b
    assert t_compat["c"] == t.c
    assert t_compat["d"] == t.d
    assert np.all(t_compat["e"] == t.e)
    assert t_compat["f"] == t.f

    t_space = t.to_obs_space()

    assert t_space == gym.spaces.Dict(
        {
            "a": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "b": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
            "c": gym.spaces.Tuple(
                [
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                ]
            ),
            "d": gym.spaces.Tuple(
                [
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                ]
            ),
            "e": gym.spaces.Box(-np.inf, np.inf, (3,), np.float32),
            "f": gym.spaces.Dict(
                {
                    "x": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    "y": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                    "z": gym.spaces.Box(-np.inf, np.inf, (1,), np.float32),
                }
            ),
        }
    )

    assert t_space.contains(t_compat)

    # String is not currently a supported type
    @dataclass
    class Type(Supertype):
        s: str = "s"

    t = Type()

    with pytest.raises(ValueError):
        t.to_obs_space_compatible_type()

    with pytest.raises(ValueError):
        t.to_obs_space()
