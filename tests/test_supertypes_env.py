import pytest

import phantom as ph

from . import MockAgent, MockEnv, MockSampler


def test_agent_supertypes_in_env_1():
    # USING STANDARD AGENT_SUPERTYPES PARAMETER STYLE
    agents = [MockAgent("a1"), MockAgent("a2")]

    network = ph.Network(agents)

    s1 = MockSampler(1.0)
    s2 = MockSampler(2.0)

    agent_supertypes = {
        "a1": MockAgent.Supertype(type_value=s1),
        "a2": MockAgent.Supertype(type_value=s2),
    }

    env = ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)

    assert set(env._samplers) == set([s1, s2])

    assert env.agents["a1"].type == None
    assert env.agents["a1"].supertype == agent_supertypes["a1"]

    assert env.agents["a2"].type == None
    assert env.agents["a2"].supertype == agent_supertypes["a2"]

    assert env.agents["a1"].supertype.type_value == s1
    assert env.agents["a2"].supertype.type_value == s2

    env.reset()

    assert env.agents["a1"].type == MockAgent.Supertype(1.0)
    assert env.agents["a2"].type == MockAgent.Supertype(2.0)


def test_agent_supertypes_in_env_2():
    # USING DICTIONARY AGENT_SUPERTYPES PARAMETER STYLE
    agents = [MockAgent("a1"), MockAgent("a2")]

    network = ph.Network(agents)

    s1 = MockSampler(1.0)
    s2 = MockSampler(2.0)

    agent_supertypes = {
        "a1": {"type_value": s1},
        "a2": {"type_value": s2},
    }

    env = ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)

    assert set(env._samplers) == set([s1, s2])

    assert env.agents["a1"].type == None
    assert env.agents["a1"].supertype == MockAgent.Supertype(type_value=s1)

    assert env.agents["a2"].type == None
    assert env.agents["a2"].supertype == MockAgent.Supertype(type_value=s2)

    assert env.agents["a1"].supertype.type_value == s1
    assert env.agents["a2"].supertype.type_value == s2

    env.reset()

    assert env.agents["a1"].type == MockAgent.Supertype(1.0)
    assert env.agents["a2"].type == MockAgent.Supertype(2.0)


def test_agent_supertypes_in_env_bad():
    agents = [MockAgent("a1"), MockAgent("a2")]

    network = ph.Network(agents)

    agent_supertypes = {"a1": {"wrong": 1.0}, "a2": {}}

    with pytest.raises(Exception):
        ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)


def test_env_supertype_in_env_1():
    # USING STANDARD ENV_SUPERTYPES PARAMETER STYLE
    s1 = MockSampler(1.0)

    env_supertype = MockEnv.Supertype(type_value=s1)

    env = MockEnv(env_supertype=env_supertype)

    assert set(env._samplers) == set([s1])

    assert env.env_type == None
    assert env.env_supertype == MockEnv.Supertype(s1)

    env.reset()

    assert env.env_type == MockEnv.Supertype(1.0)


def test_env_supertype_in_env_2():
    # USING DICTIONARY ENV_SUPERTYPES PARAMETER STYLE
    s1 = MockSampler(1.0)

    env_supertype = MockEnv.Supertype(type_value=s1)

    env = MockEnv(env_supertype={"type_value": s1})

    assert set(env._samplers) == set([s1])

    assert env.env_type == None
    assert env.env_supertype == env_supertype

    env.reset()

    assert env.env_type == MockEnv.Supertype(1.0)


def test_env_supertype_in_env_bad():
    with pytest.raises(Exception):
        MockEnv(env_supertype={"xxx": 0.0})
