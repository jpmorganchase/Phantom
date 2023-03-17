from dataclasses import dataclass

import pytest

import phantom as ph

from . import MockEnv, MockSampler, MockStrategicAgent


def test_agent_supertypes_in_env_1():
    # USING STANDARD AGENT_SUPERTYPES PARAMETER STYLE
    agents = [MockStrategicAgent("a1"), MockStrategicAgent("a2")]

    network = ph.Network(agents)

    s1 = MockSampler(0)
    s2 = MockSampler(10)

    agent_supertypes = {
        "a1": MockStrategicAgent.Supertype(type_value=s1),
        "a2": MockStrategicAgent.Supertype(type_value=s2),
    }

    # sampler sampled 1st time
    env = ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)

    assert set(env._samplers) == set([s1, s2])

    assert env.agents["a1"].supertype == agent_supertypes["a1"]
    assert env.agents["a1"].type == MockStrategicAgent.Supertype(1)

    assert env.agents["a2"].supertype == agent_supertypes["a2"]
    assert env.agents["a2"].type == MockStrategicAgent.Supertype(11)

    assert env.agents["a1"].supertype.type_value == s1
    assert env.agents["a2"].supertype.type_value == s2

    # sampler sampled 2nd time
    env.reset()

    assert env.agents["a1"].type == MockStrategicAgent.Supertype(2)
    assert env.agents["a2"].type == MockStrategicAgent.Supertype(12)


def test_agent_supertypes_in_env_2():
    # USING DICTIONARY AGENT_SUPERTYPES PARAMETER STYLE
    agents = [MockStrategicAgent("a1"), MockStrategicAgent("a2")]

    network = ph.Network(agents)

    s1 = MockSampler(0)
    s2 = MockSampler(10)

    agent_supertypes = {
        "a1": {"type_value": s1},
        "a2": {"type_value": s2},
    }

    # sampler sampled 1st time
    env = ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)

    assert set(env._samplers) == set([s1, s2])

    assert env.agents["a1"].type == MockStrategicAgent.Supertype(1)
    assert env.agents["a1"].supertype == MockStrategicAgent.Supertype(type_value=s1)

    assert env.agents["a2"].type == MockStrategicAgent.Supertype(11)
    assert env.agents["a2"].supertype == MockStrategicAgent.Supertype(type_value=s2)

    assert env.agents["a1"].supertype.type_value == s1
    assert env.agents["a2"].supertype.type_value == s2

    # sampler sampled 2nd time
    env.reset()

    assert env.agents["a1"].type == MockStrategicAgent.Supertype(2)
    assert env.agents["a2"].type == MockStrategicAgent.Supertype(12)


def test_agent_supertypes_in_env_bad():
    agents = [MockStrategicAgent("a1"), MockStrategicAgent("a2")]

    network = ph.Network(agents)

    agent_supertypes = {"a1": {"wrong": 1.0}, "a2": {}}

    with pytest.raises(Exception):
        ph.PhantomEnv(1, network, agent_supertypes=agent_supertypes)


def test_env_supertype_in_env_1():
    # USING STANDARD ENV_SUPERTYPES PARAMETER STYLE
    s1 = MockSampler(0)

    env_supertype = MockEnv.Supertype(type_value=s1)

    # sampler sampled 1st time
    env = MockEnv(env_supertype=env_supertype)

    assert set(env._samplers) == set([s1])

    assert env.env_type == None
    assert env.env_supertype == MockEnv.Supertype(s1)

    # sampler sampled 2nd time
    env.reset()

    assert env.env_type == MockEnv.Supertype(2)


def test_env_supertype_in_env_2():
    # USING DICTIONARY ENV_SUPERTYPES PARAMETER STYLE
    s1 = MockSampler(0)

    env_supertype = MockEnv.Supertype(type_value=s1)

    # sampler sampled 1st time
    env = MockEnv(env_supertype={"type_value": s1})

    assert set(env._samplers) == set([s1])

    assert env.env_type == None
    assert env.env_supertype == env_supertype

    # sampler sampled 2nd time
    env.reset()

    assert env.env_type == MockEnv.Supertype(2)


def test_env_supertype_in_env_bad():
    with pytest.raises(Exception):
        MockEnv(env_supertype={"xxx": 0.0})


def test_env_type_passed_to_agent():
    class MockAgent(ph.Agent):
        def __init__(self, *args, num_steps=None, **kwargs):
            super().__init__(*args, **kwargs)

            self.num_steps = num_steps

            self.param = 0.0

        def generate_messages(self, ctx):
            self.param = ctx.env_view.supertype_param

    class MockEnv(ph.PhantomEnv):
        @dataclass
        class Supertype(ph.Supertype):
            param: float = 0.0

        @dataclass(frozen=True)
        class View(ph.EnvView):
            supertype_param: float

        def view(self, agent_views):
            return self.View(
                self.current_step,
                self.current_step / self.num_steps,
                self.env_type.param,
            )

        def __init__(self, **kwargs):
            network = ph.StochasticNetwork([MockAgent("a1")])

            super().__init__(num_steps=10, network=network, **kwargs)

    env = MockEnv(env_supertype=MockEnv.Supertype(MockSampler(0.0)))

    # sampler value = 1.0 (env.__init__())

    env.reset()

    # sampler value = 2.0 (env.reset())

    env.step({})

    assert env["a1"].param == 2.0

    env.reset()

    # sampler value = 3.0 (env.reset())

    env.step({})

    assert env["a1"].param == 3.0
