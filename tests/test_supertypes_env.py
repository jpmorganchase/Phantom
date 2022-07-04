from dataclasses import dataclass

import phantom as ph


class StaticSampler(ph.utils.samplers.Sampler[float]):
    def __init__(self, value: float) -> None:
        self.value = value

    def sample(self) -> float:
        return self.value


@dataclass
class MockSupertype(ph.Supertype):
    type_value: float


def test_supertypes_env():
    agents = [ph.Agent("a1"), ph.Agent("a2")]

    network = ph.Network(agents)

    s1 = StaticSampler(1.0)
    s2 = StaticSampler(2.0)

    agent_supertypes = {
        "a1": MockSupertype(type_value=s1),
        "a2": MockSupertype(type_value=s2),
    }

    env_supertype = MockSupertype(type_value=s1)

    env = ph.PhantomEnv(1, network, env_supertype, agent_supertypes)

    assert set(env._samplers) == set([s1, s2])

    assert env.env_type == None
    assert env.env_supertype == env_supertype

    assert env.agents["a1"].type == None
    assert env.agents["a1"].supertype == agent_supertypes["a1"]

    assert env.agents["a2"].type == None
    assert env.agents["a2"].supertype == agent_supertypes["a2"]

    assert env.agents["a1"].supertype.type_value == env.env_supertype.type_value

    env.reset()

    assert env.env_type == MockSupertype(1.0)
    assert env.agents["a1"].type == MockSupertype(1.0)
    assert env.agents["a2"].type == MockSupertype(2.0)
