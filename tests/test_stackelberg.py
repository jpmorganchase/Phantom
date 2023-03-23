"""
Test the StackelbergEnv class with two agents.
"""

import numpy as np
import phantom as ph

from . import MockStrategicAgent


def test_stackelberg_env():
    agents = [MockStrategicAgent("leader"), MockStrategicAgent("follower")]

    network = ph.Network(agents)

    env = ph.StackelbergEnv(3, network, ["leader"], ["follower"])

    assert env.reset() == {"leader": np.array([0])}

    assert env.agents["leader"].compute_reward_count == 0
    assert env.agents["leader"].encode_obs_count == 1
    assert env.agents["leader"].decode_action_count == 0

    assert env.agents["follower"].compute_reward_count == 0
    assert env.agents["follower"].encode_obs_count == 0
    assert env.agents["follower"].decode_action_count == 0

    step = env.step({"leader": np.array([0])})

    assert step.observations == {"follower": np.array([1 / 3])}
    assert step.rewards == {}
    assert step.terminations == {"leader": False, "follower": False, "__all__": False}
    assert step.truncations == {"leader": False, "follower": False, "__all__": False}
    assert step.infos == {"follower": {}}

    assert env.agents["leader"].compute_reward_count == 1
    assert env.agents["leader"].encode_obs_count == 1
    assert env.agents["leader"].decode_action_count == 1

    assert env.agents["follower"].compute_reward_count == 0
    assert env.agents["follower"].encode_obs_count == 1
    assert env.agents["follower"].decode_action_count == 0

    step = env.step({"follower": np.array([0])})

    assert step.observations == {"leader": np.array([2 / 3])}
    assert step.rewards == {"leader": 0.0}
    assert step.terminations == {"leader": False, "follower": False, "__all__": False}
    assert step.truncations == {"leader": False, "follower": False, "__all__": False}
    assert step.infos == {"leader": {}}

    assert env.agents["leader"].compute_reward_count == 1
    assert env.agents["leader"].encode_obs_count == 2
    assert env.agents["leader"].decode_action_count == 1

    assert env.agents["follower"].compute_reward_count == 1
    assert env.agents["follower"].encode_obs_count == 1
    assert env.agents["follower"].decode_action_count == 1

    step = env.step({"leader": np.array([0])})

    assert step.observations == {"follower": np.array([1])}
    assert step.rewards == {"leader": 0.0, "follower": 0.0}
    assert step.terminations == {"leader": False, "follower": False, "__all__": False}
    assert step.truncations == {"leader": False, "follower": False, "__all__": True}
    assert step.infos == {"follower": {}}

    assert env.agents["leader"].compute_reward_count == 2
    assert env.agents["leader"].encode_obs_count == 2
    assert env.agents["leader"].decode_action_count == 2

    assert env.agents["follower"].compute_reward_count == 1
    assert env.agents["follower"].encode_obs_count == 2
    assert env.agents["follower"].decode_action_count == 1
