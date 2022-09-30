from typing import Any, Dict, Mapping

import gym
from ray import rllib

from ...agents import AgentID
from ...env import PhantomEnv


class RLlibEnvWrapper(rllib.MultiAgentEnv):
    """
    Wrapper around a :class:`PhantomEnv` that provides compatibility with the RLlib
    :class:`MultiAgentEnv` interface.
    """

    def __init__(self, env: PhantomEnv) -> None:
        self.env = env

        self.env.reset()

        self._agent_ids = set(
            agent.id
            for agent in self.env.network.agents.values()
            if agent.action_space is not None and agent.observation_space is not None
        )

        self.action_space = gym.spaces.Dict(
            {
                agent_id: env.agents[agent_id].action_space
                for agent_id in self._agent_ids
            }
        )

        self.observation_space = gym.spaces.Dict(
            {
                agent_id: env.agents[agent_id].observation_space
                for agent_id in self._agent_ids
            }
        )

        rllib.MultiAgentEnv.__init__(self)

    def step(self, action_dict: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        return self.env.step(action_dict)

    def reset(self) -> Dict[AgentID, Any]:
        return self.env.reset()

    def is_done(self) -> bool:
        return self.env.is_done()

    def __getattr__(self, name: str) -> Any:
        return getattr(self.env, name)

    def __getitem__(self, agent_id: AgentID) -> AgentID:
        return self.env.__getitem__(agent_id)

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"
