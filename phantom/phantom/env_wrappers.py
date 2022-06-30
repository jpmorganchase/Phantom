from typing import Any, Dict, List, Mapping, Optional

import mercury as me
from ray.rllib import MultiAgentEnv

from .env import PhantomEnv
from .supertype import BaseSupertype
from .utils import collect_instances_of_type
from .utils.samplers import BaseSampler


class BaseEnvWrapper(MultiAgentEnv):
    def __init__(self, env: PhantomEnv) -> None:
        self.env = env

    def __getattr__(self, name: str) -> Any:
        if name.startswith("_"):
            raise AttributeError(f"attempted to get missing private attribute '{name}'")
        return getattr(self.env, name)

    def __getitem__(self, actor_id: me.ID) -> me.actors.Actor:
        return self.env.__getitem__(actor_id)

    def step(self, actions: Mapping[me.ID, Any]) -> PhantomEnv.Step:
        return self.env.step(actions)

    def reset(self) -> Dict[me.ID, Any]:
        return self.env.reset()

    def is_done(self) -> bool:
        return self.env.is_done()

    def __str__(self):
        return f"<{type(self).__name__}{self.env}>"


class SharedSupertypeEnvWrapper(BaseEnvWrapper):
    def __init__(
        self,
        env: PhantomEnv,
        env_supertype: Optional[BaseSupertype] = None,
        agent_supertypes: Optional[Dict[me.ID, BaseSupertype]] = None,
    ) -> None:
        super().__init__(env)

        agent_supertypes = agent_supertypes or {}

        self._samplers = self._collect_samplers(
            env_supertype, agent_supertypes, self.env.network
        )
        self._env_supertype = env_supertype
        self._agent_supertypes = agent_supertypes

        for agent_id, supertype in self._agent_supertypes.items():
            self.agents[agent_id].supertype = supertype

    def reset(self) -> Dict[me.ID, Any]:
        # Sample from samplers
        for sampler in self._samplers:
            sampler.value = sampler.sample()

        # Update and apply env type
        if self._env_supertype is not None:
            self.env.env_type = self._env_supertype.sample()

            if "__ENV" in self.env.network.actor_ids:
                self.env.network.actors["__ENV"].env_type = self.env.env_type

        return self.env.reset()

    def _collect_samplers(
        self,
        env_supertype: BaseSupertype,
        agent_supertypes: Dict[me.ID, BaseSupertype],
        network: me.Network,
    ):
        # Collect all instances of classes that inherit from BaseSampler from the env
        # supertype and the agent supertypes into a flat list. We make sure that the list
        # contains only one reference to each sampler instance.
        samplers = collect_instances_of_type(BaseSampler, env_supertype)

        for agent_supertype in agent_supertypes.values():
            samplers += collect_instances_of_type(BaseSampler, agent_supertype)

        # Network samplers should be on env?
        if isinstance(network, me.StochasticNetwork):
            samplers += collect_instances_of_type(
                BaseSampler, network._base_connections
            )

        # The environment needs access to the list of samplers so it can generate new
        # values in each step.
        return samplers
