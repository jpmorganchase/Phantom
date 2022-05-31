from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
)

from .env import PhantomEnv
from .types import AgentID


class FSMPhantomEnv(PhantomEnv):
    """
    Base Phantom environment.

    Usage:

        >>> env = PhantomEnv({ ... })
        >>> env.reset()
        <Observation: dict>
        >>> env.step({ ... })
        <Step: 4-tuple>

    Attributes:
        network: A Mercury Network class or derived class describing the connections
            between agents and agents in the environment.
        num_steps: The maximum number of steps the environment allows per episode.
    """

    # def __init__(
    #     self,
    #     num_steps: int,
    #     network: Optional[Network] = None,
    #     env_supertype: Optional[ST] = None,
    #     agent_supertypes: Optional[Mapping[AgentID, Supertype]] = None,
    # ) -> None:
    #     super().__init__(num_steps, network, env_supertype, agent_supertypes)

    def reset(
        self, initial_agents: Optional[List[AgentID]] = None
    ) -> Dict[AgentID, Any]:
        self._apply_samplers()

        self._agent_ids = set(self.agent_ids)

        # Set clock back to time step 0
        self.current_step = 0

        # Reset network and call reset method on all agents in the network.
        # Message samplers should be called here from the respective agent's reset method.
        self.network.reset()
        self.network.resolve()

        # Reset the agents' done status
        self._dones = set()

        # Generate initial observations.
        observations: Dict[AgentID, Any] = {}

        for agent_id, agent in self.network.agents.items():
            if agent.takes_actions() and (
                initial_agents is None or agent_id in initial_agents
            ):
                ctx = self.network.context_for(agent_id)
                observations[agent_id] = agent.encode_obs(ctx)

        return observations

    def step(
        self,
        actions: Mapping[AgentID, Any],
        acting_agents: Optional[List[AgentID]] = None,
        rewarded_agents: Optional[List[AgentID]] = None,
    ) -> "PhantomEnv.Step":
        self.current_step += 1

        # Handle the updates due to active/strategic behaviours:
        for agent_id, action in actions.items():
            ctx = self.network.context_for(agent_id)
            messages = self.network.agents[agent_id].decode_action(ctx, action)

            for receiver_id, message in messages:
                self.network.send(agent_id, receiver_id, message)

        # Resolve the messages on the network and perform mutations:
        self.pre_message_resolution()
        self.network.resolve(self.view)
        self.post_message_resolution()

        # Compute the output for rllib:
        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, Any] = {}
        dones: Dict[AgentID, bool] = {}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        for agent_id, agent in self.network.agents.items():
            if agent.takes_actions():
                ctx = self.network.context_for(agent_id)

                if agent_id not in self._dones:
                    if acting_agents is None or agent_id in acting_agents:
                        observations[agent_id] = agent.encode_obs(ctx)
                        infos[agent_id] = agent.collect_infos(ctx)

                    if rewarded_agents is None or agent_id in rewarded_agents:
                        rewards[agent_id] = agent.compute_reward(ctx)

                dones[agent_id] = agent.is_done(ctx)

                if dones[agent_id]:
                    self._dones.add(agent_id)

        dones["__all__"] = self.is_done()

        return self.Step(
            observations=observations, rewards=rewards, dones=dones, infos=infos
        )
