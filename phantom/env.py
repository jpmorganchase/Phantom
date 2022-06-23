from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Set,
)

from .agents import Agent
from .network import Network
from .supertype import Supertype
from .types import AgentID
from .utils.samplers import Sampler


@dataclass(frozen=True)
class EnvView:
    """Immutable references to public :class:`phantom.PhantomEnv` state."""


class PhantomEnv:
    """
    Base Phantom environment.

    Usage:

        >>> env = PhantomEnv({ ... })
        >>> env.reset()
        <Observation: dict>
        >>> env.step({ ... })
        <Step: 4-tuple>

    Attributes:
        num_steps: The maximum number of steps the environment allows per episode.
        network: A Network class or derived class describing the connections between
            agents and agents in the environment.
    """

    class Step(NamedTuple):
        """Type alias for the PhantomEnv::step return object."""

        observations: Dict[AgentID, Any]
        rewards: Dict[AgentID, float]
        dones: Dict[AgentID, bool]
        infos: Dict[AgentID, Any]

    def __init__(
        self,
        num_steps: int,
        network: Optional[Network] = None,
        env_supertype: Optional[Supertype] = None,
        agent_supertypes: Optional[Mapping[AgentID, Supertype]] = None,
    ) -> None:
        self.network = network or Network()
        self.current_step = 0
        self.num_steps = num_steps

        self.env_supertype: Optional[Supertype] = None
        self.env_type: Optional[Supertype] = None

        self._dones: Set[AgentID] = set()
        self._agent_ids = set(self.agent_ids)

        # Collect all instances of classes that inherit from BaseSampler from the env
        # supertype and the agent supertypes into a flat list. We make sure that the list
        # contains only one reference to each sampler instance.
        self._samplers: List[Sampler] = []

        if env_supertype is not None:
            # Extract samplers from env supertype dict
            for value in env_supertype.__dict__.values():
                if isinstance(value, Sampler) and value not in self._samplers:
                    self._samplers.append(value)

            self.env_supertype = env_supertype

        if agent_supertypes is not None:
            for agent_id, agent_supertype in agent_supertypes.items():
                # Extract samplers from agent supertype dict
                for value in agent_supertype.__dict__.values():
                    if isinstance(value, Sampler) and value not in self._samplers:
                        self._samplers.append(value)

                agent = self.network.agents[agent_id]
                agent.supertype = agent_supertype

        # Generate initial sampled values in samplers
        for sampler in self._samplers:
            sampler.value = sampler.sample()

    @property
    def agents(self) -> Dict[AgentID, Agent]:
        """Return a mapping of agent IDs to agents in the environment."""
        return self.network.agents

    @property
    def agent_ids(self) -> List[AgentID]:
        """Return a list of the IDs of the agents in the environment."""
        return list(self.network.agent_ids)

    @property
    def n_agents(self) -> int:
        """Return the number of agents in the environment."""
        return len(self.agent_ids)

    def view(self, agent_id: Optional[AgentID] = None) -> EnvView:
        """Return an immutable view to the environment's public state."""
        return EnvView()

    def pre_message_resolution(self) -> None:
        """Perform internal, pre-message resolution updates to the environment."""
        ctxs = {
            agent_id: self.network.context_for(agent_id) for agent_id in self.agent_ids
        }

        for agent_id, ctx in ctxs.items():
            self.agents[agent_id].pre_message_resolution(ctx)

    def post_message_resolution(self) -> None:
        """Perform internal, post-message resolution updates to the environment."""
        ctxs = {
            agent_id: self.network.context_for(agent_id) for agent_id in self.agent_ids
        }

        for agent_id, ctx in ctxs.items():
            self.agents[agent_id].post_message_resolution(ctx)

    def reset(self) -> Dict[AgentID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the :attr:`clock` and the :attr:`network`. This
        includes all the agents in the network.

        Returns:
            A dictionary mapping Agent IDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """
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
            if agent.action_space is not None:
                ctx = self.network.context_for(agent_id)
                observations[agent_id] = agent.encode_obs(ctx)

        return observations

    def step(self, actions: Mapping[AgentID, Any]) -> "PhantomEnv.Step":
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
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
            if agent.action_space is not None:
                ctx = self.network.context_for(agent_id)

                if agent.is_done(ctx):
                    # if agent is just done we send the last obs, rew, info
                    if agent_id not in self._dones:
                        self._dones.add(agent_id)
                        observations[agent_id] = agent.encode_obs(ctx)
                        rewards[agent_id] = agent.compute_reward(ctx)
                        infos[agent_id] = agent.collect_infos(ctx)
                        dones[agent_id] = True
                    # otherwise just ignore
                else:
                    observations[agent_id] = agent.encode_obs(ctx)
                    rewards[agent_id] = agent.compute_reward(ctx)
                    infos[agent_id] = agent.collect_infos(ctx)
                    dones[agent_id] = False

        dones["__all__"] = self.is_done()

        return self.Step(
            observations=observations, rewards=rewards, dones=dones, infos=infos
        )

    def is_done(self) -> bool:
        """
        Implements the logic to decide when the episode is completed
        """
        is_at_max_step = (
            self.num_steps is not None and self.current_step == self.num_steps
        )

        return is_at_max_step or len(self._dones) == len(self.network.agents)

    def __getitem__(self, agent_id: AgentID) -> Agent:
        return self.network[agent_id]

    def _apply_samplers(self):
        for sampler in self._samplers:
            sampler.value = sampler.sample()

        if self.env_supertype is not None:
            self.env_type = self.env_supertype.sample()

        for agent in self.network.agents.values():
            if agent.supertype is not None:
                agent.type = agent.supertype.sample()
