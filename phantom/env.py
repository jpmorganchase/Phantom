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
from .context import Context
from .network import Network
from .supertype import Supertype
from .types import AgentID
from .utils.samplers import Sampler
from .views import EnvView


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
        env_supertype: Optional Supertype class instance for the environment. If this is
            set, it will be sampled from and the :attr:`env_type` property set on the
            class with every call to :meth:`reset()`.
        agent_supertypes: Optional mapping of agent IDs to Supertype class instances. If
            these are set, each supertype will be sampled from and the :attr:`type`
            property set on the related agent with every call to :meth:`reset()`.
    """

    class Step(NamedTuple):
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

        # Keep track of which agents are already done so we know to not continue to
        # send back obs, rewards, etc...
        self._dones: Set[AgentID] = set()

        # Context objects are generated for all active agents once at the start of each
        # step and stored for use across functions
        self._ctxs: Dict[AgentID, Context] = {}

        # Collect all instances of classes that inherit from BaseSampler from the env
        # supertype and the agent supertypes into a flat list. We make sure that the
        # list contains only one reference to each sampler instance.
        self._samplers: List[Sampler] = []

        if env_supertype is not None:
            # The env will manage sampling the supertype values
            env_supertype._managed = True

            # Extract samplers from env supertype dict
            for value in env_supertype.__dict__.values():
                if isinstance(value, Sampler) and value not in self._samplers:
                    self._samplers.append(value)

            self.env_supertype = env_supertype

        if agent_supertypes is not None:
            for agent_id, agent_supertype in agent_supertypes.items():
                # The env will manage sampling the supertype values
                agent_supertype._managed = True

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

    def view(self) -> EnvView:
        """Return an immutable view to the environment's public state."""
        return EnvView(self.current_step)

    def pre_message_resolution(self) -> None:
        """Perform internal, pre-message resolution updates to the environment."""
        for ctx in self._ctxs.values():
            ctx.agent.pre_message_resolution(ctx)

    def post_message_resolution(self) -> None:
        """Perform internal, post-message resolution updates to the environment."""
        for ctx in self._ctxs.values():
            ctx.agent.post_message_resolution(ctx)

    def resolve_network(self) -> None:
        self.pre_message_resolution()
        self.network.resolve(self._ctxs)
        self.post_message_resolution()

    def reset(self, sample_supertypes: bool = True) -> Dict[AgentID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the step count and the :attr:`network`. This includes all the
        agents in the network.

        Arguments:
            sample_supertypes: If set to False will not automatically sample supertypes
                and apply agent and environment types.

        Returns:
            A dictionary mapping Agent IDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """
        # Reset clock
        self.current_step = 0

        # Sample from supertypes and apply to agent and env type objects
        if sample_supertypes:
            self.sample_supertypes()

        # Reset network and call reset method on all agents in the network
        self.network.reset()
        self.resolve_network()

        # Reset the agents' done statuses stored by the environment
        self._dones = set()

        # Pre-generate all contexts for agents taking actions
        env_view = self.view()
        ctxs = [
            self.network.context_for(agent.id, env_view)
            for agent in self.agents.values()
            if agent.takes_actions
        ]

        # Generate initial observations for agents taking actions
        obs = {ctx.agent.id: ctx.agent.encode_observation(ctx) for ctx in ctxs}
        return {k: v for k, v in obs.items() if v is not None}

    def step(self, actions: Mapping[AgentID, Any]) -> "PhantomEnv.Step":
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.

        Returns:
            A :class:`PhantomEnv.Step` object containing observations, rewards, dones
            and infos.
        """
        # Increment clock
        self.current_step += 1

        # Generate all contexts for all agents taking actions / generating messages
        env_view = self.view()
        self._ctxs = {
            agent_id: self.network.context_for(agent_id, env_view)
            for agent_id in self.agent_ids
            if agent_id not in self._dones
        }

        # Decode action/generate messages for agents and send to the network
        for agent_id, ctx in self._ctxs.items():
            if agent_id in actions:
                messages = ctx.agent.decode_action(ctx, actions[agent_id]) or []
            else:
                messages = ctx.agent.generate_messages(ctx) or []

            for receiver_id, message in messages:
                self.network.send(agent_id, receiver_id, message)

        # Resolve the messages on the network and perform mutations:
        self.resolve_network()

        # Compute the output for rllib:
        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, Any] = {}
        dones: Dict[AgentID, bool] = {}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        # Pre-generate all context objects for acting agents
        for agent_id, ctx in self._ctxs.items():
            if ctx.agent.takes_actions:
                obs = ctx.agent.encode_observation(ctx)
                if obs is not None:
                    observations[agent_id] = obs
                    infos[agent_id] = ctx.agent.collect_infos(ctx)
                    rewards[agent_id] = ctx.agent.compute_reward(ctx)
                    dones[agent_id] = ctx.agent.is_done(ctx)

                    if dones[agent_id]:
                        self._dones.add(agent_id)

        dones["__all__"] = self.is_done()

        return self.Step(
            observations=observations, rewards=rewards, dones=dones, infos=infos
        )

    def is_done(self) -> bool:
        """Implements the logic to decide when the episode is completed."""
        is_at_max_step = (
            self.num_steps is not None and self.current_step == self.num_steps
        )

        return is_at_max_step or len(self._dones) == len(self.network.agents)

    def __getitem__(self, agent_id: AgentID) -> Agent:
        return self.network[agent_id]

    def sample_supertypes(self) -> None:
        for sampler in self._samplers:
            sampler.value = sampler.sample()

        if self.env_supertype is not None:
            self.env_type = self.env_supertype.sample()

        for agent in self.network.agents.values():
            if agent.supertype is not None:
                agent.type = agent.supertype.sample()
