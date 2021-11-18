from typing import (
    Any,
    AnyStr,
    Collection,
    Dict,
    Iterable,
    Mapping,
    NamedTuple,
    Optional,
)

import mercury as me
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from .agent import Agent
from .clock import Clock
from .packet import Mutation
from .supertype import BaseSupertype


class EnvironmentActor(me.actors.SyncActor):
    """
    Pseudo-actor used for calculating and storing global environment state for
    use by other actors and agents. This actor should be added to the environment
    via the PhantomEnv.__init__ 'environment_actor' argument. When this actor is
    added to an environment it is automatically connected to every actor and agent.

    The pre-resolution and post-resolution methods on this actor are called before
    any other actor during each step.

    This actor should not be used for sending or receiving messages.
    """

    ID = "__ENV"

    def __init__(self):
        super().__init__(self.ID)

        self.env_type: Optional[BaseSupertype] = None


class PhantomEnv(MultiAgentEnv):
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
            between agents and actors in the environment.
        clock: A Phantom Clock defining the episode length and episode step size.
        n_steps: Alternative to providing a Clock instance.
        environment_actor: An optional actor that has access to global environment
            information.
        seed: A random number generator seed to use (optional).
    """

    env_name: str = "phantom"

    class Step(NamedTuple):
        """Type alias for the PhantomEnv::step return object."""

        observations: Mapping[me.ID, Any]
        rewards: Mapping[me.ID, float]
        terminals: Mapping[me.ID, bool]
        infos: Mapping[me.ID, Any]

    def __init__(
        self,
        network: me.Network,
        clock: Optional[Clock] = None,
        n_steps: Optional[int] = None,
        environment_actor: Optional[EnvironmentActor] = None,
    ) -> None:
        if clock is None:
            if n_steps is None:
                raise ValueError(
                    "Must provide either a clock instance or n_steps value when creating PhantomEnv"
                )

            clock = Clock(0, n_steps, 1)

        self.network: me.Network = network
        self.clock: Clock = clock
        self.agents: Dict[me.ID, Agent] = {
            aid: actor
            for aid, actor in network.actors.items()
            if isinstance(actor, Agent)
        }
        self._dones = set()
        self._samplers = []
        self._env_supertype = None

        if environment_actor is not None:
            # Connect the environment actor to all existing actors
            self.network.add_actor(environment_actor)
            self.network.add_connections_between(
                [environment_actor.id], list(network.actor_ids)
            )

            # Create a function to override the existing network.resolve method
            # with. This custom method allows proper handling of the environment
            # actor. This is a temporary fix until a better way of handling
            # environment state is implemented.
            def resolve() -> None:
                env_actor_ctx = self.network.context_for(EnvironmentActor.ID)

                ctx_map = {
                    actor_id: self.network.context_for(actor_id)
                    for actor_id in self.network.actors
                    if actor_id != EnvironmentActor.ID
                }

                environment_actor.pre_resolution(env_actor_ctx)

                for ctx in ctx_map.values():
                    ctx.actor.pre_resolution(ctx)

                self.network.resolver.resolve(self.network)

                environment_actor.post_resolution(env_actor_ctx)

                for ctx in ctx_map.values():
                    ctx.actor.post_resolution(ctx)

                self.network.resolver.reset()

            # Override network.resolve method
            self.network.resolve = resolve

    @property
    def agent_ids(self) -> Collection[me.ID]:
        """Return a list of the IDs of the agents in the environment."""
        return self.agents.keys()

    @property
    def n_agents(self) -> int:
        """Return the number of agents in the environment."""
        return len(self.agent_ids)

    def reset(self) -> Dict[me.ID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the :attr:`clock` and the :attr:`network`. This
        includes all the agents in the network.

        Returns:
            A dictionary mapping Agent IDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """
        # Set clock back to time step 0
        self.clock.reset()

        # Reset network and call reset method on all actors in the network.
        # Message samplers should be called here from the respective actor's reset method.
        self.network.reset()
        self.network.resolve()

        # Reset the agents' done status
        self._dones = set()

        # Generate initial observations.
        observations: Dict[me.ID, Any] = dict()

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            observations[aid] = agent.encode_obs(ctx)

        return observations

    def step(self, actions: Mapping[me.ID, Any]) -> "PhantomEnv.Step":
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        self.clock.tick()

        mut_map: Dict[me.ID, Iterable[Mutation]] = {}

        # Handle the updates due to active/strategic behaviours:
        for aid, action in actions.items():
            ctx = self.network.context_for(aid)
            packet = self.agents[aid].decode_action(ctx, action)
            mut_map[aid] = packet.mutations

            self.network.send_from(aid, packet.messages)

        # Resolve the messages on the network and perform mutations:
        self.network.resolve()

        # Apply mutations:
        for actor_id, mutations in mut_map.items():
            ctx = self.network.context_for(actor_id)

            for mut in mutations:
                ctx.actor.handle_mutation(ctx, mut)

        # Compute the output for rllib:
        observations: Dict[me.ID, Any] = dict()
        rewards: Dict[me.ID, Any] = dict()
        dones: Dict[me.ID, bool] = dict()
        infos: Dict[me.ID, Dict[AnyStr, Any]] = dict()

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            if agent.is_done(ctx):
                # if agent is just done we send the last obs, rew, info
                if aid not in self._dones:
                    self._dones.add(aid)
                    observations[aid] = agent.encode_obs(ctx)
                    rewards[aid] = agent.compute_reward(ctx)
                    infos[aid] = agent.collect_infos(ctx)
                    dones[aid] = True
                # otherwise just ignore
            else:
                observations[aid] = agent.encode_obs(ctx)
                rewards[aid] = agent.compute_reward(ctx)
                infos[aid] = agent.collect_infos(ctx)
                dones[aid] = False

        dones["__all__"] = self.is_done()

        return self.Step(
            observations=observations, rewards=rewards, terminals=dones, infos=infos
        )

    def is_done(self) -> bool:
        """
        Implements the logic to decide when the episode is completed
        """
        return self.clock.is_terminal or len(self._dones) == len(self.agents)

    def seed(self, seed: int) -> None:
        """
        Set the random seed of the environment.

        Arguments:
            seed: The seed used by numpy to generate a deterministic set of
                random values.
        """
        self.np_random, self.sid = seeding.np_random(seed)

    def __getitem__(self, actor_id: me.ID) -> me.actors.Actor:
        return self.network[actor_id]
