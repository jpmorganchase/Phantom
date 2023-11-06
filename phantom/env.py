from collections import defaultdict
from typing import (
    Any,
    Dict,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Set,
    Tuple,
)

import gymnasium as gym

from .agents import Agent, StrategicAgent
from .context import Context
from .network import Network
from .supertype import Supertype
from .telemetry import logger
from .types import AgentID
from .utils.samplers import Sampler
from .views import AgentView, EnvView


class PhantomEnv(gym.Env):
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
        terminations: Dict[AgentID, bool]
        truncations: Dict[AgentID, bool]
        infos: Dict[AgentID, Any]

    def __init__(
        self,
        num_steps: int,
        network: Optional[Network] = None,
        env_supertype: Optional[Supertype] = None,
        agent_supertypes: Optional[Mapping[AgentID, Supertype]] = None,
    ) -> None:
        self.network = network or Network()
        self._current_step = 0
        self.num_steps = num_steps

        self.env_supertype: Optional[Supertype] = None
        self.env_type: Optional[Supertype] = None

        # Keep track of which strategic agents are already terminated/truncated so we
        # know to not continue to send back obs, rewards, etc...
        self._terminations: Set[AgentID] = set()
        self._truncations: Set[AgentID] = set()

        # Context objects are generated for all active agents once at the start of each
        # step and stored for use across functions.
        self._ctxs: Dict[AgentID, Context] = {}

        # Collect all instances of classes that inherit from BaseSampler from the env
        # supertype and the agent supertypes into a flat list. We make sure that the
        # list contains only one reference to each sampler instance.
        self._samplers: List[Sampler] = []

        if env_supertype is not None:
            if isinstance(env_supertype, dict):
                env_supertype = self.Supertype(**env_supertype)
            else:
                assert isinstance(env_supertype, self.Supertype)

            # The env will manage sampling the supertype values
            env_supertype._managed = True

            # Extract samplers from env supertype dict
            for value in env_supertype.__dict__.values():
                if isinstance(value, Sampler) and value not in self._samplers:
                    self._samplers.append(value)

            self.env_supertype = env_supertype

        if agent_supertypes is not None:
            for agent_id, agent_supertype in agent_supertypes.items():
                if isinstance(agent_supertype, dict):
                    agent_supertype = self.agents[agent_id].Supertype(**agent_supertype)
                # TODO: fix, temporarily disabled as AgentClass.Supertype changed to __main__.Supertype
                # else:
                #     assert isinstance(agent_supertype, self.agents[agent_id].Supertype)

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
            sampler.sample()

        # Apply sampled supertype values to agent types
        for agent in self.agents.values():
            agent.reset()

    @property
    def current_step(self) -> int:
        """Return the current step of the environment."""
        return self._current_step

    @property
    def n_agents(self) -> int:
        """Return the number of agents in the environment."""
        return len(self.agent_ids)

    @property
    def agents(self) -> Dict[AgentID, Agent]:
        """Return a mapping of agent IDs to agents in the environment."""
        return self.network.agents

    @property
    def agent_ids(self) -> List[AgentID]:
        """Return a list of the IDs of the agents in the environment."""
        return list(self.network.agent_ids)

    @property
    def strategic_agents(self) -> List[StrategicAgent]:
        """Return a list of agents that take actions."""
        return [a for a in self.agents.values() if isinstance(a, StrategicAgent)]

    @property
    def non_strategic_agents(self) -> List[Agent]:
        """Return a list of agents that do not take actions."""
        return [a for a in self.agents.values() if not isinstance(a, StrategicAgent)]

    @property
    def strategic_agent_ids(self) -> List[AgentID]:
        """Return a list of the IDs of the agents that take actions."""
        return [a.id for a in self.agents.values() if isinstance(a, StrategicAgent)]

    @property
    def non_strategic_agent_ids(self) -> List[AgentID]:
        """Return a list of the IDs of the agents that do not take actions."""
        return [a.id for a in self.agents.values() if not isinstance(a, StrategicAgent)]

    def view(self, agent_views: Dict[AgentID, AgentView]) -> EnvView:
        """Return an immutable view to the environment's public state."""
        return EnvView(self.current_step, self.current_step / self.num_steps)

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

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Any], Dict[str, Any]]:
        """
        Reset the environment and return an initial observation.

        This method resets the step count and the :attr:`network`. This includes all the
        agents in the network.

        Args:
            seed: An optional seed to use for the new episode.
            options : Additional information to specify how the environment is reset.

        Returns:
            - A dictionary mapping Agent IDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
            - A dictionary with auxillary information, equivalent to the info dictionary
                in `env.step()`.
        """
        logger.log_reset()

        super().reset(seed=seed, options=options)

        # Reset the clock
        self._current_step = 0

        # Sample from supertype shared sampler objects
        for sampler in self._samplers:
            sampler.sample()

        if self.env_supertype is not None:
            self.env_type = self.env_supertype.sample()

        # Reset the network and call reset method on all agents in the network
        self.network.reset()

        # Reset the strategic agents' termination/truncation statuses stored by the
        # environment
        self._terminations = set()
        self._truncations = set()

        # Generate all contexts for agents taking actions
        self._make_ctxs(self._acting_agents)

        # Generate initial observations for agents taking actions
        obs = {
            aid: ctx.agent.encode_observation(ctx) for aid, ctx in self._ctxs.items()
        }

        logger.log_observations(obs)

        unbatched_obs = {}

        for aid, obs in obs.items():
            unbatched_obs.update(self._unbatch_obs(aid, obs))

        return {k: v for k, v in unbatched_obs.items() if v is not None}, {}

    def step(self, actions: Mapping[AgentID, Any]) -> "PhantomEnv.Step":
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.

        Returns:
            A :class:`PhantomEnv.Step` object containing observations, rewards,
            terminations, truncations and infos.
        """
        self._step_1(actions)

        self.resolve_network()

        observations, rewards, terminations, truncations, infos = self._step_2(
            self.strategic_agent_ids, self.strategic_agent_ids
        )

        if terminations["__all__"] or truncations["__all__"]:
            logger.log_episode_done()

        return self.Step(observations, rewards, terminations, truncations, infos)

    def _step_1(self, actions: Mapping[AgentID, Any]):
        """Internal method."""
        # Increment the clock
        self._current_step += 1

        logger.log_step(self.current_step, self.num_steps)
        logger.log_actions(actions)
        logger.log_start_decoding_actions()

        # Generate contexts for all agents taking actions / generating messages
        self._make_ctxs(self.agent_ids)

        # Decode action/generate messages for agents and send to the network
        stacked_ids = defaultdict(list)

        for aid in actions.keys():
            if aid.startswith("__stacked__"):
                i, aid = aid[11:].split("__", 1)
                stacked_ids[aid].append(int(i))

        for aid in self.agents:
            if aid in self._terminations or aid in self._truncations:
                continue

            ctx = self._ctxs[aid]

            if aid in actions:
                if aid in stacked_ids:
                    actions[aid] = [
                        actions[f"__stacked__{idx}__{aid}"]
                        for idx in sorted(stacked_ids[aid])
                    ]

                messages = ctx.agent.decode_action(ctx, actions[aid]) or []
            else:
                messages = ctx.agent.generate_messages(ctx) or []

            for receiver_id, message in messages:
                self.network.send(aid, receiver_id, message)

    def _step_2(self, observing_agents: List[AgentID], rewarded_agents: List[AgentID]):
        """Internal method."""
        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, Any] = {}
        terminations: Dict[AgentID, bool] = {}
        truncations: Dict[AgentID, bool] = {}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        for aid in self.strategic_agent_ids:
            if aid in self._terminations or aid in self._truncations:
                continue

            ctx = self._ctxs[aid]

            if aid in observing_agents:
                obs = ctx.agent.encode_observation(ctx)

                if obs is not None:
                    infos2 = ctx.agent.collect_infos(ctx)
                    unbatched_obs = self._unbatch_obs(aid, obs)
                    observations.update(unbatched_obs)

                    if len(unbatched_obs) == 1:
                        infos[aid] = infos2
                    else:
                        infos.update(
                            {
                                f"__stacked__{i}__{aid}": infos2
                                for i in range(len(unbatched_obs))
                            }
                        )

            if aid in rewarded_agents:
                rewards2 = ctx.agent.compute_reward(ctx)

                if isinstance(rewards2, (float, int)):
                    rewards[aid] = rewards2
                elif len(rewards2) == 1:
                    rewards[aid] = rewards2[0]
                else:
                    rewards.update(
                        {f"__stacked__{i}__{aid}": r for i, r in enumerate(rewards2)}
                    )

            is_terminated = ctx.agent.is_terminated(ctx)
            is_truncated = ctx.agent.is_truncated(ctx)

            terminations[aid] = is_terminated
            truncations[aid] = is_truncated

            if is_terminated:
                self._terminations.add(aid)

            if is_truncated:
                self._truncations.add(aid)

        terminations["__all__"] = self.is_terminated()
        truncations["__all__"] = self.is_truncated()

        logger.log_step_values(observations, rewards, terminations, truncations, infos)
        logger.log_metrics(self)

        return (observations, rewards, terminations, truncations, infos)

    def render(self) -> None:
        return None

    def is_terminated(self) -> bool:
        """Implements the logic to decide when the episode is terminated."""
        return len(self._terminations) == len(self.strategic_agents)

    def is_truncated(self) -> bool:
        """Implements the logic to decide when the episode is truncated."""
        is_at_max_step = (
            self.num_steps is not None and self.current_step == self.num_steps
        )

        return is_at_max_step or len(self._truncations) == len(self.strategic_agents)

    @property
    def _acting_agents(self) -> Sequence[AgentID]:
        return self.strategic_agent_ids

    def _make_ctxs(self, agent_ids: Sequence[AgentID]) -> None:
        """Internal method."""
        env_view = self.view(
            {agent_id: agent.view() for agent_id, agent in self.agents.items()}
        )

        self._ctxs = {
            aid: self.network.context_for(aid, env_view)
            for aid in agent_ids
            if aid not in self._terminations and aid not in self._truncations
        }

    def _unbatch_obs(self, aid: AgentID, obs: Any) -> Dict[str, Any]:
        """Internal method."""
        observations = {aid: obs}

        if isinstance(obs, list):
            obs_space = self.agents[aid].observation_space
            if not obs_space.contains(obs):
                if obs_space.contains(obs[0]):
                    if len(obs) == 1:
                        observations[aid] = obs[0]
                    else:
                        for i, ob in enumerate(obs):
                            observations[f"__stacked__{i}__{aid}"] = ob

                        del observations[aid]
                else:
                    raise ValueError(
                        f"Detected batched observations but with wrong shape, expected: {obs_space}, got: {obs[0]} with dtype={obs[0].dtype}"
                    )

        return observations

    def __getitem__(self, agent_id: AgentID) -> Agent:
        return self.network[agent_id]
