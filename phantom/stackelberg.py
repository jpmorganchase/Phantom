from typing import Any, Dict, Mapping, Optional, Sequence, Tuple

import gymnasium as gym

from .env import PhantomEnv
from .network import Network
from .supertype import Supertype
from .telemetry import logger
from .types import AgentID


class StackelbergEnv(PhantomEnv):
    """
    An environment modelling a Stackelberg game/competition.

    Arguments:
        num_steps: The maximum number of steps the environment allows per episode.
        network: A Network class or derived class describing the connections between
            agents and agents in the environment.
        leader_agents: A list of Agent IDs to use as 'leaders'.
        follower_agents: A list of Agent IDs to use as 'followers'.
        env_supertype: Optional Supertype class instance for the environment. If this is
            set, it will be sampled from and the :attr:`env_type` property set on the
            class with every call to :meth:`reset()`.
        agent_supertypes: Optional mapping of agent IDs to Supertype class instances. If
            these are set, each supertype will be sampled from and the :attr:`type`
            property set on the related agent with every call to :meth:`reset()`.
    """

    def __init__(
        self,
        num_steps: int,
        network: Network,
        leader_agents: Sequence[AgentID],
        follower_agents: Sequence[AgentID],
        env_supertype: Optional[Supertype] = None,
        agent_supertypes: Optional[Mapping[AgentID, Supertype]] = None,
    ) -> None:
        super().__init__(num_steps, network, env_supertype, agent_supertypes)

        # Validate leader and follower agent ID lists
        for aid in leader_agents + follower_agents:
            assert aid in network.agent_ids, f"Agent '{aid}' not in network"

        for aid in leader_agents:
            assert aid not in follower_agents, f"Agent '{aid}' not in network"

        self.leader_agents = leader_agents
        self.follower_agents = follower_agents

        self._rewards: Dict[AgentID, Optional[float]] = {}

    def reset(
        self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[Dict[AgentID, Any], Dict[str, Any]]:
        """
        Reset the environment and return initial observations from the leader agents.

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
        gym.Env.reset(self, seed=seed, options=options)

        # Reset the clock
        self._current_step = 0

        # Generate initial sampled values in samplers
        for sampler in self._samplers:
            sampler.sample()

        if self.env_supertype is not None:
            self.env_type = self.env_supertype.sample()

        # Reset the network and call reset method on all agents in the network
        self.network.reset()

        logger.log_reset(self)

        # Reset the strategic agents' termination/truncation statuses stored by the
        # environment
        self._terminations = set()
        self._truncations = set()

        self._rewards = {aid: None for aid in self.strategic_agent_ids}

        # Generate all contexts for strategic leader agents
        self._make_ctxs(
            [aid for aid in self.leader_agents if aid in self.strategic_agent_ids]
        )

        # Generate initial observations for strategic leader agents
        obs = {
            ctx.agent.id: ctx.agent.encode_observation(ctx)
            for ctx in self._ctxs.values()
        }

        logger.log_observations(obs)

        return {k: v for k, v in obs.items() if v is not None}, {}

    def step(self, actions: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.

        Returns:
            A :class:`PhantomEnv.Step` object containing observations, rewards,
            terminations, truncations and infos.
        """
        # Increment the clock
        self._current_step += 1

        logger.log_step(self.current_step, self.num_steps)
        logger.log_actions(actions)
        logger.log_start_decoding_actions()

        # Generate contexts for all agents taking actions / generating messages
        self._make_ctxs(self.agent_ids)

        acting_agents, next_acting_agents = (
            (self.leader_agents, self.follower_agents)
            if self.current_step % 2 == 1
            else (self.follower_agents, self.leader_agents)
        )

        # Decode action/generate messages for agents and send to the network
        self._handle_acting_agents(acting_agents, actions)

        self.resolve_network()

        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, float] = {}
        terminations: Dict[AgentID, bool] = {}
        truncations: Dict[AgentID, bool] = {}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        for aid in self.strategic_agent_ids:
            if aid in self._terminations or aid in self._truncations:
                continue

            ctx = self._ctxs[aid]

            if aid in next_acting_agents:
                obs = ctx.agent.encode_observation(ctx)
                if obs is not None:
                    observations[aid] = obs
                    infos[aid] = ctx.agent.collect_infos(ctx)

            if aid in acting_agents:
                self._rewards[aid] = ctx.agent.compute_reward(ctx)

            terminations[aid] = ctx.agent.is_terminated(ctx)
            truncations[aid] = ctx.agent.is_truncated(ctx)

            if terminations[aid]:
                self._terminations.add(aid)

            if truncations[aid]:
                self._truncations.add(aid)

        logger.log_step_values(observations, rewards, terminations, truncations, infos)
        logger.log_metrics(self)

        terminations["__all__"] = self.is_terminated()
        truncations["__all__"] = self.is_truncated()

        if terminations["__all__"] or truncations["__all__"]:
            logger.log_episode_done()

            # This is the terminal step, return most recent observations, rewards and
            # infos from all agents.
            return self.Step(
                observations, self._rewards, terminations, truncations, infos
            )

        # Otherwise not in terminal step:
        rewards = {
            aid: self._rewards[aid]
            for aid in observations
            if self._rewards[aid] is not None
        }

        return self.Step(observations, rewards, terminations, truncations, infos)
