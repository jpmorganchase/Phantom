from typing import *

import mercury as me
import numpy as np
import phantom as ph


class AlternateStepEnv(ph.PhantomEnv):
    """
    A subclass of PhantomEnv that provides a way to alternate action taking
    between two sets of agents.

    Attributes:
        network: A Mercury Network class or derived class describing the connections
            between agents and actors in the environment.
        odd_step_filter: Callable that returns True if the given agent should be
            used in odd steps.
        odd_step_filter: Callable that returns True if the given agent should be
            used in even steps.
        clock: A Phantom Clock defining the episode length and episode step size.
        environment_actor: An optional actor that has access to global environment
            information.
        policy_grouping: A mapping between custom policy name and list of agents
            sharing the policy (optional).
        seed: A random number generator seed to use (optional).
    """

    def __init__(
        self,
        network: me.Network,
        odd_step_filter: Callable[[ph.Agent], bool],
        even_step_filter: Callable[[ph.Agent], bool],
        clock: Optional[ph.Clock] = None,
        n_steps: Optional[int] = None,
        environment_actor: Optional[ph.EnvironmentActor] = None,
        policy_grouping: Optional[Mapping[str, List[str]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            network, clock, n_steps, environment_actor, policy_grouping, seed
        )

        self.odd_step_filter = odd_step_filter
        self.even_step_filter = even_step_filter

    def reset(self) -> Dict[me.ID, Any]:
        """Reset the market environment and return an initial observation.

        This method resets the :attr:`clock` and the :attr:`network`. This
        includes all the agents in the network.
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
        observations: Dict[me.ID, np.ndarray] = dict()
        rewards: Dict[me.ID, float] = dict()

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            if self.odd_step_filter(agent):
                observations[aid] = agent.encode_obs(ctx)
            if self.even_step_filter(agent):
                rewards[aid] = agent.compute_reward(ctx)

        self.current_step = self.Step({}, rewards, {}, {})

        if len(observations) == 0:
            raise Exception(
                "No observations returned by odd step agents during Env.reset"
            )

        return observations

    def step(self, actions: Mapping[me.ID, Any]) -> ph.PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Args:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        self.clock.tick()

        if self.clock.elapsed % 2 == 1:
            observations = self._odd_step(actions)
            self.current_step.observations.update(observations)
        else:
            observations, rewards = self._even_step(actions)
            self.current_step.rewards.update(rewards)
            self.current_step.observations.update(observations)

        infos: Dict[me.ID, Any] = dict()

        if self.clock.is_terminal:
            for aid in self.current_step.observations.keys():
                ctx = self.network.context_for(aid)
                infos[aid] = self.agents[aid].collect_infos(ctx)

            return self.Step(
                observations=self.current_step.observations,
                rewards=self.current_step.rewards,
                terminals={"__all__": self.clock.is_terminal},
                infos=infos,
            )
        else:
            for aid in observations.keys():
                ctx = self.network.context_for(aid)
                infos[aid] = self.agents[aid].collect_infos(ctx)

            return self.Step(
                observations=observations,
                rewards={
                    aid: self.current_step.rewards[aid] for aid in observations.keys()
                },
                terminals={"__all__": self.clock.is_terminal},
                infos=infos,
            )

    def _odd_step(self, actions: Mapping[me.ID, Any]) -> Dict[me.ID, Any]:
        # Handle the updates due to active/strategic behaviours:
        for aid, action in actions.items():
            if self.odd_step_filter(self.agents[aid]):
                ctx = self.network.context_for(aid)
                packet = self.agents[aid].decode_action(ctx, action)

                self.network.send_from(aid, packet.messages)

        # Resolve the messages on the network and perform mutations:
        self.network.resolve()

        # Compute the output for rllib:
        observations: Dict[me.ID, Any] = dict()

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            if self.even_step_filter(self.agents[aid]):
                observations[aid] = agent.encode_obs(ctx)

        return observations

    def _even_step(
        self, actions: Mapping[me.ID, Any]
    ) -> Tuple[Dict[me.ID, Any], Dict[me.ID, float]]:
        # Handle the updates due to active/strategic behaviours:
        for aid, action in actions.items():
            if self.even_step_filter(self.agents[aid]):
                ctx = self.network.context_for(aid)
                packet = self.agents[aid].decode_action(ctx, action)

                self.network.send_from(aid, packet.messages)

        # Resolve the messages on the network and perform mutations:
        self.network.resolve()

        # Compute the output for rllib:
        observations: Dict[me.ID, Any] = dict()
        rewards: Dict[me.ID, Any] = dict()

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)
            rewards[aid] = agent.compute_reward(ctx)
            if self.odd_step_filter(self.agents[aid]):
                observations[aid] = agent.encode_obs(ctx)

        return observations, rewards
