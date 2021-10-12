from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Mapping,
    Set,
)

import mercury as me
import numpy as np

from .clock import Clock
from .env import EnvironmentActor, PhantomEnv
from .packet import Mutation


StateID = Hashable
StateHandler = Callable[[], StateID]


class fsm_state:
    """
    Decorator used in the FiniteStateMachineEnv and TurnBasedEnv to declare the
    FSM structure and assign handler functions to states.
    """

    def __init__(
        self,
        state_id: StateID,
        next_states: Optional[Iterable[StateID]] = None,
        take_actions_subset: Optional[Iterable[me.ID]] = None,
        calc_rewards_subset: Optional[Iterable[me.ID]] = None,
    ) -> None:
        """
        Arguments:
            state_id: The name of this state.
            next_states: The states that this state can transition to.
            take_actions_subset: If provided, only the agents given will take an action
                at the start of the step for this state. If not provided, all agents
                will take actions.
            calc_rewards_subset: If provided, only the agents given will calculate and
                return a reward at the end of the step for this state. If not provided,
                all agents will calculate and return a reward.
        """
        self.state_id = state_id
        self.next_states = next_states
        self.take_actions_subset = take_actions_subset
        self.calc_rewards_subset = calc_rewards_subset
        self._handler: Optional[StateHandler] = None

    def __call__(self, fn: Callable[..., Optional[StateID]]):
        setattr(fn, "_decorator", self)
        self._handler = fn

        return fn


class FSMValidationError(Exception):
    """
    Error raised when validating the FSM when initialising the FiniteStateMachineEnv.
    """

    pass


class FSMRuntimeError(Exception):
    """
    Error raised when validating FSM state changes when running an episode using the
    FiniteStateMachineEnv.
    """

    pass


class FiniteStateMachineEnv(PhantomEnv, ABC):
    """
    Base environment class that allows implementation of a finite state machine to
    handle complex environment multi-step setups.

    Use the fsm_state decorator on handler methods within subclasses of this class to
    register states to the FSM.

    State IDs can be anything type that is hashable, eg. strings, ints, enums.

    Attributes:
        initial_state: The initial starting state of the FSM. When the reset() method is
            called the environment is initialised into this state.
        network: A Mercury Network class or derived class describing the connections
            between agents and actors in the environment.
        clock: A Phantom Clock defining the episode length and episode step size.
        n_steps: Alternative to providing a Clock instance.
        environment_actor: An optional actor that has access to global environment
            information.
        policy_grouping: A mapping between custom policy name and list of agents
            sharing the policy (optional).
        seed: A random number generator seed to use (optional).
    """

    def __init__(
        self,
        # custom:
        initial_state: StateID,
        # from phantom env:
        network: me.Network,
        clock: Optional[Clock] = None,
        n_steps: Optional[int] = None,
        environment_actor: Optional[EnvironmentActor] = None,
        policy_grouping: Optional[Mapping[str, List[str]]] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__(
            network, clock, n_steps, environment_actor, policy_grouping, seed
        )

        self.initial_state: StateID = initial_state

        self._rewards: Dict[me.ID, float] = {}
        self._observations: Dict[me.ID, np.ndarray] = {}
        self._dones: Set[me.ID] = set()
        self._infos: Dict[me.ID, Dict[str, Any]] = {}

        self._states: Dict[StateID, fsm_state] = {}
        self.current_state: Optional[StateID] = None

        # Register all states via fsm_state decorator
        for attr in dir(self):
            attr = getattr(self, attr)
            if callable(attr):
                fn = attr
                if hasattr(fn, "_decorator"):
                    if fn._decorator.state_id in self._states:
                        raise FSMValidationError(
                            f"Found multiple states with ID '{fn._decorator.state_id}'"
                        )

                    self._states[fn._decorator.state_id] = fn._decorator

        # Check there is at least one state
        if len(self._states) == 0:
            raise FSMValidationError(
                f"Found no registered states. Please use the 'fsm_state' decorator."
            )

        # Check initial state is valid
        if self.initial_state not in self._states:
            raise FSMValidationError(
                f"Initial state '{self.initial_state}' is not a valid state"
            )

        # Check all 'next states' are valid
        for state in self._states.values():
            for next_state in state.next_states:
                if next_state not in self._states:
                    raise FSMValidationError(
                        f"Next state '{next_state}' given in state '{state.state_id}' is not a valid state"
                    )

    def reset(self) -> Dict[me.ID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the :attr:`clock` and the :attr:`network`. This
        includes all the agents in the network.

        Returns:
            A dictionary mapping Agent IDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """

        self._rewards = {}
        self._observations = {}
        self._dones = set()
        self._infos = {}

        self.current_state = self.initial_state

        # Set clock back to time step 0
        self.clock.reset()
        # Reset network and call reset method on all actors in the network.
        # Message samplers should be called here from the respective actor's reset method.
        self.network.reset()
        self.network.resolve()

        # Generate initial observations.
        observations: Dict[me.ID, Any] = {}

        take_actions_subset = self._states[self.initial_state].take_actions_subset

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            if take_actions_subset is None or aid in take_actions_subset:
                observations[aid] = agent.encode_obs(ctx)

            self._rewards[aid] = agent.compute_reward(ctx)

        return observations

    def step(self, actions: Mapping[me.ID, Any]) -> PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        self.clock.tick()

        mutations: Dict[me.ID, Iterable[Mutation]] = {}

        # Handle the updates due to active/strategic behaviours:
        for aid, action in actions.items():
            # if aid in acted_agents:
            ctx = self.network.context_for(aid)
            packet = self.agents[aid].decode_action(ctx, action)
            mutations[aid] = packet.mutations

            self.network.send_from(aid, packet.messages)

        old_state = self.current_state

        self.current_state = self._states[self.current_state]._handler(self)

        if self.current_state not in self._states[old_state].next_states:
            raise FSMRuntimeError(
                f"FiniteStateMachineEnv attempted invalid transition from '{old_state}' to {self.current_state}"
            )

        # Apply mutations:
        for actor_id, actor_mutations in mutations.items():
            ctx = self.network.context_for(actor_id)

            for mutation in actor_mutations:
                ctx.actor.handle_mutation(ctx, mutation)

        # Compute the output for rllib:
        observations: Dict[me.ID, Any] = {}
        rewards: Dict[me.ID, Any] = {}
        terminals: Dict[me.ID, bool] = {"__all__": False}
        infos: Dict[me.ID, Dict[str, Any]] = {}

        take_actions_subset = self._states[self.current_state].take_actions_subset
        calc_rewards_subset = self._states[self.current_state].calc_rewards_subset

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            terminals[aid] = agent.is_done(ctx)

            if terminals[aid]:
                self._dones.add(aid)

            if take_actions_subset is None or aid in take_actions_subset:
                observations[aid] = agent.encode_obs(ctx)
                infos[aid] = agent.collect_infos(ctx)

            if calc_rewards_subset is None or aid in calc_rewards_subset:
                rewards[aid] = agent.compute_reward(ctx)

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        if self.current_state is None or self.is_done():
            # This is the terminal state, return most recent observations, rewards and
            # infos from all agents.
            terminals = {"__all__": True}

            for aid in self.agents:
                terminals[aid] = aid in self._dones

            return self.Step(
                observations=self._observations,
                rewards=self._rewards,
                terminals=terminals,
                infos=self._infos,
            )
        else:
            rewards = {aid: self._rewards[aid] for aid in observations.keys()}

            return self.Step(observations, rewards, terminals, infos)
