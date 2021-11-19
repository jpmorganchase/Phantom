from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    Optional,
    Mapping,
    Set,
)

import mercury as me
import numpy as np

from .clock import Clock
from .env import EnvironmentActor, PhantomEnv
from .packet import Mutation


StageID = Hashable
StageHandler = Callable[[], StageID]


class FSMStage:
    """
    Decorator used in the :class:`FiniteStateMachineEnv` to declare the finite state
    machine structure and assign handler functions to stages.

    A 'stage' corresponds to a state in the finite state machine, however to avoid any
    confusion with Environment states we refer to them as stages.

    Attributes:
        stage_id: The name of this stage.
        next_stages: The stages that this stage can transition to.
        acting_agents: If provided, only the given agents will make observations at the
            end of the previous step and take actions in that steps. If not provided, all
            agents will make observations and take actions.
        rewarded_agents: If provided, only the given agents will calculate and return a
            reward at the end of the step for this stage. If not provided, all agents
            will calculate and return a reward.
        handler: Environment class method to be called when the FSM enters this stage.
    """

    def __init__(
        self,
        stage_id: StageID,
        next_stages: Optional[Iterable[StageID]] = None,
        acting_agents: Optional[Iterable[me.ID]] = None,
        rewarded_agents: Optional[Iterable[me.ID]] = None,
        handler: Optional[StageHandler] = None,
    ) -> None:
        self.stage_id = stage_id
        self.next_stages = next_stages
        self.acting_agents = acting_agents
        self.rewarded_agents = rewarded_agents
        self.handler: Optional[StageHandler] = handler

    def __call__(self, fn: Callable[..., Optional[StageID]]):
        setattr(fn, "_decorator", self)
        self.handler = fn

        return fn


class FSMValidationError(Exception):
    """
    Error raised when validating the FSM when initialising the
    :class:`FiniteStateMachineEnv`.
    """


class FSMRuntimeError(Exception):
    """
    Error raised when validating FSM stage changes when running an episode using the
    :class:`FiniteStateMachineEnv`.
    """


class FiniteStateMachineEnv(PhantomEnv, ABC):
    """
    Base environment class that allows implementation of a finite state machine to
    handle complex environment multi-step setups.

    This class should not be used directly and instead should be subclassed.

    Use the :class:`FSMStage` decorator on handler methods within subclasses of this
    class to register stages to the FSM.

    A 'stage' corresponds to a state in the finite state machine, however to avoid any
    confusion with Environment states we refer to them as stages.

    Stage IDs can be anything type that is hashable, eg. strings, ints, enums.

    Attributes:
        initial_stage: The initial starting stage of the FSM. When the reset() method is
            called the environment is initialised into this stage.
        network: A Mercury Network class or derived class describing the connections
            between agents and actors in the environment.
        clock: A Phantom Clock defining the episode length and episode step size.
        n_steps: Alternative to providing a Clock instance.
        environment_actor: An optional actor that has access to global environment
            information.
        policy_grouping: A mapping between custom policy name and list of agents
            sharing the policy (optional).
        seed: A random number generator seed to use (optional).
        stages: List of FSM stages. FSM stages can be defined via this list or
            alternatively via the :class:`FSMStage` decorator.
    """

    def __init__(
        self,
        # fsm env specific:
        initial_stage: StageID,
        # from phantom env:
        network: me.Network,
        clock: Optional[Clock] = None,
        n_steps: Optional[int] = None,
        environment_actor: Optional[EnvironmentActor] = None,
        seed: Optional[int] = None,
        # fsm env specific:
        stages: Optional[Iterable[FSMStage]] = None,
    ) -> None:
        super().__init__(network, clock, n_steps, environment_actor, seed)

        self.initial_stage: StageID = initial_stage

        self._rewards: Dict[me.ID, float] = {}
        self._observations: Dict[me.ID, np.ndarray] = {}
        self._dones: Set[me.ID] = set()
        self._infos: Dict[me.ID, Dict[str, Any]] = {}

        self._stages: Dict[StageID, FSMStage] = {}
        self.current_stage: Optional[StageID] = None

        # Register stages via optional class initialiser list
        if stages is not None:
            for stage in stages:
                if stage.stage_id not in self._stages:
                    self._stages[stage.stage_id] = stage

        # Register stages via FSMStage decorator
        for attr in dir(self):
            attr = getattr(self, attr)
            if callable(attr):
                fn = attr
                if hasattr(fn, "_decorator"):
                    if fn._decorator.stage_id in self._stages:
                        raise FSMValidationError(
                            f"Found multiple stages with ID '{fn._decorator.stage_id}'"
                        )

                    self._stages[fn._decorator.stage_id] = fn._decorator

        # Check there is at least one stage
        if len(self._stages) == 0:
            raise FSMValidationError(
                "No registered stages. Please use the 'FSMStage' decorator or the stage_definitions init parameter"
            )

        # Check initial stage is valid
        if self.initial_stage not in self._stages:
            raise FSMValidationError(
                f"Initial stage '{self.initial_stage}' is not a valid stage"
            )

        # Check all 'next stages' are valid
        for stage in self._stages.values():
            for next_stage in stage.next_stages:
                if next_stage not in self._stages:
                    raise FSMValidationError(
                        f"Next stage '{next_stage}' given in stage '{stage.stage_id}' is not a valid stage"
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

        self.current_stage = self.initial_stage

        # Set clock back to time step 0
        self.clock.reset()
        # Reset network and call reset method on all actors in the network.
        # Message samplers should be called here from the respective actor's reset method.
        self.network.reset()
        self.network.resolve()

        # Generate initial observations.
        observations: Dict[me.ID, Any] = {}

        acting_agents = self._stages[self.initial_stage].acting_agents

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            if acting_agents is None or aid in acting_agents:
                observations[aid] = agent.encode_obs(ctx)

            self._rewards[aid] = None

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

        handler = self._stages[self.current_stage].handler
        if hasattr(handler, "__self__"):
            # If the FSMStage is defined with the stage definitions the handler will be
            # a bound method of the env class.
            next_stage = handler()
        else:
            # If the FSMStage is defined as a decorator the handler will be an unbound
            # function.
            next_stage = handler(self)

        if next_stage not in self._stages[self.current_stage].next_stages:
            raise FSMRuntimeError(
                f"FiniteStateMachineEnv attempted invalid transition from '{self.current_stage}' to {next_stage}"
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

        next_acting_agents = self._stages[next_stage].acting_agents
        rewarded_agents = self._stages[self.current_stage].rewarded_agents

        for aid, agent in self.agents.items():
            ctx = self.network.context_for(aid)

            terminals[aid] = agent.is_done(ctx)

            if terminals[aid]:
                self._dones.add(aid)

            if next_acting_agents is None or aid in next_acting_agents:
                observations[aid] = agent.encode_obs(ctx)
                infos[aid] = agent.collect_infos(ctx)

            if rewarded_agents is None or aid in rewarded_agents:
                rewards[aid] = agent.compute_reward(ctx)

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        self.current_stage = next_stage

        if self.current_stage is None or self.is_done():
            # This is the terminal stage, return most recent observations, rewards and
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
