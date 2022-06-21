import logging
from abc import ABC
from typing import (
    Any,
    Callable,
    Dict,
    Iterable,
    Optional,
    Mapping,
    Set,
    Tuple,
)

import mercury as me
import numpy as np

from ..clock import Clock
from ..env import EnvironmentActor, PhantomEnv
from ..packet import Mutation
from ..typedefs import PolicyID
from .actor import FSMActor
from .agent import FSMAgent
from .typedefs import EnvStageHandler, StageID

logger = logging.getLogger(__name__)


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
        handler: Optional[EnvStageHandler] = None,
    ) -> None:
        self.stage_id = stage_id
        self.next_stages = next_stages
        self.acting_agents = acting_agents
        self.rewarded_agents = rewarded_agents
        self.handler: Optional[EnvStageHandler] = handler

    def __call__(self, handler_fn: Callable[..., Optional[StageID]]):
        setattr(handler_fn, "_decorator", self)
        self.handler = handler_fn

        return handler_fn


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
        # fsm env specific:
        stages: Optional[Iterable[FSMStage]] = None,
    ) -> None:
        super().__init__(network, clock, n_steps, environment_actor)

        self.initial_stage: StageID = initial_stage

        self._rewards: Dict[PolicyID, float] = {}
        self._observations: Dict[PolicyID, np.ndarray] = {}
        self._dones: Set[me.ID] = set()
        self._infos: Dict[PolicyID, Dict[str, Any]] = {}

        self._stages: Dict[StageID, FSMStage] = {}
        self.current_stage: Optional[StageID] = None
        self.previous_stage: Optional[StageID] = None

        self.policy_agent_handler_map: Dict[
            PolicyID, Tuple[me.ID, Optional[StageID]]
        ] = {}

        # Register stages via optional class initialiser list
        if stages is not None:
            for stage in stages:
                if stage.stage_id not in self._stages:
                    self._stages[stage.stage_id] = stage

        # Register stages via FSMStage decorator
        for attr in dir(self):
            attr = getattr(self, attr)
            if callable(attr):
                handler_fn = attr
                if hasattr(handler_fn, "_decorator"):
                    if handler_fn._decorator.stage_id in self._stages:
                        raise FSMValidationError(
                            f"Found multiple stages with ID '{handler_fn._decorator.stage_id}'"
                        )

                    self._stages[handler_fn._decorator.stage_id] = handler_fn._decorator

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

        # Check stages without handler have exactly one next stage
        for stage in self._stages.values():
            if len(stage.next_stages) != 1:
                raise FSMValidationError(
                    f"Stage '{stage.stage_id}' without handler must have exactly one next stage (got {len(stage.next_stages)})"
                )

    def reset(self) -> Dict[PolicyID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the :attr:`clock` and the :attr:`network`. This
        includes all the agents in the network.

        Returns:
            A dictionary mapping PolicyIDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """

        self._rewards = {}
        self._observations = {}
        self._dones = set()
        self._infos = {}

        self.current_stage = self.initial_stage

        logger.info("FSMEnv reset to '%s' stage", self.current_stage)

        # Set clock back to time step 0
        self.clock.reset()
        # Reset network and call reset method on all actors in the network.
        # Message samplers should be called here from the respective actor's reset method.
        self.network.reset()
        self.network.resolve()

        # Generate initial observations.
        observations: Dict[PolicyID, Any] = {}

        self.policy_agent_handler_map = {}

        # Handle pre-stage hooks in actor/agent StageHandlers.
        for actor_id, actor in self.network.actors.items():
            if isinstance(actor, (FSMActor, FSMAgent)):
                if self.current_stage in actor.stage_handlers:
                    ctx = self.network.context_for(actor_id)
                    handler = actor.stage_handlers[self.current_stage]
                    handler.pre_stage_hook(actor, self.current_stage, ctx)

        for agent_id, agent in self.agents.items():
            ctx = self.network.context_for(agent_id)

            if isinstance(agent, FSMAgent):
                for stage_id, handler in agent.stage_handlers.items():
                    policy_id = f"{agent_id}__{stage_id}"

                    self.policy_agent_handler_map[policy_id] = (agent, handler)

                    if stage_id == self.current_stage:
                        observations[policy_id] = handler.encode_obs(
                            agent, self.current_stage, ctx
                        )

                        logger.info(
                            "Returning initial observation for agent '%s'", agent_id
                        )

                    self._rewards[policy_id] = None

            else:
                self.policy_agent_handler_map[agent_id] = (agent, None)

                acting_agents = self._stages[self.current_stage].acting_agents
                if acting_agents is None or agent_id in acting_agents:
                    observations[agent_id] = agent.encode_obs(ctx)

                self._rewards[agent_id] = None

        return observations

    def step(self, actions: Mapping[PolicyID, Any]) -> PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        self.clock.tick()

        mutations: Dict[me.ID, Iterable[Mutation]] = {}

        # Handle the updates due to active/strategic behaviours:
        for policy_name, action in actions.items():
            agent, handler = self.policy_agent_handler_map[policy_name]

            logger.info("Received actions for agent '%s'", agent.id)

            ctx = self.network.context_for(agent.id)

            if handler is None:
                packet = agent.decode_action(ctx, action)
            else:
                packet = handler.decode_action(agent, self.current_stage, ctx, action)

            mutations[agent.id] = packet.mutations

            self.network.send_from(agent.id, packet.messages)

        env_handler = self._stages[self.current_stage].handler

        if env_handler is None:
            # If no handler has been set, manually resolve the network messages.
            self.resolve_network()
            next_stage = self._stages[self.current_stage].next_stages[0]
        elif hasattr(env_handler, "__self__"):
            # If the FSMStage is defined with the stage definitions the handler will be
            # a bound method of the env class.
            next_stage = env_handler()
        else:
            # If the FSMStage is defined as a decorator the handler will be an unbound
            # function.
            next_stage = env_handler(self)

        # Handle post-stage hooks in actor/agent StageHandlers.
        for actor_id, actor in self.network.actors.items():
            if isinstance(actor, (FSMActor, FSMAgent)):
                if self.current_stage in actor.stage_handlers:
                    ctx = self.network.context_for(actor_id)
                    handler = actor.stage_handlers[self.current_stage]
                    handler.post_stage_hook(actor, self.current_stage, ctx)

        logger.info("~" * 80)
        logger.info(
            "FSMEnv progressed from '%s' stage to '%s' stage",
            self.current_stage,
            next_stage,
        )

        # Handle pre-stage hooks in actor/agent StageHandlers.
        for actor_id, actor in self.network.actors.items():
            if isinstance(actor, (FSMActor, FSMAgent)):
                if next_stage in actor.stage_handlers:
                    ctx = self.network.context_for(actor_id)
                    handler = actor.stage_handlers[next_stage]
                    handler.pre_stage_hook(actor, next_stage, ctx)

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
        observations: Dict[PolicyID, Any] = {}
        rewards: Dict[PolicyID, float] = {}
        terminals: Dict[me.ID, bool] = {"__all__": False}
        infos: Dict[PolicyID, Dict[str, Any]] = {}

        for agent_id, agent in self.agents.items():
            ctx = self.network.context_for(agent_id)

            terminals[agent_id] = agent.is_done(ctx)

            if terminals[agent_id]:
                self._dones.add(agent_id)

            if isinstance(agent, FSMAgent):
                for stage_id, handler in agent.stage_handlers.items():
                    policy_id = f"{agent_id}__{stage_id}"

                    if stage_id == next_stage:
                        logger.info(
                            "Encoding observations for agent '%s' and stage '%s'",
                            agent_id,
                            stage_id,
                        )

                        observations[policy_id] = handler.encode_obs(
                            agent, self.current_stage, ctx
                        )
                        infos[policy_id] = agent.collect_infos(ctx)

                    if policy_id in actions:
                        logger.info(
                            "Computing reward for agent '%s' and stage '%s'",
                            agent_id,
                            stage_id,
                        )

                        rewards[policy_id] = handler.compute_reward(
                            agent, self.current_stage, ctx
                        )

            else:
                acting_agents = self._stages[next_stage].acting_agents
                if acting_agents is None or agent_id in acting_agents:
                    logger.info("Encoding observations for agent '%s'", agent_id)
                    observations[agent_id] = agent.encode_obs(ctx)
                    infos[agent_id] = agent.collect_infos(ctx)

                rewarded_agents = self._stages[self.current_stage].rewarded_agents
                if rewarded_agents is None or agent_id in rewarded_agents:
                    logger.info("Computing reward for agent '%s'", agent_id)
                    rewards[agent_id] = agent.compute_reward(ctx)

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        self.previous_stage, self.current_stage = self.current_stage, next_stage

        if self.current_stage is None or self.is_done():
            # This is the terminal stage, return most recent observations, rewards and
            # infos from all agents.
            terminals = {"__all__": True}

            for agent_id in self.agents:
                terminals[agent_id] = agent_id in self._dones

            return self.Step(
                observations=self._observations,
                rewards=self._rewards,
                terminals=terminals,
                infos=self._infos,
            )

        # Otherwise not in terminal stage:
        rewards = {agent_id: self._rewards[agent_id] for agent_id in observations}

        return self.Step(observations, rewards, terminals, infos)

    def resolve_network(self) -> None:
        """
        Used as a replacement to the Mercury ``Network.resolve`` method to allow control
        of the environment actor's pre- and post-resolution hooks.
        """
        ctx_map = {
            actor_id: self.network.context_for(actor_id)
            for actor_id in self.network.actors
            if actor_id != EnvironmentActor.ID
        }

        if self.environment_actor is not None:
            env_actor_ctx = self.network.context_for(EnvironmentActor.ID)
            self.environment_actor.pre_resolution(env_actor_ctx)

        for actor_id, ctx in ctx_map.items():
            actor = self.network.actors[actor_id]
            if isinstance(actor, (FSMActor, FSMAgent)):
                if self.current_stage in actor.stage_handlers:
                    handler = actor.stage_handlers[self.current_stage]
                    handler.pre_msg_resolution_hook(
                        actor, self.current_stage, ctx_map[actor_id]
                    )
            else:
                ctx.actor.pre_resolution(ctx)

        self.network.resolver.resolve(self.network)

        if self.environment_actor is not None:
            self.environment_actor.post_resolution(env_actor_ctx)

        for actor_id, ctx in ctx_map.items():
            actor = self.network.actors[actor_id]
            if isinstance(actor, (FSMActor, FSMAgent)):
                if self.current_stage in actor.stage_handlers:
                    handler = actor.stage_handlers[self.current_stage]
                    handler.post_msg_resolution_hook(
                        actor, self.current_stage, ctx_map[actor_id]
                    )
            else:
                ctx.actor.post_resolution(ctx)

        self.network.resolver.reset()