from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple

from .env import PhantomEnv
from .network import Network
from .supertype import Supertype
from .telemetry import logger
from .types import AgentID, PolicyID, StageID
from .views import EnvView


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


class FSMStage:
    """
    Decorator used in the :class:`FiniteStateMachineEnv` to declare the finite state
    machine structure and assign handler functions to stages.

    A 'stage' corresponds to a state in the finite state machine, however to avoid any
    confusion with Environment states we refer to them as stages.

    Attributes:
        id: The name of this stage.
        next_stages: The stages that this stage can transition to.
        acting_agents: If provided, only the given agents will make observations at the
            end of the previous step and take actions in that steps. If not provided,
            all agents will make observations and take actions.
        rewarded_agents: If provided, only the given agents will calculate and return a
            reward at the end of the step for this stage. If not provided, a reward will
            be computed for all acting agents for the current stage.
        handler: Environment class method to be called when the FSM enters this stage.
    """

    def __init__(
        self,
        stage_id: StageID,
        next_stages: Optional[Sequence[StageID]] = None,
        acting_agents: Optional[Sequence[AgentID]] = None,
        rewarded_agents: Optional[Sequence[AgentID]] = None,
        handler: Optional[Callable[[], StageID]] = None,
    ) -> None:
        self.id = stage_id
        self.next_stages = next_stages or []
        self.acting_agents = acting_agents
        self.rewarded_agents = rewarded_agents
        self.handler = handler

    def __call__(self, handler_fn: Callable[..., Optional[StageID]]):
        setattr(handler_fn, "_decorator", self)
        self.handler = handler_fn

        return handler_fn


@dataclass(frozen=True)
class FSMEnvView(EnvView):
    """
    Extension of the :class:`EnvView` class that records the current stage that the
    environment is in.
    """

    stage: StageID


class FiniteStateMachineEnv(PhantomEnv):
    """
    Base environment class that allows implementation of a finite state machine to
    handle complex environment multi-step setups.
    This class should not be used directly and instead should be subclassed.
    Use the :class:`FSMStage` decorator on handler methods within subclasses of this
    class to register stages to the FSM.

    A 'stage' corresponds to a state in the finite state machine, however to avoid any
    confusion with Environment states we refer to them as stages.
    Stage IDs can be anything type that is hashable, eg. strings, ints, enums.

    Arguments:
        num_steps: The maximum number of steps the environment allows per episode.
        network: A Network class or derived class describing the connections between
            agents and agents in the environment.
        initial_stage: The initial starting stage of the FSM. When the reset() method is
            called the environment is initialised into this stage.
        env_supertype: Optional Supertype class instance for the environment. If this is
            set, it will be sampled from and the :attr:`env_type` property set on the
            class with every call to :meth:`reset()`.
        agent_supertypes: Optional mapping of agent IDs to Supertype class instances. If
            these are set, each supertype will be sampled from and the :attr:`type`
            property set on the related agent with every call to :meth:`reset()`.
        stages: List of FSM stages. FSM stages can be defined via this list or
            alternatively via the :class:`FSMStage` decorator.
    """

    def __init__(
        self,
        # from phantom env:
        num_steps: int,
        network: Network,
        # fsm env specific:
        initial_stage: StageID,
        # from phantom env:
        env_supertype: Optional[Supertype] = None,
        agent_supertypes: Optional[Mapping[AgentID, Supertype]] = None,
        # fsm env specific:
        stages: Optional[Sequence[FSMStage]] = None,
    ) -> None:
        super().__init__(num_steps, network, env_supertype, agent_supertypes)

        self.initial_stage = initial_stage

        self._rewards: Dict[PolicyID, Optional[float]] = {}
        self._observations: Dict[PolicyID, Any] = {}
        self._infos: Dict[PolicyID, Dict[str, Any]] = {}

        self._stages: Dict[StageID, FSMStage] = {}

        self.policy_agent_handler_map: Dict[
            PolicyID, Tuple[AgentID, Optional[StageID]]
        ] = {}

        # Register stages via optional class initialiser list
        for stage in stages or []:
            if stage.id not in self._stages:
                self._stages[stage.id] = stage

        # Register stages via FSMStage decorator
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
            if callable(attr):
                handler_fn = attr
                if hasattr(handler_fn, "_decorator"):
                    if handler_fn._decorator.id in self._stages:
                        raise FSMValidationError(
                            f"Found multiple stages with ID '{handler_fn._decorator.id}'"
                        )

                    self._stages[handler_fn._decorator.id] = handler_fn._decorator

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
                        f"Next stage '{next_stage}' given in stage '{stage.id}' is not a valid stage"
                    )

        # Check stages without handler have exactly one next stage
        for stage in self._stages.values():
            if len(stage.next_stages) != 1:
                raise FSMValidationError(
                    f"Stage '{stage.id}' without handler must have exactly one next stage (got {len(stage.next_stages)})"
                )

        self.current_stage = self.initial_stage
        self.previous_stage: Optional[StageID] = None

    def view(self) -> FSMEnvView:
        """Return an immutable view to the FSM environment's public state."""
        return FSMEnvView(
            self.current_step, self.current_step / self.num_steps, self.current_stage
        )

    def reset(self) -> Dict[AgentID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the step count and the :attr:`network`. This includes all the
        agents in the network.

        Returns:
            A dictionary mapping AgentIDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """

        logger.log_reset()

        # Reset clock and stage
        self.current_step = 0
        self.current_stage = self.initial_stage

        # Sample from supertype shared sampler objects
        for sampler in self._samplers:
            sampler.sample()

        if self.env_supertype is not None:
            self.env_type = self.env_supertype.sample()

        self._views = {
            agent_id: agent.view() for agent_id, agent in self.agents.items()
        }

        # Generate views for use of the environment itself for generating it's own view
        self._views = {
            agent_id: agent.view() for agent_id, agent in self.agents.items()
        }

        # Reset network and call reset method on all agents in the network.
        self.network.reset()
        self.resolve_network()

        # Reset the agents' done statuses stored by the environment
        self._dones = set()

        # Set initial null reward values
        self._rewards = {aid: None for aid in self.strategic_agent_ids}

        # Initial acting agents are either those specified in stage acting agents or
        # else all acting agents
        acting_agents = (
            self._stages[self.current_stage].acting_agents or self.strategic_agent_ids
        )

        # Pre-generate all contexts for agents taking actions
        env_view = self.view()
        self._ctxs = {
            aid: self.network.context_for(aid, env_view)
            for aid in acting_agents
            if aid in self.strategic_agent_ids
        }

        # Generate initial sampled values in samplers
        for sampler in self._samplers:
            sampler.sample()

        # Generate initial observations for agents taking actions
        obs = {
            ctx.agent.id: ctx.agent.encode_observation(ctx)
            for ctx in self._ctxs.values()
        }

        logger.log_observations(obs)

        return {k: v for k, v in obs.items() if v is not None}

    def step(self, actions: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        # Increment clock
        self.current_step += 1

        logger.log_step(self.current_step, self.num_steps)

        self._views = {
            agent_id: agent.view() for agent_id, agent in self.agents.items()
        }

        # Pre-generate all contexts for all agents taking actions / generating messages
        env_view = self.view()
        self._ctxs = {
            aid: self.network.context_for(aid, env_view)
            for aid in self.agents
            if aid not in self._dones
        }

        logger.log_actions(actions)
        logger.log_start_decoding_actions()

        # Generate views for use of the environment itself for generating it's own view
        self._views = {
            agent_id: agent.view() for agent_id, agent in self.agents.items()
        }

        # Decode action/generate messages for agents and send to the network
        for aid in actions.keys():
            ctx = self._ctxs[aid]
            messages = ctx.agent.decode_action(ctx, actions[aid]) or []

            for receiver_id, message in messages:
                self.network.send(aid, receiver_id, message)

        for aid in self.non_strategic_agent_ids:
            if (
                self._stages[self.current_stage].acting_agents is None
                or aid in self._stages[self.current_stage].acting_agents
            ):
                ctx = self._ctxs[aid]
                messages = ctx.agent.generate_messages(ctx) or []

                for receiver_id, message in messages:
                    self.network.send(aid, receiver_id, message)

        env_handler = self._stages[self.current_stage].handler

        if env_handler is None:
            # If no handler has been set, manually resolve the network messages.
            self.resolve_network()

            next_stages = self._stages[self.current_stage].next_stages

            if len(next_stages) == 0:
                raise ValueError(
                    f"Current stage '{self.current_stage}' does not have an env handler or a next stage defined"
                )

            next_stage = next_stages[0]

        elif hasattr(env_handler, "__self__"):
            # If the FSMStage is defined with the stage definitions the handler will be
            # a bound method of the env class.
            next_stage = env_handler()

        else:
            # If the FSMStage is defined as a decorator the handler will be an unbound
            # function.
            next_stage = env_handler(self)

        if next_stage not in self._stages[self.current_stage].next_stages:
            raise FSMRuntimeError(
                f"FiniteStateMachineEnv attempted invalid transition from '{self.current_stage}' to {next_stage}"
            )

        # Compute the output for rllib:
        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, float] = {}
        dones: Dict[AgentID, bool] = {"__all__": False}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        if self._stages[self.current_stage].rewarded_agents is None:
            rewarded_agents = self.strategic_agent_ids
            next_acting_agents = self.strategic_agent_ids
        else:
            rewarded_agents = self._stages[self.current_stage].rewarded_agents
            next_acting_agents = self._stages[next_stage].acting_agents

        for aid in self.strategic_agent_ids:
            if aid in self._dones:
                continue

            ctx = self._ctxs[aid]

            if aid in next_acting_agents:
                obs = ctx.agent.encode_observation(ctx)
                if obs is not None:
                    observations[aid] = obs
                    infos[aid] = ctx.agent.collect_infos(ctx)

            if aid in rewarded_agents:
                rewards[aid] = ctx.agent.compute_reward(ctx)

            dones[aid] = ctx.agent.is_done(ctx)

            if dones[aid]:
                self._dones.add(aid)

        logger.log_step_values(observations, rewards, dones, infos)
        logger.log_metrics(self)

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        logger.log_fsm_transition(self.current_stage, next_stage)

        self.previous_stage, self.current_stage = self.current_stage, next_stage

        if self.current_stage is None or self.is_done():
            logger.log_episode_done()

            # This is the terminal stage, return most recent observations, rewards and
            # infos from all agents.
            dones["__all__"] = True

            return self.Step(
                observations=self._observations,
                rewards=self._rewards,
                dones=dones,
                infos=self._infos,
            )

        # Otherwise not in terminal stage:
        rewards = {aid: self._rewards[aid] for aid in observations}

        return self.Step(observations, rewards, dones, infos)
