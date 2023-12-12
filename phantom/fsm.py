from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from .env import PhantomEnv
from .network import Network
from .supertype import Supertype
from .telemetry import logger
from .types import AgentID, StageID
from .views import AgentView, EnvView


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
        acting_agents: The agents that will take an action at the end of the steps that
            belong to this stage..
        rewarded_agents: If provided, only the given agents will calculate and return a
            reward at the end of the step for this stage. If not provided, a reward will
            be computed for all acting agents for the current stage.
        next_stages: The stages that this stage can transition to.
        handler: Environment class method to be called when the FSM enters this stage.
    """

    def __init__(
        self,
        stage_id: StageID,
        acting_agents: List[AgentID],
        rewarded_agents: Optional[List[AgentID]] = None,
        next_stages: Optional[List[StageID]] = None,
        handler: Optional[Callable[..., StageID]] = None,
    ) -> None:
        self.id = stage_id
        self.acting_agents = acting_agents
        self.rewarded_agents = rewarded_agents
        self.next_stages = next_stages or []
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

        self._initial_stage = initial_stage

        self._rewards: Dict[AgentID, Optional[float]] = {}
        self._observations: Dict[AgentID, Any] = {}
        self._infos: Dict[AgentID, Dict[str, Any]] = {}

        self._stages: Dict[StageID, FSMStage] = {}

        self._current_stage = self.initial_stage
        self.previous_stage: Optional[StageID] = None

        # Register stages via optional class initialiser list
        for stage in stages or []:
            if stage.id not in self._stages:
                self._stages[stage.id] = stage

        # Register stages via FSMStage decorator
        for attr_name in dir(self):
            if attr_name != "_acting_agents":
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
            if len(stage.next_stages) != 1 and stage.handler is None:
                raise FSMValidationError(
                    f"Stage '{stage.id}' without handler must have exactly one next stage (got {len(stage.next_stages)})"
                )

    @property
    def initial_stage(self) -> StageID:
        """Returns the initial stage of the FSM Env."""
        return self._initial_stage

    @property
    def current_stage(self) -> StageID:
        """Returns the current stage of the FSM Env."""
        return self._current_stage

    def is_fsm_deterministic(self) -> bool:
        """Returns true if all stages are followed by exactly one stage."""
        return all(len(s.next_stages) == 1 for s in self._stages.values())

    def view(self, agent_views: Dict[AgentID, Optional[AgentView]]) -> FSMEnvView:
        """Return an immutable view to the FSM environment's public state."""
        return FSMEnvView(
            self.current_step, self.current_step / self.num_steps, self.current_stage
        )

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
            - An optional dictionary with auxillary information, equivalent to the info
            dictionary in `env.step()`.
        """
        # Set initial null reward values
        self._rewards = defaultdict(lambda: None)

        return super().reset()

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
        self._step_1(actions)

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

        current_stage = self._stages[self.current_stage]

        if next_stage not in current_stage.next_stages:
            raise FSMRuntimeError(
                f"FiniteStateMachineEnv attempted invalid transition from '{self.current_stage}' to {next_stage}"
            )

        observations: Dict[AgentID, Any] = {}
        rewards: Dict[AgentID, float] = {}
        terminations: Dict[AgentID, bool] = {}
        truncations: Dict[AgentID, bool] = {}
        infos: Dict[AgentID, Dict[str, Any]] = {}

        if current_stage.rewarded_agents is None:
            rewarded_agents = self.strategic_agent_ids
            next_acting_agents = self.strategic_agent_ids
        else:
            rewarded_agents = current_stage.rewarded_agents
            next_acting_agents = self._stages[next_stage].acting_agents

        observations, rewards, terminations, truncations, infos = self._step_2(
            next_acting_agents, rewarded_agents
        )

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        logger.log_fsm_transition(self.current_stage, next_stage)

        self.previous_stage, self._current_stage = self.current_stage, next_stage

        if (
            self.current_stage is None
            or terminations["__all__"]
            or truncations["__all__"]
        ):
            logger.log_episode_done()

            # This is the terminal stage, return most recent observations, rewards and
            # infos from all agents.
            return self.Step(
                observations=self._observations,
                rewards=self._rewards,
                terminations=terminations,
                truncations=truncations,
                infos=self._infos,
            )

        # Otherwise not in terminal stage:
        rewards = {aid: self._rewards[aid] for aid in observations}

        return self.Step(observations, rewards, terminations, truncations, infos)

    def validate(self) -> None:
        """
        Validate the environment by executing a number of steps that sufficiently covers
        the features of the environment.
        """
        obs, _ = self.reset()

        for _ in range(self.num_steps):
            actions = {aid: self.agents[aid].action_space.sample() for aid in obs}
            obs, _, done, _, _ = self.step(actions)

            if done["__all__"]:
                break

    @property
    def _acting_agents(self) -> Sequence[AgentID]:
        return [
            aid
            for aid in self._stages[self.current_stage].acting_agents
            if aid in self.strategic_agent_ids
        ]
