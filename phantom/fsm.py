from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)

from .env import PhantomEnv
from .network import Network
from .supertype import Supertype
from .types import AgentID, PolicyID, StageID
from .view import EnvView


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
        stage_id: The name of this stage.
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
        self.stage_id = stage_id
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

    Attributes:
        initial_stage: The initial starting stage of the FSM. When the reset() method is
            called the environment is initialised into this stage.
        num_steps: The maximum number of steps the environment allows per episode.
        network: A Network class or derived class describing the connections between
            agents and agents in the environment.
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

        self.initial_stage: StageID = initial_stage

        self._rewards: Dict[PolicyID, Optional[float]] = {}
        self._observations: Dict[PolicyID, Any] = {}
        self._dones: Set[AgentID] = set()
        self._infos: Dict[PolicyID, Dict[str, Any]] = {}

        self._stages: Dict[StageID, FSMStage] = {}
        self.current_stage: Optional[StageID] = None
        self.previous_stage: Optional[StageID] = None

        self.policy_agent_handler_map: Dict[
            PolicyID, Tuple[AgentID, Optional[StageID]]
        ] = {}

        # Register stages via optional class initialiser list
        if stages is not None:
            for stage in stages:
                if stage.stage_id not in self._stages:
                    self._stages[stage.stage_id] = stage

        # Register stages via FSMStage decorator
        for attr_name in dir(self):
            attr = getattr(self, attr_name)
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

    def view(self, agent_id: Optional[AgentID] = None) -> FSMEnvView:
        """Return an immutable view to the environment's public state. TODO"""
        return FSMEnvView(self.current_stage)

    def reset(self) -> Dict[AgentID, Any]:
        """
        Reset the environment and return an initial observation.

        This method resets the step count and the :attr:`network`. This
        includes all the agents in the network.

        Returns:
            A dictionary mapping AgentIDs to observations made by the respective
            agents. It is not required for all agents to make an initial observation.
        """
        self.current_step = 0
        self.current_stage = self.initial_stage

        self._apply_samplers()

        # Reset network and call reset method on all agents in the network.
        # Message samplers should be called here from the respective agent's reset method.
        self.network.reset()
        self.resolve_network()

        # Reset the agents' done status
        self._dones = set()

        # Generate initial observations.
        observations: Dict[AgentID, Any] = {}

        # TODO: enforce at least one acting agent?
        acting_agents = self._stages[self.current_stage].acting_agents

        for agent_id, agent in self.network.agents.items():
            if agent.action_space is not None:
                if acting_agents is None or agent_id in acting_agents:
                    ctx = self.network.context_for(agent_id)
                    observations[agent_id] = agent.encode_observation(ctx)

                self._rewards[agent_id] = None

        return observations

    def step(self, actions: Mapping[AgentID, Any]) -> PhantomEnv.Step:
        """
        Step the simulation forward one step given some set of agent actions.

        Arguments:
            actions: Actions output by the agent policies to be translated into
                messages and passed throughout the network.
        """
        self.current_step += 1

        # Handle the updates due to active/strategic behaviours:
        for agent_id, action in actions.items():
            agent = self.agents[agent_id]

            ctx = self.network.context_for(agent_id)

            messages = agent.decode_action(ctx, action)

            for receiver_id, message in messages:
                self.network.send(agent_id, receiver_id, message)

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

        for agent_id, agent in self.agents.items():
            ctx = self.network.context_for(agent_id)

            dones[agent_id] = agent.is_done(ctx)

            if dones[agent_id]:
                self._dones.add(agent_id)

            acting_agents = self._stages[next_stage].acting_agents
            if acting_agents is None or agent_id in acting_agents:
                observations[agent_id] = agent.encode_observation(ctx)
                infos[agent_id] = agent.collect_infos(ctx)

            rewarded_agents = self._stages[self.current_stage].rewarded_agents
            if (
                rewarded_agents is not None and agent_id in rewarded_agents
            ) or agent_id in acting_agents:
                reward = agent.compute_reward(ctx)
                if reward is not None:
                    rewards[agent_id] = reward

        self._observations.update(observations)
        self._rewards.update(rewards)
        self._infos.update(infos)

        self.previous_stage, self.current_stage = self.current_stage, next_stage

        if self.current_stage is None or self.is_done():
            # This is the terminal stage, return most recent observations, rewards and
            # infos from all agents.
            dones = {"__all__": True}

            for agent_id in self.agents:
                dones[agent_id] = agent_id in self._dones

            return self.Step(
                observations=self._observations,
                rewards=self._rewards,
                dones=dones,
                infos=self._infos,
            )

        # Otherwise not in terminal stage:
        rewards = {agent_id: self._rewards[agent_id] for agent_id in observations}

        return self.Step(observations, rewards, dones, infos)
