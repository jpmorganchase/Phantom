from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Optional,
    Tuple,
    TypeVar,
    Union,
)

import gym

from .agents import Agent, AgentID
from .env import PhantomEnv
from .policy import Policy


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")


class SingleAgentEnvAdapter(gym.Env):
    def __init__(
        self,
        env: PhantomEnv,
        agent_id: AgentID,
        other_policies: Mapping[AgentID, Tuple[Policy, Mapping[str, Any]]],
        config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        config = config or {}

        self._env = env(**config)

        # Check selected agent exists
        assert agent_id in self._env.agent_ids

        # Check selected agent isn't given policy
        assert agent_id not in other_policies.keys()

        # Check all acting agents have policies
        acting_agents = set(
            aid for aid, a in self._env.agents.items() if a.takes_actions()
        )
        policies = set(list(other_policies.keys()) + [agent_id])

        assert acting_agents == policies

        self._agent_id = agent_id

        self._other_policies = {
            agent_id: policy_class(
                self._env[agent_id].observation_space,
                self._env[agent_id].action_space,
                policy_config,
            )
            for agent_id, (policy_class, policy_config) in other_policies.items()
        }

        self._actions: Dict[AgentID, Any] = {}
        self._observations: Dict[AgentID, Any] = {}

        super().__init__()

    @property
    def active_agent(self) -> AgentID:
        return self._agent_id

    @property
    def agents(self) -> Dict[AgentID, Agent]:
        """Return a mapping of agent IDs to agents in the environment."""
        return self._env.agents

    @property
    def agent_ids(self) -> List[AgentID]:
        """Return a list of the IDs of the agents in the environment."""
        return self._env.agent_ids

    @property
    def n_agents(self) -> int:
        """Return the number of agents in the environment."""
        return self._env.n_agents

    @property
    def current_step(self) -> int:
        """Return the current step of the environment."""
        return self._env.current_step

    @property
    def action_space(self) -> gym.Space:
        """Return the action space of the selected env agent."""
        return self._env[self._agent_id].action_space

    @property
    def observation_space(self) -> gym.Space:
        """Return the observation space of the selected env agent."""
        return self._env[self._agent_id].observation_space

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, dict]:
        """Run one timestep of the environment's dynamics.
        When end of episode is reached, you are responsible for calling :meth:`reset` to reset this environment's state.
        Accepts an action and returns a tuple `(observation, reward, done, info)`.
        Args:
            action (ActType): an action provided by the agent
        Returns:
            observation (object): this will be an element of the environment's :attr:`observation_space`.
                This may, for instance, be a numpy array containing the positions and velocities of certain objects.
            reward (float): The amount of reward returned as a result of taking the action.
            done (bool): A boolean value for if the episode has ended, in which case further :meth:`step` calls will return undefined results.
                A done signal may be emitted for different reasons: Maybe the task underlying the environment was solved successfully,
                a certain timelimit was exceeded, or the physics simulation has entered an invalid state.
            info (dictionary): A dictionary that may contain additional information regarding the reason for a ``done`` signal.
                `info` contains auxiliary diagnostic information (helpful for debugging, learning, and logging).
                This might, for instance, contain: metrics that describe the agent's performance state, variables that are
                hidden from observations, information that distinguishes truncation and termination or individual reward terms
                that are combined to produce the total reward
        """

        self._actions = {
            agent_id: policy.compute_action(self._observations[agent_id])
            for agent_id, policy in self._other_policies.items()
        }

        self._actions[self._agent_id] = action

        step = self._env.step(self._actions)

        self._observations = step.observations

        return (
            step.observations[self._agent_id],
            step.rewards[self._agent_id],
            step.dones[self._agent_id],
            step.infos[self._agent_id],
        )

    def reset(self) -> Union[ObsType, Tuple[ObsType, dict]]:
        """Resets the environment to an initial state and returns an initial
        observation.
        Note that this function should not reset the environment's random
        number generator(s); random variables in the environment's state should
        be sampled independently between multiple calls to `reset()`. In other
        words, each call of `reset()` should yield an environment suitable for
        a new episode, independent of previous episodes.
        Returns:
            observation (object): the initial observation.
        """

        # TODO: update function interface when gym version is updated

        self._observations = self._env.reset()

        return self._observations[self._agent_id]
