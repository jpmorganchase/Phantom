from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Optional,
    Tuple,
)

import mercury as me

from ..fsm import StageID


@dataclass
class AgentStep:
    """
    Describes a step taken by a single agent in an episode.
    """

    observation: Any
    reward: float
    done: bool
    info: Dict[str, Any]
    action: Any
    stage: Optional[StageID]


@dataclass
class Step:
    """
    Describes a step taken in an episode.
    """

    observations: Dict[me.ID, Any]
    rewards: Dict[me.ID, float]
    dones: Dict[me.ID, bool]
    infos: Dict[me.ID, Dict[str, Any]]
    actions: Dict[me.ID, Any]
    stage: Optional[StageID]


@dataclass
class EpisodeTrajectory:
    """
    Class describing all the actions, observations, rewards, infos and dones of a single
    episode.
    """

    observations: List[Dict[me.ID, Any]]
    rewards: List[Dict[me.ID, float]]
    dones: List[Dict[me.ID, bool]]
    infos: List[Dict[me.ID, Dict[str, Any]]]
    actions: List[Dict[me.ID, Any]]
    stages: Optional[List[StageID]]

    def observations_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Any]]:
        """
        Helper method to filter all observations for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter observations for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            return (step_obs.get(agent_id, None) for step_obs in self.observations)
        else:
            return (
                step_obs.get(agent_id, None)
                for step_obs, stage in zip(self.observations, self.stages)
                if stage in stages
            )

    def rewards_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[float]]:
        """
        Helper method to filter all rewards for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter rewards for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            return (step_rewards.get(agent_id, None) for step_rewards in self.rewards)
        else:
            return (
                step_rewards.get(agent_id, None)
                for step_rewards, stage in zip(self.rewards, self.stages)
                if stage in stages
            )

    def dones_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[bool]]:
        """
        Helper method to filter all 'dones' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'dones' for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            return (step_dones.get(agent_id, None) for step_dones in self.dones)
        else:
            return (
                step_dones.get(agent_id, None)
                for step_dones, stage in zip(self.dones, self.stages)
                if stage in stages
            )

    def infos_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Dict[str, Any]]]:
        """
        Helper method to filter all 'infos' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'infos' for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            return (step_infos.get(agent_id, None) for step_infos in self.infos)
        else:
            return (
                step_infos.get(agent_id, None)
                for step_infos, stage in zip(self.infos, self.stages)
                if stage in stages
            )

    def actions_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Any]]:
        """
        Helper method to filter all actions for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter actions for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            return (step_actions.get(agent_id, None) for step_actions in self.actions)
        else:
            return (
                step_actions.get(agent_id, None)
                for step_actions, stage in zip(self.actions, self.stages)
                if stage in stages
            )

    def steps_for_agent(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[AgentStep]:
        """
        Helper method to filter all steps for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter steps for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            indices = range(len(self.actions))
        else:
            indices = (i for i, stage in enumerate(self.stages) if stage in stages)

        return (
            AgentStep(
                self.observations[i].get(agent_id, None),
                self.rewards[i].get(agent_id, None),
                self.dones[i].get(agent_id, None),
                self.infos[i].get(agent_id, None),
                self.actions[i].get(agent_id, None),
                self.stages[i] if self.stages is not None else None,
            )
            for i in indices
        )

    def count_actions(
        self, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """
        Helper method to count the occurances of all actions for all agents.

        Arguments:
            stages: Optionally filter by multiple stages.
        """
        if stages is None:
            filtered_actions = (
                action
                for step_actions in self.actions
                for action in step_actions.values()
            )
        else:
            filtered_actions = (
                action
                for step_actions, stage in zip(self.actions, self.stages)
                for action in step_actions.values()
                if stage in stages
            )

        return Counter(filtered_actions).most_common()

    def count_agent_actions(
        self, agent_id: me.ID, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """
        Helper method to count the occurances of all actions for a single agents.

        Arguments:
            agent_id: The ID of the agent to count actions for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            filtered_actions = (
                step_actions.get(agent_id, None) for step_actions in self.actions
            )
        else:
            filtered_actions = (
                step_actions.get(agent_id, None)
                for step_actions, stage in zip(self.actions, self.stages)
                if stage in stages
            )

        return Counter(filtered_actions).most_common()

    def __getitem__(self, index: int):
        """
        Returns a step for a given index in the episode.
        """
        try:
            return Step(
                self.observations[index],
                self.rewards[index],
                self.dones[index],
                self.infos[index],
                self.actions[index],
                self.stages[index],
            )
        except:
            KeyError(f"Index {index} not valid for trajectory")
