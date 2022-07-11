from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np

from ..message import Message
from ..types import AgentID, StageID


@dataclass
class AgentStep:
    """
    Describes a step taken by a single agent in an episode.
    """

    i: int
    observation: Optional[Any]
    reward: Optional[float]
    done: bool
    info: Optional[Dict[str, Any]]
    action: Optional[Any]
    stage: Optional[StageID] = None


@dataclass
class Step:
    """
    Describes a step taken in an episode.
    """

    i: int
    observations: Dict[AgentID, Any]
    rewards: Dict[AgentID, float]
    dones: Dict[AgentID, bool]
    infos: Dict[AgentID, Dict[str, Any]]
    actions: Dict[AgentID, Any]
    messages: Optional[List[Message]] = None
    stage: Optional[StageID] = None


@dataclass
class Rollout:
    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    rollout_params: Dict[str, Any]
    steps: Optional[List[Step]]
    metrics: Dict[str, np.ndarray]

    def observations_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Any]]:
        """
        Helper method to filter all observations for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter observations for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            return (step.observations.get(agent_id, None) for step in self.steps)

        return (
            step.observations.get(agent_id, None)
            for step in self.steps
            if step.stage in stages
        )

    def rewards_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[float]]:
        """
        Helper method to filter all rewards for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter rewards for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            return (step.rewards.get(agent_id, None) for step in self.steps)

        return (
            step.rewards.get(agent_id, None)
            for step in self.steps
            if step.stage in stages
        )

    def dones_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[bool]]:
        """
        Helper method to filter all 'dones' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'dones' for.
            stages: Optionally also filter by multiple stages.
        """

        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            return (step.dones.get(agent_id, None) for step in self.steps)

        return (
            step.dones.get(agent_id, None)
            for step in self.steps
            if step.stage in stages
        )

    def infos_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Dict[str, Any]]]:
        """
        Helper method to filter all 'infos' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'infos' for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            return (step.infos.get(agent_id, None) for step in self.steps)

        return (
            step.infos.get(agent_id, None)
            for step in self.steps
            if step.stage in stages
        )

    def actions_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[Optional[Any]]:
        """
        Helper method to filter all actions for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter actions for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            return (step.actions.get(agent_id, None) for step in self.steps)

        return (
            step.actions.get(agent_id, None)
            for step in self.steps
            if step.stage in stages
        )

    def steps_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> Iterator[AgentStep]:
        """
        Helper method to filter all steps for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter steps for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            steps = self.steps
        else:
            steps = [step for step in self.steps if step.stage in stages]

        return (
            AgentStep(
                step.i,
                step.observations.get(agent_id, None),
                step.rewards.get(agent_id, None),
                step.dones[agent_id],
                step.infos.get(agent_id, None),
                step.actions.get(agent_id, None),
                step.stage,
            )
            for step in steps
        )

    def count_actions(
        self, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """
        Helper method to count the occurances of all actions for all agents.

        Arguments:
            stages: Optionally filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            filtered_actions = (
                action for step in self.steps for action in step.actions.values()
            )
        else:
            filtered_actions = (
                action
                for step in self.steps
                for action in step.actions.values()
                if step.stage in stages
            )

        return Counter(filtered_actions).most_common()

    def count_agent_actions(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """
        Helper method to count the occurances of all actions for a single agents.

        Arguments:
            agent_id: The ID of the agent to count actions for.
            stages: Optionally also filter by multiple stages.
        """
        if self.steps is None:
            raise ValueError("Trajectory not stored in rollout")

        if stages is None:
            filtered_actions = (step.actions.get(agent_id, None) for step in self.steps)
        else:
            filtered_actions = (
                step.actions.get(agent_id, None)
                for step in self.steps
                if step.stage in stages
            )

        return Counter(filtered_actions).most_common()

    def __getitem__(self, index: int):
        """
        Returns a step for a given index in the episode.
        """
        if self.steps is None:
            raise KeyError("Trajectory not stored in rollout")

        try:
            return self.steps[index]
        except KeyError:
            KeyError(f"Index {index} not valid for trajectory")
