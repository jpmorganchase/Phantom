import io
import json
from collections import Counter
from dataclasses import asdict, dataclass
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Mapping,
    Optional,
    Tuple,
)

import numpy as np
import pandas as pd

from ..message import Message
from ..types import AgentID, StageID


@dataclass(frozen=True)
class AgentStep:
    """Describes a step taken by a single agent in an episode."""

    i: int
    observation: Optional[Any]
    reward: Optional[float]
    terminated: bool
    truncated: bool
    info: Optional[Dict[str, Any]]
    action: Optional[Any]
    stage: Optional[StageID] = None


@dataclass(frozen=True)
class Step:
    """Describes a step taken in an episode."""

    i: int
    observations: Dict[AgentID, Any]
    rewards: Dict[AgentID, float]
    terminations: Dict[AgentID, bool]
    truncations: Dict[AgentID, bool]
    infos: Dict[AgentID, Dict[str, Any]]
    actions: Dict[AgentID, Any]
    messages: Optional[List[Message]] = None
    stage: Optional[StageID] = None


@dataclass(frozen=True)
class Rollout:
    rollout_id: int
    repeat_id: int
    env_config: Mapping[str, Any]
    rollout_params: Dict[str, Any]
    steps: List[Step]
    metrics: Dict[str, np.ndarray]

    def observations_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[Any]]:
        """Helper method to filter all observations for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter observations for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.observations.get(agent_id, None)
            for step in self.steps
            if (drop_nones is False or agent_id in step.observations)
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def rewards_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[float]]:
        """Helper method to filter all rewards for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter rewards for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.rewards.get(agent_id, None)
            for step in self.steps
            if (
                drop_nones is False
                or (agent_id in step.rewards and step.rewards[agent_id] is not None)
            )
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def terminations_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[bool]]:
        """Helper method to filter all 'terminations' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'terminations' for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.terminations.get(agent_id, None)
            for step in self.steps
            if (drop_nones is False or agent_id in step.terminations)
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def truncations_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[bool]]:
        """Helper method to filter all 'truncations' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'truncations' for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.truncations.get(agent_id, None)
            for step in self.steps
            if (drop_nones is False or agent_id in step.truncations)
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def infos_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[Dict[str, Any]]]:
        """Helper method to filter all 'infos' for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter 'infos' for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.infos.get(agent_id, None)
            for step in self.steps
            if (drop_nones is False or agent_id in step.infos)
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def actions_for_agent(
        self,
        agent_id: AgentID,
        drop_nones: bool = False,
        stages: Optional[Iterable[StageID]] = None,
    ) -> List[Optional[Any]]:
        """Helper method to filter all actions for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter actions for.
            drop_nones: Drops any None values if True.
            stages: Optionally also filter by multiple stages.
        """
        return [
            step.actions.get(agent_id, None)
            for step in self.steps
            if (drop_nones is False or agent_id in step.actions)
            and (stages is None or (step.stage is not None and step.stage in stages))
        ]

    def steps_for_agent(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> List[AgentStep]:
        """Helper method to filter all steps for a single agent.

        Arguments:
            agent_id: The ID of the agent to filter steps for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            steps = self.steps
        else:
            steps = [
                step
                for step in self.steps
                if (step.stage is not None and step.stage in stages)
            ]

        return [
            AgentStep(
                step.i,
                step.observations.get(agent_id, None),
                step.rewards.get(agent_id, None),
                step.terminations.get(agent_id, False),
                step.truncations.get(agent_id, False),
                step.infos.get(agent_id, None),
                step.actions.get(agent_id, None),
                step.stage,
            )
            for step in steps
        ]

    def count_actions(
        self, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """Helper method to count the occurances of all actions for all agents.

        Arguments:
            stages: Optionally filter by multiple stages.
        """
        if stages is None:
            filtered_actions = (
                action for step in self.steps for action in step.actions.values()
            )
        else:
            filtered_actions = (
                action
                for step in self.steps
                for action in step.actions.values()
                if (step.stage is not None and step.stage in stages)
            )

        return Counter(filtered_actions).most_common()

    def count_agent_actions(
        self, agent_id: AgentID, stages: Optional[Iterable[StageID]] = None
    ) -> List[Tuple[Any, int]]:
        """Helper method to count the occurances of all actions for a single agents.

        Arguments:
            agent_id: The ID of the agent to count actions for.
            stages: Optionally also filter by multiple stages.
        """
        if stages is None:
            filtered_actions = (step.actions.get(agent_id, None) for step in self.steps)
        else:
            filtered_actions = (
                step.actions.get(agent_id, None)
                for step in self.steps
                if (step.stage is not None and step.stage in stages)
            )

        return Counter(filtered_actions).most_common()

    def __getitem__(self, index: int):
        """Returns a step for a given index in the episode."""
        try:
            return self.steps[index]
        except KeyError:
            raise KeyError(f"Index {index} not valid for trajectory")


def rollouts_to_dataframe(
    rollouts: Iterable[Rollout],
    avg_over_repeats: bool = True,
    index_value_precision: Optional[int] = None,
) -> pd.DataFrame:
    """
    Converts a list of Rollouts into a MultiIndex DataFrame with rollout params as the
    indexes and metrics as the columns.

    Arguments:
        rollouts: The list/iterator of Phantom Rollout objects to use.
        avg_over_repeats: If True will average all metric values over each set of
            repeats. This is very useful for reducing the overall data size if
            individual rollouts are not required.
        index_value_precision: If given will round the index values to the given
            precision and convert to strings. This can be useful for avoiding floating
            point inaccuracies when indexing (eg. 2.0 != 2.000000001).

    Returns:
        A Pandas DataFrame containing the results.
    """

    # Consume iterator (if applicable), throw away everything except params and metrics
    rollouts2 = [(rollout.rollout_params, rollout.metrics) for rollout in rollouts]

    index_cols = list(rollouts2[0][0].keys())

    df = pd.DataFrame([{**params, **metrics} for params, metrics in rollouts2])

    if index_value_precision is not None:
        for col in index_cols:
            df[col] = df[col].round(index_value_precision).astype(str)

    if len(index_cols) > 0:
        if avg_over_repeats:
            df = df.groupby(index_cols).mean().reset_index()

        df = df.set_index(index_cols)

    return df


def rollouts_to_jsonl(
    rollouts: Iterable[Rollout],
    file_obj: io.TextIOBase,
    human_readable: bool = False,
) -> None:
    """
    Writes multiple rollouts to a file using the JSONL (JSON Lines) format.

    Arguments:
        rollouts: The list/iterator of Phantom Rollout objects to use.
        file_obj: A writable file object to output to.
        human_readable: If True the output will be 'pretty printed'.
    """
    for rollout in rollouts:
        json.dump(
            rollout,
            file_obj,
            indent=2 if human_readable else None,
            cls=RolloutJSONEncoder,
        )
        file_obj.write("\n")
        file_obj.flush()


class RolloutJSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.number):
            return int(o)
        if isinstance(o, Rollout):
            return asdict(o)
        if isinstance(o, Step):
            return asdict(o)

        return json.JSONEncoder.default(self, o)
