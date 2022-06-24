from abc import ABC
from dataclasses import dataclass

from .types import AgentID


@dataclass(frozen=True)
class View(ABC):
    """
    TODO: improve docstring with rationale
    """


@dataclass(frozen=True)
class AgentView(View):
    """
    Immutable references to public :class:`phantom.Agent` state.
    """

    agent_id: AgentID


@dataclass(frozen=True)
class EnvView(View):
    """
    Immutable references to public :class:`phantom.PhantomEnv`.
    """
