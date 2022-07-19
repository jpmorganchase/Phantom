from abc import ABC
from dataclasses import dataclass

from .types import AgentID


@dataclass(frozen=True)
class View(ABC):
    """
    Base class for the View class hierarchy. Implementations should subclass either
    :class:`AgentView`, :class:`EnvView` or :class:`FSMEnvView`.

    Views are used to share state between agents (and the Env) in a formalised manner
    and in a way that is easier than using request and response messages.

    Views should be created via the calling of the agent/env's :meth:`view()` method.
    Views can be tailored to particular agents, i.e. the view given can depend on the
    agent that the view is being given to.
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
    Immutable references to public :class:`phantom.PhantomEnv` state.
    """

    current_step: int
