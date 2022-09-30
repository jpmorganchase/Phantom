from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from .types import AgentID
from .views import AgentView, EnvView

if TYPE_CHECKING:
    from .agents import Agent


@dataclass(frozen=True)
class Context:
    """
    Representation of the local neighbourhood around a focal agent node.

    This class is designed to provide context about an agent's local neighbourhood. In
    principle this could be extended to something different to a star graph, but for now
    this is how we define context.

    Attributes:
        agent: Focal node of the ego network.
        agent_views: A collection of view objects, each one associated with an adjacent
            agent.
        env_view: A view object associated with the environment.
    """

    agent: "Agent"
    agent_views: Dict[AgentID, Optional[AgentView]]
    env_view: EnvView

    @property
    def neighbour_ids(self) -> List[AgentID]:
        """List of IDs of the neighbouring agents."""
        return list(self.agent_views.keys())

    def __getitem__(self, view_id: str) -> Any:
        if view_id == "ENV":
            return self.env_view

        return self.agent_views[view_id]

    def __contains__(self, view_id: str) -> bool:
        return view_id == "ENV" or view_id in self.agent_views
