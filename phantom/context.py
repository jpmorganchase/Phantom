from typing import Any, Dict, Hashable, List, Optional, Mapping, TYPE_CHECKING

from .types import AgentID
from .view import AgentView, EnvView, View

if TYPE_CHECKING:
    from .agents import Agent
    from .network import Network


class Context:
    """Representation of the local neighbourhood around a focal agent node.

    This class is designed to provide context about an agent's local
    neighbourhood. In principle this could be extended to something
    different to a star graph, but for now this is how we define context.

    Arguments:
        agent: Focal node of the ego network.
        agent_views: A collection of view objects, each one associated with an adjacent
            agent.
        env_views: An optional view object associated with the environment.
        subnet: The subgraph representing the ego network of the agent.

    Attributes:
        agent: Focal node of the ego network.
        views: A collection of view objects, each one associated with an adjacent agent
            or the environment.
    """

    def __init__(
        self,
        agent: "Agent",
        agent_views: Mapping[AgentID, Optional[AgentView]],
        env_view: Optional[EnvView],
        subnet: "Network",
    ) -> None:
        self.agent = agent
        self.views: Dict[Hashable, Optional[View]] = dict(agent_views)
        self.views["ENV"] = env_view

        self._subnet = subnet

    @property
    def neighbour_ids(self) -> List[AgentID]:
        """List of IDs of the neighbouring agents."""
        return list(self.views.keys())

    def __getitem__(self, view_id: str) -> Any:
        return self.views[view_id]

    def __contains__(self, view_id: str) -> bool:
        return view_id in self.views
