from typing import Any, Dict, Iterator, Mapping, Optional, Union, TYPE_CHECKING

from .types import AgentID

if TYPE_CHECKING:
    from .agents import Agent, View
    from .env import EnvView
    from .network import Network


class Context:
    """Representation of the local neighbourhood around a focal agent node.

    This class is designed to provide context about an agent's local
    neighbourhood. In principle this could be extended to something
    different to a star graph, but for now this is how we define context.

    Arguments:
        agent: Focal node of the ego network.
        views: A collections of view objects, each one associated with an
            adjacent agent.
        subnet: The subgraph representing the ego network of the agent.

    Attributes:
        agent: Focal node of the ego network.
        views: A collections of view objects, each one associated with an
            adjacent agent.
    """

    # TODO: should env view be part of agent views or separate?

    def __init__(
        self,
        agent: "Agent",
        agent_views: Mapping[AgentID, "View"],
        env_view: Optional["EnvView"],
        subnet: "Network",
    ) -> None:
        self.agent = agent
        self.views: Dict[str, Union["View", "EnvView"]] = dict(
            ENV=env_view, **agent_views
        )

        self._subnet = subnet

    @property
    def neighbour_ids(self) -> Iterator[str]:
        """List of IDs of the neighbouring agents."""
        return iter(self.views.keys())

    def __getitem__(self, view_id: str) -> Any:
        return self.views[view_id]

    def __contains__(self, view_id: str) -> bool:
        return view_id in self.views
