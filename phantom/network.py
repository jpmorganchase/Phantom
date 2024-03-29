import warnings
from copy import deepcopy
from itertools import chain, product
from typing import (
    cast,
    Callable,
    Dict,
    Iterable,
    KeysView,
    List,
    Optional,
    Mapping,
    Sequence,
    Tuple,
    Type,
    Union,
)

import numpy as np
import networkx as nx

from .agents import Agent
from .context import Context
from .message import Message, MsgPayload
from .resolvers import BatchResolver, Resolver
from .telemetry import logger
from .types import AgentID
from .views import EnvView


class NetworkError(Exception):
    pass


class Network:
    """P2P messaging network.

    This class is responsible for monitoring connections and tracking
    state/flow between adjacent agents in a peer-to-peer network. The
    underlying representation is based on dictionaries via the NetworkX
    library.

    Arguments:
        agents: Optional list of agents to add to the network.
        resolver: Optional custom resolver to use, by default will use the BatchResolver
            with a `round_limit` of 2.
        connections: Optional initial list of connections to create in the network.
        ignore_connection_errors: If True will not raise errors if an attempt is made
            to send a message along an non-existant connection.
        enforce_msg_payload_checks: If True will ensure that accepted agent types given
            with the `@msg_payload` decorator are enforced.

    Attributes:
        agents: Mapping between IDs and the corresponding agents in the
            network.
        graph: Directed graph modelling the connections between agents.
    """

    def __init__(
        self,
        agents: Optional[Iterable[Agent]] = None,
        resolver: Optional[Resolver] = None,
        connections: Optional[Iterable[Tuple[AgentID, AgentID]]] = None,
        ignore_connection_errors: bool = False,
        enforce_msg_payload_checks: bool = True,
    ) -> None:
        self.graph = nx.DiGraph()
        self.agents: Dict[AgentID, Agent] = {}
        self.resolver = resolver or BatchResolver()
        self.ignore_connection_errors = ignore_connection_errors
        self.enforce_msg_payload_checks = enforce_msg_payload_checks

        self._has_raised_msg_payload_deprecation_warning = False

        if agents is not None:
            self.add_agents(agents)

        if connections is not None:
            for connection in connections:
                self.add_connection(*connection)

    @property
    def agent_ids(self) -> KeysView[AgentID]:
        """Iterator over the IDs of active agents in the network."""
        return self.agents.keys()

    def add_agent(self, agent: Agent) -> None:
        """Add a new agent node to the network.

        Arguments:
            agent: The new agent instance type to be added.
        """
        if agent.id in self.agent_ids:
            raise ValueError(f"Agent with ID = '{agent.id}' already exists.")

        self.agents[agent.id] = agent

        self.graph.add_node(agent.id)

    def add_agents(self, agents: Iterable[Agent]) -> None:
        """Add new agent nodes to the network.

        Arguments:
            agents: An iterable object over the agents to be added.
        """
        for agent in agents:
            self.add_agent(agent)

    def add_connection(self, u: AgentID, v: AgentID) -> None:
        """Connect the agents with IDs :code:`u` and :code:`v`.

        Arguments:
            u: One agent's ID.
            v: The other agent's ID.
        """
        if u not in self.agent_ids:
            raise ValueError(f"Agent with ID = '{u}' does not exist.")

        if v not in self.agent_ids:
            raise ValueError(f"Agent with ID = '{v}' does not exist.")

        self.graph.add_edge(u, v)
        self.graph.add_edge(v, u)

    def add_connections_from(self, ebunch: Iterable[Tuple[AgentID, AgentID]]) -> None:
        """Connect all agent ID pairs in :code:`ebunch`.

        Arguments:
            ebunch: Pairs of vertices to be connected.
        """
        for u, v in ebunch:
            self.add_connection(u, v)

    def add_connections_between(
        self, us: Iterable[AgentID], vs: Iterable[AgentID]
    ) -> None:
        """Connect all agents in :code:`us` to all agents in :code:`vs`.

        Arguments:
            us: Collection of nodes.
            vs: Collection of nodes.
        """
        self.add_connections_from(product(us, vs))

    def add_connections_with_adjmat(
        self, agent_ids: Sequence[AgentID], adjacency_matrix: np.ndarray
    ) -> None:
        """Connect a subset of agents to one another via an adjacency matrix.

        Arguments:
            agent_ids: Sequence of agent IDs that correspond to each dimension of
                the adjacency matrix.
            adjacency_matrix: A square, symmetric, hollow matrix with entries
                in {0, 1}. A value of 1 indicates a connection between two
                agents.
        """
        num_nodes = adjacency_matrix.shape[0]

        if len(agent_ids) != num_nodes:
            raise ValueError(
                "Number of agent IDs doesn't match adjacency matrix dimensions."
            )

        if len(set(adjacency_matrix.shape)) != 1:
            raise ValueError("Adjacency matrix must be square.")

        if not (adjacency_matrix.transpose() == adjacency_matrix).all():
            raise ValueError("Adjacency matrix must be symmetric.")

        if not (np.abs(adjacency_matrix.diagonal() - 0.0) < 1e-5).all():
            raise ValueError("Adjacency matrix must be hollow.")

        for i, agent_id in enumerate(agent_ids):
            self.add_connections_between(
                [agent_id],
                [agent_ids[j] for j in range(num_nodes) if adjacency_matrix[i, j] > 0],
            )

    def reset(self) -> None:
        """Reset the message queues along each edge."""
        self.resolver.reset()

        for agent in self.agents.values():
            agent.reset()

    def subnet_for(self, agent_id: AgentID) -> "Network":
        """Returns a Sub Network associated with a given agent

        Arguments:
            agent_id: The ID of the focal agent
        """
        network = Network.__new__(Network)

        network.graph = self.graph.subgraph(
            chain(
                iter((agent_id,)),
                self.graph.successors(agent_id),
                self.graph.predecessors(agent_id),
            )
        )

        network.agents = {aid: self.agents[aid] for aid in network.graph.nodes}
        network.resolver = deepcopy(self.resolver)
        network.resolver.reset()

        return network

    def context_for(self, agent_id: AgentID, env_view: EnvView) -> Context:
        """Returns the local context for agent :code:`agent_id`.

        Here we define a neighbourhood as being the first-order ego-graph with
        :code:`agent_id` set as the focal node.

        Arguments:
            agent_id: The ID of the focal agent.
        """
        agent_views = {
            neighbour_id: self.agents[neighbour_id].view(agent_id)
            for neighbour_id in self.graph.neighbors(agent_id)
        }

        return Context(self.agents[agent_id], agent_views, env_view)

    def has_edge(self, sender_id: AgentID, receiver_id: AgentID) -> bool:
        """Returns whether two agents are connected.

        Arguments:
            sender_id: The sender ID.
            receiver_id: The receiver ID.
        """
        return (sender_id, receiver_id) in self.graph.edges

    def send(
        self,
        sender_id: AgentID,
        receiver_id: AgentID,
        payload: MsgPayload,
    ) -> None:
        """Send message batches across the network.

        Arguments:
            sender_id: The sender ID.
            receiver_id: The receiver ID.
            payload: The contents of the message.
        """
        if not self.ignore_connection_errors and not self.has_edge(
            sender_id, receiver_id
        ):
            raise NetworkError(f"No connection between {sender_id} and {receiver_id}.")

        if self.enforce_msg_payload_checks:
            self._enforce_payload_checks(sender_id, receiver_id, payload)

        self.resolver.push(Message(sender_id, receiver_id, payload))

    def resolve(self, contexts: Mapping[AgentID, Context]) -> None:
        """Resolve all messages in the network and clear volatile memory.

        Arguments:
            contexts: The current contexts for all agents for the current step.
        """
        logger.log_start_resolving_msgs()

        self.resolver.resolve(self, contexts)
        self.resolver.reset()

    def get_agents_where(self, pred: Callable[[Agent], bool]) -> Dict[AgentID, Agent]:
        """Returns the set of agents in the network that satisfy a predicate.

        Arguments:
            pred: The filter predicate; should return :code:`True` iff the
                agent **should** be included in the set. This method is
                akin to the standard Python function :code:`filter`.
        """
        return {
            agent_id: self.agents[agent_id]
            for agent_id in self.graph.nodes
            if pred(self.agents[agent_id])
        }

    def get_agents_with_type(self, agent_type: Type) -> Dict[AgentID, Agent]:
        """Returns a collection of agents in the network with a given type.

        Arguments:
            agent_type: The class type of agents to include in the set.
        """
        return self.get_agents_where(lambda a: isinstance(a, agent_type))

    def get_agents_without_type(self, agent_type: Type) -> Dict[AgentID, Agent]:
        """Returns a collection of agents in the network without a given type.

        Arguments:
            agent_type: The class type of agents you want to exclude.
        """
        return self.get_agents_where(lambda a: not isinstance(a, agent_type))

    def _enforce_payload_checks(self, sender_id, receiver_id, payload):
        """Internal method."""
        if not hasattr(payload, "_sender_types") or not hasattr(
            payload, "_receiver_types"
        ):
            if isinstance(payload, MsgPayload):
                if not self._has_raised_msg_payload_deprecation_warning:
                    warnings.warn(
                        "MsgPayload type is deprecated. In future, use the @msg_payload decorator",
                        DeprecationWarning,
                    )
                    self._has_raised_msg_payload_deprecation_warning = True
                return

            raise NetworkError(
                f"Message payloads sent across the network must use the 'msg_payload' decorator (bad payload = '{payload}')"
            )

        sender, receiver = self.agents[sender_id], self.agents[receiver_id]

        if (
            payload._sender_types is not None
            and sender.__class__.__name__ not in payload._sender_types
        ):
            raise NetworkError(
                f"Message payload of type '{payload.__class__.__name__}' cannot be sent by agent with type '{sender.__class__.__name__:}' (expected one of {payload._sender_types})"
            )

        if (
            payload._receiver_types is not None
            and receiver.__class__.__name__ not in payload._receiver_types
        ):
            raise NetworkError(
                f"Message payload of type '{payload.__class__.__name__}' cannot be received by agent with type '{receiver.__class__.__name__:}' (expected one of {payload._receiver_types})"
            )

    def __getitem__(self, agent_id: AgentID) -> Agent:
        return self.agents[agent_id]

    def __len__(self) -> int:
        return len(self.graph)


class StochasticNetwork(Network):
    """Stochastic P2P messaging network.

    This class builds on the base Network class but adds the ability to resample the
    connectivity of all connections.

    Arguments:
        agents: Optional list of agents to add to the network.
        resolver: Optional custom resolver to use, by default will use the BatchResolver
            with a `round_limit` of 2.
        connections: Optional initial list of connections to create in the network.
        ignore_connection_errors: If True will not raise errors if an attempt is made
            to send a message along an non-existant connection.
        enforce_msg_payload_checks: If True will ensure that accepted agent types given
            with the `@msg_payload` decorator are enforced.

    Attributes:
        agents: Mapping between IDs and the corresponding agents in the
            network.
        graph: Directed graph modelling the connections between agents.
    """

    def __init__(
        self,
        agents: Optional[Iterable[Agent]] = None,
        resolver: Optional[Resolver] = None,
        connections: Optional[Iterable[Tuple[AgentID, AgentID]]] = None,
        ignore_connection_errors: bool = False,
        enforce_msg_payload_checks: bool = True,
    ) -> None:
        super().__init__(
            agents,
            resolver,
            connections,
            ignore_connection_errors,
            enforce_msg_payload_checks,
        )

        self._base_connections: List[Tuple[AgentID, AgentID, float]] = []

    def add_connection(self, u: AgentID, v: AgentID, rate: float = 1.0) -> None:
        """Connect the agents with IDs :code:`u` and :code:`v`.

        Arguments:
            u: One agent's ID.
            v: The other agent's ID.
            rate: The connectivity of this connection.
        """

        if np.random.random() < rate:
            self.graph.add_edge(u, v)
            self.graph.add_edge(v, u)

        self._base_connections.append((u, v, rate))

    def add_connections_from(
        self,
        ebunch: Iterable[
            Union[Tuple[AgentID, AgentID], Tuple[AgentID, AgentID, float]]
        ],
    ) -> None:
        """Connect all agent ID pairs in :code:`ebunch`.

        Arguments:
            ebunch: Pairs of vertices to be connected.
        """
        for connection in ebunch:
            n = len(connection)

            if n == 2:
                u, v = cast(Tuple[AgentID, AgentID], connection)

                self.add_connection(u, v)

            elif n == 3:
                u, v, r = cast(Tuple[AgentID, AgentID, float], connection)

                self.add_connection(u, v, r)

            else:
                raise ValueError(f"Ill-formatted connection tuple {connection}.")

    def add_connections_between(
        self,
        us: Iterable[AgentID],
        vs: Iterable[AgentID],
        rate: float = 1.0,
    ) -> None:
        """Connect all agents in :code:`us` to all agents in :code:`vs`.

        Arguments:
            us: Collection of nodes.
            vs: Collection of nodes.
            rate: The connectivity given to all connections.
        """

        for u, v in product(us, vs):
            self.add_connection(u, v, rate)

    def resample_connectivity(self) -> None:
        self.graph = nx.DiGraph()

        for agent in self.agents.values():
            self.graph.add_node(agent.id)

        for u, v, rate in self._base_connections:
            if np.random.random() < rate:
                self.graph.add_edge(u, v)
                self.graph.add_edge(v, u)

    def reset(self) -> None:
        self.resample_connectivity()

        Network.reset(self)
