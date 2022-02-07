from itertools import chain, product
from typing import (
    cast,
    Any,
    Callable,
    Dict,
    Iterable,
    Iterator,
    KeysView,
    List,
    Mapping,
    Sequence,
    Set,
    Tuple,
    Type,
    Union,
)

import numpy as np
import networkx as nx

from . import linalg
from .actors import Actor, View
from .core import ID
from .message import Payload
from .resolvers import Resolver


IntoPath = Union[str, Tuple[str, ...], "Path"]


class Path:
    """Utility class for manipulating actor paths.

    Arguments:
        path: Input to be parsed into a Path.

    Attributes:
        parts: The path split by :code:`Path.SPLITTER`.
    """

    SPLITTER = "::"

    def __init__(self, path: IntoPath) -> None:
        if isinstance(path, str):
            self.parts = tuple(path.split(Path.SPLITTER))

        elif isinstance(path, tuple):
            self.parts = path

        elif isinstance(path, Path):
            self.parts = path.parts

        else:
            raise ValueError(f"Unknown path type {type(path)}.")

    def __getitem__(self, index: int) -> str:
        return self.parts[index]

    def __iter__(self):
        return iter(self.parts)

    def __len__(self) -> int:
        return len(self.parts)

    def __repr__(self) -> str:
        return self.SPLITTER.join(list(self.parts))

    def __hash__(self) -> int:
        return hash(self.parts)

    def __eq__(self, other: "Path") -> bool:
        return self.parts == other.parts


class Groups:
    """Utility class for handling actor ID groups.

    This class is optimised for compute over memory by storing _everything_
    rather than relying on recursive implementations. For a flat structure,
    this will make no difference, but when we have subgroups this will ensure
    we have amortised constant-time access to the actor IDs in groups.

    Attributes:
        groups: Mapping between group paths and their contained actor IDs.
        subgroups: Mapping between group paths and their contained sub-group
            Paths.
    """

    def __init__(self) -> None:
        self.groups: Dict[int, Set[ID]] = dict()
        self.subgroups: Dict[int, Set[Path]] = dict()

    def is_group(self, path: IntoPath) -> bool:
        """Return true iff the path corresponds to a defined group.

        Arguments:
            path: The (possible) group path.
        """
        return Path(path) in self.groups

    def add(self, path: IntoPath, aids: Iterable[ID]) -> None:
        """Add a new group to the set of groups.

        Arguments:
            path: The new group's path.
            aids: An iterable object with entries given by actor IDs.
        """
        path = Path(path)

        # Check if the specified path is already a defined group:
        if path in self.groups:
            raise KeyError(f"Group already exists with path {path}.")

        # Check if the specified path coincides with a specified actor:
        g = self.groups.get(Path(path[:-1]), None)

        if g is not None and path[-1] in g:
            raise KeyError(f"Specified path, {path}, maps to an actor.")

        # Iterate over all sub-paths, going from root to leaf:
        for i in range(1, len(path) + 1):
            subpath = Path(path[:i])

            g = self.groups.get(subpath, None)

            # If this subpath is not a current group, then add it:
            if g is None:
                self.groups[subpath] = set(aids)
                self.subgroups[subpath] = set(
                    Path(path[:j]) for j in range(i + 1, len(path) + 1)
                )

            # If this subpath is already a defined group, then update it:
            else:
                g.update(aids)

                self.subgroups[subpath].update(
                    Path(path[:j]) for j in range(i + 1, len(path) + 1)
                )

    def assign(self, path: IntoPath, aids: Iterable[ID]) -> None:
        """Assig a set of actor IDs to a given group.

        Arguments:
            path: The existing group's path.
            aids: An iterable object with entries given by actor IDs.
        """
        path = Path(path)

        # Check if the specified path is already a defined group:
        if path not in self.groups:
            raise KeyError(f"No group exists with path {path}.")

        # Iterate over all sub-paths, going from root to leaf:
        for i in range(1, len(path) + 1):
            subpath = Path(path[:i])

            self.groups[path].update(aids)

    def remove(self, path: IntoPath) -> None:
        """Remove an existing group.

        Arguments:
            path: The existing group's path.
        """
        path = Path(path)

        # Check if the specified path is already a defined group:
        if path not in self.groups:
            raise KeyError(f"No group exists with path {path}.")

        g = self.groups[path]

        # Iterate over all sub-paths, going from leaf to root:
        for i in range(len(path) - 1, 0, -1):
            sp = Path(path[:i])
            sh = sp

            self.groups[sh].difference_update(g)
            self.subgroups[sh].remove(path)

        for sp in self.subgroups[path]:
            del self.groups[sp]
            del self.subgroups[sp]

        del self.groups[path]
        del self.subgroups[path]

    def clear(self) -> None:
        """Clear the set of groups completely."""
        self.groups.clear()
        self.subgroups.clear()

    def __str__(self) -> str:
        return str(self.groups)

    def __getitem__(self, path: IntoPath):
        path = Path(path)

        if path not in self.groups:
            raise KeyError(f"No group exists with path {path}.")

        return self.groups[path]


class NetworkError(Exception):
    pass


class Network:
    """P2P messaging network.

    This class is responsible for monitoring connections and tracking
    state/flow between adjacent actors in a peer-to-peer network. The
    underlying representation is based on dictionaries via the NetworkX
    library.

    Arguments:
        actors: Optional list of actors to add to the network.

    Attributes:
        actors: Mapping between IDs and the corresponding actors in the
            network.
        graph: Directed graph modelling the connections between actors.
    """

    class Context:
        """Representation of the local neighbourhood around a focal actor node.

        This class is designed to provide context about an actor's local
        neighbourhood. In principle this could be extended to something
        different to a star graph, but for now this is how we define context.

        Arguments:
            actor: Focal node of the ego network.
            views: A collections of view objects, each one associated with an
                adjacent actor.
            subnet: The subgraph representing the ego network of the actor.

        Attributes:
            actor: Focal node of the ego network.
            views: A collections of view objects, each one associated with an
                adjacent actor.
        """

        def __init__(
            self, actor: Actor, views: Mapping[ID, View], subnet: "Network"
        ) -> None:
            self.actor: Actor = actor
            self.views: Mapping[ID, View] = views

            self._subnet: "Network" = subnet

        @property
        def neighbour_ids(self) -> Iterator[ID]:
            """List of IDs of the neighbouring actors."""
            return iter(self.views.keys())

        def __getitem__(self, actor_id: ID) -> Any:
            return self.views[actor_id]

        def __contains__(self, actor_id: ID) -> bool:
            return actor_id in self.views

    def __init__(self, resolver: Resolver, actors: List[Actor] = list()) -> None:
        self.resolver = resolver

        self.graph: nx.DiGraph = nx.DiGraph()
        self.actors: Dict[ID, Actor] = dict()

        self.groups = Groups()

        self.add_actors(actors)

    @property
    def actor_ids(self) -> KeysView[ID]:
        """Iterator over the IDs of active actors in the network."""
        return self.actors.keys()

    def add_actor(self, actor: Actor) -> None:
        """Add a new actor node to the network.

        Arguments:
            actor: The new actor instance type to be added.
        """
        self.actors[actor.id] = actor

        self.graph.add_node(actor.id)

    def add_actors(self, actors: Iterable[Actor]) -> None:
        """Add new actor nodes to the network.

        Arguments:
            actors: An iterable object over the actors to be added.
        """
        for actor in actors:
            self.add_actor(actor)

    def add_connection(self, u: ID, v: ID, **attr: Any) -> None:
        """Connect the actors with IDs :code:`u` and :code:`v`.

        Arguments:
            u: One actor's ID.
            v: The other actor's ID.
        """
        self.graph.add_edge(u, v, *attr)
        self.graph.add_edge(v, u, *attr)

    def add_connections_from(
        self, ebunch: Iterable[Tuple[ID, ID]], **attrs: Any
    ) -> None:
        """Connect all actor ID pairs in :code:`ebunch`.

        Arguments:
            ebunch: Pairs of vertices to be connected.
        """
        for u, v in ebunch:
            self.add_connection(u, v, **attrs)

    def add_connections_between(
        self, us: Iterable[ID], vs: Iterable[ID], **attrs: Any
    ) -> None:
        """Connect all actors in :code:`us` to all actors in :code:`vs`.

        Arguments:
            us: Collection of nodes.
            vs: Collection of nodes.
        """
        self.add_connections_from(product(us, vs), **attrs)

    def add_connections_with_adjmat(
        self, aids: Sequence[ID], adjacency_matrix: np.ndarray, **attrs: Any
    ) -> None:
        """Connect a subset of actors to one another via an adjacency matrix.

        Arguments:
            aids: Sequence of actor IDs that correspond to each dimension of
                the adjacency matrix.
            adjacency_matrix: A square, symmetric, hollow matrix with entries
                in {0, 1}. A value of 1 indicates a connection between two
                actors.
        """
        num_nodes = adjacency_matrix.shape[0]

        if len(aids) != num_nodes:
            raise ValueError(
                "Number of actor IDs doesn't match adjacency" " matrix dimensions."
            )

        if not linalg.is_square(adjacency_matrix):
            raise ValueError("Adjacency matrix must be square.")

        if not linalg.is_symmetric(adjacency_matrix):
            raise ValueError("Adjacency matrix must be symmetric.")

        if not linalg.is_hollow(adjacency_matrix):
            raise ValueError("Adjacency matrix must be hollow.")

        for i, aid in enumerate(aids):
            self.add_connections_between(
                [aid],
                [aids[j] for j in range(num_nodes) if adjacency_matrix[i, j] > 0],
                **attrs,
            )

    def reset(self) -> None:
        """Reset the message queues along each edge."""
        self.resolver.reset()

        for actor in self.actors.values():
            actor.reset()

    def subnet_for(self, actor_id: ID) -> "Network":
        """Returns a Sub Network associated with a given actor

        Arguments:
            actor_id: The ID of the focal actor
        """
        network: Network = Network.__new__(Network)

        network.graph = self.graph.subgraph(
            chain(
                iter((actor_id,)),
                self.graph.successors(actor_id),
                self.graph.predecessors(actor_id),
            )
        )

        network.actors = {aid: self.actors[aid] for aid in network.graph.nodes}
        network.resolver = self.resolver

        return network

    def context_for(self, actor_id: ID) -> "Network.Context":
        """Returns the local context for actor :code:`actor_id`.

        Here we define a neighbourhood as being the first-order ego-graph with
        :code:`actor_id` set as the focal node.

        Arguments:
            actor_id: The ID of the focal actor.
        """
        subnet = self.subnet_for(actor_id)
        views = {
            neighbour_id: self.actors[neighbour_id].view(actor_id)
            for neighbour_id in subnet.graph.neighbors(actor_id)
        }

        return Network.Context(self.actors[actor_id], views, subnet)

    def send(self, all_payloads: Mapping[ID, Mapping[ID, Iterable[Payload]]]) -> None:
        """Send payload batches across the network.

        Arguments:
            payloads: Mapping from senders, to receivers, to payloads.
        """
        for sender_id, payload_map in all_payloads.items():
            for receiver_id, payloads in payload_map.items():
                if (sender_id, receiver_id) not in self.graph.edges:
                    raise NetworkError(
                        f"No connection between {self.actors[sender_id]} "
                        f"and {self.actors[receiver_id]}."
                    )

                self.resolver.push(sender_id, receiver_id, payloads)

    def send_from(
        self, sender_id: ID, payload_map: Mapping[ID, Iterable[Payload]]
    ) -> None:
        """Send payloads across the network from a given actor.

        Arguments:
            payloads: Mapping from receiver IDs to payloads.
        """
        for receiver_id, payloads in payload_map.items():
            if (sender_id, receiver_id) not in self.graph.edges:
                raise NetworkError(
                    f"No connection between {self.actors[sender_id]} "
                    f"and {self.actors[receiver_id]}."
                )

            self.resolver.push(sender_id, receiver_id, payloads)

    def send_to(
        self, receiver_id: ID, payload_map: Mapping[ID, Iterable[Payload]]
    ) -> None:
        """Send payloads across the network to a given actor.

        Arguments:
            payloads: Mapping from sender IDs to payloads.
        """
        for sender_id, payloads in payload_map.items():
            if (sender_id, receiver_id) not in self.graph.edges:
                raise NetworkError(
                    f"No connection between {self.actors[sender_id]} "
                    f"and {self.actors[receiver_id]}."
                )

            self.resolver.push(sender_id, receiver_id, payloads)

    def resolve(self) -> None:
        """Resolve all messages in the network and clear volatile memory.

        This process does three things in a strictly sequential manner:
            1. Execute the chosen resolver to handle messages in the network.
            2. Clear all edges of any remaining messages instances.
        """
        ctx_map = {actor_id: self.context_for(actor_id) for actor_id in self.actors}

        for ctx in ctx_map.values():
            ctx.actor.pre_resolution(ctx)

        self.resolver.resolve(self)

        for ctx in ctx_map.values():
            ctx.actor.post_resolution(ctx)

        self.resolver.reset()

    def get_actors_where(self, pred: Callable[[Actor], bool]) -> Dict[ID, Actor]:
        """Returns the set of actors in the network that satisfy a predicate.

        Arguments:
            pred: The filter predicate; should return :code:`True` iff the
                actor **should** be included in the set. This method is
                akin to the standard Python function :code:`filter`.
        """
        return {
            actor_id: self.actors[actor_id]
            for actor_id in self.graph.nodes
            if pred(self.actors[actor_id])
        }

    def get_actors_with_type(self, cls: Type) -> Dict[ID, Actor]:
        """Returns a collection of actors in the network with a given type.

        Arguments:
            cls: The class type of actors to include in the set.
        """
        return self.get_actors_where(lambda a: isinstance(a, cls))

    def get_actors_without_type(self, cls: Type) -> Dict[ID, Actor]:
        """Returns a collection of actors in the network without a given type.

        Arguments:
            cls: The class type of actors you want to exclude.
        """
        return self.get_actors_where(lambda a: not isinstance(a, cls))

    def __getitem__(self, k: ID) -> Actor:
        return self.actors[k]

    def __len__(self) -> int:
        return len(self.graph)


class StochasticNetwork(Network):
    """Stochastic P2P messaging network.

    This class builds on the base Mercury Network class but adds the ability to
    resample the connectivity of all connections.

    Arguments:
        actors: Optional list of actors to add to the network.

    Attributes:
        actors: Mapping between IDs and the corresponding actors in the
            network.
        graph: Directed graph modelling the connections between actors.
    """

    def __init__(self, resolver: Resolver, actors: List[Actor] = list()) -> None:
        Network.__init__(self, resolver, actors)

        self._base_connections: List[Tuple[ID, ID, float, Any]] = list()

    def add_connection(self, u: ID, v: ID, rate: float = 1.0, **attr: Any) -> None:
        """Connect the actors with IDs :code:`u` and :code:`v`.

        Arguments:
            u: One actor's ID.
            v: The other actor's ID.
            rate: The connectivity of this connection.
        """

        if np.random.random() < rate:
            self.graph.add_edge(u, v, *attr)
            self.graph.add_edge(v, u, *attr)

        self._base_connections.append((u, v, rate, attr))

    def add_connections_from(
        self,
        ebunch: Iterable[Union[Tuple[ID, ID], Tuple[ID, ID, float]]],
        **attrs: Any,
    ) -> None:
        """Connect all actor ID pairs in :code:`ebunch`.

        Arguments:
            ebunch: Pairs of vertices to be connected.
        """
        for connection in ebunch:
            n = len(connection)

            if n == 2:
                u, v = cast(Tuple[ID, ID], connection)

                self.add_connection(u, v, **attrs)

            elif n == 3:
                u, v, r = cast(Tuple[ID, ID, float], connection)

                self.add_connection(u, v, r, **attrs)

            else:
                raise ValueError(
                    "Ill-formatted connection tuple {}.".format(connection)
                )

    def add_connections_between(
        self,
        us: Iterable[ID],
        vs: Iterable[ID],
        rate: float = 1.0,
        **attrs: Any,
    ) -> None:
        """Connect all actors in :code:`us` to all actors in :code:`vs`.

        Arguments:
            us: Collection of nodes.
            vs: Collection of nodes.
            rate: The connectivity given to all connections.
        """

        for u, v in product(us, vs):
            self.add_connection(u, v, rate, **attrs)

    def resample_connectivity(self) -> None:
        self.graph = nx.DiGraph()

        for actor in self.actors.values():
            self.graph.add_node(actor.id)

        for u, v, rate, attrs in self._base_connections:
            if np.random.random() < rate:
                self.graph.add_edge(u, v, *attrs)
                self.graph.add_edge(v, u, *attrs)

    def reset(self) -> None:
        self.resample_connectivity()

        Network.reset(self)
