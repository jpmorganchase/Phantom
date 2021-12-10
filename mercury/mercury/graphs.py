from typing import (
    overload,
    Any,
    Dict,
    Hashable,
    Iterable,
    Iterator,
    ItemsView,
    KeysView,
    MutableMapping,
    Optional,
    Set,
    Tuple,
    Type,
    TypeVar,
    Union,
    ValuesView,
)
from collections import defaultdict
from copy import deepcopy
from dataclasses import dataclass
from itertools import chain


#################################################
# Type helpers for DiGraph
#################################################
NodeType = TypeVar("NodeType", bound=Hashable)

_AttributeValue = Any
_EdgeValue = MutableMapping[str, _AttributeValue]
_NodeValue = MutableMapping[NodeType, _EdgeValue]
_GraphValue = Union[_NodeValue[NodeType], _EdgeValue, _AttributeValue]

_EdgeAttributeKey = Tuple[NodeType, NodeType, str]
_EdgeKey = Tuple[NodeType, NodeType]
_NodeKey = Type[NodeType]
_GraphKey = Union[_NodeKey[NodeType], _EdgeKey[NodeType], _EdgeAttributeKey[NodeType]]

_GraphDataType = MutableMapping[NodeType, _NodeValue[NodeType]]
_RaiseKeyError = object()


class DiGraph(_GraphDataType[NodeType]):
    """Directional Graph class behaving like a Pythond dictionary but allowing more
    graph oriented actions such as accessing edge data, ...

    This data structure offers more flexibility than a classic dict of dicts of dicts used to store the graph data.
    The structure can be accessed with different keys to facilitate the interface with the graph. It can also be accessed
    the old way as well.
        - Access the neighbors and edges associated with a single node
            Example:
                graph: DiGraph := A --{'att1': True}--> B
                graph[A] -> {B: {'att1': True}}
        - Access the data on a given edge
            Example:
                graph: DiGraph := A --{'att1': True}--> B
                graph[(A, B)] -> {'att1': True}
                graph[A][B] -> {'att1': True}
        - Access a specific attribute on a given edge
            Example:
                graph: DiGraph := A --{'att1': True}--> B
                graph[(A, B, 'att1')] -> True
                graph[A][B]['att1'] -> True

    All the methods available on a dictionary are also available on this data structure and supports the 3 different types of keys.
    i.e Node, Edge, EdgeAttribute

    Some graph helper functions have also been implemented such as:
        - :code:`add_node`
        - :code:`add_edge`
        - :code:`successors`
        - :code:`predecessors`
        - :code:`neighbors`
        - :code:`subgraph`
        - :code:`subgraph_for`

    """

    @dataclass
    class EdgeCollection(Iterable[Tuple[NodeType, NodeType]]):
        edges: Tuple[Tuple[NodeType, NodeType], _EdgeValue]

        def __iter__(self) -> Iterator[Tuple[NodeType, NodeType]]:
            return (t[0] for t in self.edges)

        def data(
            self,
        ) -> Iterator[Tuple[Tuple[NodeType, NodeType], _EdgeValue]]:
            return ((u, v, d) for (u, v), d in self.edges)

    def __init__(
        self,
        incoming_data=Union[
            Iterable[NodeType],
            MutableMapping[_GraphKey[NodeType], _GraphValue[NodeType]],
        ],
    ):
        self._init()

        if isinstance(incoming_data, (list, tuple)):
            self.fromkeys(incoming_data)
        elif isinstance(incoming_data, dict):
            self.update(incoming_data)

    def _init(self):
        self._data: _GraphDataType[NodeType] = defaultdict(self._edge_factory)
        self._successors: Dict[NodeType, Set[NodeType]] = defaultdict(set)
        self._predecessors: Dict[NodeType, Set[NodeType]] = defaultdict(set)

    @staticmethod
    def _edge_factory():
        return defaultdict(dict)

    def __getstate__(self):
        return deepcopy(self._data)

    def __setstate__(self, state):
        self._init()
        self._data = state
        self._successors.update({k: set(v.keys()) for k, v in self._data.items()})
        _ = [
            self._predecessors[v].add(k)
            for k, vs in self._successors.items()
            for v in vs
        ]

    ############################################
    # overload methods for type hints
    ############################################

    @overload
    def __getitem__(self, k: _NodeKey[NodeType]) -> _NodeValue[NodeType]:
        ...

    @overload
    def __getitem__(self, k: _EdgeKey[NodeType]) -> _EdgeValue:
        ...

    @overload
    def __getitem__(self, k: _EdgeAttributeKey[NodeType]) -> _AttributeValue:
        ...

    @overload
    def get(
        self, k: _NodeKey[NodeType], default: Optional[_NodeValue[NodeType]] = None
    ) -> Optional[_NodeValue[NodeType]]:
        ...

    @overload
    def get(
        self, k: _EdgeKey[NodeType], default: Optional[_EdgeValue] = None
    ) -> Optional[_EdgeValue]:
        ...

    @overload
    def get(
        self,
        k: _EdgeAttributeKey[NodeType],
        default: Optional[_AttributeValue] = None,
    ) -> Optional[_AttributeValue]:
        ...

    @overload
    def pop(
        self,
        k: _NodeKey[NodeType],
        default: Union[_NodeValue[NodeType], object] = _RaiseKeyError,
    ) -> _NodeValue[NodeType]:
        ...

    @overload
    def pop(
        self,
        k: _EdgeKey[NodeType],
        default: Union[_EdgeValue, object] = _RaiseKeyError,
    ) -> _EdgeValue:
        ...

    @overload
    def pop(
        self,
        k: _EdgeAttributeKey[NodeType],
        default: Union[_AttributeValue, object] = _RaiseKeyError,
    ) -> _AttributeValue:
        ...

    @overload
    def __setitem__(self, k: _NodeKey[NodeType], v: _NodeValue[NodeType]) -> None:
        ...

    @overload
    def __setitem__(self, k: _EdgeKey[NodeType], v: _EdgeValue) -> None:
        ...

    @overload
    def __setitem__(self, k: _EdgeAttributeKey[NodeType], v: _AttributeValue) -> None:
        ...

    @overload
    def setdefault(
        self, k: _NodeKey[NodeType], default: Optional[_NodeValue[NodeType]] = None
    ) -> _NodeValue[NodeType]:
        ...

    @overload
    def setdefault(
        self, k: _EdgeKey[NodeType], default: Optional[_EdgeValue] = None
    ) -> _EdgeValue:
        ...

    @overload
    def setdefault(
        self,
        k: _EdgeAttributeKey[NodeType],
        default: Optional[_AttributeValue] = None,
    ) -> _AttributeValue:
        ...

    @overload
    def update(
        self, data: MutableMapping[_EdgeAttributeKey[NodeType], _AttributeValue]
    ) -> None:
        ...

    @overload
    def update(self, data: MutableMapping[_EdgeKey[NodeType], _EdgeValue]) -> None:
        ...

    @overload
    def update(
        self, data: MutableMapping[_NodeKey[NodeType], _NodeValue[NodeType]]
    ) -> None:
        ...

    @overload
    def fromkeys(
        self,
        iterable: Iterable[_EdgeAttributeKey[NodeType]],
        value: Optional[_AttributeValue] = None,
    ):
        ...

    @overload
    def fromkeys(
        self,
        iterable: Iterable[_EdgeKey[NodeType]],
        value: Optional[_EdgeValue] = None,
    ):
        ...

    @overload
    def fromkeys(
        self,
        iterable: Iterable[_NodeKey[NodeType]],
        value: Optional[_NodeValue[NodeType]] = None,
    ):
        ...

    ############################################
    # dict methods override
    ############################################

    def __getitem__(self, k: _GraphKey[NodeType]) -> _GraphValue[NodeType]:
        if not isinstance(k, tuple):
            k = (k,)

        if len(k) == 1:
            self._update_neighbors(k)
            return self._data[k[0]]
        elif len(k) == 2:
            self._update_neighbors(k)
            return self._data[k[0]][k[1]]
        elif len(k) == 3 and (k[0], k[1]) in self:
            self._update_neighbors(k)
            return self._data[k[0]][k[1]][k[2]]

        raise KeyError(k)

    def get(
        self, k: _GraphKey[NodeType], default: Optional[_GraphValue[NodeType]] = None
    ) -> Optional[_GraphValue[NodeType]]:
        if k not in self:
            return default
        return self[k]

    def __setitem__(self, k: _GraphKey[NodeType], v: _GraphValue[NodeType]) -> None:
        if not isinstance(k, tuple):
            k = (k,)

        if len(k) == 1:
            if not isinstance(v, dict):
                raise ValueError(
                    'The value must be of type {<neighbor>: {"attr": value}}'
                )

            data_: _GraphDataType[NodeType] = deepcopy(self._data)
            successors_: Dict[NodeType, Set[NodeType]] = deepcopy(self._successors)
            predecessors_: Dict[NodeType, Set[NodeType]] = deepcopy(self._predecessors)

            if k[0] not in self:
                successors_[k[0]] = successors_.default_factory()

            for n, d in v.items():
                if not isinstance(d, dict):
                    raise ValueError(
                        'The value must be of type {<neighbor>: {"attr": value}}'
                    )

                if (k[0], n) not in self:
                    successors_[k[0]].add(n)
                    predecessors_[n].add(k[0])
                data_[k[0]][n].update(d)

            self._update_neighbors(k)
            self._data = data_
            self._successors = successors_
            self._predecessors = predecessors_

        elif len(k) == 2:
            if not isinstance(v, dict):
                raise ValueError('The value must be of type {"attr": value}')

            self._update_neighbors(k)
            self._data[k[0]][k[1]].update(v)

        elif len(k) == 3:
            self._update_neighbors(k)
            self._data[k[0]][k[1]][k[2]] = v

    def setdefault(
        self, k: _GraphKey[NodeType], default: Optional[_GraphValue[NodeType]] = None
    ) -> _GraphValue[NodeType]:
        if not isinstance(k, tuple):
            k = (k,)

        if k not in self:
            if default is None:
                if len(k) == 1:
                    default = defaultdict(dict)
                elif len(k) == 2:
                    default = dict()
            self[k] = default
        return self[k]

    def __delitem__(self, k: _GraphKey[NodeType]) -> None:
        if not isinstance(k, tuple):
            k = (k,)

        if k not in self:
            raise KeyError(k)

        if len(k) == 1:
            del self._data[k[0]]
            _ = [self._delete_predecessor(s, k[0]) for s in self._successors.pop(k[0])]
        elif len(k) == 2:
            del self._data[k[0]][k[1]]
            self._successors[k[0]].remove(k[1])
            self._delete_predecessor(k[1], k[0])
        elif len(k) == 3:
            del self._data[k[0]][k[1]][k[2]]
        else:
            raise KeyError(k)

    def pop(
        self,
        k: _GraphKey[NodeType],
        default: Union[_GraphValue[NodeType], object] = _RaiseKeyError,
    ) -> _GraphValue[NodeType]:
        if k not in self:
            if default is _RaiseKeyError:
                raise KeyError(k)
            return default

        ret = self[k]
        del self[k]

        return ret

    def __iter__(self) -> Iterator[NodeType]:
        return self._data.__iter__()

    def __len__(self) -> int:
        return len(set(self._successors.keys()).union(set(self._predecessors.keys())))

    def __contains__(self, k: _GraphKey[NodeType]) -> bool:
        is_present = True

        if not isinstance(k, tuple):
            k = (k,)

        if len(k) >= 1:
            is_present &= k[0] in self._data

        if is_present and len(k) >= 2:
            is_present &= k[1] in self._data[k[0]]

        if is_present and len(k) == 3:
            is_present &= k[2] in self._data[k[0]][k[1]]

        if is_present and len(k) > 3:
            is_present = False

        return is_present

    def __eq__(self, o: object) -> bool:
        # Note that the __eq__ method must be implemented for the objects contained
        return isinstance(o, DiGraph) and self._data == o._data

    def __ne__(self, o: object) -> bool:
        return not self.__eq__(o)

    def clear(self) -> None:
        self._init()
        return self

    def keys(self) -> KeysView[NodeType]:
        return tuple((f, t) for f, td in self._data.items() for t in td.keys())

    def values(self) -> ValuesView[_NodeValue]:
        return tuple(d for f, td in self._data.items() for d in td.values())

    def items(self) -> ItemsView[NodeType, _NodeValue]:
        return tuple(((f, t), d) for f, td in self._data.items() for t, d in td.items())

    def update(
        self, data: MutableMapping[_GraphKey[NodeType], _GraphValue[NodeType]]
    ) -> None:
        if self._is_a_three_layers_dict(data):
            _ = [
                self[(f, t)].update(d) for f, td in data.items() for t, d in td.items()
            ]
        elif self._is_a_two_layers_dict(data) and all(
            isinstance(k, tuple) and len(k) == 2 for k in data.keys()
        ):
            _ = [self[(f, t)].update(d) for (f, t), d in data.items()]
        elif isinstance(data, dict) and all(
            isinstance(k, tuple) and len(k) == 3 for k in data.keys()
        ):
            _ = [self[(f, t)].update({a: v}) for (f, t, a), v in data.items()]
        else:
            raise ValueError("Unknown data input format")

    def fromkeys(
        self,
        iterable: Iterable[_GraphKey[NodeType]],
        value: Optional[_GraphValue[NodeType]] = None,
    ):
        _ = [self.setdefault(k, value) for k in iterable]

    def _update_neighbors(self, k: _NodeKey[NodeType]):
        if not isinstance(k, tuple):
            k = (k,)

        if len(k) == 1:
            if k[0] not in self:
                self._successors[k[0]] = self._successors.default_factory()
        elif len(k) == 2 or len(k) == 3:
            if (k[0], k[1]) not in self:
                self._successors[k[0]].add(k[1])
                self._predecessors[k[1]].add(k[0])

    def _delete_predecessor(self, s: NodeType, p: NodeType):
        self._predecessors[s].remove(p)
        if not self._predecessors[s]:
            del self._predecessors[s]

    @classmethod
    def _is_a_three_layers_dict(cls, data):
        return (
            isinstance(data, dict)
            and data.values()  # the overall struct is a dict
            and isinstance(next(iter(data.values())), dict)
            and next(iter(data.values())).values()  # the second layer is also a dict
            and isinstance(
                next(iter(next(iter(data.values())).values())), dict
            )  # and the third layer too
        )

    @classmethod
    def _is_a_two_layers_dict(cls, data):
        return (
            isinstance(data, dict)
            and data.values()  # the first layer is a dict
            and isinstance(next(iter(data.values())), dict)  # and the second layer too
        )

    ############################################
    # graph specific methods
    ############################################

    @property
    def nodes(self) -> Iterator[NodeType]:
        """Property returning the nodes in the graph

        Returns:
            Iterator[NodeType]: The iterator of nodes in the graph
        """
        return iter(set(self._successors.keys()).union(set(self._predecessors.keys())))

    @property
    def edges(self) -> "DiGraph.EdgeCollection":
        """Return the edges data structure

        Returns:
            DiGraph.EdgeCollection: The collection of edges and their data
        """
        return DiGraph.EdgeCollection(edges=self.items())

    def add_node(self, node: NodeType) -> None:
        """Add a node to the graph

        Args:
            node (NodeType): The node to add
        """
        self.setdefault(node)

    def add_edge(self, u: NodeType, v: NodeType, **attr: Any) -> None:
        """Add an edge to the graph.

        Note:
            If the nodes do not exist they are added to the graph.

        Args:
            u (NodeType): The `from` node
            v (NodeType): The `to` node
        """
        self[(u, v)] = dict(**attr)

    def has_node(self, node: NodeType) -> bool:
        """Check whether or not a node exists in the graph

        Args:
            node (NodeType): The node to check for

        Returns:
            bool: True if the node is in the graph, False otherwise
        """
        return node in self._successors or node in self._predecessors

    def has_edge(self, u: NodeType, v: NodeType) -> bool:
        """Check whether or not a directed edge exists in the graph

        Args:
            u (NodeType): the `from` node of the edge to check for
            v (NodeType): the `to` node of the edge to check for

        Returns:
            bool: True if the edge exists in the graph, False otherwise
        """
        return (u, v) in self

    def successors(self, node: NodeType) -> Iterator[NodeType]:
        """The successors of the given node

        Args:
            node (NodeType): The node for which we want the successors

        Returns:
            Iterator[NodeType]: Iterator of the successors nodes
        """
        return iter(self._successors[node])

    def predecessors(self, node: NodeType) -> Iterator[NodeType]:
        """The predecessors of the given node

        Args:
            node (NodeType): The node for which we want the predecessors

        Returns:
            Iterator[NodeType]: Iterator of the predecessors nodes
        """
        return iter(self._predecessors[node])

    def neighbors(self, node: NodeType) -> Iterator[NodeType]:
        """The neighbors of the given node (i.e successors + predecessors)

        Args:
            node (NodeType): The node for which we want the neighbors

        Returns:
            Iterator[NodeType]: Iterator of the neighbors nodes
        """
        return chain(self.successors(node), self.predecessors(node))

    def subgraph(self, nodes: Iterable[NodeType]) -> "DiGraph":
        """Create the graph induced by the given nodes

        Returns:
            DiGraph: The subgraph induced by the nodes
        """
        subgraph_ = {}
        for n in nodes:
            subgraph_.update(
                {(n, v): self[(n, v)] for v in self._successors.get(n, [])}
            )
            subgraph_.update(
                {(v, n): self[(v, n)] for v in self._predecessors.get(n, [])}
            )
        return DiGraph(subgraph_)

    def subgraph_for(self, node: NodeType) -> "DiGraph":
        """Create the graph induced only by the given node and its neighbors

        Returns:
            DiGraph: The subgraph induced by the node
        """
        subgraph_ = {}

        subgraph_.update(
            {(node, v): self[(node, v)] for v in self._successors.get(node, [])}
        )
        subgraph_.update(
            {(v, node): self[(v, node)] for v in self._predecessors.get(node, [])}
        )

        return DiGraph(subgraph_)
