import pickle5 as pickle

import mercury as me
import pytest


class StringNodeDataStructure(me.graphs.DiGraph[str]):
    pass


@pytest.fixture
def empty_graph():
    g = StringNodeDataStructure()
    return g


@pytest.fixture
def filled_graph():
    g = StringNodeDataStructure()
    g[("a", "b", "c")] = 1
    return g


class TestSetItem:
    def test_set_item_single_key(self, empty_graph):
        empty_graph["from"] = {"to": {"value": 1}}

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "value" in empty_graph._data["from"]["to"]
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1

    def test_set_item_single_key_more_complex(self, empty_graph):
        empty_graph["from"] = {"to1": {"value": 1}, "to2": {"value1": 1, "value2": 2}}

        assert 1 == empty_graph._data["from"]["to1"]["value"]
        assert 1 == empty_graph._data["from"]["to2"]["value1"]
        assert 2 == empty_graph._data["from"]["to2"]["value2"]
        assert len(empty_graph) == 3
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 2

    def test_set_item_single_key_wrong_format_raises(self, empty_graph):
        with pytest.raises(ValueError):
            empty_graph["from"] = 1

        assert len(empty_graph) == 0

    def test_set_item_pair_key(self, empty_graph):
        empty_graph[("from", "to")] = {"value": 1}

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "value" in empty_graph._data["from"]["to"]
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1

    def test_set_item_pair_key_wrong_format_raises(self, empty_graph):
        with pytest.raises(ValueError):
            empty_graph[("from", "to")] = 1

        assert len(empty_graph) == 0

    def test_set_item_triplet_key(self, empty_graph):
        empty_graph[("from", "to", "value")] = 1

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "value" in empty_graph._data["from"]["to"]
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1


class TestGetItem:
    def test_get_item_single_key(self, filled_graph):
        val = filled_graph["a"]

        assert isinstance(val, dict)
        assert "b" in val
        assert isinstance(val["b"], dict)
        assert "c" in val["b"]
        assert val["b"]["c"] == True
        assert len(filled_graph) == 2

    def test_get_single_key(self, filled_graph):
        val = filled_graph.get("a", True)

        assert isinstance(val, dict)
        assert "b" in val
        assert isinstance(val["b"], dict)
        assert "c" in val["b"]
        assert val["b"]["c"] == True
        assert len(filled_graph) == 2

    def test_get_item_single_key_missing(self, filled_graph):
        val = filled_graph["1"]

        assert isinstance(val, dict)
        assert len(val) == 0
        assert len(filled_graph) == 3

    def test_get_single_key_missing(self, filled_graph):
        val = filled_graph.get("1", True)

        assert val == True
        assert len(filled_graph) == 2

    def test_get_item_pair_key(self, filled_graph):
        val = filled_graph[("a", "b")]

        assert isinstance(val, dict)
        assert "c" in val
        assert val["c"] == True
        assert len(filled_graph) == 2

    def test_get_pair_key(self, filled_graph):
        val = filled_graph.get(("a", "b"), True)

        assert isinstance(val, dict)
        assert "c" in val
        assert val["c"] == True
        assert len(filled_graph) == 2

    def test_get_item_pair_key_missing(self, filled_graph):
        val = filled_graph[("1", "2")]

        assert isinstance(val, dict)
        assert len(val) == 0
        assert len(filled_graph) == 4

    def test_get_pair_key_missing(self, filled_graph):
        val = filled_graph.get(("1", "2"), True)

        assert val == True
        assert len(filled_graph) == 2

    def test_get_item_triplet_key(self, filled_graph):
        val = filled_graph[("a", "b", "c")]

        assert val == 1
        assert len(filled_graph) == 2

    def test_get_triplet_key(self, filled_graph):
        val = filled_graph.get(("a", "b", "c"), True)

        assert val == 1
        assert len(filled_graph) == 2

    def test_get_item_triple_key_missing(self, filled_graph):
        with pytest.raises(KeyError):
            filled_graph[("1", "2", "3")]

        assert len(filled_graph) == 2

    def test_get_triple_key_missing(self, filled_graph):
        val = filled_graph.get(("1", "2", "3"), True)

        assert val == True
        assert len(filled_graph) == 2

    def test_get_item_bad_key_missing(self, filled_graph):
        with pytest.raises(KeyError):
            filled_graph[("1", "2", "3", "4")]

        assert len(filled_graph) == 2

    def test_get_bad_key_missing(self, filled_graph):
        val = filled_graph.get(("1", "2", "3", "4"), True)

        assert val == True
        assert len(filled_graph) == 2


class TestSetdefault:
    def test_setdefault_single_key(self, empty_graph):
        val = empty_graph.setdefault("from", {"to": {"value": 1}})

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "to" in val
        assert "value" in empty_graph._data["from"]["to"]
        assert "value" in val["to"]
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert 1 == val["to"]["value"]
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1

    def test_setdefault_single_key_wrong_format(self, empty_graph):
        with pytest.raises(ValueError):
            _ = empty_graph.setdefault("from", 1)

        assert len(empty_graph) == 0

    def test_setdefault_single_key_no_default(self, empty_graph):
        val = empty_graph.setdefault("from")

        assert isinstance(val, dict)
        assert isinstance(val["b"], dict)

    def test_setdefault_pair_key(self, empty_graph):
        val = empty_graph.setdefault(("from", "to"), {"value": 1})

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "value" in empty_graph._data["from"]["to"]
        assert "value" in val
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert 1 == val["value"]
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1

    def test_setdefault_pair_key_wrong_format_raises(self, empty_graph):
        with pytest.raises(ValueError):
            _ = empty_graph.setdefault(("from", "to"), 1)

        assert len(empty_graph) == 0

    def test_setdefault_pair_key_no_default(self, empty_graph):
        val = empty_graph.setdefault(("from", "to"))

        assert isinstance(val, dict)

    def test_setdefault_triplet_key(self, empty_graph):
        val = empty_graph.setdefault(("from", "to", "value"), 1)

        assert "from" in empty_graph._data
        assert "to" in empty_graph._data["from"]
        assert "value" in empty_graph._data["from"]["to"]
        assert 1 == empty_graph._data["from"]["to"]["value"]
        assert 1 == val
        assert len(empty_graph) == 2
        assert len(empty_graph._successors) == 1
        assert len(empty_graph._predecessors) == 1

    def test_setdefault_triplet_key_no_default(self, empty_graph):
        val = empty_graph.setdefault(("from", "to", "value"))

        assert val is None


class TestContains:
    def test_contains_single_key(self, filled_graph):
        assert "a" in filled_graph
        assert ("a",) in filled_graph
        assert "1" not in filled_graph
        assert ("1",) not in filled_graph

    def test_contains_pair_key(self, filled_graph):
        assert ("a", "b") in filled_graph
        assert ("1", "2") not in filled_graph
        assert ("a", "2") not in filled_graph

    def test_contains_triplet_key(self, filled_graph):
        assert ("a", "b", "c") in filled_graph
        assert ("1", "2", "3") not in filled_graph
        assert ("a", "2", "3") not in filled_graph
        assert ("a", "b", "3") not in filled_graph

    def test_contains_bad_key(self, filled_graph):
        assert ("a", "b", "c", "d") not in filled_graph
        assert ("1", "2", "3", "4") not in filled_graph


class TestDelItem:
    def test_delitem_single_key(self, filled_graph):
        assert len(filled_graph) == 2

        del filled_graph["a"]

        assert "a" not in filled_graph
        assert len(filled_graph) == 0

    def test_delitem_single_key_missing(self, filled_graph):
        assert len(filled_graph) == 2

        with pytest.raises(KeyError):
            del filled_graph["1"]

        assert len(filled_graph) == 2

    def test_delitem_pair_key(self, filled_graph):
        assert len(filled_graph) == 2

        del filled_graph[("a", "b")]

        assert "a" in filled_graph
        assert ("a", "b") not in filled_graph
        assert len(filled_graph) == 1

    def test_delitem_pair_key_missing(self, filled_graph):
        assert len(filled_graph) == 2

        with pytest.raises(KeyError):
            del filled_graph[("1", "2")]

        assert len(filled_graph) == 2

    def test_delitem_triplet_key(self, filled_graph):
        assert len(filled_graph) == 2

        del filled_graph[("a", "b", "c")]

        assert "a" in filled_graph
        assert ("a", "b") in filled_graph
        assert ("a", "b", "c") not in filled_graph
        assert len(filled_graph) == 2

    def test_delitem_triplet_key_missing(self, filled_graph):
        assert len(filled_graph) == 2

        with pytest.raises(KeyError):
            del filled_graph[("1", "2", "3")]

        assert len(filled_graph) == 2

    def test_delitem_bad_key(self, filled_graph):
        assert len(filled_graph) == 2

        with pytest.raises(KeyError):
            del filled_graph[("a", "b", "c", "d")]

        assert len(filled_graph) == 2


class TestPop:
    def test_pop_single_key(self, filled_graph):
        val = filled_graph.pop("a")

        assert "a" not in filled_graph
        assert len(filled_graph) == 0
        assert "b" in val
        assert isinstance(val["b"], dict)
        assert "c" in val["b"]
        assert 1 == val["b"]["c"]

    def test_pop_single_key_missing_no_default(self, filled_graph):
        with pytest.raises(KeyError):
            _ = filled_graph.pop("1")

        assert len(filled_graph) == 2

    def test_pop_single_key_missing_default(self, filled_graph):
        val = filled_graph.pop("1", True)

        assert len(filled_graph) == 2
        assert val is True

    def test_pop_pair_key(self, filled_graph):
        val = filled_graph.pop(("a", "b"))

        assert len(filled_graph) == 1
        assert "a" in filled_graph
        assert ("a", "b") not in filled_graph
        assert isinstance(val, dict)
        assert "c" in val
        assert 1 == val["c"]

    def test_pop_pair_key_missing_no_default(self, filled_graph):
        with pytest.raises(KeyError):
            _ = filled_graph.pop(("1", "2"))

        assert len(filled_graph) == 2

    def test_pop_pair_key_missing_default(self, filled_graph):
        val = filled_graph.pop(("1", "2"), True)

        assert len(filled_graph) == 2
        assert val is True

    def test_pop_triplet_key(self, filled_graph):
        val = filled_graph.pop(("a", "b", "c"))

        assert len(filled_graph) == 2
        assert "a" in filled_graph
        assert ("a", "b") in filled_graph
        assert ("a", "b", "c") not in filled_graph
        assert 1 == val

    def test_pop_triplet_key_missing_no_default(self, filled_graph):
        with pytest.raises(KeyError):
            _ = filled_graph.pop(("1", "2", "3"))

        assert len(filled_graph) == 2

    def test_pop_triplet_key_missing_default(self, filled_graph):
        val = filled_graph.pop(("1", "2", "3"), True)

        assert len(filled_graph) == 2
        assert val is True

    def test_pop_bad_key_no_default(self, filled_graph):
        with pytest.raises(KeyError):
            _ = filled_graph.pop(("a", "b", "c", "d"))

        assert len(filled_graph) == 2

    def test_pop_bad_key_default(self, filled_graph):
        val = filled_graph.pop(("a", "b", "c", "d"), True)

        assert len(filled_graph) == 2
        assert val is True


class TestUpdate:
    def test_update_full_dict(self, empty_graph):
        empty_graph.update({"1": {"2": {"3": 4}}})

        assert len(empty_graph) == 2
        assert "1" in empty_graph
        assert ("1", "2") in empty_graph
        assert ("1", "2", "3") in empty_graph
        assert empty_graph[("1", "2", "3")] == 4

    def test_update_full_dict_complex(self, empty_graph):
        empty_graph.update({"1": {"2.1": {"3.1": 4}, "2.2": {"3.2": 4}}})
        empty_graph.update({"1": {"2.1": {"3.1": 4.1}}})
        empty_graph.update({"1": {"2.2": {"3.2": 4.2}}})
        empty_graph.update({"1": {"2.2": {"3.22": 4.22}}})
        empty_graph.update({"1": {"2.3": {"3.3": 4.3}}})

        assert len(empty_graph) == 4
        assert "1" in empty_graph
        assert ("1", "2.1") in empty_graph
        assert ("1", "2.2") in empty_graph
        assert ("1", "2.3") in empty_graph
        assert ("1", "2.1", "3.1") in empty_graph
        assert ("1", "2.2", "3.2") in empty_graph
        assert ("1", "2.2", "3.22") in empty_graph
        assert ("1", "2.3", "3.3") in empty_graph
        assert empty_graph[("1", "2.1", "3.1")] == 4.1
        assert empty_graph[("1", "2.2", "3.2")] == 4.2
        assert empty_graph[("1", "2.2", "3.22")] == 4.22
        assert empty_graph[("1", "2.3", "3.3")] == 4.3

    def test_update_2_layers_dict(self, empty_graph):
        empty_graph.update({("1", "2"): {"3": 4}})

        assert len(empty_graph) == 2
        assert "1" in empty_graph
        assert ("1", "2") in empty_graph
        assert ("1", "2", "3") in empty_graph
        assert empty_graph[("1", "2", "3")] == 4

    def test_update_1_layer_dict(self, empty_graph):
        empty_graph.update({("1", "2", "3"): 4})

        assert len(empty_graph) == 2
        assert "1" in empty_graph
        assert ("1", "2") in empty_graph
        assert ("1", "2", "3") in empty_graph
        assert empty_graph[("1", "2", "3")] == 4

    def test_update_bad_input_2_layers(self, empty_graph):
        with pytest.raises(ValueError):
            empty_graph.update({"1": {"3": 4}})

        assert len(empty_graph) == 0

    def test_update_bad_input_1_layer(self, empty_graph):
        with pytest.raises(ValueError):
            empty_graph.update({"1": 4})

        assert len(empty_graph) == 0

    def test_update_empty(self, empty_graph):
        empty_graph.update({})

        assert len(empty_graph) == 0


class TestOther:
    def test_iter(self, filled_graph):
        assert list(filled_graph) == ["a"]

    def test_len(self, filled_graph):
        assert len(filled_graph) == 2

    def test_eq(self, filled_graph):
        g2 = StringNodeDataStructure()
        g2[("a", "b", "c")] = 1
        assert filled_graph == g2

    def test_neq(self, filled_graph):
        g2 = StringNodeDataStructure()
        g2[("a", "b", "c")] = 2
        assert filled_graph != g2

    def test_clear(self, filled_graph, empty_graph):
        filled_graph.clear()

        assert filled_graph == empty_graph

    def test_keys(self, filled_graph):
        val = filled_graph.keys()

        assert val == (("a", "b"),)

    def test_values(self, filled_graph):
        val = filled_graph.values()

        assert val == ({"c": 1},)

    def test_items(self, filled_graph):
        val = filled_graph.items()

        assert val == ((("a", "b"), {"c": 1}),)

    def test_fromkeys(self, empty_graph):
        empty_graph.fromkeys(["1", "2"])

        assert len(empty_graph) == 2
        assert "1" in empty_graph
        assert "2" in empty_graph

    def test_edges(self, filled_graph):
        assert list(filled_graph.edges) == [("a", "b")]

    def test_edges_data(self, filled_graph):
        assert list(filled_graph.edges.data()) == [("a", "b", {"c": 1})]

    def test_has_edge(self, filled_graph):
        assert filled_graph.has_edge("a", "b")
        assert not filled_graph.has_edge("b", "c")

    def test_has_node(self, filled_graph):
        assert filled_graph.has_node("a")
        assert filled_graph.has_node("b")
        assert not filled_graph.has_node("c")

    def test_add_node(self, empty_graph):
        empty_graph.add_node("a")
        assert "a" in empty_graph

    def test_subgraph(self, filled_graph):
        filled_graph[("1", "2")] = {"3": 4}

        g = filled_graph.subgraph(iter(["a"]))

        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "1" not in g.nodes
        assert "2" not in g.nodes
        assert g[("a", "b", "c")] == 1

    def test_subgraph_2(self, filled_graph):
        filled_graph[("1", "2")] = {"3": 4}

        g = filled_graph.subgraph(iter(["a", "b"]))

        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "1" not in g.nodes
        assert "2" not in g.nodes
        assert g[("a", "b", "c")] == 1

    def test_subgraph_3(self, filled_graph):
        filled_graph[("1", "2")] = {"3": 4}

        g = filled_graph.subgraph(iter(["a", "b", "1"]))

        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "1" in g.nodes
        assert "2" in g.nodes
        assert g[("a", "b", "c")] == 1
        assert g[("1", "2", "3")] == 4

    def test_subgraph_missing(self, filled_graph):
        filled_graph[("1", "2")] = {"3": 4}

        g = filled_graph.subgraph(iter(["a", "b", "m"]))

        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "1" not in g.nodes
        assert "2" not in g.nodes
        assert g[("a", "b", "c")] == 1

    def test_subgraph_for(self, filled_graph):
        filled_graph[("1", "2")] = {"3": 4}

        g = filled_graph.subgraph_for("a")

        assert "a" in g.nodes
        assert "b" in g.nodes
        assert "1" not in g.nodes
        assert "2" not in g.nodes
        assert g[("a", "b", "c")] == 1

    def test_neighbors(self, filled_graph):
        n = filled_graph.neighbors("a")

        assert {"b"} == set(n)

    def test_neighbors_2(self, filled_graph):
        filled_graph[("a", "1")] = {"2": 2}
        n = filled_graph.neighbors("a")

        assert {"b", "1"} == set(n)

    def test_neighbors_3(self, filled_graph):
        filled_graph[("b", "1")] = {"2": 2}
        n = filled_graph.neighbors("b")

        assert {"a", "1"} == set(n)

    def test_serialize_deserialize(self, filled_graph):
        g = pickle.loads(pickle.dumps(filled_graph))

        assert "a" in g
        assert ("a", "b") in g
        assert ("a", "b", "c") in g
        assert g["a"]["b"]["c"] == 1
        assert {"b"} == set(g.neighbors("a"))
        assert {"a"} == set(g.predecessors("b"))
        assert {"b"} == set(g.successors("a"))
