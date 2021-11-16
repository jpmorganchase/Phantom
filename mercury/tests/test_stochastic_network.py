from mercury import StochasticNetwork
from mercury.actors import Actor
from mercury.resolvers import UnorderedResolver


def test_stochastic_network_1():
    net = StochasticNetwork(UnorderedResolver(), [Actor("A"), Actor("B")])

    net.add_connection("A", "B", 1.0)

    assert net.graph.has_edge("A", "B")
    assert net.graph.has_edge("B", "A")

    net.resample_connectivity()

    assert net.graph.has_edge("A", "B")
    assert net.graph.has_edge("B", "A")


def test_stochastic_network_2():
    net = StochasticNetwork(UnorderedResolver(), [Actor("A"), Actor("B")])

    net.add_connection("A", "B", 0.0)

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")


def test_stochastic_network_3():
    net = StochasticNetwork(UnorderedResolver(), [Actor("A"), Actor("B")])

    net.add_connections_from([("A", "B", 0.0)])

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")


def test_stochastic_network_4():
    net = StochasticNetwork(UnorderedResolver(), [Actor("A"), Actor("B")])

    net.add_connections_between(["A"], ["B"], rate=0.0)

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")
