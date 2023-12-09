import pytest

from phantom import Agent, StochasticNetwork
from phantom.resolvers import BatchResolver
from .. import IncrementingComparableSampler


@pytest.fixture
def net():
    return StochasticNetwork([Agent("A"), Agent("B")], BatchResolver(2))


def test_stochastic_network_1(net):
    net.add_connection("A", "B", 1.0)

    assert net.graph.has_edge("A", "B")
    assert net.graph.has_edge("B", "A")

    net.resample_connectivity()

    assert net.graph.has_edge("A", "B")
    assert net.graph.has_edge("B", "A")


def test_stochastic_network_2(net):
    net.add_connection("A", "B", 0.0)

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")


def test_stochastic_network_3(net):
    net.add_connections_from([("A", "B", 0.0)])

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")


def test_stochastic_network_4(net):
    net.add_connections_between(["A"], ["B"], rate=0.0)

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")


def test_stochastic_network_5(net):
    net.add_connection("A", "B", rate=IncrementingComparableSampler(1.0))

    assert net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert net.graph.has_edge("A", "B")


def test_stochastic_network_6(net):
    net.add_connection("A", "B", rate=IncrementingComparableSampler(0.0))

    assert not net.graph.has_edge("A", "B")

    net.resample_connectivity()

    assert not net.graph.has_edge("A", "B")
