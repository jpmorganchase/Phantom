import pytest

from . import MockAgent


@pytest.fixture
def mock_agent():
    return MockAgent("Agent")


def test_repr(mock_agent):
    assert str(mock_agent) == "[MockAgent Agent]"
