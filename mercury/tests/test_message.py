import pytest

from mercury import Message
from dataclasses import FrozenInstanceError


def test_immutability():
    m = Message("sender", "receiver", None)

    with pytest.raises(FrozenInstanceError):
        m.foo = "bar"
