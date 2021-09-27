import pytest

import phantom as ph
import numpy as np


@pytest.fixture
def encoder():
    return ph.encoders.ElapsedTime()


class DummyContext:
    class DummyActor:
        def __init__(self, clock):
            self.clock = clock

    def __init__(self, clock):
        self.clock = clock

    @property
    def actor(self):
        return self.DummyActor(self.clock)


def test_encode(encoder):
    clock = ph.Clock(0, 100, 1)
    ctx = DummyContext(clock)

    np.testing.assert_array_equal(encoder.encode(ctx), np.array([0.0]))

    clock._step = 50

    np.testing.assert_array_equal(encoder.encode(ctx), np.array([0.5]))
