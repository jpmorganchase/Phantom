import numpy as np
import phantom as ph


def test_uniform_range():
    range_ = ph.utils.ranges.UniformRange(
        start=0.0,
        end=10.0,
        step=1.0,
    )

    assert (range_.values() == np.array([0,1,2,3,4,5,6,7,8,9])).all()

def test_linspace_range():
    range_ = ph.utils.ranges.LinspaceRange(
        start=0.0,
        end=10.0,
        n=11,
    )

    assert (range_.values() == np.array([0,1,2,3,4,5,6,7,8,9,10])).all()

