import pytest

import numpy as np
import mercury as me


def test_is_square():
    assert me.linalg.is_square(np.eye(5))
    assert not me.linalg.is_square(np.zeros((2, 4)))


def test_is_symmetric():
    assert me.linalg.is_symmetric(np.eye(5))


def test_is_skew_symmetric():
    assert not me.linalg.is_skew_symmetric(np.eye(5))


@pytest.mark.parametrize("i", list(range(5)))
def test_is_hollow_elementwise(i):
    mat = np.zeros((5, 5))

    assert me.linalg.is_hollow(mat)

    mat[i, i] = 1.0

    assert not me.linalg.is_hollow(mat)
