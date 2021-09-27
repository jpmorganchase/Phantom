import numpy as np
from phantom.numerics import (
    is_equal,
    is_greater,
    is_greater_equal,
    is_less,
    is_less_equal,
    project,
)


ATOL = 1e-9


def test_is_equal():
    a = 234.12
    b = a + 1e-2
    assert not is_equal(a, b, atol=1e-12)
    assert is_equal(a, b, atol=1e-1)

    a = 234.12
    b = a + ATOL * 0.9
    assert is_equal(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 0.9
    assert np.array_equal(is_equal(a, b), np.array([True, True]))

    a = 234.12
    b = a + ATOL * 1.1
    assert not is_equal(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 1.1
    assert np.array_equal(is_equal(a, b), np.array([False, False]))

    a = np.ones((100))

    assert np.all(is_greater(a, 0))


def test_is_greater():
    a = 234.12
    b = a + 1e-2
    assert is_greater(b, a, atol=1e-12)
    assert not is_greater(b, a, atol=1e-1)

    a = 234.12
    b = a + ATOL * 1.1
    assert is_greater(b, a)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 1.1
    assert np.array_equal(is_greater(b, a), np.array([True, True]))

    a = 234.12
    b = a + ATOL * 0.9
    assert not is_greater(b, a)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 0.9
    assert np.array_equal(is_greater(b, a), np.array([False, False]))


def test_is_greater_equal():
    a = 234.12
    b = a + 1e-2

    assert is_greater_equal(b, a, atol=1e-12)
    assert is_greater_equal(b, a, atol=1e-1)

    a = 234.12
    b = a + ATOL * 1.1
    assert is_greater_equal(b, a)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 1.1
    assert np.array_equal(is_greater_equal(b, a), np.array([True, True]))

    a = 234.12
    b = a + ATOL * 0.9
    assert is_greater_equal(b, a)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 0.9
    assert np.array_equal(is_greater_equal(b, a), np.array([True, True]))

    a = 234.12
    b = a - ATOL * 1.1
    assert not is_greater_equal(b, a)

    a = np.array([234.12, 2e-6])
    b = a - ATOL * 1.1
    assert np.array_equal(is_greater_equal(b, a), np.array([False, False]))


def test_is_less():
    a = 234.12
    b = a + 1e-2
    assert is_less(a, b, atol=1e-12)
    assert not is_less(a, b, atol=1e-1)

    a = 234.12
    b = a + ATOL * 1.1
    assert is_less(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 1.1
    assert np.array_equal(is_less(a, b), np.array([True, True]))

    a = 234.12
    b = a + ATOL * 0.9
    assert not is_less(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 0.9
    assert np.array_equal(is_less(a, b), np.array([False, False]))


def test_is_less_equal():
    a = 234.12
    b = a + 1e-2
    assert is_less_equal(a, b, atol=1e-12)
    assert is_less_equal(a, b, atol=1e-1)

    a = 234.12
    b = a + ATOL * 1.1
    assert is_less_equal(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 1.1
    assert np.array_equal(is_less_equal(a, b), np.array([True, True]))

    a = 234.12
    b = a + ATOL * 0.9
    assert is_less_equal(a, b)

    a = np.array([234.12, 2e-6])
    b = a + ATOL * 0.9
    assert np.array_equal(is_less_equal(a, b), np.array([True, True]))

    a = 234.12
    b = a - ATOL * 1.1
    assert not is_less_equal(a, b)

    a = np.array([234.12, 2e-6])
    b = a - ATOL * 1.1
    assert np.array_equal(is_less_equal(a, b), np.array([False, False]))


def test_nans():
    assert not is_equal(1, np.nan)
    assert not is_equal(1, -np.nan)

    assert not is_greater(1, np.nan)
    assert not is_greater(1, -np.nan)

    assert not is_greater_equal(1, np.nan)
    assert not is_greater_equal(1, -np.nan)

    assert not is_less(1, np.nan)
    assert not is_less(1, -np.nan)

    assert not is_less_equal(1, np.nan)
    assert not is_less_equal(1, -np.nan)


def test_infs():
    assert not is_equal(1, np.inf)
    assert not is_equal(1, -np.inf)

    assert not is_greater(1, np.inf)
    assert is_greater(1, -np.inf)

    assert not is_greater_equal(1, np.inf)
    assert is_greater_equal(1, -np.inf)

    assert is_less(1, np.inf)
    assert not is_less(1, -np.inf)

    assert is_less_equal(1, np.inf)
    assert not is_less_equal(1, -np.inf)


def test_project():
    assert project(1.001, 2) == 1.0
    assert project(1.001, 4) == 1.001

    assert np.isnan(project(np.nan, 4))

    np.testing.assert_array_equal(
        project(np.array([0.22, np.nan, 0.3]), 1), np.array([0.2, np.nan, 0.3])
    )
