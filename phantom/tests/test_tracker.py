import pytest

from phantom import Tracker


def test_init():
    tracker = Tracker(0.0)
    assert tracker.current == 0.0
    assert len(tracker.history) == 0

    tracker = Tracker(0.0, 1.0)
    assert tracker.current == 0.0
    assert len(tracker.history) == 1
    assert tracker.history[0] == 1.0

    with pytest.raises(TypeError):
        _ = Tracker()


def test_cache():
    tracker = Tracker(0.0, max_history_length=10)

    assert len(tracker.history) == 0

    tracker.current = 1.0
    tracker.cache()

    assert tracker.current == 1.0
    assert len(tracker.history) == 1
    assert list(tracker.history) == [1.0]

    tracker.current = 2.0
    tracker.cache()

    assert tracker.current == 2.0
    assert len(tracker.history) == 2
    assert list(tracker.history) == [2.0, 1.0]


def test_maxlen():
    tracker = Tracker(0.0, max_history_length=1)

    assert len(tracker.history) == 0

    tracker.cache()
    assert len(tracker.history) == 1

    tracker.cache()
    assert len(tracker.history) == 1

    tracker = Tracker(0.0, max_history_length=3)

    assert len(tracker.history) == 0

    tracker.cache()
    assert len(tracker.history) == 1

    tracker.cache()
    assert len(tracker.history) == 2

    tracker.cache()
    assert len(tracker.history) == 3

    tracker.cache()
    assert len(tracker.history) == 3


def test_reset_1():
    tracker = Tracker(0.0)

    tracker.current == 5.0

    assert tracker.current == 0.0


def test_reset_2():
    tracker = Tracker(0.0)

    tracker.cache()
    tracker.cache()
    tracker.reset()

    assert len(tracker.history) == 0


def test_reset_3():
    tracker = Tracker(0.0, 1.0)

    tracker.cache()
    tracker.cache()
    tracker.reset()

    assert tracker.current == 0.0
    assert len(tracker.history) == 1
    assert tracker.history[0] == 1.0


def test_slice():
    tracker = Tracker(5.0, max_history_length=5)

    assert tracker.get_recent(0) == []

    with pytest.raises(ValueError):
        tracker.get_recent(1)

    tracker.current = 10.0
    tracker.cache()
    assert tracker.get_recent(1) == [10.0]

    tracker.current = 20.0
    tracker.cache()
    assert tracker.get_recent(2) == [20.0, 10.0]


def test_previous():
    tracker = Tracker(0.0)

    with pytest.raises(ValueError):
        _ = tracker.previous

    tracker.cache()

    assert tracker.previous == 0.0


def test_diff():
    tracker = Tracker(5.0)

    with pytest.raises(ValueError):
        _ = tracker.diff

    tracker.cache()

    tracker.current = 10.0

    assert tracker.diff == 5.0


def test_abs():
    tracker = Tracker(-5.0)

    assert abs(tracker) == 5.0


def test_add():
    tracker = Tracker(5.0)

    assert tracker + 5.0 == 10.0


def test_iadd():
    tracker = Tracker(5.0)

    tracker += 5.0

    assert tracker.current == 10.0


def test_sub():
    tracker = Tracker(5.0)

    assert tracker - 5.0 == 0.0


def test_isub():
    tracker = Tracker(5.0)

    tracker -= 5.0

    assert tracker.current == 0.0


def test_imul():
    tracker = Tracker(5.0)

    tracker *= 5.0

    assert tracker.current == 25.0


def test_itruediv():
    tracker = Tracker(5.0)

    tracker /= 2.0

    assert tracker.current == 2.5


def test_eq():
    tracker1 = Tracker(5.0)
    tracker2 = Tracker(5.0)

    assert tracker1 == tracker2

    tracker1.cache()

    assert tracker1 == tracker2


def test_ne():
    tracker1 = Tracker(5.0)
    tracker2 = Tracker(2.0)

    assert tracker1 != tracker2


def test_lt():
    tracker1 = Tracker(2.0)
    tracker2 = Tracker(5.0)

    assert tracker1 < tracker2


def test_le():
    tracker1 = Tracker(2.0)
    tracker2 = Tracker(5.0)

    assert tracker1 <= tracker1
    assert tracker1 <= tracker2


def test_gt():
    tracker1 = Tracker(5.0)
    tracker2 = Tracker(2.0)

    assert tracker1 > tracker2


def test_ge():
    tracker1 = Tracker(5.0)
    tracker2 = Tracker(2.0)

    assert tracker1 >= tracker1
    assert tracker1 >= tracker2


def test_repr():
    tracker = Tracker(0.0)

    assert str(tracker) == "Tracker(0.0)"
