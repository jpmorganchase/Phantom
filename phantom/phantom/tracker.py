from copy import deepcopy
from collections import deque
from typing import Deque, Generic, List, Optional, TypeVar, Union

import numpy as np

from .numerics import is_equal, is_less, is_less_equal, is_greater_equal, is_greater


V = TypeVar("V", int, float, np.ndarray)


class Tracker(Generic[V]):
    """Utility for tracking the current and previous value of a quantity.

    The class implements many of the standard numerical operations which means
    you can treat an instance of :class:`Tracker` as if it were just a
    numerical value. Note: any of these operations are only applied to the
    current value.

    Arguments:
        default_current_value: The default value. This is used as an initial value on
            init and reset.
        default_previous_value: An optional default previous value.
        max_history_length: The maximum length of history to track.

    Attributes:
        current: The most recent value stored in the tracker.
        previous: The second most recent value stored in the tracker.
        diff: The difference between the current and previous value.
        history: A list of previous values stored in the tracker, history[0] most recent.
    """

    def __init__(
        self,
        default_current_value: V,
        default_previous_value: Optional[V] = None,
        max_history_length: int = 1,
    ) -> None:
        self._default_current_value: V = default_current_value
        self._default_previous_value: Optional[V] = default_previous_value

        self.current: V = deepcopy(self._default_current_value)
        self.history: Deque[V] = deque(maxlen=max(max_history_length, 1))

        if self._default_previous_value is not None:
            self.history.appendleft(deepcopy(self._default_previous_value))

    def cache(self) -> "Tracker":
        """Cache the current value by saving it into the history."""
        self.history.appendleft(deepcopy(self.current))
        return self

    def update(self, value: V) -> None:
        """Cache the current value and sets a new current value."""
        self.history.appendleft(self.current)
        self.current = value

    def reset(self) -> None:
        """Clears the history and sets the current value to the preset default value."""
        self.current = deepcopy(self._default_current_value)
        self.history.clear()

        if self._default_previous_value is not None:
            self.history.appendleft(deepcopy(self._default_previous_value))

    def get_recent(self, n: int) -> List[V]:
        """
        Returns the most recent N values from the history.

        Implemented as the Python deque container does not implement slicing.
        """
        if n > len(self.history):
            raise ValueError(
                "Number of items requested is out of range of the length of the history."
            )

        return [self.history[i] for i in range(n)]

    @property
    def previous(self) -> V:
        if len(self.history) < 1:
            raise ValueError("No previous value stored in Tracker history.")

        return self.history[0]

    @property
    def diff(self) -> V:
        if len(self.history) < 1:
            raise ValueError("Cannot compare present value to empty history.")

        return self.current - self.history[0]

    def __abs__(self) -> V:
        return abs(self.current)

    def __add__(self, value: V) -> V:
        return self.current + value

    def __iadd__(self, value: V) -> "Tracker[V]":
        self.current += value
        return self

    def __sub__(self, value: V) -> V:
        return self.current - value

    def __isub__(self, value: V) -> "Tracker[V]":
        self.current -= value
        return self

    def __imul__(self, value: V) -> "Tracker[V]":
        self.current *= value
        return self

    def __itruediv__(self, value: V) -> "Tracker[V]":
        self.current /= value
        return self

    def __eq__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return is_equal(self.current, other.current)

        return is_equal(self.current, other)

    def __ne__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return not is_equal(self.current, other.current)

        return not is_equal(self.current, other)

    def __lt__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return is_less(self.current, other.current)

        return is_less(self.current, other)

    def __le__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return is_less_equal(self.current, other.current)

        return is_less_equal(self.current, other)

    def __gt__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return is_greater(self.current, other.current)

        return is_greater(self.current, other)

    def __ge__(self, other: Union[V, "Tracker[V]"]) -> bool:
        if isinstance(other, Tracker):
            return is_greater_equal(self.current, other.current)

        return is_greater_equal(self.current, other)

    def __repr__(self) -> str:
        return f"Tracker({self.current})"
