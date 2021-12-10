from abc import abstractmethod, ABC
from typing import cast, Generic, TypeVar


T = TypeVar("T")


class TimeTypeVar(Generic[T], ABC):
    @abstractmethod
    def __ge__(self, other: T) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __add__(self, other: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def __sub__(self, other: T) -> T:
        raise NotImplementedError

    @abstractmethod
    def __mul__(self, other: int) -> T:
        raise NotImplementedError


Time = TypeVar("Time", bound=TimeTypeVar)


class Clock(Generic[Time]):
    """
    Utility class for keeping track of time elapsed.

    This class amounts to a finite state machine with deterministic
    transitions. Let :math:`t_0` be the :py:attr:`start_time`, :math:`\\Delta
    t` be the :py:attr:`increment`, and :math:`T` be the
    :py:attr:`terminal_time`.  Then we have :math:`N = \\left\\lfloor\\frac{T -
    t_0}{\\Delta t}\\right\\rfloor + 1` states, with :math:`p(t_i \\rightarrow
    t_{i+1}) = 1 \\forall i \\in \\{0\\ldots N\\}` and :math:`p(T \\rightarrow
    T) = 1`, where :math:`T = t_N`

    .. note::
        This class is :py:class:`typing.Generic` over the specific type used to
        represent time. That is, you can use this class to monitor time in
        :math:`\\mathbb{R}` or :math:`\\mathbb{Z}`, for example.

    Usage::

        >>> clock = Clock(0.0, 1.0, 0.1)
        >>> clock.tick(5)
        >>> clock.tick(3)
        0.8
        >>> clock.reset()
        0.0

    Attributes:
        start_time: The initial time on the clock :math:`t_0`.
        terminal_time: The time at which the clock moves to a terminal state
            :math:`T`.
        increment: The change :math:`\\Delta t` between time steps.
    """

    def __init__(self, start_time: Time, terminal_time: Time, increment: Time) -> None:
        self.start_time: Time = start_time
        self.terminal_time: Time = terminal_time
        self.increment: Time = increment

        self._step: int = 0

    @property
    def elapsed(self) -> Time:
        """The total time elasped since :math:`t_0`."""
        return cast(Time, self.increment * self._step)

    @property
    def n_steps(self) -> int:
        """The total number of timesteps that the clock will go through"""
        return int((self.terminal_time - self.start_time) / self.increment)

    @property
    def time(self) -> Time:
        """The current time on the clock :math:`t`."""
        new_time = self.start_time + self.elapsed

        return min(new_time, self.terminal_time)

    @property
    def is_terminal(self) -> bool:
        """
        Indicates whether the clock has reached the terminal time :math:`T`
        (:py:attr:`terminal_time`).
        """
        return self.time >= self.terminal_time

    def reset(self) -> Time:
        """
        Resets the clock back to :math:`t_0` (:py:attr:`start_time`) and
        returns the new time.
        """
        self._step = 0

        return self.time

    def tick(self, n: int = 1) -> Time:
        """
        Step the clock forward :math:`n` ticks and return the new time.

        Args:
            n: The number of elapsed ticks.
        """
        self._step += n

        return self.time
