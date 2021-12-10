import math
from typing import Optional, Union

import numba
import numpy as np


ATOL = 1e-9
ATOL_DIGITS = -int(math.log10(ATOL))

NumericType = Union[int, float, np.ndarray]


@numba.jit
def is_equal(a: NumericType, b: NumericType, atol: float = ATOL) -> bool:
    """Returns true iff a is within atol of b."""
    return np.logical_not(
        np.logical_or(
            np.logical_or(np.greater(b, a + atol), np.greater(a, b + atol)),
            np.logical_or(np.isnan(a), np.isnan(b)),
        )
    )


@numba.jit
def is_greater(a: NumericType, b: NumericType, atol: float = ATOL) -> bool:
    """Returns true iff a is greater than b + atol."""
    return np.greater(a, b + atol)


@numba.jit
def is_less(a: NumericType, b: NumericType, atol: float = ATOL) -> bool:
    """Returns true iff a is less than b - atol."""
    return np.greater(b, a + atol)


@numba.jit
def is_greater_equal(a: NumericType, b: NumericType, atol: float = ATOL) -> bool:
    """Returns true iff a is greater than or equal to b (wrt. atol)."""
    return np.logical_not(
        np.logical_or(np.greater(b, a + atol), np.logical_or(np.isnan(a), np.isnan(b)))
    )


@numba.jit
def is_less_equal(a: NumericType, b: NumericType, atol: float = ATOL) -> bool:
    """Returns true iff a is less than or equal to b (wrt. atol)."""
    return np.logical_not(
        np.logical_or(np.greater(a, b + atol), np.logical_or(np.isnan(a), np.isnan(b)))
    )


def project(a: NumericType, decimals: Optional[int] = None) -> NumericType:
    """Round a number to its best possible tolerance"""
    return np.round(a, decimals or ATOL_DIGITS)
