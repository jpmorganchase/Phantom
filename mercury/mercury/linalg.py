from typing import cast

import numpy as np


def is_square(mat: np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a square matrix."""
    return len(set(mat.shape)) == 1


def is_symmetric(mat: np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a symmetric matrix."""
    return cast(bool, (mat.transpose() == mat).all())


def is_skew_symmetric(mat: np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a skew-symmetric matrix."""
    return cast(bool, (mat.transpose() == -mat).all())


def is_hollow(mat: np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a hollow matrix."""
    return cast(bool, (np.abs(mat.diagonal() - 0.0) < 1e-5).all())
