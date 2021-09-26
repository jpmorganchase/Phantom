import numpy as _np
import typing as _t


def is_square(mat: _np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a square matrix."""
    return len(set(mat.shape)) == 1


def is_symmetric(mat: _np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a symmetric matrix."""
    return _t.cast(bool, (mat.transpose() == mat).all())


def is_skew_symmetric(mat: _np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a skew-symmetric matrix."""
    return _t.cast(bool, (mat.transpose() == -mat).all())


def is_hollow(mat: _np.ndarray) -> bool:
    """Returns true iff :code:`mat` is a hollow matrix."""
    return _t.cast(bool, (_np.abs(mat.diagonal() - 0.0) < 1e-5).all())
