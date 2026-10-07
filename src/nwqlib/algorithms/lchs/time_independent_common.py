"""Shared input validation for time-independent LCHS."""

from __future__ import annotations

from typing import Any

import numpy as np

from nwqlib._validation import finite_real


def as_square_matrix(matrix: Any) -> np.ndarray:
    """Convert an input object into a square complex matrix."""

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix A must be a square two-dimensional array")
    if not np.all(np.isfinite(array)):
        raise ValueError("matrix A must contain only finite values")
    return array


def validate_final_time(final_time: Any) -> float:
    """Return a normalized positive, finite LCHS evolution time."""

    normalized = finite_real(final_time, "final_time")
    if normalized <= 0.0:
        raise ValueError("final_time must be positive and finite")
    return normalized


__all__ = [
    "as_square_matrix",
    "validate_final_time",
]
