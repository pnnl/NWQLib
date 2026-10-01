"""Neutral absolute structural predicates for dense matrices."""

from __future__ import annotations

from typing import Any

import numpy as np


def is_unitary(matrix: Any, *, atol: float = 1.0e-8) -> bool:
    """Return whether ``matrix`` is unitary within absolute ``atol``.

    The test is ``max |(U^dagger U - I)_ij| <= atol`` entrywise. Unitarity
    fixes the scale of the matrix, so the window is dimensionless. The
    default ``1e-8`` is an untuned window registered in
    docs/ENGINEERING_CONSTANTS.md. Revisit with another precision, larger
    admitted dimensions or a derived arithmetic error bound.
    """

    array = np.asarray(matrix, dtype=complex)
    return (
        array.ndim == 2
        and array.shape[0] == array.shape[1]
        and np.allclose(
            array.conj().T @ array,
            np.eye(array.shape[0]),
            atol=atol,
            rtol=0.0,
        )
    )
