"""Dense matrices of the QLS shortcut (projector complement, Dalzell augmentation), without circuit imports."""

from typing import Any
import numpy as np


def projector_complement_matrix(state: Any, *, dimension: int | None = None) -> np.ndarray:
    """Return the projector `I - |state><state|` onto the complement of a state, as a dense matrix.

    The QLS shortcut forms `G_t = Q_{b'} A_t` with `Q_b = I - b b^dagger`
    (Dalzell, arXiv:2406.12086v2, Eqs. (2) and (11)). The input is
    normalized here, and a larger `dimension` zero-pads it.

    Args:
        state (array_like): Nonzero state vector.
        dimension (int | None): Default `None`, which uses the length of
            `state`. A larger value zero-pads the state.

    Returns:
        projector (numpy.ndarray): The complex `dimension`-square matrix
            `I - |s><s|` for the normalized, padded state `s`.

    Raises:
        ValueError: If the state has zero norm or is longer than
            `dimension`.
    """

    vector = np.asarray(state, dtype=complex).reshape(-1)
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ValueError("projector_complement state must have nonzero norm")
    vector = vector / norm
    # An explicit dimension=0 must reach the size check below, not silently
    # fall back to the state's own size.
    size = vector.size if dimension is None else int(dimension)
    if vector.size > size:
        raise ValueError("projector_complement state is larger than the target dimension")
    padded = np.zeros(size, dtype=complex)
    padded[: vector.size] = vector
    return np.eye(size, dtype=complex) - np.outer(padded, padded.conj())


def dalzell_augmented_matrix(encoded_matrix: Any, t_value: float) -> np.ndarray:
    """Return ``A_t = [[A_encoded, 0], [0, 1/t]]`` zero-padded to twice the dimension.

    Dalzell (arXiv:2406.12086v2), Eq. (8), defines the ``(d + 1)``-square
    ``A_t`` with ``1/t`` in the new diagonal entry. The returned matrix is
    ``2d``-square, with the remaining ``d - 1`` rows and columns zero, so a
    power-of-two ``d`` stays a power of two. The zero padding adds only zero
    singular values.
    """

    matrix = np.asarray(encoded_matrix, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Dalzell augmentation requires a square matrix")
    dimension = matrix.shape[0]
    if dimension < 1:
        raise ValueError("Dalzell augmentation requires a positive dimension")
    t_value = float(t_value)
    if t_value <= 0.0:
        raise ValueError("Dalzell augmentation needs positive t")
    augmented = np.zeros((2 * dimension, 2 * dimension), dtype=complex)
    augmented[:dimension, :dimension] = matrix
    augmented[dimension, dimension] = 1.0 / t_value
    return augmented
