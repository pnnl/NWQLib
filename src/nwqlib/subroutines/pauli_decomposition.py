"""Pauli decomposition helpers for matrix-valued algorithm inputs.

An ``n``-qubit matrix is ``A = sum_P c_P P`` over Pauli strings ``P`` with
``c_P = 2**-n Tr(P A)``, because Pauli strings are Hermitian, unitary and
orthogonal in the trace inner product. Labels follow Qiskit, whose last
character acts on qubit 0. Every Pauli string has operator norm one, so the
coefficient 1-norm bounds ``||A||_2``, and it is the LCU subnormalization
of the Pauli block encoding.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import fsum
from typing import Any, Sequence

import numpy as np
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

from nwqlib._validation import finite_real
from nwqlib.subroutines._power_of_two import (
    is_power_of_two as _is_power_of_two,
    next_power_of_two as _next_power_of_two,
)


# Relative cutoff for decomposition roundoff, scaled by the largest
# coefficient. The pruned coefficient mass is returned so product-formula
# budgets can deduct it. Registered in docs/ENGINEERING_CONSTANTS.md.
PAULI_COEFFICIENT_RTOL = 1.0e-12


# Single-qubit factors used by Pauli SELECT, with Y = [[0, -i], [i, 0]].
_LOCAL_PAULI_FACTORS = {
    "I": np.eye(2, dtype=complex),
    "X": np.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.asarray([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=complex),
}


def _as_square_matrix(matrix: Any) -> np.ndarray:
    """Return a square complex matrix."""

    array = np.asarray(matrix, dtype=complex)
    if array.ndim != 2 or array.shape[0] != array.shape[1]:
        raise ValueError("matrix must be a square two-dimensional array")
    return array


@dataclass(frozen=True, kw_only=True)
class PauliTerm:
    """Single Pauli term in a matrix decomposition.

    Args:
        label: Pauli string label using Qiskit's ordering convention.
        coefficient: Complex coefficient multiplying the Pauli string.
    """

    label: str
    coefficient: complex


@dataclass(frozen=True, kw_only=True)
class PauliDecomposition:
    """Sparse Pauli decomposition of a matrix.

    Args:
        terms: Nonzero Pauli terms defining the represented operator after selected pruning.
        input_dimension: Original matrix dimension.
        operator_dimension: Dimension actually decomposed; it exceeds input_dimension
            exactly when the input was zero-padded to a power-of-two dimension.
        num_qubits: Number of qubits represented by the Pauli labels.
        atol: Resolved absolute coefficient cutoff after combining atol and rtol.
            It is not a bound on the difference from the original matrix.

    Raises:
        ValueError: If a term's label is not a string of ``num_qubits``
            letters from ``IXYZ``.
    """

    terms: tuple[PauliTerm, ...]
    input_dimension: int
    operator_dimension: int
    num_qubits: int
    atol: float

    def __post_init__(self) -> None:
        # Planning reads the letters of each label directly, so a label of
        # another length or alphabet would plan the census of another operator.
        for term in self.terms:
            label = term.label
            if not isinstance(label, str):
                raise ValueError(f"Pauli label must be a string, not {type(label).__name__}")
            if len(label) != self.num_qubits or label.strip("IXYZ"):
                raise ValueError(f"Pauli label {label[:80]!r} must be {self.num_qubits} letters from IXYZ")

    def to_sparse_pauli_op(self) -> SparsePauliOp:
        """Return the decomposition as a Qiskit ``SparsePauliOp``."""

        from qiskit.quantum_info import SparsePauliOp

        if not self.terms:
            return SparsePauliOp.from_list([("I" * self.num_qubits, 0.0)])
        return SparsePauliOp.from_list(
            [(term.label, term.coefficient) for term in self.terms]
        )

def prune_pauli_terms_relative(
    terms: Sequence[PauliTerm],
    *,
    rtol: float = PAULI_COEFFICIENT_RTOL,
) -> tuple[tuple[PauliTerm, ...], float]:
    """Prune relatively negligible Pauli terms and bound the removed operator.

    Args:
        terms: Exact Pauli-decomposition terms to inspect.
        rtol: Relative cutoff applied to the largest coefficient magnitude.

    Returns:
        The kept terms and ``sum(abs(c))`` over pruned coefficients. Since every
        Pauli string has operator norm one, the latter bounds the operator norm
        of the removed part.

    Raises:
        ValueError: If ``rtol`` is negative or non-finite.
    """

    tolerance = float(rtol)
    if not np.isfinite(tolerance) or tolerance < 0.0:
        raise ValueError("rtol must be non-negative and finite")
    resolved_terms = tuple(terms)
    max_magnitude = max(
        (abs(complex(term.coefficient)) for term in resolved_terms),
        default=0.0,
    )
    cutoff = tolerance * max_magnitude
    kept = tuple(
        term
        for term in resolved_terms
        if abs(complex(term.coefficient)) > cutoff
    )
    pruned_mass = float(
        fsum(
            abs(complex(term.coefficient))
            for term in resolved_terms
            if abs(complex(term.coefficient)) <= cutoff
        )
    )
    return kept, pruned_mass


def decompose_matrix_to_pauli(
    matrix: Any,
    *,
    atol: float = 0.0,
    rtol: float | None = None,
    pad_to_power_of_two: bool = False,
) -> PauliDecomposition:
    """Decompose an explicit matrix with the shared I/X/Y/Z block transform.

    This helper prepares matrix inputs for LCHS and later circuit-building
    paths. Matrix dimensions must be powers of two unless explicit zero padding
    is requested. ``operators._pauli.pauli_coefficients`` owns the transform,
    including overflow-safe averages and vectorized level traversal. No matrix
    norm or reference decomposition is added to choose the pruning scale.
    Rounding already present in a densely assembled matrix can appear as tiny
    coefficients. Explicit ``rtol`` or ``prune_pauli_terms_relative`` can remove
    those terms, with the omitted coefficient mass reported by the latter.

    Args:
        matrix: Square complex matrix to decompose.
        atol: Explicit absolute coefficient cutoff. Zero preserves every
            computed nonzero coefficient by default.
        rtol: Relative cutoff against the largest computed coefficient.
            ``None`` uses zero. The combined cutoff is max(atol, rtol*scale).
        pad_to_power_of_two: Whether to zero-pad non-power-of-two matrices to
            the next power-of-two dimension before decomposition.

    Returns:
        PauliDecomposition containing nonzero terms and embedding metadata.

    Raises:
        ValueError: If the matrix is not square, the tolerance is negative, or a
            non-power-of-two dimension is provided without padding, or a
            coefficient magnitude exceeds the finite binary64 range.
    """

    atol = finite_real(atol, "atol")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")
    rtol = 0.0 if rtol is None else finite_real(rtol, "rtol")
    if rtol < 0.0:
        raise ValueError("rtol must be non-negative")
    array = _as_square_matrix(matrix)
    input_dimension = array.shape[0]
    operator_dimension = input_dimension
    if not _is_power_of_two(input_dimension):
        if not pad_to_power_of_two:
            raise ValueError(
                "Pauli decomposition requires a power-of-two dimension unless "
                "pad_to_power_of_two=True"
            )
        operator_dimension = _next_power_of_two(input_dimension)
        padded_array = np.zeros((operator_dimension, operator_dimension), dtype=complex)
        padded_array[:input_dimension, :input_dimension] = array
        array = padded_array

    from nwqlib.operators._pauli import pauli_coefficients

    # Qiskit's zero-tolerance path can lose tiny nonzero subtrees. The shared
    # block transform tests exact zero without squaring a tiny magnitude.
    coefficients = pauli_coefficients(array)
    try:
        scale = max((abs(value) for _, value in coefficients), default=0.0)
    except OverflowError as error:
        raise ValueError("Pauli coefficient magnitude exceeds the finite binary64 range") from error
    cutoff = max(atol, rtol*scale)
    terms = tuple(
        PauliTerm(label=label, coefficient=coefficient)
        for label, coefficient in coefficients if abs(coefficient) > cutoff
    )
    return PauliDecomposition(
        terms=terms,
        input_dimension=input_dimension,
        operator_dimension=operator_dimension,
        num_qubits=int(np.log2(operator_dimension)),
        atol=cutoff,
    )


__all__ = [
    "PAULI_COEFFICIENT_RTOL",
    "PauliDecomposition",
    "PauliTerm",
    "decompose_matrix_to_pauli",
    "prune_pauli_terms_relative",
]
