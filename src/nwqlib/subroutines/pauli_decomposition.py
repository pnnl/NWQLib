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
    """One term `c P` of a Pauli decomposition, with `P` a Qiskit Pauli label.

    Build it with keyword arguments, for example
    `PauliTerm(label="ZI", coefficient=0.5)`. Both arguments are required.

    Args:
        label: Pauli string label using Qiskit's ordering convention.
        coefficient: Complex coefficient multiplying the Pauli string.
    """

    label: str
    coefficient: complex


@dataclass(frozen=True, kw_only=True)
class PauliDecomposition:
    """A matrix written as a sum of Pauli strings, with its dimensions and the coefficient cutoff used.

    [`decompose_matrix_to_pauli`][nwqlib.subroutines.pauli_decomposition.decompose_matrix_to_pauli]
    returns it. It can also be built with keyword arguments, all required,
    to pass a Pauli sum to
    [`build_block_encoding`][nwqlib.subroutines.block_encoding.build_block_encoding].
    The operator is `sum_j c_j P_j` over `terms`.

    Args:
        terms: Nonzero Pauli terms defining the represented operator after any pruning.
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
    """Remove Pauli terms that are small relative to the largest, and return a bound on the removed operator.

    A term is removed when `|c| <= rtol * max_j |c_j|` and kept otherwise.

    Args:
        terms (Sequence[PauliTerm]): Exact Pauli-decomposition terms to
            inspect, such as `decomposition.terms`.
        rtol (float): Default `1e-12`. Relative cutoff applied to the
            largest coefficient magnitude.

    Returns:
        kept (tuple[PauliTerm, ...]): The kept terms.
        removed_mass (float): `sum(abs(c))` over the removed coefficients.
            Since every Pauli string has operator norm one, it bounds the
            operator norm of the removed part.

    Raises:
        ValueError: If ``rtol`` is negative or non-finite.

    Examples:
        The `1e-14 XI` term of `0.5 ZZ + 1e-14 XI` is below `1e-12` times
        the largest coefficient:

        >>> from qiskit.quantum_info import SparsePauliOp
        >>> from nwqlib.subroutines.pauli_decomposition import (
        ...     decompose_matrix_to_pauli, prune_pauli_terms_relative)
        >>> A = SparsePauliOp(["ZZ", "XI"], [0.5, 1e-14]).to_matrix()
        >>> kept, removed = prune_pauli_terms_relative(
        ...     decompose_matrix_to_pauli(A).terms)
        >>> print([term.label for term in kept], removed)
        ['ZZ'] 1e-14
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
    """Write a square matrix as a sum of Pauli strings, `A = sum_P c_P P` with `c_P = 2**-n Tr(P A)`.

    This helper prepares matrix inputs for LCHS and later circuit-building
    paths. Matrix dimensions must be powers of two unless explicit zero
    padding is requested. NWQLib's I/X/Y/Z block transform computes the
    coefficients with overflow-safe componentwise averages at each level and
    omits only computed exact-zero coefficients before the cutoff below. It
    avoids the tiny-magnitude loss of Qiskit's zero-tolerance decomposition.
    For dimension D and width q,
    the arithmetic is `O(q D**2)` with `O(D**2)` array workspace. No matrix
    norm or reference decomposition is added to choose the pruning scale.
    Rounding already present in a densely assembled matrix can appear as tiny
    coefficients. Explicit ``rtol`` or ``prune_pauli_terms_relative`` can remove
    those terms, with the omitted coefficient mass reported by the latter.

    Args:
        matrix (array_like): Square complex matrix to decompose.
        atol (float): Default `0.0`. Explicit absolute coefficient cutoff,
            in the units of the matrix. Zero keeps every computed nonzero
            coefficient.
        rtol (float | None): Default `None`, which uses zero. Relative
            cutoff against the largest computed coefficient. The combined
            cutoff is `max(atol, rtol * max_P |c_P|)`, and a term is kept
            when `|c_P|` exceeds it.
        pad_to_power_of_two (bool): Default `False`. Whether to zero-pad a
            non-power-of-two matrix to the next power-of-two dimension
            before decomposition.

    Returns:
        decomposition (PauliDecomposition): The kept terms in
            `decomposition.terms`, the dimensions, and the resolved cutoff
            in `decomposition.atol`.

    Raises:
        ValueError: If the matrix is not square, the tolerance is negative, or a
            non-power-of-two dimension is provided without padding, or a
            coefficient magnitude exceeds the finite binary64 range.

    Examples:
        `[[1, 0.5], [0.5, -1]]` is `0.5 X + Z`:

        >>> from nwqlib.subroutines.pauli_decomposition import (
        ...     decompose_matrix_to_pauli)
        >>> decomposition = decompose_matrix_to_pauli([[1.0, 0.5], [0.5, -1.0]])
        >>> print([(t.label, t.coefficient) for t in decomposition.terms])
        [('X', (0.5+0j)), ('Z', (1+0j))]
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
