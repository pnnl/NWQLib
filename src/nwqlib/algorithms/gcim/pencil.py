"""Shared GCiM pair, quadrature and projected-matrix mathematics.

Native acquisition belongs to its actual circuit owner. These kernels neither
prepare a state nor evaluate a Hamiltonian.

The matrix elements are ``H_ij = <phi_i|A|phi_j>`` and ``S_ij = <phi_i|phi_j>``
of Zheng et al., Phys. Rev. Research 5, 023200 (2023), Eqs. (14)-(15), in the
Pauli-sum form of Eqs. (33)-(34) (numbering of arXiv:2212.09205v1). Here a
quadrature is the real or imaginary part of one Hadamard-test amplitude, as
in Zheng et al. (2024), arXiv:2312.07691v3, Appendix G, Eqs. (G4)-(G5), which
treat the real part. Those equations give the ancilla outcome probabilities
``1/2 +- Re<phi_i|P|phi_j>/2``, so the ancilla ``Z`` expectation is
``Re<phi_i|P|phi_j>``. An ``S^dagger`` on the ancilla before the final
Hadamard gate turns it into ``Im<phi_i|P|phi_j>``.
"""

from dataclasses import dataclass
from math import fsum
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class MatrixElement:
    """One scalar Hadamard observation of an upper-triangle transition amplitude.

    ``name`` is the experiment name and the join key between planned settings
    and observed values, so it must encode the matrix, pair, Pauli label and
    quadrature exactly once.

    Attributes:
        matrix: ``overlap`` for ``S_ij`` or ``hamiltonian`` for one Pauli term
            of ``H_ij``.
        basis_pair: ``(i, j)`` with ``i <= j``. The observed amplitude is
            ``<phi_i|P|phi_j>``.
        pauli_label: Pauli word ``P``, the identity word for an overlap.
        coefficient: Real Pauli coefficient ``c_k`` that multiplies the
            observed amplitude in ``H_ij``, 1.0 for an overlap.
        quadrature: ``real`` or ``imag`` part of the amplitude.
    """

    matrix: Literal["overlap", "hamiltonian"]
    basis_pair: tuple[int, int]
    pauli_label: str
    coefficient: float
    quadrature: Literal["real", "imag"]

    @property
    def name(self):
        left, right = self.basis_pair
        return f"{self.matrix}_{left}_{right}_{self.pauli_label}_{self.quadrature}"


def matrix_element_count(pair_count, diagonal_pair_count, nonidentity_terms):
    """Count scalar observations before any pair or setting expansion.

    An off-diagonal pair needs real and imaginary parts of ``S_ij`` and of
    every nonidentity Pauli term. A diagonal pair needs only the real part of
    each Pauli term, because normalized columns give ``S_ii = 1`` and
    Hermiticity makes ``H_ii`` real. The count must equal the population that
    ``matrix_elements`` yields, since planning admits it before expansion.
    """
    return (
        2 * (pair_count - diagonal_pair_count) * (1 + nonidentity_terms)
        + diagonal_pair_count * nonidentity_terms
    )


def matrix_elements(basis_size, terms, num_qubits):
    """Yield every scalar Hadamard observation of one ``basis_size`` pencil.

    ``terms`` holds the nonidentity ``(label, real coefficient)`` Pauli terms
    of the operator. Pairs ``(i, j)`` with ``i <= j`` run row by row. An
    off-diagonal pair yields the real and imaginary overlap, then both parts
    of each term. A diagonal pair yields only the real part of each term.
    ``S_ii = 1`` for normalized columns, and each ``<phi_i|P|phi_i>`` is real
    because every Pauli word is Hermitian, so those scalars carry no new
    information. The identity term is
    never measured (``assemble_pencil``). ``matrix_element_count`` gives the
    size of this population.
    """
    identity = "I" * num_qubits
    pairs = ((left, right) for left in range(basis_size) for right in range(left, basis_size))
    for left, right in pairs:
        if left != right:
            for quadrature in ("real", "imag"):
                yield MatrixElement("overlap", (left, right), identity, 1.0, quadrature)
        for label, coefficient in terms:
            for quadrature in ("real",) if left == right else ("real", "imag"):
                yield MatrixElement(
                    "hamiltonian", (left, right), label, float(coefficient), quadrature
                )


def assemble_pencil(basis_size, terms, num_qubits, values):
    """Return ``(H0, S)`` from every upper-triangle pair, without the identity term.

    Callers pass the complete ``matrix_elements`` population. Fixed analysis
    returns a partial result without calling this kernel, and ADAPT acquires
    each query or raises for missing data before assembly. A missing scalar
    still raises here instead of becoming a plausible zero.

    For ``A = cI + sum_k c_k P_k`` the identity term contributes
    ``<phi_i|cI|phi_j> = c S_ij``, so the projected matrix ``H = H0 + c S``
    reuses the acquired overlap. ``H0`` holds only the nonidentity terms.
    Callers solve ``(H0, S)`` with ``identity_shift=c``, which adds c to the
    Ritz values after the ill-conditioned solve (``_solve_projected_pencil``
    explains the accuracy gained), and record ``physical_hamiltonian``. The
    lower triangle is the conjugate of the upper triangle, which makes both
    returned matrices Hermitian by construction.
    """
    h0 = np.full((basis_size, basis_size), complex(np.nan, np.nan))
    overlap = np.full_like(h0, complex(np.nan, np.nan))
    identity = "I" * num_qubits
    pairs = ((left, right) for left in range(basis_size) for right in range(left, basis_size))
    for left, right in pairs:
        if left == right:
            s = 1.0 + 0.0j
        else:
            name = f"overlap_{left}_{right}_{identity}"
            s = complex(values[f"{name}_real"], values[f"{name}_imag"])
        real, imag = [], []
        for label, coefficient in terms:
            name = f"hamiltonian_{left}_{right}_{label}"
            real.append(coefficient * values[f"{name}_real"])
            if left != right:
                imag.append(coefficient * values[f"{name}_imag"])
        h = complex(fsum(real), fsum(imag))
        h0[left, right], h0[right, left] = h, h.conjugate()
        overlap[left, right], overlap[right, left] = s, s.conjugate()
    # One stored overlap, one algebraic covariance identity. There is no
    # independent identity-term measurement or coordinate-identity shift.
    return h0, overlap


def physical_hamiltonian(h0, overlap, identity_shift):
    """Return the recorded projected matrix ``H = H0 + c S`` in the problem's energy unit.

    ``H0`` omits the identity term c*I. The solve itself uses ``H0`` and
    receives c separately, so this matrix only enters the saved pencil.
    """
    return h0 + identity_shift * overlap if identity_shift else h0


def _projected_solve_bytes(basis_size):
    """Return the byte allowance for assembling and solving one ``basis_size`` pencil.

    ``512 b**2`` bytes is 32 complex128 ``b``-by-``b`` arrays, an untuned
    allowance. ``_solve_projected_pencil`` creates about a dozen such arrays:
    the acquired ``H`` and ``S``, their Hermitian parts, the overlap
    eigenvectors, the kept vectors and orthogonalizer ``X``, the product
    ``X^dagger H X`` and its eigenvectors, and the Ritz coefficients. The rest
    leaves room for the caller's copies. FixedGCIM and ADAPT share this law
    (docs/ENGINEERING_CONSTANTS.md, "Fixed GCIM projected size law"). It
    excludes LAPACK workspace and Python object overhead.
    """
    return 512 * basis_size * basis_size


def _projected_solve_work(basis_size):
    """Return the operation-count limit for solving one ``basis_size`` pencil.

    ``16 b**3`` is an untuned allowance for the dense cubic steps: the
    Hermitian eigendecompositions of ``S`` and of ``X^dagger H X``, the
    products that form ``X^dagger H X`` and the Ritz coefficients, and the
    ``S``-normalization of every Ritz vector, which costs two quadratic forms
    of ``b**2`` terms per vector. ``32 b**2`` covers the entrywise scans for
    Hermiticity and scaling, the residual of the lowest pair and its final
    normalization check. The limit is checked before the solve and is not
    measured CPU time.
    """
    return 32 * basis_size * basis_size + 16 * basis_size * basis_size * basis_size


def pair_count(basis_size, nonidentity_terms):
    """Count the exact pair acquisitions of one ``basis_size`` pencil.

    For M needed pairs including D diagonal pairs and L nonidentity terms,
    the exact per-pair route acquires ``(M - D) + D*1_(L>0)``: one joint
    state per off-diagonal pair and one system state per diagonal pair, whose
    reduction returns both ``H0_ij`` and ``S_ij``. A complete b-state pencil
    with L > 0 has ``M = b(b+1)/2`` and ``D = b``, so ``b(b+1)/2``
    acquisitions, against ``b**2 L + b(b-1)`` Hadamard observations of the
    per-element route. A logical scalar is no longer synonymous with an
    acquisition.
    """
    pairs = basis_size * (basis_size + 1) // 2
    return (pairs - basis_size) + (basis_size if nonidentity_terms else 0)


def pair_acquisitions(basis_size, nonidentity_terms):
    """Yield the exact route's upper-triangle pairs ``(left, right)`` row by row.

    A diagonal pair is acquired only when L > 0: with no nonidentity term its
    H0 entry is algebraically zero and its normalized overlap is one.
    """
    for left in range(basis_size):
        for right in range(left, basis_size):
            if left != right or nonidentity_terms:
                yield left, right


def assemble_pair_pencil(basis_size, values):
    """Return ``(H0, S)`` from the acquired ``(H0_ij, S_ij)`` of every upper-triangle pair.

    ``values[(i, j)]`` holds the complex pair entries in the requested
    ``(i, j)`` orientation. Lower triangles follow by conjugation once. The
    normalized-basis convention sets ``S_ii = 1``; the separately acquired
    raw diagonal norms stay evidence and do not rescale the saved blocks.
    Imaginary diagonal parts are zero algebraically for Hermitian P, so a
    diagonal H0 keeps the real part of its acquired value, which has no
    larger error relative to its real Hermitian target. A diagonal pair
    that was not acquired (L = 0) has H0 exactly zero. A missing
    off-diagonal pair raises instead of becoming a plausible zero.
    """
    h0 = np.full((basis_size, basis_size), complex(np.nan, np.nan))
    overlap = np.full_like(h0, complex(np.nan, np.nan))
    for left in range(basis_size):
        for right in range(left, basis_size):
            if left == right:
                acquired = values.get((left, right))
                h0[left, left] = 0.0 if acquired is None else acquired[0].real
                overlap[left, left] = 1.0
                continue
            h, s = values[(left, right)]
            h0[left, right], h0[right, left] = h, h.conjugate()
            overlap[left, right], overlap[right, left] = s, s.conjugate()
    return h0, overlap
