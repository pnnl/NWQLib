"""Shared projected Hermitian eigensolve and finite-shot pencil diagnostics.

These operations act only on projected matrices; they do not construct basis
states, evaluate operators, or certify physical accuracy of sampled estimates.
FixedGCIM, ADAPT and Lanczos all reduce their acquired (H, S) pairs here, so
the admission rules, rank cutoff and reported diagnostics mean the same thing
in every projected method.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from nwqlib._numerics import stable_vector_norm


# Numerical-null resolution for projected overlaps. This is a rank
# selection policy, not an input-error bound. Revisit it with the precision
# policy (docs/ENGINEERING_CONSTANTS.md).
DEFAULT_OVERLAP_EIGENVALUE_CUTOFF = 1.0e-12


@dataclass(frozen=True, kw_only=True)
class _GeneralizedEigenResult:
    """Solution of one thresholded pencil H c = E S c of size m with kept rank r.

    H and S below are the matrices as solved, their Hermitian parts when
    symmetrize_matrices is set (the default). With identity_shift=c_I the
    solve uses H0, and H below is H0 + c_I*S. The S-normalization of the
    vectors and overlap_normalization_error use the supplied S.

    Attributes:
        eigenvalues: Ascending Ritz values, shape (r,), in the units of H.
        eigenvectors: S-normalized coefficient vectors as columns, shape (m, r).
        ground_state_coefficients: First column of eigenvectors, the lowest Ritz vector.
        overlap_eigenvalues: Ascending eigenvalues of S before thresholding, shape (m,).
        kept_overlap_rank: r, the number of overlap eigenvalues above the cutoff.
        residual_norm: ||H c - E S c||_2 of the lowest pair.
        generalized_eigenpair_backward_error: Dimensionless residual_norm/((||H||_F + |E| ||S||_F) ||c||).
        overlap_normalization_error: |c^dagger S c - 1| of the lowest vector with the supplied S.
        overlap_eigenvectors: Eigenvectors of S, shape (m, m), kept only on request, else None.
    """

    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    ground_state_coefficients: np.ndarray
    overlap_eigenvalues: np.ndarray
    kept_overlap_rank: int
    residual_norm: float
    generalized_eigenpair_backward_error: float
    overlap_normalization_error: float
    overlap_eigenvectors: np.ndarray | None = field(default=None, repr=False)


def _eigenpair_backward_error(residual_norm, hamiltonian_norm, overlap_matrix_norm, coefficient_norm, ground_energy):
    """Return ``||Hc - ESc|| / ((||H||_F + |E| ||S||_F) ||c||)`` without overflow.

    This is the dimensionless ``projected_backward_error`` defined in the GCiM
    guide's diagnostics table. The denominator is formed in logarithms because
    ``|E| ||S||_F`` can overflow binary64 for large energy units even when the
    ratio itself is representable.
    """
    if residual_norm == 0.0:
        return 0.0
    else:
        log_hamiltonian_norm = (
            math.log(hamiltonian_norm) if hamiltonian_norm > 0.0 else -math.inf
        )
        log_energy_overlap_norm = (
            math.log(abs(ground_energy)) + math.log(overlap_matrix_norm)
            if ground_energy != 0.0 and overlap_matrix_norm > 0.0
            else -math.inf
        )
        log_denominator = float(
            np.logaddexp(log_hamiltonian_norm, log_energy_overlap_norm)
        ) + math.log(coefficient_norm)
        if not math.isfinite(log_denominator):
            raise RuntimeError("generalized-eigenpair backward-error scale is unavailable")
        return math.exp(
            math.log(residual_norm) - log_denominator
        )


def _hermitian_part(matrix: np.ndarray) -> np.ndarray:
    """Average conjugate entries without overflowing or halving subnormals first."""

    return _safe_average(matrix, matrix.conj().T)


def _safe_average(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    """Average finite components without overflowing or halving subnormals first."""

    result = np.empty(left.shape, dtype=np.result_type(left, right, np.complex128))
    for output, left_component, right_component in (
        (result.real, left.real, right.real),
        (result.imag, left.imag, right.imag),
    ):
        with np.errstate(over="ignore"):
            np.add(left_component, right_component, out=output, dtype=output.dtype)
        overflow = ~np.isfinite(output)
        np.divide(output, 2.0, out=output, where=~overflow)
        output[overflow] = left_component[overflow] / 2.0 + right_component[overflow] / 2.0
    return result


def gram_formation_allowance(length, diagonal_sum):
    """Return a first-order bound on ||S - V^dagger V||_2 for binary64 Gram entries.

    Each computed S_ij = <v_i|v_j> is a complex inner product of ``length``
    terms. In any summation order its error is at most
    g*sum_k |v_ki||v_kj| with g = sqrt(2)*gamma_(2*length) and
    gamma_n = n*u/(1 - n*u). This is the complex inner-product bound derived
    above exact_probability_window in _validation.py, from Higham, Accuracy
    and Stability of Numerical Algorithms, 2nd ed., SIAM 2002,
    doi:10.1137/1.9780898718027, Lemma 3.1 and
    Eq. (3.5). By Cauchy-Schwarz the sum is at most a_i*a_j with
    a_i = ||v_i||, so the error matrix E satisfies |E| <= g*a*a^T entrywise.
    A nonnegative matrix bounds the spectral norm of every matrix it
    dominates entrywise, hence ||E||_2 <= g*||a||**2 = g*sum_i ||v_i||**2.
    Each computed diagonal entry is at least (1 - g)*||v_i||**2, so
    g*diagonal_sum/(1 - g) bounds that sum through the computed
    ``diagonal_sum``, the trace of S. By Weyl's inequality every eigenvalue
    of the computed S lies within this allowance of an eigenvalue of the
    exact Gram matrix, and those are nonnegative. Averaging S with its
    conjugate transpose keeps the bound, and the rounding of that average is
    covered by the eigensolver term of _solve_projected_pencil.
    Revisit if the Gram products change their length or leave binary64
    (docs/ENGINEERING_CONSTANTS.md).
    """
    from nwqlib._validation import UNIT_ROUNDOFF

    rounding = 2 * length * UNIT_ROUNDOFF
    factor = math.sqrt(2) * rounding / (1 - rounding)
    return factor * diagonal_sum / (1 - factor)


def entrywise_gram_allowance(entry_errors):
    """Return an outward spectral bound from symmetric entrywise errors.

    entry_errors contains finite nonnegative bounds e_ij on a Hermitian
    error E. Its spectral norm equals its spectral radius, bounded by the
    induced infinity norm, so ||E||_2 <= max_i sum_j e_ij. Each binary64
    entry is summed exactly and the largest sum is converted upward.
    This adds no assumption of statistical independence. The caller owns
    the entry bounds, diagonal convention and completeness checks.
    """
    from fractions import Fraction

    bound = max(
        (sum((Fraction(float(value)) for value in row), Fraction(0))
         for row in entry_errors),
        default=Fraction(0),
    )
    try:
        result = float(bound)
    except OverflowError:
        return math.inf
    if math.isfinite(result) and Fraction(result) < bound:
        result = math.nextafter(result, math.inf)
    return result


def _as_square_pencil(hamiltonian_matrix, overlap_matrix):
    """Admit matching nonempty square matrices before any preprocessing."""
    h_matrix = np.asarray(hamiltonian_matrix, dtype=complex)
    s_matrix = np.asarray(overlap_matrix, dtype=complex)
    if (
        h_matrix.shape != s_matrix.shape
        or h_matrix.ndim != 2
        or h_matrix.shape[0] != h_matrix.shape[1]
        or h_matrix.shape[0] == 0
    ):
        raise ValueError("hamiltonian_matrix and overlap_matrix must be square matrices with the same shape")
    return h_matrix, s_matrix


def _overlap_norm_squared(coefficients, overlap, overlap_magnitudes, roundoff_factor):
    """Return c^dagger S c when it is a positive real number up to roundoff, else None.

    For Hermitian S the exact value is real. The computed imaginary part is
    admitted when it is at most roundoff_factor * a.T @ A @ a, with
    a=|Re c|+|Im c| and A=overlap_magnitudes=|Re S|+|Im S|. A nonfinite value,
    a nonpositive real part or a larger imaginary part returns None.
    """
    value = complex(coefficients.conj().T @ overlap @ coefficients)
    coefficient_magnitudes = np.abs(coefficients.real) + np.abs(coefficients.imag)
    imaginary_roundoff = roundoff_factor * float(
        coefficient_magnitudes @ overlap_magnitudes @ coefficient_magnitudes
    )
    if (
        not np.isfinite(value)
        or value.real <= 0.0
        or not math.isfinite(imaginary_roundoff)
        or abs(value.imag) > imaginary_roundoff
    ):
        return None
    return value


def _solve_projected_pencil(
    hamiltonian_matrix: Any,
    overlap_matrix: Any,
    *,
    overlap_eigenvalue_cutoff: float = DEFAULT_OVERLAP_EIGENVALUE_CUTOFF,
    symmetrize_matrices: bool = True,
    overlap_input_tolerance: float = 0.0,
    identity_shift: float = 0.0,
    _sampled_overlap: bool = False,
    _keep_overlap_eigenvectors: bool = False,
) -> tuple[_GeneralizedEigenResult | None, np.ndarray | None, str | None, float | None]:
    """Solve a projected Hermitian generalized eigenproblem.

    This is the discretized Hill-Wheeler eigenproblem ``H f = E S f`` of
    Zheng et al., Phys. Rev. Research 5, 023200 (2023), Eq. (13), with the
    matrix elements of Eqs. (14)-(15) (numbering of arXiv:2212.09205v1). For
    Lanczos it is the Krylov pencil of Kirby, Motta and Mezzacapo, arXiv
    2208.00567v4, Eq. (10), p. 4.
    Solves ``H c = E S c`` by canonical (Lowdin) orthogonalization:
    diagonalize ``S = V diag(s) V^dagger``, discard directions with
    ``s <= overlap_eigenvalue_cutoff``, form ``X = V_kept diag(s_kept^-1/2)``,
    and diagonalize ``X^dagger H X`` (Rayleigh-Ritz in the kept
    subspace). Coefficient vectors are returned S-normalized,
    ``c^dagger S c = 1``.

    The truncation is the thresholding procedure of Epperly, Lin and
    Nakatsukasa, arXiv:2110.07492v2, Algorithm 1.1 (Sec. 1.2, p. 5). It keeps
    the overlap eigenvalues strictly above the threshold, as here. Their
    Theorem 2.7 (p. 14) analyzes its stability under noise in H and S. Zheng et al.
    (2024), arXiv:2312.07691v3, Appendix F, Eqs. (F1)-(F3), apply the same
    positive-eigenvalue truncation to GCIM pencils. The absolute cutoff is a
    rank policy (``DEFAULT_OVERLAP_EIGENVALUE_CUTOFF``). The theorem's
    accuracy statements need a threshold tied to the noise level, which this
    function does not choose.

    A deterministic Gram matrix is positive semidefinite, so a negative
    eigenvalue beyond roundoff means the input is not a Gram matrix and the
    solve is refused. A sampled overlap can be indefinite, so that path keeps
    only the positive directions above the cutoff.

    An identity term c_I*I of the operator contributes c_I*S to H and c_I to
    every Ritz value. In ``X^dagger (H0 + c_I S) X`` the part
    ``X^dagger (c_I S) X`` is c_I times the identity in exact arithmetic, but
    its computed entries carry roundoff of order
    |c_I|*u*||X||**2 = |c_I|*u/s_min for the smallest kept overlap eigenvalue
    s_min, so on an ill-conditioned S the Ritz error grows with |c_I|. A
    caller that knows c_I passes the matrix H0 without it and
    ``identity_shift=c_I``. The solve then uses (H0, S) and adds c_I to each
    eigenvalue afterwards, one rounding per eigenvalue.

    Args:
        hamiltonian_matrix: Projected Hamiltonian matrix ``H``.
        overlap_matrix: Projected overlap matrix ``S``.
        overlap_eigenvalue_cutoff: Eigenvalue cutoff used to remove numerical
            null directions from ``S``.
        symmetrize_matrices: Whether to Hermitian-symmetrize ``H`` and ``S``
            after the raw matrices pass the Hermitian check within roundoff.
        overlap_input_tolerance: Absolute spectral admission allowance for the
            input representation, independent of the positive rank cutoff. Its
            default zero assumes only matrix arithmetic roundoff. A caller
            declaring a numerical resolution must document it and not call it
            a certified bound. ``gram_formation_allowance`` derives it for a
            Gram matrix formed from binary64 inner products.
        identity_shift: Coefficient c_I of an identity term that
            ``hamiltonian_matrix`` omits. The returned eigenvalues belong to
            the pencil (H0 + c_I S, S), and so do the residual and backward
            error. The residual is computed as ``||H0 c - E0 S c||`` with
            E0 = E - c_I, which equals ``||(H0 + c_I S) c - E S c||`` in exact
            arithmetic.
        _sampled_overlap: Keep only the positive directions of a sampled ``S``
            above the cutoff instead of rejecting a deterministic Gram matrix
            with a negative mode beyond its admission tolerance.
        _keep_overlap_eigenvectors: Keep the already computed overlap
            eigenbasis for internal consumers that differentiate the
            regularized solve. Ordinary solves do not keep it.

    Returns:
        ``(result, overlap_eigenvalues, failure_reason, overlap_psd_tolerance)``.
        ``result`` is the _GeneralizedEigenResult, or None. In that case
        ``failure_reason`` names the cause for nonfinite
        or non-Hermitian matrices, a negative deterministic overlap mode, no
        usable positive mode, a nonfinite effective Hamiltonian or an invalid
        S-normalization. ``overlap_eigenvalues`` is the ascending spectrum of
        S as solved (its Hermitian part by default), or None when the input
        is rejected before that eigensolve. ``overlap_psd_tolerance`` is the allowance for a negative
        deterministic overlap eigenvalue, overlap_input_tolerance plus
        m*eps*max|s|, or None before it is computed.

    Raises:
        ValueError: If the matrices are not matching nonempty square matrices,
            or the cutoff, input tolerance or identity shift is outside its
            domain.
    """

    h_matrix, s_matrix = _as_square_pencil(hamiltonian_matrix, overlap_matrix)
    if not np.isfinite(overlap_eigenvalue_cutoff) or overlap_eigenvalue_cutoff <= 0.0:
        raise ValueError("overlap_eigenvalue_cutoff must be finite and positive")
    for name, matrix in (("hamiltonian", h_matrix), ("overlap", s_matrix)):
        if not np.all(np.isfinite(matrix)):
            return None, None, f"nonfinite_{name}_matrix", None
    if not math.isfinite(overlap_input_tolerance) or overlap_input_tolerance < 0.0:
        raise ValueError("overlap_input_tolerance must be finite and nonnegative")
    if not math.isfinite(identity_shift):
        raise ValueError("identity_shift must be finite")

    # Scaling avoids overflow. The n*eps tolerance admits arithmetic roundoff
    # in projected matrices, without using the overlap cutoff to admit asymmetry.
    roundoff_tolerance = float(h_matrix.shape[0] * np.finfo(float).eps)
    overlap_psd_tolerance = None
    for name, matrix in (("overlap", s_matrix), ("hamiltonian", h_matrix)):
        scale = float(max(np.max(np.abs(matrix.real)), np.max(np.abs(matrix.imag))))
        if scale > 0.0:
            scaled_real = matrix.real / scale
            scaled_imag = matrix.imag / scale
            defect = np.hypot(
                scaled_real - scaled_real.T, scaled_imag + scaled_imag.T
            )
            if not np.all(np.isfinite(defect)) or np.max(defect) > roundoff_tolerance:
                return None, None, f"non_hermitian_{name}_matrix", overlap_psd_tolerance
    if symmetrize_matrices:
        h_matrix = _hermitian_part(h_matrix)
        s_matrix = _hermitian_part(s_matrix)
    normalization_overlap = np.asarray(overlap_matrix, dtype=complex)

    overlap_eigenvalues, overlap_eigenvectors = np.linalg.eigh(s_matrix)
    # Hermitian eigensolver roundoff scales with the spectral norm, which can
    # exceed every entry by a factor n for a legal rank-one Gram matrix.
    # Weyl's inequality separates the caller's absolute input admission allowance
    # from the eigensolver's scale-dependent arithmetic floor and rank cutoff.
    overlap_psd_tolerance = overlap_input_tolerance + roundoff_tolerance * float(np.max(np.abs(overlap_eigenvalues)))
    if not _sampled_overlap and np.min(overlap_eigenvalues) < -overlap_psd_tolerance:
        return None, overlap_eigenvalues, "negative_deterministic_overlap_eigenvalue", overlap_psd_tolerance
    # Epperly et al., arXiv:2110.07492v2, Algorithm 1.1: keep the eigenvectors
    # with D_ii > epsilon.
    keep = overlap_eigenvalues > overlap_eigenvalue_cutoff
    if not np.any(keep):
        return None, overlap_eigenvalues, "no_usable_overlap_subspace", overlap_psd_tolerance

    kept_vectors = overlap_eigenvectors[:, keep]
    kept_values = overlap_eigenvalues[keep]
    orthogonalizer = kept_vectors * (1.0 / np.sqrt(kept_values))
    effective_hamiltonian = orthogonalizer.conj().T @ h_matrix @ orthogonalizer
    if not np.all(np.isfinite(effective_hamiltonian)):
        return None, overlap_eigenvalues, "nonfinite_effective_hamiltonian", overlap_psd_tolerance
    processed_effective_hamiltonian = _hermitian_part(effective_hamiltonian)
    eigenvalues, reduced_eigenvectors = np.linalg.eigh(processed_effective_hamiltonian)
    del processed_effective_hamiltonian
    coefficient_vectors = orthogonalizer @ reduced_eigenvectors
    overlap_magnitudes = np.abs(normalization_overlap.real) + np.abs(normalization_overlap.imag)
    # Expanding both complex contractions into real dot products gives a
    # gamma_(4n) allowance times a.T @ A @ a, with a=|Re c|+|Im c| and
    # A=|Re S|+|Im S|. eps is twice unit roundoff, adding a safety margin.
    norm_roundoff_factor = 4.0 * roundoff_tolerance / (1.0 - 4.0 * roundoff_tolerance)
    # Rescale every column to c^dagger S c = 1 with the supplied S.
    for column in range(coefficient_vectors.shape[1]):
        squared_norm = _overlap_norm_squared(
            coefficient_vectors[:, column], normalization_overlap, overlap_magnitudes, norm_roundoff_factor
        )
        if squared_norm is None:
            return None, overlap_eigenvalues, "invalid_overlap_normalization", overlap_psd_tolerance
        coefficient_vectors[:, column] /= math.sqrt(squared_norm.real)

    ground_coefficients = coefficient_vectors[:, 0]
    ground_energy = float(np.real_if_close(eigenvalues[0]))
    residual_norm = stable_vector_norm(
        h_matrix @ ground_coefficients - ground_energy * s_matrix @ ground_coefficients
    )
    if identity_shift:
        # Report the pencil (H0 + c_I S, S) whose Ritz values are E0 + c_I. The
        # residual above is already that pencil's residual, since the
        # c_I*S*c terms cancel in exact arithmetic, and the backward error
        # scales it by that pencil.
        eigenvalues = eigenvalues + identity_shift
        ground_energy += identity_shift
        hamiltonian_norm = stable_vector_norm((h_matrix + identity_shift * s_matrix).ravel())
    else:
        hamiltonian_norm = stable_vector_norm(h_matrix.ravel())
    overlap_matrix_norm = stable_vector_norm(s_matrix.ravel())
    coefficient_norm = stable_vector_norm(ground_coefficients)
    generalized_eigenpair_backward_error = _eigenpair_backward_error(
        residual_norm, hamiltonian_norm, overlap_matrix_norm, coefficient_norm, ground_energy,
    )
    # Recompute c^dagger S c after rescaling. Its distance from one is the
    # reported normalization error.
    overlap_norm = _overlap_norm_squared(
        ground_coefficients, normalization_overlap, overlap_magnitudes, norm_roundoff_factor
    )
    if overlap_norm is None:
        return None, overlap_eigenvalues, "invalid_overlap_normalization", overlap_psd_tolerance
    overlap_normalization_error = float(abs(overlap_norm - 1.0))
    result = _GeneralizedEigenResult(
        eigenvalues=np.real_if_close(eigenvalues),
        eigenvectors=coefficient_vectors,
        ground_state_coefficients=ground_coefficients,
        overlap_eigenvalues=np.real_if_close(overlap_eigenvalues),
        kept_overlap_rank=int(np.sum(keep)),
        residual_norm=residual_norm,
        generalized_eigenpair_backward_error=generalized_eigenpair_backward_error,
        overlap_normalization_error=overlap_normalization_error,
        overlap_eigenvectors=overlap_eigenvectors if _keep_overlap_eigenvectors else None,
    )
    return result, overlap_eigenvalues, None, overlap_psd_tolerance


def _sampled_pencil_diagnostics(
    *,
    eigensolver: _GeneralizedEigenResult | None,
    solver_failure_reason: str | None,
    enclosure: tuple[float, float],
    relative_tolerance: float,
) -> dict[str, Any]:
    """Compare a sampled positive-subspace Ritz value with an operator enclosure.

    The enclosure and its relative comparison window are supplied by the
    consumer. The comparison does not certify physical accuracy, and it reuses
    the existing eigensolve without further solves. For exact H and S every
    Ritz value lies in ``[lambda_min, lambda_max]`` of the operator, which any
    valid enclosure contains. Sampled matrices carry no such guarantee, so a
    value outside the window marks a physically implausible pencil while the
    solve itself may still be numerically sound.
    """

    lower, upper = enclosure
    ritz_estimate = None if eigensolver is None else float(eigensolver.eigenvalues[0])
    enclosure_violation = (
        None
        if ritz_estimate is None
        else max(lower - ritz_estimate, ritz_estimate - upper, 0.0)
    )
    enclosure_scale = max(abs(lower), abs(upper))
    enclosure_passed = None if enclosure_violation is None else enclosure_violation <= relative_tolerance * enclosure_scale
    failure_reason = solver_failure_reason or ("ritz_outside_operator_enclosure" if enclosure_passed is False else None)
    return {
        "enclosure_bounds": list(enclosure),
        "enclosure_violation": enclosure_violation,
        "enclosure_relative_tolerance": relative_tolerance,
        "enclosure_passed": enclosure_passed,
        "failure_reason": failure_reason,
    }
