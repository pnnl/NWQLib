"""Shared projected-pencil checks independent of a normalized generating basis."""

import numpy as np

from nwqlib._projected_eigensolver import _sampled_pencil_diagnostics, _solve_projected_pencil


def test_sampled_unnormalized_gram_keeps_positive_mode_and_enclosure_diagnostic():
    # Independent diagonal pencil: kept E=8/4=2 and c=(0, 1/2),
    # hence c^dag S c=1. The negative sampled mode is discarded, not repaired.
    overlap = np.diag([-0.25, 4.0])
    hamiltonian = np.diag([-7.0, 8.0])
    # Rank selection cannot waive deterministic PSD admission, even with an
    # enormous cutoff that keeps the positive mode after rescaling.
    for scale, cutoff in ((1.0, 0.5), (2.0**40, 2.0**40)):
        failed, _, reason, _ = _solve_projected_pencil(
            scale * hamiltonian,
            scale * overlap,
            overlap_eigenvalue_cutoff=cutoff,
            overlap_input_tolerance=4e-12,
        )
        assert failed is None and reason == "negative_deterministic_overlap_eigenvalue"
    result, _, failure, _ = _solve_projected_pencil(
        hamiltonian,
        overlap,
        overlap_eigenvalue_cutoff=0.5,
        _sampled_overlap=True,
    )
    assert failure is None
    assert result.kept_overlap_rank == 1
    # These powers-of-two operations are exact in binary arithmetic.
    np.testing.assert_array_equal(result.eigenvalues, [2.0])
    np.testing.assert_array_equal(np.abs(result.ground_state_coefficients), [0.0, 0.5])
    assert result.residual_norm == 0.0
    assert result.overlap_normalization_error == 0.0
    assert result.generalized_eigenpair_backward_error == 0.0
    diagnostics = _sampled_pencil_diagnostics(
        eigensolver=result,
        solver_failure_reason=failure,
        enclosure=(-1.0, 1.0),
        relative_tolerance=1e-8,
    )
    assert diagnostics["enclosure_violation"] == 1.0
    assert diagnostics["enclosure_passed"] is False
    assert diagnostics["failure_reason"] == "ritz_outside_operator_enclosure"


def test_deterministic_input_resolution_preserves_rank_and_raw_hermitian_admission():
    # A dyadic Walsh basis has exact prescribed spectrum (0,0,1,2).
    # A small negative null perturbation is admitted only when the caller
    # declares that input resolution, not when it merely increases rank cutoff.
    basis = np.array([[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]]) / 2
    overlap = (basis * [-(2.0**-40), 0.0, 1.0, 2.0]) @ basis.T
    hamiltonian = (basis * [0.0, 0.0, -1.0, 4.0]) @ basis.T
    strict, _, failure, _ = _solve_projected_pencil(hamiltonian, overlap)
    assert strict is None and failure == "negative_deterministic_overlap_eigenvalue"
    admitted, _, failure, _ = _solve_projected_pencil(
        hamiltonian,
        overlap,
        overlap_input_tolerance=4e-12,
    )
    assert failure is None and admitted.kept_overlap_rank == 2
    # Exact kept ratios are -1/1 and 4/2; all nonzero gaps are O(1).
    np.testing.assert_allclose(admitted.eigenvalues, [-1.0, 2.0], rtol=0.0, atol=1e-13)
    np.testing.assert_allclose(
        abs(basis[:, :2].T @ admitted.eigenvectors), 0.0, rtol=0.0, atol=1e-13
    )
    np.testing.assert_allclose(
        admitted.eigenvectors.T.conj() @ overlap @ admitted.eigenvectors,
        np.eye(2),
        rtol=0.0,
        atol=1e-13,
    )
    # Raw sampled evidence must itself be Hermitian. Admitting sampling noise
    # does not permit averaging a non-Hermitian matrix before checking it.
    h = np.array([[0.0, 1.0], [1.0, 0.0]])
    s = np.array([[1.0, 0.75], [0.25, 1.0]])
    raw, _, reason, _ = _solve_projected_pencil(h, s, _sampled_overlap=True)
    assert raw is None and reason == "non_hermitian_overlap_matrix"


def test_identity_shift_reports_the_pencil_that_contains_it():
    import math
    import pytest
    from nwqlib._projected_eigensolver import _eigenpair_backward_error

    # The diagonal pencil has Ritz values 2 and -1 from H0 = diag(8, -1),
    # S = diag(4, 1). The shift c = 16 is exact, so the reported values are
    # E0 + c and the residual of (H0 + cS, S) equals the residual of (H0, S),
    # here zero because every product is exact. The backward error uses the
    # norms of the pencil with the shift.
    h0, s = np.diag([8., -1.]), np.diag([4., 1.])
    result, _, failure, _ = _solve_projected_pencil(h0, s, identity_shift=16.)
    assert failure is None
    np.testing.assert_array_equal(result.eigenvalues, [15., 18.])
    assert result.residual_norm == 0.0
    # A perturbed Hamiltonian gives a nonzero residual, the same with and
    # without the shift, and a backward error scaled by ||H0 + cS||_F and E0 + c.
    h1 = h0 + np.array([[0., 1e-3], [1e-3, 0.]])
    plain, _, _, _ = _solve_projected_pencil(h1, s)
    shifted, _, _, _ = _solve_projected_pencil(h1, s, identity_shift=16.)
    assert shifted.residual_norm == plain.residual_norm > 0
    np.testing.assert_array_equal(shifted.eigenvalues, plain.eigenvalues + 16.)
    coefficient_norm = float(np.linalg.norm(shifted.ground_state_coefficients))
    expected = _eigenpair_backward_error(shifted.residual_norm, float(np.linalg.norm(h1 + 16. * s)),
                                         float(np.linalg.norm(s)), coefficient_norm,
                                         float(shifted.eigenvalues[0]))
    assert shifted.generalized_eigenpair_backward_error == pytest.approx(expected, rel=1e-14, abs=0)
    with pytest.raises(ValueError, match="identity_shift"):
        _solve_projected_pencil(h0, s, identity_shift=math.inf)


def test_gram_formation_allowance_admits_roundoff_and_still_refuses_a_non_gram_matrix():
    from nwqlib._projected_eigensolver import gram_formation_allowance

    # Three normalized columns of length 2**14 give an allowance of about
    # sqrt(2)*gamma_(2**15)*3 = 1.5e-11. A matrix with an eigenvalue of
    # -1e-9 is not a Gram matrix at this resolution and stays refused, while
    # -1e-12 is inside the allowance.
    basis = np.array([[1, 1, 1], [1, -1, 0], [1, 1, -2]]) / np.sqrt([3., 2., 6.])
    for negative, refused in ((-1e-9, True), (-1e-12, False)):
        overlap = (basis.T * [negative, 1., 2.]) @ basis
        result, _, failure, _ = _solve_projected_pencil(
            np.eye(3), overlap, overlap_input_tolerance=gram_formation_allowance(2**14, 3.))
        assert (failure == "negative_deterministic_overlap_eigenvalue") == refused


def test_entrywise_gram_allowance_does_not_round_a_row_sum_below_its_exact_value():
    from nwqlib._projected_eigensolver import entrywise_gram_allowance

    # The exact row sum 1 + 2**-52 is a binary64 number, but summing the row
    # in binary64 rounds 1 + 2**-53 back to 1 and returns 1, below the bound
    # ||E||_2 <= max_i sum_j e_ij it must state.
    row = [1.0, 2.0**-53, 2.0**-53]
    assert entrywise_gram_allowance([row, [0.0, 0.0, 0.0]]) == 1 + 2.0**-52
    # A row sum between two binary64 numbers is converted to the upper one:
    # 1 + 2**-54 lies between 1 and 1 + 2**-52 and rounds to nearest as 1.
    assert entrywise_gram_allowance([[1.0, 2.0**-54]]) == 1 + 2.0**-52
