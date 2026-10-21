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
