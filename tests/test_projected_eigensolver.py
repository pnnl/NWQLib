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
    assert result.offset_free_eigenvalues == (2.0,)
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
    assert result.offset_free_eigenvalues == (-1., 2.)
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

    # With c = 2**53 the Ritz values 1/4 and 1/2 of H0 = diag(1/4, 1/2),
    # S = I both round to 2**53, a tie of the physical values, while the
    # offset-free values keep 1/4, 1/2 and their gap 1/4. solve_pencil
    # carries them into the ProjectedPencil, which checks only their number.
    from nwqlib.algorithms.gcim.fixed_basis import solve_pencil

    tie = np.diag([.25, .5])
    witness, _, failure, _ = _solve_projected_pencil(tie, np.eye(2), identity_shift=2.**53)
    assert failure is None and witness.offset_free_eigenvalues == (.25, .5)
    np.testing.assert_array_equal(witness.eigenvalues, [2.**53, 2.**53])
    pencil = solve_pencil(tie, np.eye(2), cutoff=1e-12, sampled=False, identity_shift=2.**53)
    assert pencil.eigenvalues == (2.**53, 2.**53) and pencil.offset_free_eigenvalues == (.25, .5)
    with pytest.raises(ValueError, match="offset_free_eigenvalues"):
        pencil.revise(offset_free_eigenvalues=(.25,))


def test_backward_error_is_unavailable_and_an_overflowing_hermitian_part_is_refused():
    import pytest
    from nwqlib._projected_eigensolver import _eigenpair_backward_error, _hermitian_part

    # 0.125 / ((2 + 0.5 * 4) * 1) = 1/32 exactly.
    assert _eigenpair_backward_error(0.125, 2., 4., 1., 0.5) == 0.03125
    # |E| ||S||_F = 1e309 has no binary64 value.
    assert _eigenpair_backward_error(1., 1e200, 1e200, 1., 1e109) is None
    tiny = np.nextafter(0., 1.)
    np.testing.assert_array_equal(_hermitian_part(np.array([[1., .125], [.25, tiny]])),
                                  [[1., .1875], [.1875, tiny]])
    with pytest.raises(ValueError, match="binary64 range"):
        _solve_projected_pencil(np.diag([1.5e308, 1.]), np.eye(2))


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


def test_entrywise_gram_allowance_is_the_least_float_above_the_largest_exact_row_sum():
    # The reference sums each row exactly in rationals and rounds the largest sum upward. Rows that round down,
    # sum exactly, mix subnormal and normal entries, or exceed the binary64 range all give the reference value.
    from fractions import Fraction
    from math import inf, nextafter

    from nwqlib._projected_eigensolver import entrywise_gram_allowance

    def reference(errors):
        largest = max((sum((Fraction(float(v)) for v in row), Fraction(0)) for row in errors), default=Fraction(0))
        try:
            value = float(largest)
        except OverflowError:
            return inf
        return nextafter(value, inf) if value < inf and Fraction(value) < largest else value

    rng = np.random.default_rng(17)
    big = np.finfo(float).max
    cases = [np.zeros((0, 0)), np.zeros((3, 3)), np.full((4, 4), 0.1), np.array([[1.0, 2.0**-53], [2.0**-53, 1.0]]),
             np.array([[5e-324, 1.0, 2.0**-60]]), np.array([[big, big * 2.0**-53]]), np.array([[big / 2, big / 2]]),
             np.array([[big, big]]), *(rng.random((n, n)) * 10.0 ** rng.integers(-300, 300) for n in (2, 5, 9))]
    for errors in cases:
        assert entrywise_gram_allowance(errors) == reference(errors)
