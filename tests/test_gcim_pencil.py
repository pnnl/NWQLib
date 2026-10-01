"""Projected GCiM numerical relations; small matrices, no native execution."""

import numpy as np
import pytest
from nwqlib._projected_eigensolver import _sampled_pencil_diagnostics, _solve_projected_pencil


def test_projected_generalized_eigensolver_handles_rank_deficient_overlap() -> None:
    hamiltonian = np.diag([1.0, 2.0, 3.0])
    overlap = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ]
    )

    result = _solve_projected_pencil(
        hamiltonian, overlap, overlap_eigenvalue_cutoff=1.0e-10, symmetrize_matrices=False
    )[0]

    assert result.kept_overlap_rank == 2
    # Kept directions e0 and (e1+e2)/sqrt(2) give 1/1 and (5/2)/2.
    np.testing.assert_allclose(result.eigenvalues, [1.0, 1.25], rtol=0.0, atol=1.0e-13)
    assert result.residual_norm < 1.0e-10
    assert result.overlap_normalization_error < 1.0e-10

    close_basis = np.array([[1.0, np.exp(0.3j)], [0.0, 0.01]], dtype=complex)
    close_basis /= np.linalg.norm(close_basis, axis=0)
    close_gram = close_basis.conj().T @ close_basis
    close_gram = (close_gram + close_gram.conj().T) / 2.0
    close_result = _solve_projected_pencil(close_gram, close_gram, symmetrize_matrices=False)[0]
    assert close_result.kept_overlap_rank == 2
    np.testing.assert_allclose(close_result.eigenvalues, [1.0, 1.0], rtol=0.0, atol=1.0e-10)
    # B @ c avoids cancellation in the quadratic contraction c^dag (B^dag B) c.
    physical_columns = close_basis @ close_result.eigenvectors
    np.testing.assert_allclose(
        np.sum(np.abs(physical_columns) ** 2, axis=0), [1.0, 1.0], rtol=0.0, atol=1.0e-10
    )

    rng = np.random.default_rng(15)
    basis = rng.normal(size=(1, 3)) + 1j * rng.normal(size=(1, 3))
    basis /= np.linalg.norm(basis, axis=0)
    gram = basis.conj().T @ basis
    gram = (gram + gram.conj().T) / 2.0
    for overlap in (gram, np.ones((4, 4), dtype=complex)):
        rank_one = _solve_projected_pencil(overlap, overlap, symmetrize_matrices=False)[0]
        assert rank_one.kept_overlap_rank == 1
        assert rank_one.eigenvalues[0] == pytest.approx(1.0, rel=0.0, abs=1.0e-13)
        coefficients = rank_one.ground_state_coefficients
        assert coefficients.conj() @ overlap @ coefficients == pytest.approx(
            1.0, rel=0.0, abs=1.0e-13
        )

    # A complex Fourier basis makes the pencil spectra independent inputs.
    # Dyadic entries give exactly Hermitian pencils. Positive overlap
    # eigenvalues stay at least 0.0625 from zero, far above the cutoff.
    fourier = 1j ** np.outer(np.arange(4), np.arange(4)) / 2.0
    hamiltonian_modes = np.array([-0.25, -0.75, 2.0, 1.0])
    complex_hamiltonian = (fourier * hamiltonian_modes) @ fourier.conj().T
    for overlap_modes in (
        np.array([0.0625, 0.25, 1.0, 4.0]),
        np.array([0.0, 0.25, 1.0, 4.0]),
    ):
        complex_overlap = (fourier * overlap_modes) @ fourier.conj().T
        kept = overlap_modes > 0.0
        expected = np.sort(hamiltonian_modes[kept] / overlap_modes[kept])
        complex_result = _solve_projected_pencil(
            complex_hamiltonian, complex_overlap, symmetrize_matrices=False
        )[0]
        assert complex_result.kept_overlap_rank == np.count_nonzero(kept)
        # Machine-precision identities: the positive overlap modes have
        # condition number <= 64; 1e-12 covers eigensolve and contraction error.
        np.testing.assert_allclose(complex_result.eigenvalues, expected, rtol=0.0, atol=1.0e-12)
        np.testing.assert_allclose(
            complex_result.eigenvectors.conj().T @ complex_overlap @ complex_result.eigenvectors,
            np.eye(expected.size),
            rtol=0.0,
            atol=1.0e-12,
        )
        np.testing.assert_allclose(
            fourier[:, ~kept].conj().T @ complex_result.eigenvectors,
            0.0,
            rtol=0.0,
            atol=1.0e-12,
        )


def test_projected_generalized_eigensolver_rejects_invalid_hermitian_pencil() -> None:
    hamiltonian = np.array([[0.0, 1.0], [1.0, 0.0]])
    legal_overlap = np.array([[1.0, 0.5], [0.5, 1.0]])
    for invalid_cutoff in (0.0, -1.0, np.inf, np.nan):
        with pytest.raises(
            ValueError, match="overlap_eigenvalue_cutoff must be finite and positive"
        ):
            _solve_projected_pencil(
                hamiltonian, legal_overlap, overlap_eigenvalue_cutoff=invalid_cutoff
            )
    for invalid_shape in (np.ones(2), np.ones((2, 3)), np.empty((0, 0))):
        with pytest.raises(ValueError, match="square matrices with the same shape"):
            _solve_projected_pencil(invalid_shape, invalid_shape, symmetrize_matrices=False)
    with pytest.raises(ValueError, match="square matrices with the same shape"):
        _solve_projected_pencil(np.eye(3), legal_overlap, symmetrize_matrices=False)
    # Positive overlap, finite H: whitening alone overflows (1e308 / 1e-4).
    with np.errstate(over="ignore", invalid="ignore"):
        rejected, _, reason, _ = _solve_projected_pencil([[1.0e308]], [[1.0e-4]])
    assert rejected is None and reason == "nonfinite_effective_hamiltonian"
    legal = _solve_projected_pencil(hamiltonian, legal_overlap, symmetrize_matrices=False)[0]
    assert legal.eigenvalues[0] == pytest.approx(-2.0, rel=0.0, abs=2.0e-14)
    np.testing.assert_allclose(
        legal.eigenvectors.conj().T @ legal_overlap @ legal.eigenvectors,
        np.eye(2),
        rtol=0.0,
        atol=2.0e-14,
    )
    scale = 1.0e-310
    for h, s, options, expected in (
        (hamiltonian, [[1.0, 5.0], [0.5, 1.0]], {}, "non_hermitian_overlap_matrix"),
        (hamiltonian, [[1.0, 2.0], [2.0, 1.0]], {}, "negative_deterministic_overlap_eigenvalue"),
        (hamiltonian, [[1.0, np.nan], [np.nan, 1.0]], {}, "nonfinite_overlap_matrix"),
        ([[0.0, 5.0], [0.5, 0.0]], legal_overlap, {}, "non_hermitian_hamiltonian_matrix"),
        (
            hamiltonian,
            np.diag([-0.1, 2.0]),
            {"overlap_eigenvalue_cutoff": 0.2, "symmetrize_matrices": True},
            "negative_deterministic_overlap_eigenvalue",
        ),
        (
            np.diag([1.0, -1.0e-16]),
            np.diag([1.0, 1.0e-16 + 2.0e-16j]),
            {"overlap_eigenvalue_cutoff": 1.0e-20},
            "invalid_overlap_normalization",
        ),
        (
            scale * np.eye(2),
            scale * np.array([[1.0, 0.75], [0.25, 1.0]]),
            {"overlap_eigenvalue_cutoff": 1.0e-312},
            "non_hermitian_overlap_matrix",
        ),
    ):
        rejected, _, reason, _ = _solve_projected_pencil(
            h, s, **({"symmetrize_matrices": False} | options)
        )
        assert rejected is None and reason == expected

    smallest_float = np.nextafter(0.0, 1.0)
    # Averaging halves only after adding, so subnormal entries survive symmetrization.
    for symmetrize in (False, True):
        result = _solve_projected_pencil(
            [[smallest_float]],
            [[2.0 * smallest_float]],
            overlap_eigenvalue_cutoff=smallest_float,
            symmetrize_matrices=symmetrize,
        )[0]
        assert result.eigenvalues[0] == pytest.approx(0.5, rel=0.0, abs=1.0e-15)


def test_sampled_enclosure_uses_numerical_window_without_rejecting_positive_subspace():
    """The relation H = scale*S fixes the usable projected eigenvalue even when sampled S has a
    negative direction.
    """
    overlap = np.array([[1.0, 1.2], [1.2, 1.0]])
    for scale in (-3.0, 0.0, 1e-200, 1e200):
        result, _, failure, _ = _solve_projected_pencil(
            scale * overlap, overlap, _sampled_overlap=True
        )
        assert failure is None
        assert result.eigenvalues[0] == pytest.approx(scale, rel=2e-14, abs=0.0)
        pencil = _sampled_pencil_diagnostics(
            eigensolver=result,
            solver_failure_reason=failure,
            enclosure=(scale, scale),
            relative_tolerance=1e-8,
        )
        assert pencil["enclosure_passed"] is True
    for relative_distance, passed in ((0.5e-6, True), (2e-6, False)):
        for scale in (1e-100, 1.0, 1e100):
            eigen = _solve_projected_pencil([[scale * (1 + relative_distance)]], np.eye(1))[0]
            pencil = _sampled_pencil_diagnostics(
                eigensolver=eigen,
                solver_failure_reason=None,
                enclosure=(-scale, scale),
                relative_tolerance=1e-6,
            )
            assert pencil["enclosure_passed"] is passed


def test_projected_backward_error_uses_scaled_denominator_without_overflow():
    scale = 8e307
    h = np.array([[1.0, 0.1], [0.1, -1.0]]) * scale
    result = _solve_projected_pencil(h, np.eye(2), symmetrize_matrices=False)[0]
    coefficient = result.ground_state_coefficients
    denominator = (
        np.linalg.norm(h / scale) + abs(result.eigenvalues[0] / scale) * np.sqrt(2)
    ) * np.linalg.norm(coefficient)
    assert denominator > np.finfo(float).max / scale
    expected = (result.residual_norm / scale) / denominator
    assert result.generalized_eigenpair_backward_error == pytest.approx(
        expected, rel=1e-12, abs=0.0
    )


def test_ancilla_population_normalization_and_incremental_pricing():
    from test_adapt_primary import plan_for
    from nwqlib.core import Source
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.execution import ObservationChunk, RegisterMap
    from nwqlib.algorithms.gcim.adapt_acquisition import _point, _read_values, _PartialData
    from nwqlib.algorithms.gcim.pencil import matrix_element_count

    plan = plan_for(shots=4)
    point = _point(plan, "pair", term=0)
    identity = "sha256:" + "1" * 64
    fields = dict(run_id="r", plan_id=identity, realization_id=point.content_id, prepared_id=identity,
                  experiment="pair", setting="s", bindings=point.bindings, quantum_layout=(),
                  classical_layout=(RegisterMap(name="c", bits=(0,)),), attempt="a", job="j", chunk="c",
                  observation=ObservationSpec(kind="counts", shots=4), population="unconditional",
                  trajectories=None, source=Source(name="supplied", version="1", domain="test", reference="counts"))
    chunk = ObservationChunk.from_histogram({"0": 3, "1": 1}, returned_shots=4, **fields)
    assert next(iter(_read_values(plan, point, chunk).values())) == 0.5
    chunk = ObservationChunk.from_histogram({}, returned_shots=0, **fields)
    with pytest.raises(_PartialData, match="empty returned counts"):
        _read_values(plan, point, chunk)
    assert matrix_element_count(3, 2, 1) == 6


def test_projected_backward_error_is_invariant_under_energy_unit_scaling():
    h = np.array([[0.7, 0.2], [0.2, -0.3]])
    s = np.array([[1.0, 0.9], [0.9, 0.82]])
    base = _solve_projected_pencil(h, s, symmetrize_matrices=False)[0]
    for scale in (2.0**-40, 2.0**40):
        result = _solve_projected_pencil(scale * h, s, symmetrize_matrices=False)[0]
        assert result.eigenvalues[0] / scale == pytest.approx(
            base.eigenvalues[0], rel=2e-14, abs=0.0
        )
        assert result.residual_norm / scale == pytest.approx(
            base.residual_norm, rel=1e-10, abs=1e-28
        )
        assert result.generalized_eigenpair_backward_error == pytest.approx(
            base.generalized_eigenpair_backward_error, rel=1e-10, abs=1e-28
        )
