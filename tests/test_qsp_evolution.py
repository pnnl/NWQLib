"""Tests for the QSP/QSVT subroutines and the LCHS qsp_block_encoding wiring.

Tolerance classes follow the FRAMEWORK discipline: convention identities and
circuit known answers are machine-precision (<= 1e-12); Hamiltonian-evolution
equivalence uses the documented per-node error bound derived from the Bessel
tail, solver residuals, and the recorded rescale (method-error-bounded).
Integer degree anchors carry the recorded
Bessel-tail slack (platform-quantities rule); no wall-clock assertions.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
import scipy.linalg
import scipy.special
from qiskit.circuit import Gate
from qiskit.quantum_info import Operator

from conftest import (
    _encoded_block,
    load_docs_script,
)
from nwqlib.subroutines.block_encoding import (
    BandSpecification,
    build_block_encoding,
)
from nwqlib.subroutines.qsp import (
    QSP_OAA_QUERY_MULTIPLIER,
    QSP_SOLVER_RESIDUAL_TOLERANCE,
    build_control_diagonal_generator_encoding,
    build_qsvt_circuit,
    build_real_chebyshev_encoding,
    chebyshev_grid,
    evaluate_qsp_polynomial,
    jacobi_anger_expansion,
    solve_symmetric_qsp_phases,
    wx_phases_to_reflection,
)
from nwqlib.subroutines.qsp.evolution import build_qsp_evolution_encoding
from nwqlib.subroutines.qsp.phases import (
    chebyshev_polynomial_sup_bound,
    chebyshev_norming_sup_bound,
)

# Integer degree and analysis-terminal selections stay exactly pinned; only the
# tail bound gets a relative allowance for SciPy/libm Bessel evaluation
# differences. This 1e-12 relative class is far below the anchor-comparison
# classes used elsewhere in the suite.
JACOBI_ANGER_TAIL_ANCHOR_RTOL = 1.0e-12

# Four libm terms as large as lgamma(85) ~= 291 give a roughly 2.6e-13
# relative ceiling; 1e-11 leaves about 40x platform headroom and remains
# at least nine orders below a formula-scale regression.
JACOBI_ANGER_ANALYTIC_REMAINDER_RTOL = 1.0e-11


def _banded_tridiagonal_encoding():
    specification = BandSpecification(
        offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2
    )
    encoding = build_block_encoding(specification)
    matrix = np.zeros((4, 4), dtype=complex)
    columns = np.arange(4)
    for offset, coefficient in zip(specification.offsets, specification.coefficients):
        matrix[(columns + offset) % 4, columns] = coefficient
    return encoding, matrix


def _matrix_function(matrix: np.ndarray, values: np.ndarray) -> np.ndarray:
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    return eigenvectors @ np.diag(values(eigenvalues)) @ eigenvectors.conj().T


def test_hermitian_admission_uses_evidence_without_touching_a_large_operator(monkeypatch):
    from nwqlib.subroutines.qsp import evolution

    # The public admission must work with circuit access alone, even at a
    # width where a dense matrix would occupy 16 PiB (complex128, 25 qubits).
    class Oracle:
        alpha = 1.0
        system_qubits = 25
        metadata = {}

        @property
        def circuit(self):
            pytest.fail("Hermitian admission inspected the circuit")

    def reached_synthesis(*args, **kwargs):
        raise RuntimeError("reached scalar synthesis")

    monkeypatch.setattr(evolution, "jacobi_anger_expansion", reached_synthesis)
    oracle = Oracle()
    with pytest.warns(UserWarning, match="conditional"):
        with pytest.raises(RuntimeError, match="reached scalar synthesis"):
            evolution.build_qsp_evolution_encoding(oracle, evolution_time=.1, epsilon=.01)
    with pytest.raises(ValueError, match="evidence is unavailable"):
        evolution.build_qsp_evolution_encoding(
            oracle, evolution_time=.1, epsilon=.01, require_hermitian_evidence=True)
    for field in ("target_operator_is_hermitian", "encoded_operator_is_hermitian"):
        oracle.metadata = {field: False}
        with pytest.raises(ValueError, match="requires Hermitian"):
            evolution.build_qsp_evolution_encoding(oracle, evolution_time=.1, epsilon=.01)
    oracle.metadata = {"target_operator_is_hermitian": True, "encoded_operator_is_hermitian": True}
    with pytest.raises(RuntimeError, match="reached scalar synthesis"):
        evolution.build_qsp_evolution_encoding(
            oracle, evolution_time=.1, epsilon=.01, require_hermitian_evidence=True)


@pytest.mark.parametrize("implementation", ("pauli_lcu", "dense_dilation"))
def test_library_built_nonhermitian_input_rejects_before_qsp_synthesis(monkeypatch, implementation):
    from nwqlib.subroutines.qsp import evolution

    encoding = build_block_encoding(np.array([[0., 1.], [0., 0.]]), implementation=implementation)
    assert encoding.metadata["target_operator_is_hermitian"] is False
    monkeypatch.setattr(evolution, "jacobi_anger_expansion",
                        lambda *a, **k: pytest.fail("non-Hermitian input reached polynomial synthesis"))
    with pytest.raises(ValueError, match="requires Hermitian"):
        evolution.build_qsp_evolution_encoding(encoding, evolution_time=.3, epsilon=1e-3)
    valid = build_block_encoding(np.array([[1., 1j], [-1j, 2.]]), implementation=implementation)
    assert valid.metadata["target_operator_is_hermitian"] is True
    assert valid.metadata["encoded_operator_is_hermitian"] is True


def test_joint_generator_propagates_one_nonhermitian_child_only_at_nonzero_weight():
    from nwqlib.subroutines.qsp.evolution import build_control_diagonal_generator_encoding

    nonhermitian = build_block_encoding(np.array([[0., 1.], [0., 0.]]), implementation="pauli_lcu")
    hermitian = build_block_encoding(np.diag([1., -1.]), implementation="pauli_lcu")
    joint = build_control_diagonal_generator_encoding(nonhermitian, hermitian,
        l_diagonal=[1., 2.], h_diagonal=[1., 1.])
    assert joint.metadata["target_operator_is_hermitian"] is False
    ignored = build_control_diagonal_generator_encoding(nonhermitian, hermitian,
        l_diagonal=[0., 0.], h_diagonal=[1., 1.])
    assert ignored.metadata["target_operator_is_hermitian"] is True


def test_periodic_shift_adjoint_pairs_supply_exact_hermitian_evidence():
    # S**(-1) = (S**1).dagger, including complex conjugate coefficients.
    for coefficients, expected in (((1j, -1j), True), ((1j, 1j), False)):
        encoding = build_block_encoding(BandSpecification(
            offsets=(-1, 1), coefficients=coefficients, num_qubits=3))
        matrix = _encoded_block(encoding.circuit, encoding.num_ancillas)
        assert np.allclose(matrix, matrix.conj().T, rtol=0., atol=1e-12) == expected
        assert encoding.metadata["target_operator_is_hermitian"] is expected
        assert encoding.metadata["encoded_operator_is_hermitian"] is expected


def test_joint_generator_does_not_discard_an_imaginary_control_coefficient(monkeypatch):
    from nwqlib.subroutines.qsp import evolution

    encoding = build_block_encoding(np.diag([.5, -.25]), implementation="pauli_lcu")
    legal = build_control_diagonal_generator_encoding(encoding, None, l_diagonal=[1+0j])
    assert legal.metadata["encoded_operator_is_hermitian"] is True
    monkeypatch.setattr(evolution, "_diagonal_rotation_circuit",
                        lambda *a, **k: pytest.fail("non-Hermitian coefficients reached synthesis"))
    for imaginary in (1e-300, 1e-13, .5):
        with pytest.raises(ValueError, match="entries must be real"):
            build_control_diagonal_generator_encoding(encoding, None, l_diagonal=[1+1j*imaginary])


# ---------------------------------------------------------------------------
# (a) Wx-convention evaluation, conversions, and the symmetric phase solver
# ---------------------------------------------------------------------------


def test_all_zero_phases_realize_chebyshev_t_d() -> None:
    """Known answer (machine precision): zero phases realize T_d(x)."""

    grid = chebyshev_grid(64)
    for degree in (1, 2, 5, 9):
        realized = evaluate_qsp_polynomial(np.zeros(degree + 1), grid)
        expected = np.cos(degree * np.arccos(grid))
        assert np.max(np.abs(realized - expected)) <= 1.0e-12

    # Endpoint nodes of the degree-255 verification grid, where 1 - x^2 is
    # about 1e-8. The reference is T_d at the binary64 node in exact rational
    # arithmetic, rounded once. The allowance 4 d u, with u = 2**-53, is the
    # roundoff scale of a product of d unitary 2 x 2 factors. An absolute
    # error u in the signal's s^2 = 1 - x^2 would instead reach about
    # d^2 u / 2 here (Markov's polynomial inequality), and forming 1 - x**2
    # directly gives 1.7e-12 with NumPy 2.5.2.
    degree = 255
    grid = chebyshev_grid(64 * (degree + 1))
    nodes = np.concatenate([grid[:20], grid[-20:]])

    def exact_chebyshev(x: float) -> float:
        x = Fraction(x)
        previous, current = Fraction(1), x
        for _ in range(degree - 1):
            previous, current = current, 2 * x * current - previous
        return float(current)

    expected = np.array([exact_chebyshev(float(x)) for x in nodes])
    realized = evaluate_qsp_polynomial(np.zeros(degree + 1), nodes)
    assert np.max(np.abs(realized.real - expected)) <= 4 * degree * np.finfo(float).eps / 2
    assert np.all(realized.imag == 0.0)


def test_reflection_convention_identity() -> None:
    """[MRTC] arXiv:2105.02859v5, App. A: Wx product == i^d x reflection product, phases shifted."""

    rng = np.random.default_rng(3)
    grid = np.linspace(-0.99, 0.99, 41)
    root = np.sqrt(1.0 - grid**2)
    reflection = np.zeros((grid.size, 2, 2), dtype=complex)
    reflection[:, 0, 0] = grid
    reflection[:, 1, 1] = -grid
    reflection[:, 0, 1] = root
    reflection[:, 1, 0] = root
    for degree in (1, 2, 5):
        phases = rng.normal(size=degree + 1)
        converted, global_phase = wx_phases_to_reflection(phases)

        def phase_matrix(phi: float) -> np.ndarray:
            stack = np.zeros((grid.size, 2, 2), dtype=complex)
            stack[:, 0, 0] = np.exp(1j * phi)
            stack[:, 1, 1] = np.exp(-1j * phi)
            return stack

        product = phase_matrix(converted[0])
        for phi in converted[1:]:
            product = product @ reflection @ phase_matrix(phi)
        via_reflection = np.exp(1j * global_phase) * product[:, 0, 0]
        direct = evaluate_qsp_polynomial(phases, grid)
        assert np.max(np.abs(direct - via_reflection)) <= 1.0e-12


def test_solved_phases_reproduce_target_coefficients() -> None:
    """Known answer (machine precision): solved Re P matches the target at 1e-12."""

    for tau, parity in ((1.5, 0), (1.5, 1), (4.0, 1)):
        expansion = jacobi_anger_expansion(tau, 1.0e-6)
        coefficients = np.asarray(
            expansion.cos_coefficients if parity == 0 else expansion.sin_coefficients
        )
        degree = coefficients.size - 1
        target_bound = chebyshev_polynomial_sup_bound(coefficients)
        scale = max(1.0, target_bound)
        scale *= 1.0 + 1.0e-3  # solver margin; recorded upstream by evolution.py
        target = coefficients / scale
        solution = solve_symmetric_qsp_phases(target)
        assert solution.max_residual <= 1.0e-12
        verification_grid = chebyshev_grid(64 * (degree + 1))
        verification_residual = evaluate_qsp_polynomial(
            solution.phases, verification_grid
        ).real - np.polynomial.chebyshev.chebval(verification_grid, target)
        expected_sup_bound = chebyshev_norming_sup_bound(
            float(np.max(np.abs(verification_residual))),
            degree=degree,
            num_points=verification_grid.size,
        )
        assert solution.residual_sup_bound == pytest.approx(expected_sup_bound, rel=1.0e-13, abs=0)
        assert solution.to_dict()["residual_sup_bound"] == solution.residual_sup_bound
        # Recover the realized Chebyshev coefficients and compare directly.
        grid = chebyshev_grid(2 * (degree + 1))
        realized = evaluate_qsp_polynomial(solution.phases, grid).real
        recovered = np.polynomial.chebyshev.chebfit(grid, realized, degree)
        assert np.max(np.abs(recovered - target)) <= 1.0e-12


def test_phase_solver_rejects_invalid_targets() -> None:
    with pytest.raises(ValueError, match="definite parity"):
        solve_symmetric_qsp_phases([0.1, 0.5, 0.2])
    with pytest.raises(ValueError, match="max\\|f\\| <= 1"):
        solve_symmetric_qsp_phases([0.0, 1.4])
    # The 192-node shared grid still undersamples the endpoint maximum of
    # 1.0001*T_2, while the norming factor recovers its true supremum.
    target = np.array([0.0, 0.0, 1.0001])
    grid = chebyshev_grid(64 * target.size)
    grid_max = float(np.max(np.abs(np.polynomial.chebyshev.chebval(grid, target))))
    assert grid_max < 1.0
    assert chebyshev_polynomial_sup_bound(target) > 1.0
    with pytest.raises(ValueError, match="max\\|f\\| <= 1"):
        solve_symmetric_qsp_phases(target)


def test_chebyshev_grid_norming_bound_covers_polynomial_suprema() -> None:
    coefficients = np.zeros(33)
    coefficients[-1] = 1.0
    assert chebyshev_polynomial_sup_bound(coefficients) == pytest.approx(1.0, abs=2.0e-15)

    grid = chebyshev_grid(12)
    t_two = np.polynomial.chebyshev.chebval(grid, [0.0, 0.0, 1.0])
    assert chebyshev_norming_sup_bound(
        float(np.max(np.abs(t_two))), degree=2, num_points=12
    ) == pytest.approx(1.0, abs=2.0e-15)

    with pytest.raises(ValueError, match="num_points > degree"):
        chebyshev_norming_sup_bound(0.5, degree=3, num_points=3)

    rng = np.random.default_rng(20260709)
    for degree in range(1, 13):
        coefficients = np.zeros(degree + 1)
        coefficients[degree % 2 :: 2] = rng.normal(size=(degree + 2) // 2)
        num_points = 4 * (degree + 1)
        samples = np.polynomial.chebyshev.chebval(chebyshev_grid(num_points), coefficients)
        bound = chebyshev_norming_sup_bound(
            float(np.max(np.abs(samples))),
            degree=degree,
            num_points=num_points,
        )
        derivative_roots = np.polynomial.chebyshev.chebroots(
            np.polynomial.chebyshev.chebder(coefficients)
        )
        real_roots = derivative_roots[
            (np.abs(derivative_roots.imag) <= 1.0e-12) & (np.abs(derivative_roots.real) <= 1.0)
        ].real
        extrema = np.concatenate(([-1.0, 1.0], real_roots))
        true_sup = float(np.max(np.abs(np.polynomial.chebyshev.chebval(extrema, coefficients))))
        assert true_sup <= bound * (1.0 + 2.0e-13)


# ---------------------------------------------------------------------------
# (b) QSVT circuits: known answers, real passes, evolution synthesis
# ---------------------------------------------------------------------------


def test_qsvt_trivial_phases_block_encode_chebyshev_of_matrix() -> None:
    """Circuit-level known answer (machine precision, the probe target):

    zero Wx phases turn a block encoding of A into one of T_d(A / alpha).
    The banded U_BE is non-unitary-symmetric, so this catches alternation,
    projector-phase, conversion, and Qiskit-sign errors at once.
    """

    def chebyshev_t(degree: int):
        # Polynomial form: dense_dilation places an eigenvalue at exactly
        # +-1, where arccos is fp-fragile; chebval is not.
        coefficients = np.zeros(degree + 1)
        coefficients[degree] = 1.0
        return lambda x: np.polynomial.chebyshev.chebval(x, coefficients)

    encoding, matrix = _banded_tridiagonal_encoding()
    scaled = matrix / encoding.alpha
    for degree in (1, 2, 3):
        circuit = build_qsvt_circuit(encoding, np.zeros(degree + 1))
        block = _encoded_block(circuit, encoding.num_ancillas + 1)
        expected = _matrix_function(scaled, chebyshev_t(degree))
        assert np.max(np.abs(block - expected)) <= 1.0e-12

    dense = build_block_encoding(
        np.array([[0.3, 0.1], [0.1, -0.2]], dtype=complex), implementation="dense_dilation"
    )
    circuit = build_qsvt_circuit(dense, np.zeros(3))
    block = _encoded_block(circuit, dense.num_ancillas + 1)
    scaled = np.array([[0.3, 0.1], [0.1, -0.2]]) / dense.alpha
    expected = _matrix_function(scaled, chebyshev_t(2))
    assert np.max(np.abs(block - expected)) <= 1.0e-12


def test_qsvt_random_phases_match_classical_polynomial() -> None:
    """The circuit block equals the classically evaluated P applied to A/alpha."""

    encoding, matrix = _banded_tridiagonal_encoding()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / encoding.alpha)
    rng = np.random.default_rng(5)
    for degree in (2, 3):
        phases = rng.normal(size=degree + 1)
        circuit = build_qsvt_circuit(encoding, phases)
        block = _encoded_block(circuit, encoding.num_ancillas + 1)
        values = evaluate_qsp_polynomial(phases, eigenvalues)
        expected = eigenvectors @ np.diag(values) @ eigenvectors.conj().T
        assert np.max(np.abs(block - expected)) <= 1.0e-12


def test_real_chebyshev_pass_encodes_real_part() -> None:
    encoding, matrix = _banded_tridiagonal_encoding()
    eigenvalues, eigenvectors = np.linalg.eigh(matrix / encoding.alpha)
    rng = np.random.default_rng(7)
    for degree in (2, 3, 5):
        phases = rng.normal(size=degree + 1) * 0.4
        circuit = build_real_chebyshev_encoding(encoding, phases)
        block = _encoded_block(circuit, encoding.num_ancillas + 2)
        values = evaluate_qsp_polynomial(phases, eigenvalues).real
        expected = eigenvectors @ np.diag(values) @ eigenvectors.conj().T
        assert np.max(np.abs(block - expected)) <= 1.0e-12


def test_jacobi_anger_degree_anchors_with_recorded_slack() -> None:
    """Integer degree anchors are valid only with recorded tail slack.

    Each anchored degree clears its epsilon by the recorded slack (>= 2x
    here), so cross-platform last-ulp differences in the Bessel sums cannot
    flip the integers (FRAMEWORK platform-quantities rule).
    """

    for tau, epsilon, expected_degree in ((1.5, 1.0e-6, 8), (0.72, 1.0e-2, 3)):
        expansion = jacobi_anger_expansion(tau, epsilon)
        assert expansion.tail_slack >= 2.0
        assert expansion.degree == expected_degree
        assert expansion.tail_bound <= epsilon
        # One-step-coarser truncation must violate the bound (minimality).
        previous_tail = expansion.tail_bound + 2.0 * abs(
            float(scipy.special.jv(expansion.degree, tau))
        )
        assert previous_tail > epsilon
    parity_check = jacobi_anger_expansion(1.5, 1.0e-6)
    assert parity_check.cos_degree % 2 == 0
    assert parity_check.sin_degree % 2 == 1


@pytest.mark.parametrize(
    ("tau", "epsilon"),
    ((0.1, 1.0e-10), (5.0, 1.0e-7), (50.0, 1.0e-5)),
)
def test_jacobi_anger_records_complete_finite_and_analytic_tail(
    tau: float,
    epsilon: float,
) -> None:
    expansion = jacobi_anger_expansion(tau, epsilon)
    terminal = expansion.analysis_terminal
    first = terminal + 1
    ratio = tau / (2.0 * (terminal + 2))
    log_remainder = (
        first * np.log(tau / 2.0)
        - scipy.special.gammaln(first + 1)
        + tau**2 / (4.0 * (first + 1))
        - np.log1p(-ratio)
    )
    magnitudes = np.abs(scipy.special.jv(np.arange(terminal + 1), tau))
    remainder = float(np.exp(log_remainder))
    assert remainder > 0.0
    assert expansion.analytic_infinite_tail_bound == pytest.approx(
        4.0 * remainder,
        rel=JACOBI_ANGER_ANALYTIC_REMAINDER_RTOL,
        abs=0.0,
    )
    shared_remainder = expansion.analytic_infinite_tail_bound / 4.0
    previous_terminal = terminal - 1
    previous_first = previous_terminal + 1
    previous_ratio = tau / (2.0 * (previous_terminal + 2))
    previous_log_remainder = (
        previous_first * np.log(tau / 2.0)
        - scipy.special.gammaln(previous_first + 1)
        + tau**2 / (4.0 * (previous_first + 1))
        - np.log1p(-previous_ratio)
    )
    previous_best_tail = 2.0 * abs(float(scipy.special.jv(previous_terminal, tau))) + 4.0 * float(
        np.exp(previous_log_remainder)
    )
    assert previous_best_tail > epsilon
    even_finite = 2.0 * float(
        np.sum(magnitudes[np.arange(expansion.cos_degree + 2, terminal + 1, 2)])
    )
    odd_finite = 2.0 * float(
        np.sum(magnitudes[np.arange(expansion.sin_degree + 2, terminal + 1, 2)])
    )
    assert expansion.cos_tail_bound == even_finite + 2.0 * shared_remainder
    assert expansion.sin_tail_bound == odd_finite + 2.0 * shared_remainder
    assert expansion.tail_bound <= epsilon

    # The analytic remainder covers a much longer direct suffix independently
    # of the finite terminal used by the implementation.
    check_orders = np.arange(terminal + 1, terminal + 1001)
    direct_remainder = float(np.sum(np.abs(scipy.special.jv(check_orders, tau))))
    assert direct_remainder <= shared_remainder * (1.0 + 1.0e-12)


@pytest.mark.parametrize(
    ("tau", "epsilon", "min_degree", "expected"),
    [
        (0.1, 1.0e-3, 1, (2, 3, 4.2693346287522684e-5)),
        (5.0, 1.0e-4, 2, (12, 13, 6.152867677760108e-5)),
        (20.0, 1.0e-3, 2, (30, 34, 9.976982850285507e-4)),
        (50.0, 1.0e-3, 1, (62, 80, 8.476776396111822e-4)),
        (100.0, 1.0e-3, 1, (117, 155, 8.780965809779798e-4)),
        (200.0, 1.0e-3, 1, (219, 307, 8.72946806664447e-4)),
        (300.0, 1.0e-3, 2, (323, 458, 8.31286632429867e-4)),
    ],
)
def test_jacobi_anger_restart_preserves_selected_terminal(
    tau: float,
    epsilon: float,
    min_degree: int,
    expected: tuple[int, int, float],
) -> None:
    expansion = jacobi_anger_expansion(tau, epsilon, min_degree=min_degree, max_degree=max(256, expected[0]))
    expected_degree, expected_terminal, expected_tail = expected
    assert expansion.degree == expected_degree
    assert expansion.analysis_terminal == expected_terminal
    assert expansion.tail_bound == pytest.approx(
        expected_tail,
        rel=JACOBI_ANGER_TAIL_ANCHOR_RTOL,
        abs=0.0,
    )


def test_unrepresentable_analytic_remainder_keeps_evolution_build_available() -> None:
    tau = 1.0e-100
    expansion = jacobi_anger_expansion(tau, 1.0e-10, min_degree=2)
    assert expansion.analytic_infinite_tail_bound is None
    assert expansion.tail_bound is None and expansion.tail_slack is None

    encoding = build_block_encoding(
        np.diag([0.2, -0.1]).astype(complex),
        implementation="dense_dilation",
    )
    evolution = build_qsp_evolution_encoding(
        encoding,
        evolution_time=tau / encoding.alpha,
        epsilon=1.0e-10,
    )
    record = evolution.metadata["jacobi_anger"]
    assert record["analytic_infinite_tail_bound"] is None
    assert evolution.error_bound is None
    assert evolution.metadata["target_rescale_source"] == ("chebyshev_grid_norming_bound")


def test_qsp_evolution_does_not_retry_invalid_phase_input(monkeypatch) -> None:
    from nwqlib.subroutines.qsp import evolution

    failure = ValueError("invalid phase target")
    calls = 0

    def invalid_target(_coefficients, **_controls):
        nonlocal calls
        calls += 1
        raise failure

    monkeypatch.setattr(evolution, "solve_symmetric_qsp_phases", invalid_target)
    with pytest.raises(ValueError) as caught:
        evolution.build_qsp_evolution_encoding(
            SimpleNamespace(alpha=1.0), evolution_time=0.1, epsilon=1.0e-3
        )
    assert caught.value is failure
    assert calls == 1


@pytest.mark.parametrize("nonfinite", [False, True])
def test_qsp_evolution_retries_only_phase_convergence_failure(monkeypatch, nonfinite) -> None:
    from nwqlib.subroutines.qsp import evolution, phases

    targets = []

    def failed_optimizer(_problem, start):
        # Exercise the real final-residual rejection without running optimization.
        return (np.full_like(start, np.nan) if nonfinite else start), 0.0

    def record_phase_solve(coefficients, **controls):
        targets.append(np.asarray(coefficients).copy())
        return phases.solve_symmetric_qsp_phases(coefficients, **controls)

    monkeypatch.setattr(phases, "_solve_from", failed_optimizer)
    monkeypatch.setattr(evolution, "solve_symmetric_qsp_phases", record_phase_solve)
    with pytest.raises(ValueError, match="QSP phase solving failed") as caught:
        evolution.build_qsp_evolution_encoding(
            SimpleNamespace(alpha=1.0), evolution_time=0.1, epsilon=1.0e-3
        )
    assert isinstance(caught.value.__cause__, phases._QSPPhaseConvergenceError)
    assert len(targets) == 2
    assert np.max(np.abs(targets[1])) < np.max(np.abs(targets[0]))


def test_qsp_evolution_matches_dense_expm_within_documented_bound() -> None:
    """Per-node equivalence (method-error-bounded): QSP circuit vs expm.

    Separating instances of the joint generator with length-one diagonals:
    a supplied Hermitian generator (k L + H via dense dilation), the periodic
    heat-PDE banded instance (H = 0, banded child), and a k = 0 node (pure H).
    """


    l_matrix = np.array([[0.5, 0.1], [0.1, 0.3]], dtype=complex)
    h_matrix = np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex)
    encoding_l = build_block_encoding(l_matrix, implementation="dense_dilation")
    encoding_h = build_block_encoding(h_matrix, implementation="dense_dilation")
    shift=np.roll(np.eye(4),1,axis=0)
    heat=2*np.eye(4)-shift-shift.T  # Explicit periodic second difference, H=0.
    heat_encoding = build_block_encoding(heat)
    assert heat_encoding.implementation == "banded"

    cases = {
        "hermitian_random_kL_plus_H": (
            build_control_diagonal_generator_encoding(
                encoding_l, encoding_h, l_diagonal=[1.7], h_diagonal=[1.0]
            ),
            1.7 * l_matrix + h_matrix,
            0.5,
            1.0e-5,
        ),
        "heat_banded_kL": (
            build_control_diagonal_generator_encoding(heat_encoding, None, l_diagonal=[-0.6]),
            -0.6 * heat,
            0.3,
            1.0e-4,
        ),
        "k_zero_pure_H": (
            build_control_diagonal_generator_encoding(
                encoding_l, encoding_h, l_diagonal=[0.0], h_diagonal=[1.0]
            ),
            h_matrix,
            0.8,
            1.0e-5,
        ),
    }
    for name, (generator, target, final_time, epsilon) in cases.items():
        evolution = build_qsp_evolution_encoding(
            generator,
            evolution_time=final_time,
            epsilon=epsilon,
        )
        block = _encoded_block(evolution.circuit, evolution.num_ancillas)
        exact = scipy.linalg.expm(-1.0j * final_time * target)
        error = float(np.linalg.norm(block - exact, 2))
        assert error <= evolution.error_bound, name
        # The current cases use at most 1.43 epsilon, leaving 2.8x headroom.
        assert evolution.error_bound <= 4.0 * epsilon, name
        assert evolution.alpha == 1.0
        metadata = evolution.metadata
        assert evolution.error_bound == pytest.approx(
            metadata["oaa_amplitude_deficit"] + metadata["oaa_residual_error_bound"],
            rel=1.0e-14,
            abs=0,
        )
        assert metadata["oaa_query_multiplier"] == QSP_OAA_QUERY_MULTIPLIER
        assert metadata["block_encoding_queries"] == QSP_OAA_QUERY_MULTIPLIER * (
            metadata["jacobi_anger"]["cos_degree"] + metadata["jacobi_anger"]["sin_degree"]
        )
        assert metadata["target_rescale"] >= 1.0
        assert metadata["target_rescale_source"] == "chebyshev_grid_norming_bound"
        assert metadata["target_norming_bound"] < 1.0
        phase_solver = metadata["phase_solver"]
        polynomial_part = metadata["oaa_amplitude"] * (
            metadata["jacobi_anger"]["cos_tail_bound"]
            + metadata["jacobi_anger"]["sin_tail_bound"]
            + metadata["target_rescale"]
            * (
                phase_solver["cos"]["residual_sup_bound"]
                + phase_solver["sin"]["residual_sup_bound"]
            )
        )
        assert metadata["polynomial_part_error_bound"] == polynomial_part
        part_error = polynomial_part + metadata["child_operator_error_contribution"]
        assert metadata["oaa_residual_error_bound"] == pytest.approx(
            6.0 * part_error + 6.0 * part_error**2 + 4.0 * part_error**3,
            rel=1.0e-14,
            abs=0,
        )
        amplitude = metadata["oaa_amplitude"]
        assert metadata["oaa_amplitude_deficit"] == pytest.approx(
            1.0 - (3.0 * amplitude - 4.0 * amplitude**3), rel=1.0e-12,
            abs=0,
        )
        assert metadata["classical_preprocessing_scope"] == "qsp_phase_factor_optimization"


def test_qsp_child_operator_error_uses_physical_time_before_oaa() -> None:
    base = build_block_encoding(
        np.diag([0.2, -0.1]).astype(complex), implementation="dense_dilation"
    )
    child = replace(
        base,
        alpha=3.0,
        error_bound=0.04,
        metadata={
            **dict(base.metadata),
            "error_bound_provenance": ("abstract_operator_bound_physical_generator_frame"),
            "encoded_operator_is_hermitian": True,
        },
    )
    physical_time = 0.25
    evolution = build_qsp_evolution_encoding(
        child,
        evolution_time=physical_time,
        epsilon=1.0e-2,
    )
    metadata = evolution.metadata
    contribution = metadata["child_operator_error_contribution"]
    assert contribution == pytest.approx(physical_time * child.error_bound, rel=2e-14, abs=0)
    assert contribution != pytest.approx(metadata["effective_time_tau"] * child.error_bound, rel=2e-14, abs=0)
    part_error = metadata["polynomial_part_error_bound"] + contribution
    assert metadata["oaa_residual_error_bound"] == pytest.approx(
        6.0 * part_error + 6.0 * part_error**2 + 4.0 * part_error**3,
        rel=2e-14, abs=0,
    )
    child_record = metadata["child_block_encoding"]
    assert child_record["error_bound_provenance"] == (
        "abstract_operator_bound_physical_generator_frame"
    )
    assert child_record["encoded_operator_is_hermitian"] is True
    assert child_record["physical_time_error_bound"] == contribution


# ---------------------------------------------------------------------------
# (c)-(e) LCHS wiring: resolution, rejections, resource tiers
# ---------------------------------------------------------------------------














# ---------------------------------------------------------------------------
# Symmetric-QSP phase-solver robustness
# ---------------------------------------------------------------------------


# Single-source sweep targets: the fast-suite characterization
# must exercise bit-identical instances to the maintainer sweep, so the
# builder is loaded from the script rather than mirrored.
_sweep_module = load_docs_script("docs/scripts/qsp_phase_solver_sweep.py")
_sweep_target = _sweep_module.build_target


def _strictly_admissible_sweep_target(kind: str, degree: int, seed: int) -> np.ndarray:
    target = np.asarray(_sweep_target(kind, degree, seed=seed), dtype=float)
    grid = chebyshev_grid(4 * (degree + 1))
    samples = np.polynomial.chebyshev.chebval(grid, target)
    bound = chebyshev_norming_sup_bound(
        float(np.max(np.abs(samples))),
        degree=degree,
        num_points=grid.size,
    )
    return target * (0.98 / bound) if bound > 0.98 else target


def test_qsp_phase_solver_boundary_characterization() -> None:
    """Fast-suite seeded sweep of the solver's characterized regions.

    Deterministic reproduction class: targets use fixed seeds and the
    solver draws no random numbers, so every value below is a deterministic
    reproduction. The nearest decision threshold is the solver's own
    acceptance gate QSP_SOLVER_RESIDUAL_TOLERANCE = 1e-12. The largest
    residual on these instances is 2.09e-13 (single-term T_255), a 4.8x
    margin to that gate, and the degree-64 instances stay below 3.5e-14.
    The sup-0.5 classes converge through degree 128 in the full maintainer
    sweep. Near-unit targets are rescaled to norming bound 0.98 before the
    solve, matching the public target-premise gate rather than a sampled
    maximum.
    """

    for kind in ("single_term", "random_decay", "random_flat"):
        for degree in (8, 16, 32, 64):
            solution = solve_symmetric_qsp_phases(_sweep_target(kind, degree, seed=degree))
            assert solution.max_residual <= QSP_SOLVER_RESIDUAL_TOLERANCE

    # 0.5 T_255 has |f'| = 0.5 * 255^2 at the endpoints, the largest slope
    # below the default max_degree. Its acceptance depends on evaluating the
    # signal root sqrt((1 - x)(1 + x)) without cancellation.
    endpoint_slope = solve_symmetric_qsp_phases(_sweep_target("single_term", 255, seed=255))
    assert endpoint_slope.max_residual <= QSP_SOLVER_RESIDUAL_TOLERANCE

    hard_ok = solve_symmetric_qsp_phases(_strictly_admissible_sweep_target("near_sup", 32, seed=32))
    assert hard_ok.max_residual <= QSP_SOLVER_RESIDUAL_TOLERANCE


def test_qsp_phase_solver_rejects_nonfinite_targets() -> None:
    """Nonfinite targets must not produce finite-looking accepted phases.

    NaN coefficients sailed through the parity and sup gates (NaN
    comparisons are False) and returned finite-looking phases with
    ``max_residual = nan``; both nonfinite inputs now raise the actionable
    error instead.
    """

    nan_target = np.zeros(8)
    nan_target[7] = np.nan
    with pytest.raises(ValueError, match="finite Chebyshev coefficients"):
        solve_symmetric_qsp_phases(nan_target)

    inf_target = np.zeros(8)
    inf_target[7] = np.inf
    with pytest.raises(ValueError, match="finite Chebyshev coefficients"):
        solve_symmetric_qsp_phases(inf_target)


def test_evolution_reuses_one_child_gate_pair_across_cosine_and_sine(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count one realized child/inverse pair shared by cosine, sine and OAA without repeated
    control synthesis.
    """
    import nwqlib.subroutines.qsp.evolution as evolution

    encoding = build_block_encoding(np.diag([0.25, 0.75]))
    to_gate = encoding.circuit.to_gate
    child_gates: list[Gate] = []
    inversions: list[Gate] = []
    from nwqlib.subroutines import qiskit_compat
    gate_inverse = qiskit_compat.inverse_realized_gate
    gate_control = Gate.control
    inverting = False

    def _record_to_gate(*args: object, **kwargs: object) -> Gate:
        child_gates.append(to_gate(*args, **kwargs))
        return child_gates[-1]

    def _record_inverse(self: Gate, *args: object, **kwargs: object) -> Gate:
        nonlocal inverting
        if child_gates and self is child_gates[0]:
            inversions.append(self)
        inverting = True
        try:
            return gate_inverse(self, *args, **kwargs)
        finally:
            inverting = False

    def _record_control(self: Gate, *args: object, **kwargs: object) -> Gate:
        assert not inverting, "QSP inverse repeated control synthesis"
        return gate_control(self, *args, **kwargs)

    monkeypatch.setattr(encoding.circuit, "to_gate", _record_to_gate)
    monkeypatch.setattr(qiskit_compat, "inverse_realized_gate", _record_inverse)
    monkeypatch.setattr(Gate, "control", _record_control)
    result = evolution.build_qsp_evolution_encoding(
        encoding,
        evolution_time=0.7,
        epsilon=1.0e-3,
    )
    assert result.metadata["jacobi_anger"]["degree"] > 1
    assert len(child_gates) == len(inversions) == 1
    assert child_gates[0].label == "U_BE" and inversions[0] is child_gates[0]


def test_qsp_evolution_synthesizes_each_child_multiplexer_once(monkeypatch) -> None:
    """The controlled parity passes reuse the child's realized UCG synthesis.

    ``build_qsp_evolution_encoding`` builds the child inverse with
    ``native_ucg=False`` (``_block_gate_pair``,
    ``qiskit_compat.inverse_realized_gate``), which realizes the child's
    ``p`` complete multiplexers and reverses their factors before the two
    passes copy the child occurrences. Copies carry that realized definition,
    so the build decomposes ``p`` multiplexers at any degree and constructs
    no adjoint table, also for the OAA inverse of the controlled passes.
    Copies made before realization synthesize independently: an adjoint
    table in the inverse gives ``p(2d - 1)`` decompositions at degree ``d``, the
    ``p q_F + p q_I`` of the ``q_F = d`` forward and ``q_I = d - 1`` inverse
    child queries of the two passes.
    """
    from qiskit.circuit.library import UCGate
    from qiskit.quantum_info import SparsePauliOp

    matrix = SparsePauliOp.from_list(
        [("XZ", 0.5), ("YY", 0.3), ("ZI", -0.2), ("IX", 0.4)]
    ).to_matrix()
    encoding = build_block_encoding(matrix, implementation="pauli_lcu")
    leaves = sum(isinstance(item.operation, UCGate) for item in encoding.circuit.data)
    decompositions = []
    constructions = []
    decompose = UCGate._dec_ucg
    construct = UCGate.__init__

    def _record(self: UCGate):
        decompositions.append(self)
        return decompose(self)

    def _record_construction(self: UCGate, *args, **kwargs):
        constructions.append(self)
        construct(self, *args, **kwargs)

    monkeypatch.setattr(UCGate, "_dec_ucg", _record)
    monkeypatch.setattr(UCGate, "__init__", _record_construction)
    result = build_qsp_evolution_encoding(encoding, evolution_time=0.7, epsilon=1.0e-3)
    assert leaves > 0 and result.metadata["jacobi_anger"]["degree"] > 1
    assert len(decompositions) == leaves
    assert not constructions


def test_small_tau_evolution_synthesizes_within_bound() -> None:
    from nwqlib.subroutines.block_encoding import block_encoding_top_left

    target = 0.05 * np.diag([1.0, -1.0]).astype(complex)
    encoding = build_block_encoding(target)
    evolution = build_qsp_evolution_encoding(
        encoding,
        evolution_time=1.0,
        epsilon=0.01,
    )

    block = block_encoding_top_left(
        Operator(evolution.circuit).data, num_ancillas=evolution.num_ancillas
    )
    exact = scipy.linalg.expm(-1.0j * 1.0 * target)
    assert float(np.linalg.norm(block - exact, 2)) <= evolution.error_bound


def test_qsp_error_scalar_domains_and_signed_time():
    from types import SimpleNamespace as NS
    from nwqlib.subroutines.qsp.evolution import qsp_evolution_error_terms

    prepared = NS(scale=1., expansion=NS(cos_tail_bound=0., sin_tail_bound=0.),
                  cos_solution=NS(residual_sup_bound=0.), sin_solution=NS(residual_sup_bound=0.))
    positive = qsp_evolution_error_terms(prepared, evolution_time=2., child_error_bound=.125)
    negative = qsp_evolution_error_terms(prepared, evolution_time=-2., child_error_bound=.125)
    assert positive == negative and positive["child_error"] == .25
    for value in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            qsp_evolution_error_terms(prepared, evolution_time=1., child_error_bound=value)
        for owner, field in ((prepared.expansion, "cos_tail_bound"), (prepared.expansion, "sin_tail_bound"),
                             (prepared.cos_solution, "residual_sup_bound"),
                             (prepared.sin_solution, "residual_sup_bound")):
            setattr(owner, field, value)
            with pytest.raises(ValueError):
                qsp_evolution_error_terms(prepared, evolution_time=1., child_error_bound=0.)
            setattr(owner, field, 0.)


def test_prepared_scale_domain_preserves_unit_and_large_finite_scales():
    from dataclasses import replace
    from math import isfinite
    from sys import float_info
    from types import SimpleNamespace as NS
    from nwqlib.subroutines.qsp.evolution import QSPPreparedEvolution, qsp_evolution_error_terms

    prepared = QSPPreparedEvolution(expansion=NS(cos_tail_bound=0., sin_tail_bound=0.),
        cos_solution=NS(residual_sup_bound=0.), sin_solution=NS(residual_sup_bound=0.),
        scale=1., target_norming_bounds=(1., 0.), margin=0., attempted_margins=())
    # At s=1, a=1/2 and 3a-4a^3=1 exactly: no amplitude deficit.
    assert prepared.recovery_scale == 1.
    for time in (-2., 0., 2.):
        bounds = qsp_evolution_error_terms(prepared, evolution_time=time, child_error_bound=0.)
        assert bounds["recovery_scale"] == 1. and bounds["error_bound"] == 0.
    for changes in ({"scale": .5}, {"scale": float("inf")}, {"margin": -.1},
                    {"margin": float("nan")}, {"target_norming_bounds": (1.,)},
                    {"target_norming_bounds": (1., 0., 0.)}, {"target_norming_bounds": (-.1, 0.)},
                    {"target_norming_bounds": (float("inf"), 0.)}, {"target_norming_bounds": (1.1, 0.)}):
        with pytest.raises(ValueError):
            replace(prepared, **changes)
    # The helper also accepts duck-typed input, which cannot bypass s>=1.
    invalid = NS(scale=.5, expansion=prepared.expansion,
                 cos_solution=prepared.cos_solution, sin_solution=prepared.sin_solution)
    with pytest.raises(ValueError, match="at least one"):
        qsp_evolution_error_terms(invalid, evolution_time=0., child_error_bound=0.)
    # Scale need only enclose the supplied bounds; independently evaluated
    # margin products are not constrained to bitwise equality.
    assert replace(prepared, scale=2., margin=.1, target_norming_bounds=(1.5, 0.)).scale == 2.
    large = replace(prepared, scale=float_info.max, target_norming_bounds=(float_info.max, 0.))
    bounds = qsp_evolution_error_terms(large, evolution_time=0., child_error_bound=0.)
    assert 0 < bounds["amplitude"] < .5
    # For s>=1, 2s/3 <= 1/(3a-4a^3) <= s. Use loose strict bounds
    # away from the asymptotic endpoint to avoid subnormal rounding sensitivity.
    for recovery in (large.recovery_scale, bounds["recovery_scale"]):
        assert isfinite(recovery) and large.scale/2 < recovery <= large.scale




def test_qsp_controls_reject_before_numerical_work_and_keep_real_targets(monkeypatch):
    from nwqlib.subroutines.qsp import phases, evolution

    with monkeypatch.context() as guard:
        def unexpected(*_args, **_kwargs):
            pytest.fail("invalid input reached numerical selection")
        guard.setattr(phases, "chebyshev_grid", unexpected)
        for target, controls in (
            ([0., .5j], {}), ([0., .5], {"max_degree": 0}),
            ([0., .5], {"max_bytes": 10}), ([0., .5], {"max_evaluations": 0}),
            ([0., .5], {"residual_tolerance": float("nan")}),
            ([0., .5], {"residual_tolerance": 0}),
        ):
            with pytest.raises(ValueError):
                phases.solve_symmetric_qsp_phases(target, **controls)
        guard.setattr(scipy.special, "jv", unexpected)
        for tau, controls in ((float("inf"), {}), (1e100, {}),
                              (1., {"max_bytes": 100}), (1., {"max_degree": True})):
            with pytest.raises(ValueError):
                evolution.jacobi_anger_expansion(tau, .01, **controls)
    # A complex container whose entries are real is mathematically admissible.
    solved = phases.solve_symmetric_qsp_phases(np.array([0., .5], dtype=complex), max_degree=1)
    assert solved.max_residual <= 1e-12
    expansion = evolution.jacobi_anger_expansion(1.5, 1e-6, max_degree=8)
    assert expansion.degree == 8 and expansion.tail_bound <= 1e-6
    with pytest.raises(ValueError, match="max_degree=7"):
        evolution.jacobi_anger_expansion(1.5, 1e-6, max_degree=7)


def test_qsp_evaluation_limit_counts_actual_work_of_both_starts(monkeypatch):
    from nwqlib.subroutines.qsp import phases

    original = phases._phase_matrices
    phase_calls = 0

    def count_phase(*args):
        nonlocal phase_calls
        phase_calls += 1
        return original(*args)

    monkeypatch.setattr(phases, "_phase_matrices", count_phase)
    solved = phases.solve_symmetric_qsp_phases([0., .5])
    # For degree one, each objective or verification builds exactly two
    # phase matrices; count the actual kernel calls independently of its log.
    assert solved.evaluations == phase_calls // 2 and phase_calls % 2 == 0
    phase_calls = 0
    exact = phases.solve_symmetric_qsp_phases([0., .5], max_evaluations=solved.evaluations)
    assert exact.phases == solved.phases and phase_calls == 2 * solved.evaluations
    phase_calls = 0
    with pytest.raises(ValueError, match="max_evaluations"):
        phases.solve_symmetric_qsp_phases([0., .5], max_evaluations=solved.evaluations - 1)
    assert phase_calls == 2 * (solved.evaluations - 1)

    def no_progress(problem, start):
        problem(start)
        return start, 1.0

    monkeypatch.setattr(phases, "_solve_from", no_progress)
    monkeypatch.setattr(phases, "_damped_newton", no_progress)
    phase_calls = 0
    with pytest.raises(ValueError, match="max_evaluations"):
        phases.solve_symmetric_qsp_phases([0., .5], max_evaluations=2)
    # The L-BFGS and Newton starts use the allowance, then the final
    # verification rejects before it builds any phase matrix.
    assert phase_calls == 4


def test_qsp_evolution_evaluation_limit_is_shared_by_parities_and_margins(monkeypatch):
    from nwqlib.subroutines.qsp import evolution, phases

    solve = phases.solve_symmetric_qsp_phases
    optimize = phases._solve_from
    allowances = []
    used = []

    def recorded_solve(coefficients, **controls):
        allowances.append(controls["max_evaluations"])
        # First valid target fails after two actual objective evaluations
        # plus final verification. The second margin must keep this cost.
        if len(allowances) == 1:
            def no_convergence(problem, start):
                problem(start)
                problem(start)
                return start, 0.0
            with monkeypatch.context() as local:
                local.setattr(phases, "_solve_from", no_convergence)
                try:
                    solve(coefficients, **controls)
                except phases._QSPPhaseConvergenceError as error:
                    used.append(error.evaluations)
                    raise
        solved = solve(coefficients, **controls)
        used.append(solved.evaluations)
        return solved

    monkeypatch.setattr(evolution, "solve_symmetric_qsp_phases", recorded_solve)
    prepared = evolution.prepare_qsp_evolution(tau=.1, epsilon=.001, max_evaluations=2000)
    assert phases._solve_from is optimize
    assert prepared.attempted_margins == evolution.QSP_EVOLUTION_TARGET_MARGINS
    assert used[0] == 3 and len(used) == 3
    assert allowances == [2000, 2000 - used[0], 2000 - used[0] - used[1]]
    total = sum(used)
    allowances.clear()
    used.clear()
    exact = evolution.prepare_qsp_evolution(tau=.1, epsilon=.001, max_evaluations=total)
    assert exact == prepared and sum(used) == total
    allowances.clear()
    used.clear()
    # The last solve receives only the remainder. Its exhaustion is reported
    # under the caller's field and full cap, not the remainder.
    with pytest.raises(ValueError, match=rf"^QSP evolution exhausted Caller\.field={total - 1}$"):
        evolution.prepare_qsp_evolution(tau=.1, epsilon=.001, max_evaluations=total - 1,
                                        limit_name="Caller.field")
    assert len(allowances) == 3  # Exhaustion does not start another margin.
    with pytest.raises(ValueError, match=r"^QSP phase solving exhausted max_evaluations=1$"):
        solve([0.0, 0.5], max_evaluations=1)


def test_qsp_prepared_cache_assembly_never_reselects(monkeypatch):
    from nwqlib.subroutines.qsp import evolution

    encoding = build_block_encoding(np.diag([.2, -.1]), implementation="dense_dilation")
    prepared = evolution.prepare_qsp_evolution(tau=encoding.alpha * .2, epsilon=.001)
    build_pass = evolution._build_real_chebyshev_encoding
    selected_phases = []

    def recorded_pass(encoding, phases, **kwargs):
        selected_phases.append(tuple(phases))
        return build_pass(encoding, phases, **kwargs)

    def unexpected(*_args, **_kwargs):
        pytest.fail("assembly recomputed selected expansion or phases")

    monkeypatch.setattr(evolution, "prepare_qsp_evolution", unexpected)
    monkeypatch.setattr(evolution, "jacobi_anger_expansion", unexpected)
    monkeypatch.setattr(evolution, "solve_symmetric_qsp_phases", unexpected)
    monkeypatch.setattr(evolution, "_build_real_chebyshev_encoding", recorded_pass)
    realized = evolution.build_qsp_evolution_encoding(encoding, evolution_time=.2,
                                                      epsilon=.001, _prepared=prepared,
                                                      require_hermitian_evidence=True)
    assert realized.metadata["hermitian_premise"] == "construction_metadata"
    assert realized.circuit.num_qubits == 5
    assert selected_phases == [prepared.cos_solution.phases, prepared.sin_solution.phases]
    assert realized.metadata["phase_solver"]["cos"]["evaluations"] == prepared.cos_solution.evaluations
    block = _encoded_block(realized.circuit, realized.num_ancillas)
    target = scipy.linalg.expm(-.2j * np.diag([.2, -.1]))
    assert np.linalg.norm(block - target, 2) <= realized.error_bound


def test_unavailable_encoding_error_preserves_known_relations_and_archive(monkeypatch, tmp_path):
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.blocks.encoding import select_block_encoding
    from nwqlib.operators import operator_input
    from nwqlib.subroutines.block_encoding import BlockEncoding
    from nwqlib.subroutines.qsp import evolution

    matrix = np.diag([.2, -.1])
    known = build_block_encoding(matrix, implementation="dense_dilation")
    unknown = replace(known, error_bound=None)
    for invalid in (-1., float("nan"), float("inf")):
        with pytest.raises(ValueError):
            replace(unknown, error_bound=invalid)
    assert unknown.alpha == known.alpha and unknown.circuit is known.circuit
    selected = select_block_encoding("unknown", unknown, operator=operator_input(matrix))
    assert selected.record.semantics.epsilon is None
    assert selected.record.semantics.alpha == known.alpha
    generator = build_control_diagonal_generator_encoding(unknown, known, l_diagonal=[.3], h_diagonal=[1.])
    assert generator.error_bound is None and generator.alpha == pytest.approx(1.3 * known.alpha, rel=2e-14, abs=0)
    # A zero diagonal removes the unknown child from the actual relation.
    assert build_control_diagonal_generator_encoding(
        unknown, known, l_diagonal=[0.], h_diagonal=[1.]).error_bound == known.error_bound
    # Ideal rotation blocks are D_i/max|D_i|, so ||sum_i kron(A_i - alpha_i B_i, D_i)||
    # <= sum_i max|D_i| eps_i. Dyadic operands make that float sum exact.
    joint = build_control_diagonal_generator_encoding(
        replace(known, error_bound=.25), replace(known, error_bound=.0625),
        l_diagonal=[.25, -.5], h_diagonal=[2., 1.])
    assert joint.error_bound == .5 * .25 + 2. * .0625
    prepared = evolution.prepare_qsp_evolution(tau=.1, epsilon=.01)
    bounds = evolution.qsp_evolution_error_terms(prepared, evolution_time=.5, child_error_bound=None)
    assert bounds["child_error"] is None and bounds["error_bound"] is None
    assert bounds["polynomial_part_error"] is not None and bounds["recovery_scale"] == prepared.recovery_scale
    assert evolution.qsp_evolution_error_terms(prepared, evolution_time=0., child_error_bound=None)["child_error"] == 0.

    unavailable = evolution.build_qsp_evolution_encoding(known, evolution_time=1e-100,
                                                          epsilon=1e-10)
    assert unavailable.error_bound is None and unavailable.circuit.num_qubits == 5
    files = ArchiveFiles(tmp_path, 4_000_000)
    circuit_path = files.write_circuit("unknown.qpy", unavailable.circuit)
    data = unavailable.to_dict()
    for key in ("num_qubits", "depth", "gate_counts"):
        data.pop(key)
    files.write_json("unknown.json", data)

    def unexpected(*_args, **_kwargs):
        pytest.fail("archive load recomputed a selected numerical model")

    monkeypatch.setattr(np.linalg, "svd", unexpected)
    monkeypatch.setattr(evolution, "prepare_qsp_evolution", unexpected)
    monkeypatch.setattr(evolution, "jacobi_anger_expansion", unexpected)
    monkeypatch.setattr(evolution, "solve_symmetric_qsp_phases", unexpected)
    reopened = ArchiveFiles(tmp_path, 4_000_000)
    restored = BlockEncoding(circuit=reopened.read_circuit(circuit_path), **reopened.read_json("unknown.json"))
    assert restored.error_bound is None and restored.alpha == unavailable.alpha
    assert restored.metadata == unavailable.metadata
    np.testing.assert_allclose(Operator(restored.circuit).data, Operator(unavailable.circuit).data,
                               rtol=0., atol=1e-12)


def _selected_lchs_qsp(matrix,*,time=.1):
    from nwqlib import LinearDynamics,NormSquared,plan
    from nwqlib.algorithms.lchs import LCHS,ProviderConfig
    initial=np.zeros(len(matrix),dtype=complex)
    initial[0]=1
    return plan(LinearDynamics(A=matrix,initial_state=initial,time=time),
        method=LCHS(hamiltonian_evolution_backend='qsp_block_encoding',
            k_quadrature=ProviderConfig(implementation='signed_binary_uniform',parameters={'num_qubits':2,'lsb_position':-1})),
        output=NormSquared())


def test_lchs_qsp_selected_quantities_reach_actual_native_program():
    from nwqlib import prepare,estimate
    from nwqlib.algorithms.lchs.native import LCHS_QSP_EPSILON_FRACTION
    chosen=_selected_lchs_qsp(np.diag([.1,.2]))
    selected=chosen._native['native_data'].qsp_plan
    quad=chosen._native['native_data'].quadrature
    expected_alpha=max(abs(float(k)) for k in quad.k_nodes)*.2
    assert selected['layout']['alpha']==pytest.approx(expected_alpha,rel=1e-12,abs=0)
    assert selected['part_plans'][0].alpha==pytest.approx(.2,rel=1e-12,abs=0)
    assert selected['part_plans'][1].alpha==0
    assert selected['epsilon_he']==LCHS_QSP_EPSILON_FRACTION*chosen.method.approximation_tolerance
    expansion=jacobi_anger_expansion(expected_alpha*.1,selected['epsilon_he'],min_degree=2)
    assert selected['expansion']==expansion
    assert selected['structural_law']['block_encoding_queries']==QSP_OAA_QUERY_MULTIPLIER*(expansion.cos_degree+expansion.sin_degree)
    width=sum(register.width for register in chosen.construction.program.registers)
    assert width<=13
    assert estimate(chosen).construction_id==chosen.construction.content_id
    preview=prepare(chosen)
    assert all(circuit.num_qubits==width for circuit in preview.circuits)


def test_lchs_qsp_heat_and_shifted_input_keep_actual_selected_child():
    shift=np.roll(np.eye(4),1,axis=0)
    heat=.5*(2*np.eye(4)-shift-shift.T)
    chosen=_selected_lchs_qsp(heat,time=.2)
    selected=chosen._native['native_data'].qsp_plan
    assert selected['part_plans'][0].implementation=='banded'
    assert selected['part_plans'][1].implementation=='exact_zero'
    assert selected['part_plans'][0].alpha==pytest.approx(2.,rel=1e-12, abs=0)
    shifted=_selected_lchs_qsp(np.diag([-2.,1.]),time=1.)
    psd=_selected_lchs_qsp(np.diag([0.,3.]),time=1.)
    first,second=(choice._native['native_data'].qsp_plan for choice in (shifted,psd))
    # The original endpoints are -2 and 1. PSD conversion shifts by
    # 2 + psd_tolerance*max(|-2|,|1|), so the new upper endpoint is not 3.
    expected_shift=2.+2.*shifted.method.psd_tolerance
    assert shifted.reconstruction.psd_shift==pytest.approx(expected_shift, rel=2e-14, abs=0)
    assert first['part_plans'][0].alpha==pytest.approx(1.+expected_shift, rel=2e-14, abs=0)
    assert first['structural_law']['block_encoding_queries']==second['structural_law']['block_encoding_queries']


def test_lchs_guide_first_problem_plans_qsp_evolution():
    """The LCHS guide's first problem selects tau = 1.2422 and eps = 1e-3.

    L-BFGS from the Dong-Meng-Whaley-Lin (arXiv:2002.11649v2) start fits the
    degree-4 cosine
    target but ends at a near-singular point with residual 2e-6 on the
    degree-5 sine target (SciPy 1.18.1). The reference for each parity is the untruncated function
    divided by the selected scale. Its distance to Re P is bounded by that
    parity's Bessel tail over the scale plus the solver's norming bound on
    the residual.
    """
    from nwqlib import LinearDynamics, plan
    from nwqlib.algorithms import LCHS

    chosen = plan(
        LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1, 0], time=.1),
        method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding"),
    )
    prepared = chosen._native["native_data"].qsp_plan["prepared_evolution"]
    expansion = prepared.expansion
    assert expansion.tau == pytest.approx(1.2422, abs=1e-4)
    assert (expansion.cos_degree, expansion.sin_degree) == (4, 5)
    assert prepared.margin == 1e-3 and prepared.attempted_margins == (1e-3,)
    grid = chebyshev_grid(2048)
    for solution, function, tail in (
        (prepared.cos_solution, np.cos, expansion.cos_tail_bound),
        (prepared.sin_solution, np.sin, expansion.sin_tail_bound),
    ):
        realized = evaluate_qsp_polynomial(solution.phases, grid).real
        reference = function(expansion.tau * grid) / prepared.scale
        allowance = tail / prepared.scale + solution.residual_sup_bound
        assert np.max(np.abs(realized - reference)) <= allowance


def test_nonfinite_lbfgs_point_falls_through_to_the_newton_start(monkeypatch):
    """A NaN point from L-BFGS counts as not converged.

    On the degree-5 sine target at tau = 1.25, L-BFGS misses the tolerance
    and the Newton start from the [DMWL] point supplies the phases. When
    L-BFGS returns a NaN point instead, the solver must still run the
    Newton start and keep its finite result, so the phases agree bitwise
    with the ordinary solve.
    """
    from nwqlib.subroutines.qsp import evolution, phases

    prepared = evolution.prepare_qsp_evolution(tau=1.25, epsilon=1e-3)
    target = prepared.sin_solution.target_chebyshev_coefficients
    ordinary = phases.solve_symmetric_qsp_phases(target)

    def diverged(objective, start, **options):
        return phases.scipy.optimize.OptimizeResult(x=np.full_like(start, np.nan))

    monkeypatch.setattr(phases.scipy.optimize, "minimize", diverged)
    solved = phases.solve_symmetric_qsp_phases(target)
    assert solved.phases == ordinary.phases
    assert solved.max_residual <= QSP_SOLVER_RESIDUAL_TOLERANCE


def test_qsp_evolution_admits_the_dense_syntheses_of_both_passes_before_the_first(monkeypatch) -> None:
    # Controlling each parity pass synthesizes the dense dilation it holds
    # again, the forward unitary and, from degree two, its adjoint, and
    # Qiskit controls the synthesized gates at each of the pass's queries.
    # The syntheses counted in one build and one control step for every
    # query set the work, and one unit less refuses the next build before
    # any synthesis starts.
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_synthesis_admission import _gatewise_control_work

    matrix = np.array([[0.3, 0.1, 0.0, 0.05], [0.1, -0.2, 0.02, 0.0],
                       [0.0, 0.02, 0.1, 0.07], [0.05, 0.0, 0.07, -0.1]], dtype=complex)
    encoding = build_block_encoding(matrix, implementation="dense_dilation")
    calls = _counted_syntheses(monkeypatch)
    built = build_qsp_evolution_encoding(encoding, evolution_time=0.5, epsilon=1.0e-3)
    assert set(calls) == {3} and len(calls) in (3, 4)
    expansion = built.metadata["jacobi_anger"]
    queries = expansion["cos_degree"] + expansion["sin_degree"]
    need = sum(_dense_synthesis_work(width) for width in calls) + _gatewise_control_work(3, 1, queries)
    calls.clear()
    with pytest.raises(ValueError, match="QSP evolution dense synthesis exceeds max_work"):
        build_qsp_evolution_encoding(encoding, evolution_time=0.5, epsilon=1.0e-3, max_work=need - 1)
    assert calls == []
    build_qsp_evolution_encoding(encoding, evolution_time=0.5, epsilon=1.0e-3, max_work=need)


@pytest.mark.parametrize("route", ["gatewise", "auto"])
def test_joint_generator_admits_the_syntheses_of_its_controlled_branches(monkeypatch, route) -> None:
    # The combine qubit controls each branch, so each dense child is
    # synthesized once and its synthesized gates are controlled once. On the
    # whole-matrix route, which "auto" takes for this one control, each
    # dense child is synthesized once with the control, on three qubits. A
    # Pauli child holds no dense unitary.
    from test_dense_synthesis import _counted_syntheses, _dense_synthesis_work
    from test_synthesis_admission import _controlled_synthesis_work, _gatewise_control_work

    encoding_l = build_block_encoding(np.array([[0.5, 0.1], [0.1, 0.3]], dtype=complex),
                                      implementation="dense_dilation")
    encoding_h = build_block_encoding(np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex),
                                      implementation="dense_dilation")
    diagonals = dict(l_diagonal=[1.0, 0.5], h_diagonal=[1.0, 1.0], dense_control_route=route)
    calls = _counted_syntheses(monkeypatch)
    build_control_diagonal_generator_encoding(encoding_l, encoding_h, **diagonals)
    width = 2 if route == "gatewise" else 3
    assert calls == [width, width]
    need = 2 * (_dense_synthesis_work(2) + _gatewise_control_work(2, 1) if route == "gatewise"
                else _controlled_synthesis_work(3))
    calls.clear()
    with pytest.raises(ValueError, match="joint-generator dense synthesis exceeds max_work"):
        build_control_diagonal_generator_encoding(encoding_l, encoding_h, max_work=need - 1, **diagonals)
    assert calls == []
    build_control_diagonal_generator_encoding(encoding_l, encoding_h, max_work=need, **diagonals)
    pauli_h = build_block_encoding(np.array([[0.2, 0.05j], [-0.05j, -0.1]], dtype=complex),
                                   implementation="multiplexed_pauli")
    calls.clear()
    build_control_diagonal_generator_encoding(encoding_l, pauli_h, max_work=need // 2, **diagonals)
    assert calls == [width]
