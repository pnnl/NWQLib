"""Tests for the QSP/QSVT subroutines and the LCHS qsp_block_encoding wiring.

Tolerance classes follow the FRAMEWORK discipline: convention identities and
circuit known answers are machine-precision (<= 1e-12); Hamiltonian-evolution
equivalence uses the documented per-node error bound derived from the Bessel
tail, solver residuals, and the recorded rescale (method-error-bounded).
Tail bounds are checked against independent Bessel sums and analytic remainders.
"""

from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest
import scipy.linalg
import scipy.special
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
    imaginary_time_expansion,
    jacobi_anger_expansion,
    prepare_imaginary_time_filter,
    select_imaginary_time_degree,
    solve_laurent_qsp_phases,
    solve_wx_qsp_phases,
    wx_phases_to_reflection,
)
from nwqlib.subroutines.qsp.evolution import (
    build_qsp_evolution_encoding,
    plan_jacobi_anger_laurent,
    prepare_qubitized_target,
    with_phase_allowance,
)
from nwqlib.subroutines.qsp.phases import (
    chebyshev_polynomial_sup_bound,
    chebyshev_norming_sup_bound,
)

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


@pytest.mark.parametrize("implementation", ("pauli_lcu",))
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


def test_periodic_shift_adjoint_pairs_supply_exact_hermitian_evidence():
    # S**(-1) = (S**1).dagger, including complex conjugate coefficients.
    for coefficients, expected in (((1j, -1j), True), ((1j, 1j), False)):
        encoding = build_block_encoding(BandSpecification(
            offsets=(-1, 1), coefficients=coefficients, num_qubits=3))
        matrix = _encoded_block(encoding.circuit, encoding.num_ancillas)
        assert np.allclose(matrix, matrix.conj().T, rtol=0., atol=1e-12) == expected
        assert encoding.metadata["target_operator_is_hermitian"] is expected
        assert encoding.metadata["encoded_operator_is_hermitian"] is expected


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
    """Known answer (machine precision): solved Re P matches the target at 1e-12.

    The targets carry the QLS rescale convention, norming estimate times
    1.001, so their own norming estimate is about 1/1.001.
    """

    for tau, parity in ((1.5, 0), (1.5, 1), (4.0, 1)):
        expansion = jacobi_anger_expansion(tau, 1.0e-6)
        coefficients = np.asarray(
            expansion.cos_coefficients if parity == 0 else expansion.sin_coefficients
        )
        degree = coefficients.size - 1
        target = coefficients / (chebyshev_polynomial_sup_bound(coefficients) * (1.0 + 1.0e-3))
        solution = solve_wx_qsp_phases(target)
        assert solution.residual_sup_bound <= 1.0e-12 and solution.degree == degree
        # Recover the realized Chebyshev coefficients and compare directly.
        grid = chebyshev_grid(2 * (degree + 1))
        realized = evaluate_qsp_polynomial(solution.phases, grid).real
        recovered = np.polynomial.chebyshev.chebfit(grid, realized, degree)
        assert np.max(np.abs(recovered - target)) <= 1.0e-12

    # Nonsymmetric complex Laurent coefficients detect conjugation,
    # reversal and endpoint errors that a real-parity target cannot.
    coefficients = (.12 + .04j, .2j, .15 - .07j)
    solved = solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6)
    from qiskit import QuantumCircuit
    for theta in (-.7, .31, 2.1):
        circuit = QuantumCircuit(1)
        circuit.rx(solved.terminal_rx, 0)
        for k in range(solved.degree, -1, -1):
            left, rx, right = solved.rotation_angles[k]
            circuit.rz(right, 0)
            circuit.rx(rx, 0)
            circuit.rz(left, 0)
            if k:
                circuit.p(-theta, 0)
                if k > solved.negative_signals:
                    circuit.global_phase += theta
        actual = Operator(circuit).data
        expected = sum(c * np.exp(1j * (j - 1) * theta) for j, c in enumerate(coefficients))
        assert abs(actual[0, 0] - expected) <= 1e-12
        controlled = Operator(circuit.to_gate().control()).data
        np.testing.assert_allclose(controlled[np.ix_([0, 2], [0, 2])], np.eye(2), rtol=0, atol=1e-12)
        np.testing.assert_allclose(controlled[np.ix_([1, 3], [1, 3])], actual, rtol=0, atol=1e-12)


def test_phase_fft_doubling_stops_only_by_acceptance_ideal_length_cap_or_memory() -> None:
    """The FFT length doubles from N0 = next_power_of_two(2d + 1) until the actual residual is accepted.

    A degree-2 Laurent target with cap .6 is accepted at the first length
    whose residual meets 1e-12, after rejected shorter ones, with the same
    phases and evidence under the physical memory and under a limit that
    just admits the accepted attempt. A tolerance of 1e-30 lies below the
    binary64 residual floor, so the residual stagnates near 1e-16 from the
    first attempt; stagnation must not stop the doubling, which continues to
    the first length an ideal rule calls sufficient and then refuses with
    every attempt recorded. An explicit max_fft and a memory limit stop it
    earlier, each before allocating.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.subroutines.qsp import _nlft
    from nwqlib.subroutines.qsp.phases import _QSPPhaseConvergenceError

    coefficients, n0 = (.12 + .04j, .2j, .15 - .07j), 8
    accepted = solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6)
    tried, residuals = accepted.evidence.fft_lengths, accepted.evidence.residuals
    assert tried == tuple(n0 << k for k in range(len(tried)))
    assert residuals[-1] <= 1e-12 < min(residuals[:-1], default=1)
    tight = solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6,
                                     max_bytes=accepted.evidence.known_array_bytes)
    assert (tight.rotation_angles, tight.terminal_rx, tight.evidence) == (
        accepted.rotation_angles, accepted.terminal_rx, accepted.evidence)
    with pytest.raises(_QSPPhaseConvergenceError, match="calls FFT=.* sufficient") as failure:
        solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6, residual_tolerance=1e-30)
    evidence = failure.value.evidence
    lengths = evidence.fft_lengths
    assert lengths == tuple(n0 << k for k in range(len(lengths))) and len(lengths) > 2
    assert evidence.ideal_rule == "generic_sufficient" and len(evidence.residuals) == len(lengths)
    assert evidence.ideal_aliasing_bound <= 5e-31
    assert _nlft.ideal_rule(2, .6, 1e-30, lengths[-2]) == (None, None)
    # An ideal-rule stop is a numerical failure, also when a limit would block the next attempt too.
    assert not isinstance(failure.value, ResourceLimitError)
    with pytest.raises(_QSPPhaseConvergenceError, match="calls FFT=.* sufficient") as both:
        solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6, residual_tolerance=1e-30,
                                 max_fft=lengths[-1])
    assert both.value.evidence.fft_lengths == lengths and not isinstance(both.value, ResourceLimitError)
    # A stop before an attempt that does not fit is a capacity limit that keeps the attempts.
    with pytest.raises(_QSPPhaseConvergenceError, match="max_fft=16") as capped:
        solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6, residual_tolerance=1e-30,
                                 max_fft=16)
    assert capped.value.evidence.fft_lengths == (8, 16) and isinstance(capped.value, ResourceLimitError)
    limit = _nlft.known_array_bytes(2, 128)
    with pytest.raises(_QSPPhaseConvergenceError, match="memory limit") as limited:
        solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6, residual_tolerance=1e-30,
                                 max_bytes=limit)
    assert limited.value.evidence.fft_lengths == (8, 16, 32, 64, 128) and isinstance(limited.value, ResourceLimitError)
    assert _nlft.known_array_bytes(2, 256) > limit
    # A refusal of the first attempt made no attempt, so it has no convergence evidence.
    for first in (dict(max_bytes=_nlft.known_array_bytes(2, n0) - 1), dict(max_fft=n0 - 1)):
        with pytest.raises(ResourceLimitError) as refused:
            solve_laurent_qsp_phases(coefficients, minimum_power=-1, norm_bound=.6, **first)
        assert not isinstance(refused.value, _QSPPhaseConvergenceError)


@pytest.mark.parametrize("synthesize", [
    lambda **limits: prepare_imaginary_time_filter(imaginary_time_expansion(
        select_imaginary_time_degree(.2, 1e-6), epsilon_phase=1e-30, **limits)),
    lambda **limits: prepare_qubitized_target(with_phase_allowance(plan_jacobi_anger_laurent(
        jacobi_anger_expansion(.4, 1e-3), margin=1 / 16), epsilon_phase=1e-30, **limits)),
], ids=("imaginary_time_filter", "qubitized_target"))
def test_a_phase_wrapper_keeps_the_class_and_evidence_of_its_stop(synthesize):
    """A wrapper adds its context to a stop. A rule stop stays numerical and a stop at max_fft stays a limit refusal.

    The target 1e-30 lies far below what these binary64 attempts reach, so the doubling runs to the ideal rule.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.subroutines.qsp.phases import _QSPPhaseConvergenceError

    with pytest.raises(_QSPPhaseConvergenceError) as ruled:
        synthesize()
    assert not isinstance(ruled.value, ResourceLimitError) and ruled.value.evidence.ideal_rule is not None
    first = ruled.value.evidence.fft_lengths[0]
    with pytest.raises(_QSPPhaseConvergenceError) as capped:
        synthesize(max_fft=2 * first)
    assert isinstance(capped.value, ResourceLimitError)
    assert capped.value.evidence.fft_lengths == (first, 2 * first)
    with pytest.raises(ResourceLimitError) as start:
        synthesize(max_fft=first - 1)
    assert not isinstance(start.value, _QSPPhaseConvergenceError)


def test_chebyshev_grid_norming_bound_covers_polynomial_suprema() -> None:
    coefficients = np.zeros(33)
    coefficients[-1] = 1.0
    assert chebyshev_polynomial_sup_bound(coefficients) == pytest.approx(1.0, abs=2.0e-15)

    grid = chebyshev_grid(12)
    t_two = np.polynomial.chebyshev.chebval(grid, [0.0, 0.0, 1.0])
    assert chebyshev_norming_sup_bound(
        float(np.max(np.abs(t_two))), degree=2, num_points=12
    ) == pytest.approx(1.0, abs=2.0e-15)

    with pytest.raises(ValueError):
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
        assert true_sup <= chebyshev_polynomial_sup_bound(coefficients) * (1.0 + 2.0e-13)


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
    for degree in (0, 2, 3, 5):
        phases = rng.normal(size=degree + 1) * 0.4
        circuit = build_real_chebyshev_encoding(encoding, phases)
        block = _encoded_block(circuit, encoding.num_ancillas + 2)
        values = evaluate_qsp_polynomial(phases, eigenvalues).real
        expected = eigenvectors @ np.diag(values) @ eigenvectors.conj().T
        assert np.max(np.abs(block - expected)) <= 1.0e-12


@pytest.mark.parametrize(
    ("tau", "epsilon"),
    ((0.01, 1.0e-3), (0.1, 1.0e-10), (5.0, 1.0e-7), (50.0, 1.0e-5), (1000.0, 1.0e-3)),
)
def test_jacobi_anger_records_complete_finite_and_analytic_tail(tau: float, epsilon: float) -> None:
    expansion = jacobi_anger_expansion(tau, epsilon)
    terminal = expansion.analysis_terminal
    assert expansion.degree <= terminal
    assert expansion.tail_bound is not None  # These finite tails are representable.
    assert expansion.tail_bound <= epsilon
    # Independent long finite sums expose an omitted remainder even when
    # the selected degree equals the analysis terminal. The analytic
    # inequalities and arithmetic qualification are derived in evolution.py.
    orders = np.arange(terminal + 1001)
    magnitudes = 2.0 * np.abs(scipy.special.jv(orders, tau))
    # For these fixed fixtures, the independent finite tail proves that
    # the predecessor degree is insufficient when the floor does not fix
    # the choice. This does not assert global degree optimality.
    if expansion.degree > 1:
        assert np.sum(magnitudes[expansion.degree:]) > epsilon
    for degree, parity, bound in (
        (expansion.cos_degree, 0, expansion.cos_tail_bound),
        (expansion.sin_degree, 1, expansion.sin_tail_bound),
    ):
        direct = np.sum(magnitudes[(orders > degree) & (orders % 2 == parity)])
        assert bound is not None  # Ordinary fixtures have positive, representable tails.
        assert 0.0 < direct <= bound
    assert np.sum(magnitudes[terminal + 1:]) <= expansion.analytic_infinite_tail_bound
    # Separate exp/log evaluations at logs up to 24 incur several ulps.
    assert expansion.tail_bound == pytest.approx(
        expansion.cos_tail_bound + expansion.sin_tail_bound, rel=2e-14, abs=0)


def test_jacobi_anger_zero_time_and_positive_underflow_are_distinct():
    zero = jacobi_anger_expansion(0.0, 1e-10)
    assert zero.cos_coefficients == (1.0,) and zero.sin_coefficients == (0.0,)
    assert zero.tail_bound == 0.0 and zero.tail_log_bound is None
    tiny = jacobi_anger_expansion(1e-300, 1e-10, min_degree=2)
    assert tiny.tail_bound is None
    # The leading omitted order is 3: 2*J_3(tau)=tau^3/24+O(tau^5).
    # Corrections are below 1e-600 relative; the log arithmetic is O(1e3),
    # so a 4e-12 absolute log allowance covers binary64 evaluation only.
    assert tiny.tail_log_bound == pytest.approx(3*np.log(tiny.tau)-np.log(24), rel=0, abs=4e-12)
    assert tiny.to_dict()["tail_log_bound"] == tiny.tail_log_bound


# beta=100 is the row whose selected degree is decided by the r<=beta form of
# the Chernoff exponent; the smaller rows are decided by the r>beta form.
@pytest.mark.parametrize("beta", (0.2, 3.0, 20.0, 100.0))
def test_imaginary_time_filter_matches_scalar_exponential(beta):
    selection = select_imaginary_time_degree(beta, 1e-6)
    # No phases are fitted here; the phase allowance is only recorded.
    expansion = imaginary_time_expansion(selection, epsilon_phase=1e-6)
    grid = np.linspace(-1, 1, 101)
    # The independent scalar exponential detects the sign and the factor
    # two on positive Bessel orders. 1e-14 only covers binary64 evaluation
    # in this small, nonnegative-coefficient case, below the 1e-6 tail budget.
    physical = expansion.scale * np.asarray(expansion.target_coefficients).real
    actual = np.polynomial.chebyshev.chebval(grid, physical)
    expected = np.exp(-beta * (1 - grid))
    assert max(abs(actual - expected)) <= selection.tail_bound + 1e-14
    # Fractions sum the exact numerical inputs, independently of upward
    # binary64 accumulation. This checks the circle cap, not Bessel accuracy.
    exact_sum = sum(Fraction(value.real) for value in expansion.target_coefficients)
    assert exact_sum <= Fraction(expansion.stored_norm_bound) < 1
    assert expansion.coefficient_error_bound is None
    assert expansion.arithmetic_error_bound is None
    assert selection.tail_log_bound <= np.log(selection.epsilon_tail)
    assert selection.predecessor_tail_log_bound > np.log(selection.epsilon_tail)


def test_imaginary_time_identity_and_underflow_have_different_error_frames():
    zero = imaginary_time_expansion(select_imaginary_time_degree(0.0, 0.0), epsilon_phase=1e-6)
    assert zero.coefficients == (1.0,)  # Exact beta-zero identity.
    assert zero.scale == 1.0 and zero.selection.tail_bound == 0.0
    assert zero.coefficient_error_bound == 0.0
    assert prepare_imaginary_time_filter(zero).solution is None

    # A positive fit: the returned rotations realize Q, whose normalized
    # phase residual is multiplied by scale once. A=RZ(l)RX(a)RZ(r) and the
    # signal is diag(z,1), or diag(1,1/z) for the first negative_signals,
    # as documented by LaurentQSPPhases. For real c_k and x=cos(theta),
    # sum_k c_k T_k(x)=[Q(e^{i theta})+Q(e^{-i theta})]/2, so the phase
    # allowance moves the filter by at most scale*epsilon_phase. 1e-14 covers
    # binary64 evaluation of these degree-4 products, as in the test above.
    selected = select_imaginary_time_degree(0.2, 1e-6)
    expansion = imaginary_time_expansion(selected, epsilon_phase=1e-6, max_fft=32)
    prepared = prepare_imaginary_time_filter(expansion)
    solution = prepared.solution

    def rotation(axis, angle):
        c, s = np.cos(angle / 2), np.sin(angle / 2)
        if axis == "z":
            return np.diag([c - 1j * s, c + 1j * s])
        return np.array([[c, -1j * s], [-1j * s, c]])

    def upper_left(z):
        product = np.eye(2, dtype=complex)
        for k, (left, middle, right) in enumerate(solution.rotation_angles):
            if k:
                signal = np.diag([1, 1 / z]) if k <= solution.negative_signals else np.diag([z, 1])
                product = product @ signal
            product = product @ rotation("z", left) @ rotation("x", middle) @ rotation("z", right)
        return (product @ rotation("x", solution.terminal_rx))[0, 0]

    grid = np.linspace(-1, 1, 101)
    theta = np.arccos(grid)
    realized = [(upper_left(np.exp(1j * t)) + upper_left(np.exp(-1j * t))) / 2 for t in theta]
    error = abs(expansion.scale * np.asarray(realized) - np.exp(-0.2 * (1 - grid)))
    assert max(error) <= selected.tail_bound + expansion.scale * expansion.epsilon_phase + 1e-14

    tiny = np.nextafter(0.0, 1.0)
    selected = select_imaginary_time_degree(tiny, tiny)
    expansion = imaginary_time_expansion(selected, epsilon_phase=1e-6)
    assert selected.tail_bound is None
    # For r=2, the tiny-beta Chernoff log is log(2)-2*(log(4/beta)-1).
    # O(beta) corrections are unrepresentable, and the logarithms are O(1e3).
    expected_log = 2*np.log(tiny) - 3*np.log(2.0) + 2
    assert selected.tail_log_bound == pytest.approx(expected_log, rel=0, abs=4e-12)
    assert expansion.coefficient_error_bound is None
    assert len(expansion.target_coefficients) == selected.degree + 1
    assert expansion.actual_support == (0, 0)
    with pytest.raises(ValueError, match="positive omitted Bessel tail"):
        select_imaginary_time_degree(1.0, 0.0)
    coarse = select_imaginary_time_degree(1.0, 1.0)
    assert coarse.degree == 0 and coarse.tail_bound == 1.0


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
            metadata["jacobi_anger"]["tail_bound"]
            + metadata["target_rescale"]
            * (
                phase_solver["cos"]["residual_sup_bound"]
                + phase_solver["sin"]["residual_sup_bound"]
            )
        )
        assert metadata["polynomial_part_error_bound"] == polynomial_part
        amplitude = metadata["oaa_amplitude"]
        delta = polynomial_part / amplitude
        phi = delta + (12*amplitude**2*delta**2 + 4*amplitude**2*delta**3)/(3-4*amplitude**2)
        assert metadata["oaa_residual_error_bound"] == pytest.approx(
            (phi + metadata["child_operator_error_contribution"]) / metadata["recovery_scale"],
            rel=1.0e-14,
            abs=0,
        )
        assert metadata["oaa_amplitude_deficit"] == pytest.approx(
            1.0 - (3.0 * amplitude - 4.0 * amplitude**3), rel=1.0e-12,
            abs=0,
        )
        assert metadata["phase_arithmetic_error_bound"] is None


def test_qsp_child_operator_error_uses_physical_time_after_oaa() -> None:
    from nwqlib.subroutines.qsp.evolution import prepare_qsp_evolution, qsp_evolution_error_terms

    encoded = np.diag([0.2, -0.1]).astype(complex)
    base = build_block_encoding(
        encoded, implementation="dense_dilation"
    )
    child = replace(
        base,
        error_bound=0.04,
        metadata={
            **dict(base.metadata),
            "error_bound_provenance": ("abstract_operator_bound_physical_generator_frame"),
            "encoded_operator_is_hermitian": True,
        },
    )
    physical_time = 0.25
    prepared = prepare_qsp_evolution(tau=base.alpha*physical_time, epsilon=1.0e-2)
    evolution = build_qsp_evolution_encoding(
        child,
        evolution_time=physical_time,
        epsilon=1.0e-2,
        _prepared=prepared,
    )
    metadata = evolution.metadata
    contribution = metadata["child_operator_error_contribution"]
    assert contribution == pytest.approx(physical_time * child.error_bound, rel=2e-14, abs=0)
    assert contribution != pytest.approx(metadata["effective_time_tau"] * child.error_bound, rel=2e-14, abs=0)
    exact_child = qsp_evolution_error_terms(prepared, evolution_time=physical_time, child_error_bound=0.)
    assert metadata["compensated_recovery_error_bound"] - exact_child["compensated_recovery_error_bound"] == pytest.approx(
        contribution, rel=2e-14, abs=0)
    target = encoded + np.diag([child.error_bound, -child.error_bound])
    block = _encoded_block(evolution.circuit, evolution.num_ancillas)
    reference = scipy.linalg.expm(-1j*physical_time*target)
    assert np.linalg.norm(block*metadata["recovery_scale"] - reference, 2) <= metadata["compensated_recovery_error_bound"]

    # Independently form 3B-4BB^dagger B for a unitary F and two perturbation
    # directions. This covers adjoint/order errors and the leading coefficient
    # one in the recovered bound, without using the circuit emitter.
    delta = prepared.expansion.tail_bound + prepared.scale*(
        prepared.cos_solution.residual_sup_bound + prepared.sin_solution.residual_sup_bound)
    f = np.diag(np.exp(1j*np.array([.37, -.61])))
    a = .5/prepared.scale
    r = 3*a-4*a**3
    for perturbation in (1j*delta*f, delta*np.array([[0., 1.], [0., 0.]])):
        b = a*(f+perturbation)
        actual = (3*b-4*b@b.conj().T@b)/r
        assert np.linalg.norm(actual-f, 2) <= exact_child["compensated_recovery_error_bound"]
    assert exact_child["compensated_recovery_error_bound"] <= delta+1.5*delta**2+.5*delta**3
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
# Real-Wx phase synthesis on the maintainer sweep targets
# ---------------------------------------------------------------------------


# Single-source sweep targets: the fast-suite characterization
# must exercise bit-identical instances to the maintainer sweep, so the
# builder is loaded from the script rather than mirrored.
_sweep_target = load_docs_script("docs/scripts/qsp_phase_solver_sweep.py").build_target


def test_wx_phase_synthesis_boundary_characterization() -> None:
    """Canonical Wx synthesis meets its tolerance on the fixed parity and cancellation targets.

    The half-norm, near-unit (0.98) and endpoint-slope inputs probe different
    numerical failure modes. They do not qualify every target of these
    degrees or replace the actual residual check.
    """

    for kind in ("single_term", "random_decay", "random_flat"):
        for degree in (8, 16, 32, 64):
            solution = solve_wx_qsp_phases(_sweep_target(kind, degree, seed=degree))
            assert solution.residual_sup_bound <= QSP_SOLVER_RESIDUAL_TOLERANCE

    # 0.5 T_255 has |f'| = 0.5 * 255^2 at the endpoints, where the signal root
    # sqrt((1 - x)(1 + x)) must be evaluated without cancellation.
    endpoint_slope = solve_wx_qsp_phases(_sweep_target("single_term", 255, seed=255))
    assert endpoint_slope.residual_sup_bound <= QSP_SOLVER_RESIDUAL_TOLERANCE

    near_unit = solve_wx_qsp_phases(_sweep_target("near_sup", 32, seed=32))
    assert near_unit.residual_sup_bound <= QSP_SOLVER_RESIDUAL_TOLERANCE


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
    assert unknown.alpha == known.alpha
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

    unavailable = evolution.build_qsp_evolution_encoding(unknown, evolution_time=.1,
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
    monkeypatch.setattr(evolution, "_solve_wx_problem", unexpected)
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
    """The planned coherent evolution uses the generator at its node and child scale and the declared allowance.

    For A = diag(.1, .2) the centered child is L with alpha .2 and H is an exact zero, so the separate-parts
    generator has alpha max|k_j| * .2 on the four-node grid. The expansion is the Jacobi-Anger expansion of
    tau = alpha T_e at the row's tail allowance, which in the algorithmic frame is at most half the combined
    allowance LCHS_QSP_EPSILON_FRACTION * approximation_tolerance (coherent_selection, M7.1). The native
    program, its estimate and its prepared circuits share one width.
    """
    from nwqlib import prepare,estimate
    from nwqlib.algorithms.lchs.native import LCHS_QSP_EPSILON_FRACTION
    chosen=_selected_lchs_qsp(np.diag([.1,.2]))
    data=chosen._native['native_data']
    prepared=data.qsp_plan
    expected_alpha=max(abs(float(k)) for k in data.quadrature.k_nodes)*.2
    assert prepared.generator.components['l_child'].alpha==pytest.approx(.2,rel=1e-12,abs=0)
    assert prepared.generator.components.get('h_child') is None
    assert prepared.generator.alpha==pytest.approx(expected_alpha,rel=1e-12,abs=0)
    row=prepared.candidate
    combined=LCHS_QSP_EPSILON_FRACTION*chosen.method.approximation_tolerance
    assert row.allowance['algorithmic']==combined and 0<row.allowance['tail']<=combined/2
    expansion=prepared.phases.expansion
    assert expansion==jacobi_anger_expansion(expected_alpha*.1,row.allowance['tail'],min_degree=1)
    width=sum(register.width for register in chosen.construction.program.registers)
    assert width<=13
    assert estimate(chosen).construction_id==chosen.construction.content_id
    preview=prepare(chosen)
    assert all(circuit.num_qubits==width for circuit in preview.circuits)


def test_lchs_guide_first_problem_plans_qsp_evolution():
    """The selected polynomial approximates the evolution at the guide input's actual scale.

    The reference is the untruncated function divided by the selected scale. For shared QSVT each parity's
    distance to Re P is bounded by its Bessel tail over the scale plus the actual-angle coefficient residual.
    For the qubitized walk the block is sum_j r_j T_|j|(x) for the realized Laurent coefficients r_j, since
    J^dagger W^(+-k) J = T_k(x); its distance to exp(-i tau x)/sigma is at most delta_T/sigma plus the
    coefficient residual ||r - f||_1, because |T_k| <= 1 on [-1, 1] (T15, T17).
    """
    from nwqlib import LinearDynamics, plan
    from nwqlib.algorithms import LCHS
    from nwqlib.subroutines.qsp.phases import _laurent_reconstruct

    chosen = plan(
        LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1, 0], time=.1),
        method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding"),
    )
    prepared = chosen._native["native_data"].qsp_plan.phases
    expansion = prepared.expansion
    grid = chebyshev_grid(2048)
    if hasattr(prepared, "cos_solution"):
        for solution, function, tail in (
            (prepared.cos_solution, np.cos, expansion.cos_tail_bound),
            (prepared.sin_solution, np.sin, expansion.sin_tail_bound),
        ):
            realized = evaluate_qsp_polynomial(solution.phases, grid).real
            reference = function(expansion.tau * grid) / prepared.scale
            allowance = tail / prepared.scale + solution.residual_sup_bound
            assert np.max(np.abs(realized - reference)) <= allowance
        return
    solution = prepared.solution
    realized = _laurent_reconstruct(solution.rotation_angles, solution.terminal_rx)
    powers = np.abs(np.arange(len(realized)) - solution.negative_signals)
    block = np.cos(np.outer(np.arccos(grid), powers)) @ realized
    reference = np.exp(-1j * expansion.tau * grid) / prepared.scale
    allowance = expansion.tail_bound / prepared.scale + solution.evidence.coefficient_residual
    assert np.max(np.abs(block - reference)) <= allowance


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


def _real_pauli_plan(terms, n):
    from qiskit.quantum_info import SparsePauliOp
    from nwqlib.subroutines.block_encoding import plan_block_encoding
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    decomposition = PauliDecomposition(terms=tuple(PauliTerm(label=label, coefficient=value) for label, value in terms),
                                       input_dimension=1 << n, operator_dimension=1 << n, num_qubits=n, atol=0.)
    return (plan_block_encoding(decomposition, implementation="multiplexed_pauli"),
            SparsePauliOp.from_list(list(terms)).to_matrix())


def _supplied_dilation(matrix, defect):
    """A supplied unitary dilation (test oracle) of A + i*defect, A Hermitian: block not Hermitian, no involution.

    With ``defect=None`` the circuit is ``[[A, -B], [B, A]]``, ``B = sqrt(I - A^2)``: a Hermitian block whose
    whole unitary is not an involution. Otherwise it is the Halmos dilation of the non-Hermitian contraction.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.subroutines.block_encoding import BlockEncoding

    a = matrix if defect is None else matrix + 1j * defect
    if defect is None:
        w, v = np.linalg.eigh(a)
        b = v @ np.diag(np.sqrt(1 - w**2)) @ v.conj().T
        u = np.block([[a, -b], [b, a]])
    else:
        u = np.block([[a, scipy.linalg.sqrtm(np.eye(len(a)) - a @ a.conj().T)],
                      [scipy.linalg.sqrtm(np.eye(len(a)) - a.conj().T @ a), -a.conj().T]])
    order = np.arange(2 * len(a)).reshape(2, len(a)).T.reshape(-1)  # ancilla as the low qubit
    circuit = QuantumCircuit(int(np.log2(len(a))) + 1)
    circuit.append(UnitaryGate(u[np.ix_(order, order)]), circuit.qubits)
    return BlockEncoding(circuit=circuit, alpha=1.0, num_ancillas=1, system_qubits=circuit.num_qubits - 1,
                         error_bound=0.0 if defect is None else float(np.linalg.norm(defect, 2)),
                         implementation="supplied_dilation",
                         metadata={"target_operator_is_hermitian": True, "encoded_operator_is_hermitian": defect is None,
                                   "full_unitary_involution": False})


def _controlled_full(matrix, state):
    result = np.eye(2 * len(matrix), dtype=complex)
    rows = [state + 2 * index for index in range(len(matrix))]
    result[np.ix_(rows, rows)] = matrix
    return result


_DEFECTED = (np.array([[.3, .2 - .1j], [.2 + .1j, -.4]]), np.array([[.01, .02], [.02, -.01]]))


def _route_encoding(route, source, time, epsilon, residual_tolerance=1e-12):
    from nwqlib.subroutines.block_encoding import access_summary
    from nwqlib.subroutines.qsp.evolution import (build_qubitized_evolution, build_shared_qsvt_evolution,
                                                  prepare_jacobi_anger_gqet, prepare_jacobi_anger_laurent,
                                                  prepare_qsp_evolution)

    tau = access_summary(source).alpha * time
    if route.startswith("shared"):
        prepared = prepare_qsp_evolution(tau=tau, epsilon=epsilon, residual_tolerance=residual_tolerance)
        return build_shared_qsvt_evolution(source, prepared, evolution_time=time, amplification=route[7:])
    prepare = prepare_jacobi_anger_laurent if route == "laurent" else prepare_jacobi_anger_gqet
    target = prepare(jacobi_anger_expansion(tau, epsilon), margin=1 / 16, epsilon_phase=1e-11)
    return build_qubitized_evolution(source, target, evolution_time=time)


@pytest.mark.parametrize("route", ["shared_raw", "shared_oaa"])
def test_shared_route_phase_tolerance_is_its_allowance_after_the_tail(route):
    """Each shared route's parity tolerance is what its own recovery leaves of the route allowance A after the tail.

    With common rescale s and parity residuals r, the half block is a(F + E), a = 1/(2s), with
    ||E|| <= d = tail + s(r_c + r_s). Raw recovery leaves error at most d, so the tolerance is (A - tail)/(2s). OAA's
    recovered error is Phi_a(d) <= Phi_{1/2}(d), Phi_{1/2}(d) = d + 1.5 d**2 + .5 d**3, so the tolerance is
    (Phi_{1/2}^-1(A) - tail)/(2s), which a separate root solve evaluates here. The actual residuals, far above the
    solver's default 1e-12 at this time, enter the reported error through those same relations, within A.
    """
    from scipy.optimize import brentq
    from nwqlib import LinearDynamics, NormSquared, plan
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation

    problem = LinearDynamics(A=ingest_pauli([("II", .001), ("ZI", 20j), ("IZ", 20j), ("XX", 10j)], num_qubits=2),
                             initial_state=ingest_occupation("00", num_qubits=2), time=30.0)
    prepared = plan(problem, method=LCHS(hamiltonian_evolution_backend="qsp_block_encoding", qsp_route=route),
                    output=NormSquared(), shots=100, seed=3)._native["native_data"].qsp_plan
    row, phases = prepared.candidate, prepared.phases
    assert row.route == route
    s, tail, allowance = phases.scale, phases.expansion.tail_bound, row.allowance["algorithmic"]
    limit = allowance if route == "shared_raw" else brentq(
        lambda d: d + 1.5 * d**2 + .5 * d**3 - allowance, 0, allowance, xtol=1e-20, rtol=1e-15)
    tolerance = row.allowance["synthesis_per_parity"]
    # The library's bisection keeps its lower endpoint, and the root solve has rtol 1e-15 of about 1e-3.
    assert tolerance == pytest.approx((limit - tail) / (2 * s), rel=1e-12, abs=0)
    residuals = phases.cos_solution.residual_sup_bound, phases.sin_solution.residual_sup_bound
    assert all(1e-12 < residual <= tolerance for residual in residuals)
    d = tail + s * sum(residuals)
    a = .5 / s
    expected = d if route == "shared_raw" else d + 12 * a**2 / (3 - 4 * a**2) * d**2 + 4 * a**2 / (3 - 4 * a**2) * d**3
    assert prepared.error["algorithmic"] == pytest.approx(expected, rel=1e-14, abs=0)
    assert prepared.error["algorithmic"] <= allowance


@pytest.mark.parametrize("route", ["shared_raw", "shared_oaa"])
def test_shared_routes_recover_the_exponential_with_a_residual_above_rounding(route):
    """The recovered action stays within the route's bound when the phase residual is far above rounding.

    A wide tolerance leaves d of order 1e-3 on a Pauli generator, with tau about 14 so that its first accepted FFT is
    not exact. Here the quadratic OAA term 1.5 d**2 is far above the 1e-9 statevector allowance of these 6 qubits, so
    the actual residuals, the rescale and the OAA map all matter to the bound against scipy's exponential.
    """
    time = 15.0
    pauli, matrix = _real_pauli_plan((("XZ", .4), ("ZI", -.3), ("YY", .2)), 2)
    encoding = _route_encoding(route, pauli, time, 1e-3, residual_tolerance=1e-3)
    phase_solver = encoding.metadata["phase_solver"]
    assert max(phase_solver["cos"]["residual_sup_bound"], phase_solver["sin"]["residual_sup_bound"]) > 1e-6
    block = _encoded_block(encoding.circuit, encoding.num_ancillas)
    error = np.linalg.norm(encoding.alpha * block - scipy.linalg.expm(-1j * time * matrix), 2)
    assert error <= encoding.error_bound + 1e-9, (error, encoding.error_bound)


def test_prepared_phase_tolerance_reaches_each_public_builder():
    """A target no attempt can meet still refuses, naming the residual, and a caller without an allowance keeps 1e-12."""
    from nwqlib.subroutines.block_encoding import build_block_encoding_from_plan
    from nwqlib.subroutines.qsp.evolution import prepare_qsp_evolution

    with pytest.raises(ValueError, match="exceeds"):
        prepare_qsp_evolution(tau=2.0, epsilon=1e-6, residual_tolerance=1e-30)
    pauli, _ = _real_pauli_plan((("XZ", .4), ("ZI", -.3), ("YY", .2)), 2)
    encoding = build_block_encoding_from_plan(pauli)
    with pytest.raises(ValueError, match="exceeds"):
        build_qsp_evolution_encoding(encoding, evolution_time=.5, epsilon=1e-6, residual_tolerance=1e-30)
    prepared = prepare_qsp_evolution(tau=2.0, epsilon=1e-6)
    assert max(prepared.cos_solution.residual_sup_bound, prepared.sin_solution.residual_sup_bound) <= 1e-12


@pytest.mark.parametrize("route", ["shared_raw", "shared_oaa", "laurent", "gqet"])
def test_transformation_routes_recover_the_exponential(route):
    """Each route's recovered block, alpha times the success block, is within its bound of exp(-i t A).

    Direct access uses an asymmetric real Pauli sum on two system qubits (a full involution, Hermitian block),
    so a reversed system-wire order changes the action. The supplied dilation of a Hermitian matrix has a
    Hermitian block but U^dagger != U: the shared routes query it directly through the alternating ladder
    U, U^dagger, U, ..., and the qubitized routes, which need a full involution, lift it. The third encoding has
    an anti-Hermitian block defect of norm ~0.03 and no involution, so every route must use the generic lift,
    whose Hermitian part removes that defect; the bound adds t*delta_S once. On the shared routes the short time
    tau = 1e-4 makes standard preparation return a degree-0 cosine, the constant branch. The tolerance 1e-9
    covers statevector rounding of these circuits of at most 6 qubits.
    """
    time = .9
    shared = route.startswith("shared")
    pauli, matrix = _real_pauli_plan((("XZ", .4), ("ZI", -.3), ("YY", .2)), 2)
    hermitian = _supplied_dilation(_DEFECTED[0], None)
    lifted = _supplied_dilation(*_DEFECTED)
    cases = [(pauli, matrix, time, 1e-7, "direct" if shared else "direct_involution"),
             (hermitian, _DEFECTED[0], time, 1e-7, "direct" if shared else "generic_lift"),
             (lifted, _DEFECTED[0], time, 1e-7, "generic_lift")]
    if shared:
        cases.append((pauli, matrix, 1e-4 / pauli.alpha, 1e-8, "direct"))
    for source, target, t, epsilon, access in cases:
        encoding = _route_encoding(route, source, t, epsilon)
        assert encoding.metadata["plan"]["signal_access"] == access
        if epsilon == 1e-8:
            assert encoding.metadata["plan"]["cos_degree"] == 0  # premise of the constant-branch case
        block = _encoded_block(encoding.circuit, encoding.num_ancillas)
        error = np.linalg.norm(encoding.alpha * block - scipy.linalg.expm(-1j * t * target), 2)
        assert error <= encoding.error_bound + 1e-9, (error, encoding.error_bound)
        assert encoding.error_bound < 1e-6 + t * (source.error_bound or 0.0) + 1e-6


def test_transformation_queries_are_controlled_and_reversed_exactly(monkeypatch):
    """Controlled evolutions are the identity on every qubit in the inactive branch and the adjoint is the inverse.

    No global phase is aligned. Shared raw on the direct Pauli access is checked with an even and an odd shared
    ladder: an odd one relies on the controlled first query in the inactive branch. Shared OAA adds the two
    reflections and the restored sign -1, which must carry the controls. Qubitized Laurent on the direct
    involution controls the walk query on g and the external control, including its inverse signals. The
    qubitized GQET case uses the lift, whose controls act on X_h and the reflection only.

    The original-U queries the emitters build are also those of the plan census, which the coherent route
    prices and reports: every query gate from ``build_access_query`` is labeled with its control count and
    adjoint flag, and its uses in the built uncontrolled and singly controlled circuits must equal the plan's
    ``forward`` and ``controlled`` rows, for the shared raw ladders (plain, controlled-first and controlled-final
    queries) and the qubitized direct and lifted walks (D_U^dagger and D_U on h for every external control).
    """
    from collections import Counter
    from nwqlib.subroutines.block_encoding import joint
    from nwqlib.subroutines.qsp.evolution import (build_qubitized_evolution_query, build_shared_qsvt_evolution_query,
                                                  plan_qubitized_evolution, plan_shared_qsvt_evolution,
                                                  prepare_jacobi_anger_gqet, prepare_jacobi_anger_laurent,
                                                  prepare_qsp_evolution)
    forms, build_query = {}, joint.build_access_query

    def labeled_query(source, **options):
        gate = build_query(source, **options)
        gate.label = f"census_query_{len(forms)}"
        forms[gate.label] = (options.get("num_controls", 0), options.get("adjoint", False))
        return gate

    monkeypatch.setattr(joint, "build_access_query", labeled_query)

    def uses(gate):
        return Counter(forms[item.operation.label] for item in gate.definition.data if item.operation.label in forms)

    time = .9
    pauli, _ = _real_pauli_plan((("X", .6), ("Z", -.3)), 1)
    builders, parities = [], set()
    for epsilon in (1e-4, 1e-5):
        prepared = prepare_qsp_evolution(tau=pauli.alpha * time, epsilon=epsilon)
        parities.add((max(prepared.cos_solution.degree, prepared.sin_solution.degree) - 1) % 2)
        builders.append((lambda prepared=prepared, **kw: build_shared_qsvt_evolution_query(
            pauli, prepared, evolution_time=time, amplification="raw", **kw),
            plan_shared_qsvt_evolution(pauli, prepared, evolution_time=time, amplification="raw")))
    assert parities == {0, 1}
    builders.append((lambda prepared=prepared, **kw: build_shared_qsvt_evolution_query(
        pauli, prepared, evolution_time=time, **kw), None))
    laurent = prepare_jacobi_anger_laurent(jacobi_anger_expansion(pauli.alpha * time, 1e-4), margin=1 / 16,
                                           epsilon_phase=1e-11)
    builders.append((lambda **kw: build_qubitized_evolution_query(pauli, laurent, evolution_time=time, **kw),
                     plan_qubitized_evolution(pauli, laurent, evolution_time=time)))
    lifted = _supplied_dilation(_DEFECTED[0], None)
    target = prepare_jacobi_anger_gqet(jacobi_anger_expansion(time, 1e-4), margin=1 / 16, epsilon_phase=1e-11)
    builders.append((lambda **kw: build_qubitized_evolution_query(lifted, target, evolution_time=time, **kw),
                     plan_qubitized_evolution(lifted, target, evolution_time=time)))
    assert builders[-1][1].signal_access == "generic_lift"
    for build, plan in builders:
        gate = build()
        forward = Operator(gate).data
        census = {} if plan is None else {key: Counter({(controls, adjoint): count
                                                        for _, controls, adjoint, count in plan.query_census[key]})
                                          for key in ("forward", "controlled")}
        if plan is not None:
            assert uses(gate) == census["forward"]
        for state in (0, 1):
            controlled = build(num_controls=1, ctrl_state=state)
            if plan is not None:
                assert uses(controlled) == census["controlled"]
            assert np.max(np.abs(Operator(controlled).data - _controlled_full(forward, state))) <= 1e-10
        assert np.max(np.abs(Operator(build(adjoint=True)).data - forward.conj().T)) <= 1e-10


def test_imaginary_time_filter_recovers_the_contraction_semigroup():
    """GQET on X = -L_0/alpha recovers exp(-beta(1 - X)) within the bound of imaginary_time_filter_error_terms."""
    from nwqlib.subroutines.block_encoding import plan_convex_contraction
    from nwqlib.subroutines.qsp.evolution import (build_qubitized_evolution, imaginary_time_filter_error_terms,
                                                  imaginary_time_target)
    plan, l0 = _real_pauli_plan((("X", .5), ("Z", .2)), 1)
    alpha, time = .7, 1.3
    contraction = plan_convex_contraction(l_child=plan, alpha=alpha)
    beta = time * alpha
    expansion = imaginary_time_expansion(select_imaginary_time_degree(beta, 1e-8), epsilon_phase=1e-11)
    target = imaginary_time_target(prepare_imaginary_time_filter(expansion))
    encoding = build_qubitized_evolution(contraction, target)
    terms = imaginary_time_filter_error_terms(target, contraction_error=contraction.error_bound)
    assert encoding.error_bound == pytest.approx(terms["filter_error"], rel=1e-12)
    exact = scipy.linalg.expm(-beta * (np.eye(2) + l0 / alpha))
    block = _encoded_block(encoding.circuit, encoding.num_ancillas)
    assert np.linalg.norm(encoding.alpha * block - exact, 2) <= encoding.error_bound + 1e-9
