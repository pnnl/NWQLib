"""Actual native QLS selection, canonical phase synthesis and acquisitions, Q<=8.

Planning runs the closed-form polynomial selection and the canonical Wx
phase synthesis. The 2e-10 comparison floor is numerical headroom, not a
vendor certificate. These tests compare actual circuits with independent
polynomial recurrences on the recorded coefficients.
"""

import math
import numpy as np
import pytest
import nwqlib
from nwqlib.algorithms.qls import QLS
from nwqlib.operators import ingest_pauli
from nwqlib.problems import LinearSystem, QuadraticForm, Solution, StateVector
from nwqlib.problems.inputs import ingest_occupation, ingest_product

NUMERICAL_FLOOR = 2e-10
EPSILON = 0.1


def native_choice(
    matrix,
    rhs,
    *,
    method="qsvt_inverse",
    alpha=2.0,
    kappa=2.0,
    embedded=False,
    quadratic=False,
    compact=False,
    encoding=None,
):
    """Select a bounded native QLS fixture whose requested output makes physical recovery or
    direction-only semantics explicit.
    """
    if compact:
        state = (
            ingest_occupation((0,), num_qubits=1)
            if quadratic
            else ingest_product(rhs.reshape(1, 2))
        )
    else:
        state = rhs
    problem = LinearSystem(A=matrix, b=state)
    inverse = method == "qsvt_inverse"
    output = (
        QuadraticForm(observable=ingest_pauli((("Z", 1.0),), num_qubits=1))
        if quadratic
        else (
            Solution()
            if inverse
            else StateVector(normalization="unit", global_phase="modulo_global_phase")
        )
    )
    selected = nwqlib.plan(
        problem,
        output=output,
        method=QLS(
            solver=method,
            alpha=alpha,
            kappa=kappa,
            epsilon_inv=EPSILON,
            block_encoding=encoding or ("banded" if quadratic else "dense_dilation"),
            norm_estimate=None if inverse else 1.5,
        ),
        seed=7,
    )
    assert selected.reconstruction.width <= 8
    assert (selected.reconstruction.embedding == "hermitian_dilation") == embedded
    return selected


def execute(chosen):
    result = nwqlib.solve(chosen)
    vector = None if result.artifact is None else result.value
    if vector is not None:
        assert vector.nbytes <= 32
    return result, vector


def chebyshev_action(matrix, coefficients, vector):
    """Independent tiny recurrence, never the production polynomial action."""
    previous = vector
    total = coefficients[0] * previous
    if len(coefficients) == 1:
        return total
    current = matrix @ previous
    total = total + coefficients[1] * current
    for coefficient in coefficients[2:]:
        following = 2 * (matrix @ current) - previous
        total = total + coefficient * following
        previous, current = current, following
    return total


@pytest.mark.parametrize("case", ["inverse", "embedded", "tiny_positive_auto"])
def test_native_inverse_polynomial_action_and_physical_recovery(case):
    embedded = case == "embedded"
    factor = 1e-13 if case == "tiny_positive_auto" else 1.
    sign = 1. if case == "tiny_positive_auto" else -1.
    matrix = factor*(np.array([[1.0, 1.0], [0.0, 2.0]]) if embedded else np.diag([1.0, sign*2.0]))
    rhs = factor*np.array([1.0, 1j])
    alpha, kappa = (3.0, 4.0) if embedded else (2.0, 2.0)
    alpha *= factor
    chosen = native_choice(matrix, rhs, alpha=alpha, kappa=kappa, embedded=embedded,
        encoding="auto" if case == "tiny_positive_auto" else None)
    result, actual = execute(chosen)
    rec = chosen.reconstruction
    operator, start = matrix / alpha, rhs / np.linalg.norm(rhs)
    if embedded:
        operator = np.block([[np.zeros((2, 2)), operator], [operator.T, np.zeros((2, 2))]])
        start = np.r_[start, np.zeros(2)]
    selected = chebyshev_action(
        operator, np.array(rec.polynomial.coefficients) / rec.polynomial.rescale, start
    )
    if embedded:
        selected = selected[2:]
    scale = np.linalg.norm(rhs) * kappa * rec.polynomial.rescale / alpha
    delta = rec.phase_error
    assert np.linalg.norm(actual - scale * selected) <= scale * delta + NUMERICAL_FLOOR
    assert (
        abs(result.algorithm_success_mass - np.linalg.norm(selected) ** 2)
        <= (2 * np.linalg.norm(selected) + delta) * delta + NUMERICAL_FLOOR
    )
    # Hand inverses preserve the physical sign/complex phase and selected half.
    target = np.array([1.0 - 0.5j, 0.5j]) if embedded else np.array([1.0, sign*0.5j])
    allowance = (
        np.linalg.norm(rhs) * kappa / alpha * (EPSILON + rec.polynomial.rescale * delta)
    )
    assert np.linalg.norm(actual - target) <= allowance + NUMERICAL_FLOOR
    assert chosen.reconstruction.width == (5 if embedded else 4)


def test_phase_residual_reaches_the_executed_inverse_through_k_s_and_the_solution_norm():
    """A phase residual far above rounding enters the physical output as ``k s delta/||y||``.

    For ``A = diag(1, .1)``, alpha = 1 and k = 10, the eigenvector b's give ``||y||`` of 1 and 10. Recovery multiplies the
    executed polynomial ``P/s + r``, ``|r| <= delta``, by ``k s``, so the relative error of x against a dense solve is at
    most ``e + k s delta/||y||``, with e the polynomial's check value. The explicit target 1e-3 leaves delta about 9e-4, so
    its worst case ``e + k s delta`` exceeds ``epsilon_inv`` while the computation is valid: the summary must show that
    value, and it must not read as a passed accuracy check. The executed action differs from the exact polynomial by
    more than rounding and at most ``||b|| k s delta/alpha``, which shows delta in the frame the relation claims.
    """
    from nwqlib.algorithms.qls.verification import QLSVerification

    matrix = np.diag([1.0, 0.1])
    for rhs, norm in (([1.0, 0.0], 1.0), ([0.0, 1.0], 10.0)):
        chosen = nwqlib.plan(LinearSystem(A=matrix, b=np.array(rhs, dtype=complex)), output=Solution(), seed=7,
                             method=QLS(alpha=1.0, kappa=10.0, phase_tol=1e-3,
                                        block_encoding="dense_dilation"))
        result = nwqlib.solve(chosen)
        actual = result.value
        rec = chosen.reconstruction
        e, k, s, delta = (rec.polynomial.residual_check_value, rec.polynomial_kappa, rec.polynomial.rescale,
                          rec.phase_error)
        assert 1e-5 < delta <= 1e-3 and k == 10.0 and s > 1
        expected = np.linalg.solve(matrix, rhs)
        assert np.linalg.norm(expected) == pytest.approx(norm, rel=1e-15, abs=0)
        error = np.linalg.norm(actual - expected) / norm
        allowance = e + k * s * delta / norm
        assert error <= allowance + NUMERICAL_FLOOR
        receipt = result.verify(checks=QLSVerification(comparisons=("inverse_relative_error",)))[0]
        facts = {f.fact.quantity: f.fact.value.value for f in receipt.applications[0].facts}
        assert facts["reference.inverse_relative_error.method_allowance"] == pytest.approx(allowance, rel=1e-13, abs=0)
        assert facts["reference.inverse_relative_error.raw_relative_error"] == pytest.approx(error, rel=1e-9, abs=0)
        selected = chebyshev_action(matrix, np.array(rec.polynomial.coefficients) / s, np.array(rhs, dtype=float))
        phase_effect = np.linalg.norm(actual - k * s * selected)
        assert 1e-7 < phase_effect <= k * s * delta + NUMERICAL_FLOOR
        worst = e + k * s * delta
        text = str(result)
        assert worst > chosen.method.epsilon_inv and f"{worst:.3g}" in text
        assert f"{delta:.3g}" in text and f"{rec.phase_tolerance:.3g}" in text
        assert text.endswith("accuracy not assessed") and "PASS" not in text


def test_classical_summary_gives_the_polynomial_check_without_a_phase_component():
    """Classical execution has no phases, so an explicit target changes neither its polynomial nor its record."""
    problem = LinearSystem(A=np.diag([1.0, 0.5]), b=np.array([1.0, 1.0]))
    baseline = nwqlib.plan(problem, method=QLS(), execution="classical", seed=7)
    chosen = nwqlib.plan(problem, method=QLS(phase_tol=1e-3), execution="classical", seed=7)
    assert chosen.reconstruction.polynomial == baseline.reconstruction.polynomial
    assert chosen.reconstruction.phase_tolerance is None and chosen.reconstruction.phase_error is None
    text = str(nwqlib.solve(chosen))
    assert f"{chosen.reconstruction.polynomial.residual_check_value:.3g}" in text and "not applicable" in text


def test_native_shortcut_pair_preserves_right_singular_unit_action():
    """Right singular vectors and a separate Hermitian-dilation action independently check both
    shortcut realizations.
    """
    matrix = np.diag([1.0, 0.5])
    rhs = np.array([math.sqrt(0.3), 1j * math.sqrt(0.7)])
    bprime = np.r_[rhs, 1.0, 0.0] / math.sqrt(2)
    augmented = np.diag([1.0, 0.5, 1 / 1.5, 0.0]).astype(complex)
    g = (np.eye(4) - np.outer(bprime, bprime.conj())) @ augmented
    vectors, tolerances, widths = [], [], []
    for method in ("shortcut_native_svp", "shortcut_dilation"):
        chosen = native_choice(matrix, rhs, method=method, alpha=1.0, compact=True)
        result, actual = execute(chosen)
        rec = chosen.reconstruction
        coefficients = np.array(rec.polynomial.coefficients) / rec.polynomial.rescale
        if method == "shortcut_native_svp":
            _, singular, right = np.linalg.svd(
                g
            )  # One D4 independent reference, no full circuit operator.
            reference = right.conj().T @ (
                np.polynomial.chebyshev.chebval(singular, coefficients) * (right @ np.eye(4)[:, 2])
            )
            selected = reference[:2]
        else:
            h = np.block([[np.zeros((4, 4)), g], [g.conj().T, np.zeros((4, 4))]])
            selected = chebyshev_action(h, coefficients, np.eye(8)[:, 6])[4:6]
        norm = np.linalg.norm(selected)
        expected = selected / norm
        overlap = np.vdot(expected, actual)
        aligned = actual * np.conj(overlap) / abs(overlap)
        delta = rec.phase_error
        tolerance = 2 * delta / norm + NUMERICAL_FLOOR
        assert np.linalg.norm(aligned - expected) <= tolerance
        # Dalzell arXiv:2406.12086v2, Appendix C, Theorem 1 over ||y|| in [1, k] gives the unrescaled
        # success lower bound L in the summary's unit-direction relation, so its term 2 s delta/sqrt(L)
        # must cover this realized 2 delta/m, for m the rescaled norm of the selected vector.
        kappa, s, t, eta = rec.polynomial_kappa, rec.polynomial.rescale, rec.t, rec.polynomial.eta
        center = min(4 * t**2 * n**2 / (t**2 + n**2)**2 for n in (1.0, kappa))
        lower = center * ((1 - eta) / (1 + eta))**2
        assert math.sqrt(lower) / s <= norm
        assert f"{2 * s * delta / math.sqrt(lower):.3g}" in str(result)
        assert (
            abs(result.algorithm_success_mass - norm**2)
            <= (2 * norm + delta) * delta + NUMERICAL_FLOOR
        )
        assert result.norm_squared is None and result.physical_scale is None
        widths.append(chosen.reconstruction.width)
        vectors.append(aligned)
        tolerances.append(tolerance)
    assert np.linalg.norm(vectors[0] - vectors[1]) <= sum(tolerances)
    assert widths == [7, 8]


def test_native_banded_quadratic_form_preserves_physical_scale():
    matrix = np.array([[2.0, -1.0], [-1.0, 2.0]])
    rhs = np.array([1.0, 0.0])
    chosen = native_choice(matrix, rhs, alpha=3.0, kappa=4.0, quadratic=True, compact=True)
    result, _ = execute(chosen)
    rec = chosen.reconstruction
    branch = chebyshev_action(
        matrix / 3.0, np.array(rec.polynomial.coefficients) / rec.polynomial.rescale, rhs
    )
    scale = 4 * rec.polynomial.rescale / 3
    physical = scale * branch
    expected = float(abs(physical[0]) ** 2 - abs(physical[1]) ** 2)
    delta = rec.phase_error
    tolerance = scale**2 * (2 * np.linalg.norm(branch) + delta) * delta + NUMERICAL_FLOOR
    assert abs(result.value - expected) <= tolerance
    assert chosen.reconstruction.width == 4
