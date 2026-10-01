"""Saved independent polynomial inputs, actual native QLS acquisitions, Q<=8.

The 2e-10 comparison floor is numerical headroom, not a vendor certificate.
These tests compare actual circuits with independent polynomial recurrences.
"""

import json
import math
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import pytest
import nwqlib
from nwqlib.algorithms.qls import QLS, method as owner
from nwqlib.algorithms.qls.primary_records import InversePolynomial, ReflectionPolynomial
from nwqlib.operators import ingest_pauli
from nwqlib.problems import LinearSystem, QuadraticForm, Solution, StateVector
from nwqlib.problems.inputs import ingest_occupation, ingest_product
from nwqlib.subroutines.qsp import phases

NUMERICAL_FLOOR = 2e-10
SELECTED = json.loads((Path(__file__).parent / "data" / "qls_native_selected.json").read_text())


@pytest.fixture(autouse=True)
def fixed_selection(monkeypatch):
    def polynomial(options, kappa):
        assert options.epsilon_inv == SELECTED["epsilon"] and kappa in (2.0, 4.0)
        key = f"inverse{int(kappa)}" if options.solver == "qsvt_inverse" else "reflection2"
        assert options.solver == "qsvt_inverse" or kappa == 2.0
        table = SELECTED["tables"][key]
        fields = {
            name: value
            for name, value in table.items()
            if name not in ("phases", "phase_error")
        }
        record = InversePolynomial if options.solver == "qsvt_inverse" else ReflectionPolynomial
        return record(**fields)

    def phase(coefficients, **kwargs):
        table = next(
            table
            for table in SELECTED["tables"].values()
            if tuple(value / table["rescale"] for value in table["coefficients"])
            == tuple(coefficients)
        )
        return SimpleNamespace(
            phases=tuple(table["phases"]),
            residual_sup_bound=table["phase_error"],
            evaluations=0,
        )

    monkeypatch.setattr(owner, "select_polynomial", polynomial)
    monkeypatch.setattr(phases, "solve_symmetric_qsp_phases", phase)
    return []


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
            epsilon_inv=SELECTED["epsilon"],
            block_encoding_implementation=encoding or ("banded" if quadratic else "dense_dilation"),
            encoded_solution_norm_estimate=None if inverse else 1.5,
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
def test_native_inverse_polynomial_action_and_physical_recovery(fixed_selection, case):
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
        np.linalg.norm(rhs) * kappa / alpha * (SELECTED["epsilon"] + rec.polynomial.rescale * delta)
    )
    assert np.linalg.norm(actual - target) <= allowance + NUMERICAL_FLOOR
    assert chosen.reconstruction.width == (5 if embedded else 4)


def test_native_shortcut_pair_preserves_right_singular_unit_action(fixed_selection):
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


def test_native_banded_quadratic_form_preserves_physical_scale(fixed_selection):
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
