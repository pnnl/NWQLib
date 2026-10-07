"""Trotterization error-budget tests.

Anchors the step-count selection to the [CSTWZ] commutator-scaling bounds
(Childs, Su, Tran, Wiebe, and Zhu, PRX 11, 011020 (2021),
doi:10.1103/PhysRevX.11.011020, arXiv:1912.08854).
In the PRX numbering these are first-order Prop. 9 / Eq. (120), second-order
Prop. 10 / Eq. (121), and the Sec. V B smallest-r rule. The arXiv:1912.08854v3
preprint numbers them Prop. 15 / Eq. (145), Prop. 16 / Eq. (152) and
Sec. 5.2. Every commutator sum is recomputed here with an independent dense
numpy pass, and every ceil argument's distance to the nearest integer is
stated so the anchors sit away from flip points.
"""

from __future__ import annotations

import dataclasses
import importlib
import json
import math
from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

from nwqlib.subroutines.trotterization import (
    TROTTER_BOUND_REFERENCE,
    TrotterStepSelection,
    build_trotter_evolution_circuit,
    dense_trotter_bound_coefficient,
    evaluate_trotter_bound,
    select_trotter_step_count,
    trotter_bound_coefficient,
)
from nwqlib.subroutines.trotterization import error_budget
from nwqlib.subroutines.trotterization.error_budget import (
    common_steps,
    emitted_subtotal,
    select_common_step,
)


def test_fixed_step_bound_owner_matches_selector_and_monomial() -> None:
    operator = _anchor_two_qubit()
    time = 0.7
    single = evaluate_trotter_bound(operator, time=time, steps=1, order=2)

    for steps in (1, 2, 7):
        assert evaluate_trotter_bound(
            operator,
            time=time,
            steps=steps,
            order=2,
        ) == single / steps**2

    selection = select_trotter_step_count(
        operator,
        time=time,
        error_budget=single / 3.5**2,
        order=2,
    )
    assert selection.bound_value == evaluate_trotter_bound(
        operator,
        time=time,
        steps=selection.step_count,
        order=2,
    )
    with pytest.raises(ValueError, match="positive integer"):
        evaluate_trotter_bound(operator, time=time, steps=0, order=2)


def _anchor_two_qubit() -> SparsePauliOp:
    """Three summands, pairwise non-commuting with the ZZ head term."""

    return SparsePauliOp.from_list([("ZZ", 0.75), ("XI", 0.60), ("IX", 0.40)])


def _anchor_three_qubit() -> SparsePauliOp:
    """Transverse-field Ising chain on three qubits (two ZZ bonds, field 0.8)."""

    return SparsePauliOp.from_list(
        [("ZZI", 1.0), ("IZZ", 1.0), ("XII", 0.8), ("IXI", 0.8), ("IIX", 0.8)]
    )


# Binary64 roundoff for these anchors and seeded draws (L <= 6, d <= 8).
# Let h_i = |c_i| and tau_i = sum_{j>i}|c_j|. Define B = sum h_i*tau_i
# at order 1, or sum(h_i*tau_i**2/3 + h_i**2*tau_i/6) at order 2.
# These inputs have B/W_exact < 5. With u = 2**-53 and gamma_n = n*u/(1-n*u),
# dense construction, ||M v||/||v||, and scalar sums give
# dense <= D + gamma_256*B (gamma index <= 28*d + 3*L + 10 <= 252).
# D is the exact dense sum and D <= W_exact. trotter_bound_coefficient
# returns an outward upper bound W_up >= W_exact, rounded upward, for either
# bound variant (the relaxed one is larger still). Comparison rounding requires
# less than 1.43e-13, so 1e-12 leaves margin. This assumes finite normal
# arithmetic and a finite nonzero SVD vector. On the tight two-qubit anchor,
# losing the order-1 factor 2 drops W by 50%, and halving the C12 or C24
# contribution drops W by 36.36% or 13.64%, respectively. Revisit the scale
# ratio B/W_exact when the fixtures or their generation change
# (docs/ENGINEERING_CONSTANTS.md).
_DENSE_TAIL_SUM_REL_TOL = 1.0e-12


def _term_matrices(hamiltonian: SparsePauliOp) -> list[np.ndarray]:
    matrices = []
    for label, coefficient in hamiltonian.to_list():
        matrix = np.asarray(SparsePauliOp.from_list([(label, coefficient)]).to_matrix())
        # The roundoff derivation of _DENSE_TAIL_SUM_REL_TOL assumes exact term
        # matrices, every nonzero entry the coefficient times 1, -1, i or -i.
        entries = matrix[matrix != 0]
        assert np.all((entries.real == 0) | (entries.imag == 0))
        assert np.all(np.abs(entries) == abs(coefficient))
        matrices.append(matrix)
    return matrices


def _spectral_norm_from_below(matrix: np.ndarray) -> float:
    """||M v||_2 / ||v||_2 for the top right singular vector v of M.

    For any nonzero v, ||M v|| <= ||M||_2 ||v||, so an inaccurate SVD vector
    cannot raise the exact quotient above the norm of the matrix passed in; it
    can lower it. The bound is one-sided. The matrix passed in is the rounded
    commutator, and its construction error still relates it to the exact
    commutator. The matrix-vector product and the two vector norms round the
    quotient itself (_DENSE_TAIL_SUM_REL_TOL covers both).
    """

    vector = np.linalg.svd(matrix)[2][0].conj()
    return float(np.linalg.norm(matrix @ vector) / np.linalg.norm(vector))


def _prop9_commutator_sum(matrices: list[np.ndarray]) -> float:
    """Independent Eq. (120) sum: sum_g1 ||sum_{g2>g1} [H_g2, H_g1]||."""

    total = 0.0
    for gamma1, head in enumerate(matrices):
        accumulated = np.zeros_like(head)
        for tail_term in matrices[gamma1 + 1 :]:
            accumulated += tail_term @ head - head @ tail_term
        total += _spectral_norm_from_below(accumulated)
    return total


def _prop10_commutator_sums(matrices: list[np.ndarray]) -> tuple[float, float]:
    """Independent Eq. (121) sums (C12, C24) with T_g1 = sum_{g2>g1} H_g2."""

    c12 = 0.0
    c24 = 0.0
    for gamma1, head in enumerate(matrices):
        tail = np.zeros_like(head)
        for tail_term in matrices[gamma1 + 1 :]:
            tail += tail_term
        inner_t = tail @ head - head @ tail
        c12 += _spectral_norm_from_below(tail @ inner_t - inner_t @ tail)
        inner_h = head @ tail - tail @ head
        c24 += _spectral_norm_from_below(head @ inner_h - inner_h @ head)
    return c12, c24


def _dense_product_formula(
    matrices: list[np.ndarray], time: float, steps: int, order: int
) -> np.ndarray:
    """Dense scipy-expm realization of S_order(time / steps)**steps.

    Uses the [CSTWZ] product conventions the bounds are stated for: S_1
    applies H_1 first (gamma = 1 is the rightmost factor), and S_2 is the
    forward-then-reverse half-step palindrome of Prop. 10.
    """

    dim = matrices[0].shape[0]
    step_time = time / steps
    if order == 1:
        step = np.eye(dim, dtype=complex)
        for matrix in matrices:
            step = expm(-1j * step_time * matrix) @ step
    else:
        forward = np.eye(dim, dtype=complex)
        for matrix in matrices:
            forward = expm(-1j * (step_time / 2.0) * matrix) @ forward
        backward = np.eye(dim, dtype=complex)
        for matrix in reversed(matrices):
            backward = expm(-1j * (step_time / 2.0) * matrix) @ backward
        step = backward @ forward
    return np.linalg.matrix_power(step, steps)


# Anchor tuples: (hamiltonian builder, time, budget, expected r at order 1,
# expected r at order 2). Ceil arguments and their distances to the nearest
# integer (all >= 0.045, so double rounding noise around 1e-13 in argument
# units cannot flip a count):
#   two-qubit, t=1.3, eps=0.1:  order 1 arg = 12.6750 (distance 0.325),
#                               order 2 arg =  2.7481 (distance 0.252)
#   three-qubit, t=0.9, eps=0.05: order 1 arg = 51.8400 (distance 0.160),
#                                 order 2 arg =  6.0454 (distance 0.045)
_ANCHORS = [
    (_anchor_two_qubit, 1.3, 0.1, 13, 3),
    (_anchor_three_qubit, 0.9, 0.05, 52, 7),
]


@pytest.mark.parametrize("builder, time, budget, expected_r1, expected_r2", _ANCHORS)
def test_step_count_paper_arithmetic_anchors(
    builder, time: float, budget: float, expected_r1: int, expected_r2: int
) -> None:
    """Selection reproduces the Prop. 9/10 + Sec. V B ceil arithmetic."""

    hamiltonian = builder()
    matrices = _term_matrices(hamiltonian)
    dense_c1 = _prop9_commutator_sum(matrices)
    dense_c12, dense_c24 = _prop10_commutator_sums(matrices)

    selections = []
    for order, expected_r in ((1, expected_r1), (2, expected_r2)):
        selection = select_trotter_step_count(
            hamiltonian, time=time, error_budget=budget, order=order
        )
        selections.append(selection)
        coefficient = trotter_bound_coefficient(hamiltonian, order=order)
        argument = time**2 * coefficient / budget if order == 1 else math.sqrt(
            time**3 * coefficient / budget
        )
        assert np.isfinite(coefficient) and coefficient > 0.0
        assert selection.step_count == max(1, math.ceil(argument)) == expected_r
        dense_coefficient = (
            dense_c1 / 2.0
            if order == 1
            else dense_c12 / 12.0 + dense_c24 / 24.0
        )
        # The Pauli-triangle W bounds the dense tail sum from above, and W_up
        # bounds W. For both orders of the two-qubit anchor and order 1 of the
        # three-qubit anchor the dense sum and W are equal in exact arithmetic
        # (0.75, 0.34375 and 3.2), so the comparison allows the relative
        # roundoff _DENSE_TAIL_SUM_REL_TOL of the dense computation, far below
        # the change from any lost factor in W. The relaxed expression is at
        # least W at order 2.
        assert coefficient >= dense_coefficient * (1.0 - _DENSE_TAIL_SUM_REL_TOL)
        if order == 2:
            relaxed = trotter_bound_coefficient(hamiltonian, order=2, bound_variant="relaxed_prefix")
            assert relaxed >= dense_coefficient * (1.0 - _DENSE_TAIL_SUM_REL_TOL)
        # bound_value is the certified total r * bound(t / r), within the budget, and one step fewer exceeds it.
        assert selection.bound_value <= budget
        assert evaluate_trotter_bound(hamiltonian, time=time, steps=expected_r - 1, order=order) > budget

    first, second = selections
    assert "Prop. 9, Eq. (120)" in first.bound_formula
    assert "Prop. 10, Eq. (121)" in second.bound_formula
    assert "smallest-r rule" in first.step_formula
    assert first.synthesis_method == "lie_trotter"
    assert second.synthesis_method == "suzuki_trotter"
    assert "1912.08854" in TROTTER_BOUND_REFERENCE


@pytest.mark.parametrize("builder, time, budget, expected_r1, expected_r2", _ANCHORS)
@pytest.mark.parametrize("order", [1, 2])
def test_built_circuit_matches_dense_product_formula(
    builder, time: float, budget: float, expected_r1: int, expected_r2: int, order: int
) -> None:
    """The delegated circuit realizes exactly the bounded product formula.

    Machine-precision identity class (atol <= 1e-12): both sides are the
    same dense arithmetic — Pauli-rotation matrices versus scipy expm of
    the same summands — on <= 3 qubits; the observed gap is <= 5e-15.
    ``decompose()`` first: Operator() on the raw PauliEvolutionGate takes
    the gate's exact-exponential matrix, not the synthesized formula.

    An independent exact-exponential comparison also checks Props. 9–10
    certificate soundness on these anchors: actual <= selection.bound_value <= budget.
    """

    hamiltonian = builder()
    selection = select_trotter_step_count(
        hamiltonian, time=time, error_budget=budget, order=order
    )
    circuit = build_trotter_evolution_circuit(hamiltonian, selection)
    synthesized = Operator(circuit.decompose()).data
    manual = _dense_product_formula(
        _term_matrices(hamiltonian), time, selection.step_count, order
    )
    assert np.max(np.abs(synthesized - manual)) <= 1.0e-12
    # Independent exact evolution also tests the certificate, not only synthesis identity.
    exact = expm(-1j * time * hamiltonian.to_matrix())
    actual = float(np.linalg.norm(synthesized - exact, 2))
    assert actual <= selection.bound_value <= budget


def test_commuting_hamiltonian_has_zero_bound_and_exact_circuit() -> None:
    """All-Z summands commute: every Eq. (120)/(121) commutator vanishes.

    The sums are exactly zero in floating point (products of exact Pauli
    entries), one step suffices, and the single-step circuit equals the
    dense evolution at machine precision (identity class, atol <= 1e-12).
    """

    hamiltonian = SparsePauliOp.from_list([("ZI", 0.5), ("IZ", 0.3), ("ZZ", 0.2)])
    assert evaluate_trotter_bound(hamiltonian, time=0.7, steps=1, order=1) == 0.0
    assert evaluate_trotter_bound(hamiltonian, time=0.7, steps=1, order=2) == 0.0
    selection = select_trotter_step_count(
        hamiltonian, time=0.7, error_budget=1.0e-6, order=1
    )
    assert selection.step_count == 1
    assert selection.bound_value == 0.0
    circuit = build_trotter_evolution_circuit(hamiltonian, selection)
    exact = expm(-1j * 0.7 * hamiltonian.to_matrix())
    assert np.max(np.abs(Operator(circuit.decompose()).data - exact)) <= 1.0e-12


def test_rejects_nonpositive_inputs_and_invalid_operators() -> None:
    hamiltonian = _anchor_two_qubit()
    with pytest.raises(ValueError, match="evolution time must be positive"):
        select_trotter_step_count(hamiltonian, time=0.0, error_budget=0.1)
    with pytest.raises(ValueError, match="evolution time must be positive"):
        evaluate_trotter_bound(hamiltonian, time=-1.3, steps=1, order=2)
    with pytest.raises(ValueError, match="error budget must be positive"):
        select_trotter_step_count(hamiltonian, time=1.0, error_budget=0.0)
    with pytest.raises(ValueError, match="error budget must be positive"):
        select_trotter_step_count(hamiltonian, time=1.0, error_budget=-0.1)
    with pytest.raises(ValueError, match="formula order must be 1"):
        select_trotter_step_count(hamiltonian, time=1.0, error_budget=0.1, order=3)
    with pytest.raises(ValueError, match="formula order must be 1"):
        trotter_bound_coefficient(hamiltonian, order=3)
    with pytest.raises(ValueError, match="identity term"):
        select_trotter_step_count(
            SparsePauliOp.from_list([("II", 0.5), ("XX", 1.0)]),
            time=1.0,
            error_budget=0.1,
        )
    with pytest.raises(ValueError, match="coefficients must be real"):
        evaluate_trotter_bound(
            SparsePauliOp.from_list([("XX", 0.5 + 0.1j)]), time=1.0, steps=1, order=1
        )


@pytest.mark.parametrize("order,time,budget,scale", [
    (1, 1.0, 0.125, 1.0),
    (2, 1.0, 0.03125, 1.0),
    (2, 1.0, 2.0, 1.0),
    (1, 1.0, 1.0 / 3.0, 1.0),
    (2, 1.0, 0.5 / 9.0, 1.0),
    (1, 1.0e15, 0.7, 1.0),
    (2, 1.0e20, 0.7, 1.0),
    # Coefficients 1e100 at time 1e-110 and 1e-200 at time 1e200, where a binary64 evaluation of the bound
    # rounds to zero or its coefficient underflows.
    (2, 1.0e-110, 1.0e-34, 1.0e100),
    (1, 1.0e200, 0.5, 1.0e-200),
])
def test_step_selection_is_minimal_for_the_supplied_upper_coefficient(order, time, budget, scale) -> None:
    # For X and Z with coefficient c the Pauli triangle is W = c**2 at order 1 (C1 = 2 c**2) and W = c**3/2 at
    # order 2 (C12 = C24 = 4 c**3), an exact rational of the binary64 inputs. The recorded upper coefficient
    # W_up must dominate it (outward evaluation, error_budget.coefficient_up), and the step count must be
    # minimal for W_up: exact comparisons of B = W_up t**(order + 1) test admissibility and minimality at
    # integer and square boundaries and for integers beyond 2**53, and the published bound is the least
    # binary64 number at or above B/r**order.
    c, t = Fraction(scale), Fraction(time)
    operator = SparsePauliOp.from_list([("X", scale), ("Z", scale)])
    upper = error_budget._bound_coefficient_evaluation(operator, order=order).coefficient
    assert upper >= (c * c if order == 1 else c**3 / 2)
    exact = upper * t ** (order + 1)
    selection = select_trotter_step_count(operator, time=time, error_budget=budget, order=order)
    steps = selection.step_count
    assert exact / steps**order <= Fraction(budget)
    assert steps == 1 or exact / (steps - 1) ** order > Fraction(budget)
    assert Fraction(math.nextafter(selection.bound_value, -math.inf)) < exact / steps**order <= Fraction(
        selection.bound_value)
    assert 0.0 < selection.bound_value <= budget
    assert selection.bound_value == evaluate_trotter_bound(operator, time=time, steps=steps, order=order)


def test_selection_record_round_trips_through_json() -> None:
    selection = select_trotter_step_count(
        _anchor_two_qubit(), time=1.3, error_budget=0.1, order=2
    )
    assert isinstance(selection, TrotterStepSelection)
    payload = json.loads(json.dumps(selection.to_dict()))
    assert payload == {
        "formula_order": 2,
        "evolution_time": 1.3,
        "error_budget": 0.1,
        "bound_formula": selection.bound_formula,
        "step_formula": selection.step_formula,
        "bound_value": selection.bound_value,
        "step_count": 3,
        "synthesis_method": "suzuki_trotter",
        "bound_method": "pauli_triangle",
        "bound_value_status": "structural_upper_bound",
        "bound_variant": "exact_census",
        "algebraic_work_counts": {
            "pauli_terms": 3,
            "pair_commutation_checks": 3,
            "nested_commutation_checks": 4,
        },
        "common_step": None,
    }
    relaxed = select_trotter_step_count(
        _anchor_two_qubit(), time=1.3, error_budget=0.1, order=2, bound_variant="relaxed_prefix"
    ).to_dict()
    # The relaxed census performs the P = 3 pair tests and no nested test.
    assert relaxed["bound_variant"] == "relaxed_prefix"
    assert relaxed["algebraic_work_counts"]["nested_commutation_checks"] == 0
    assert "S_i = sum_{k > i} |c_k|" in relaxed["bound_formula"]


def test_huge_times_select_steps_and_raise_for_an_unrepresentable_fixed_bound() -> None:
    """A huge time still selects an exact step count, while a fixed-step bound beyond binary64 raises."""

    hamiltonian = SparsePauliOp.from_list([("ZZ", 0.75), ("XI", 0.6), ("IX", 0.4)])
    selection = select_trotter_step_count(hamiltonian, time=1.0e200, error_budget=0.1, order=1)
    assert selection.step_count > 2**1000 and 0.0 < selection.bound_value <= 0.1
    with pytest.raises(ValueError, match="overflows at evolution time"):
        evaluate_trotter_bound(hamiltonian, time=1.0e155, steps=1, order=1)


def test_machine_precision_imaginary_residues_are_accepted() -> None:
    """Numerically Hermitian operators pass the coefficient gate.

    Library-native routes (from_operator on exactly Hermitian matrices)
    leave ~1e-17j residues; the gate uses the same 1e-12 tolerance as the
    delegated native rotation path, so these inputs are accepted and a
    genuinely complex coefficient is still rejected.
    """

    dressed = SparsePauliOp(
        ["ZZ", "XI"], coeffs=np.array([0.5 + 1.0e-17j, 0.25 - 5.0e-18j])
    )
    bound = evaluate_trotter_bound(dressed, time=0.5, steps=1, order=1)
    assert np.isfinite(bound) and bound > 0.0
    complex_op = SparsePauliOp(["ZZ", "XI"], coeffs=np.array([0.5 + 0.1j, 0.25]))
    with pytest.raises(ValueError, match="must be real"):
        evaluate_trotter_bound(complex_op, time=0.5, steps=1, order=1)


@pytest.mark.parametrize("order", [1, 2])
def test_pauli_triangle_bound_dominates_dense_tail_sum_on_random_inputs(order: int) -> None:
    rng = np.random.default_rng(7301 + order)
    labels = ("XI", "YI", "ZI", "IX", "IY", "IZ", "XX", "XY", "XZ", "YY", "YZ", "ZZ")
    for _ in range(20):
        selected = rng.choice(labels, size=6, replace=False)
        coefficients = rng.normal(size=6)
        hamiltonian = SparsePauliOp.from_list(list(zip(selected, coefficients, strict=True)))
        matrices = _term_matrices(hamiltonian)
        if order == 1:
            dense = _prop9_commutator_sum(matrices) / 2.0
        else:
            c12, c24 = _prop10_commutator_sums(matrices)
            dense = c12 / 12.0 + c24 / 24.0
        # A draw without tail cancellation makes the two equal in exact
        # arithmetic, so the relative roundoff _DENSE_TAIL_SUM_REL_TOL is allowed.
        for variant in ("exact_census", "relaxed_prefix"):
            upper = trotter_bound_coefficient(hamiltonian, order=order, bound_variant=variant)
            assert upper >= dense * (1.0 - _DENSE_TAIL_SUM_REL_TOL)


@pytest.mark.parametrize(
    "order, terms",
    [
        (
            1,
            [
                ("IZ", 0.016657300576047057),
                ("IY", 0.20573134494682813),
                ("XI", -0.7835948765056134),
                ("ZZ", 1.2264980292859606),
                ("XZ", 0.9432005776780167),
                ("XY", -0.12182388148769023),
            ],
        ),
        (
            2,
            [
                ("YX", -1.302179506862318),
                ("XX", 0.12784040316728537),
                ("XZ", -0.3162425923435822),
                ("XI", -0.016801157504288795),
                ("ZZ", -0.85304392757358),
            ],
        ),
    ],
)
def test_tail_sum_cancellation_makes_structural_bound_strictly_looser(
    order: int, terms: list[tuple[str, float]]
) -> None:
    hamiltonian = SparsePauliOp.from_list(terms)
    structural = trotter_bound_coefficient(hamiltonian, order=order)
    matrices = _term_matrices(hamiltonian)
    if order == 1:
        dense = _prop9_commutator_sum(matrices) / 2.0
    else:
        c12, c24 = _prop10_commutator_sums(matrices)
        dense = c12 / 12.0 + c24 / 24.0
    assert dense_trotter_bound_coefficient(hamiltonian, order=order) == pytest.approx(dense, rel=2e-13, abs=0)
    assert structural > dense


def test_production_bound_uses_no_dense_operator_or_spectral_norm(monkeypatch) -> None:
    error_budget_module = importlib.import_module(
        "nwqlib.subroutines.trotterization.error_budget"
    )
    def fail(*_args, **_kwargs):
        raise AssertionError("dense validation owner reached")

    monkeypatch.setattr(SparsePauliOp, "to_matrix", fail)
    monkeypatch.setattr(error_budget_module, "_spectral_norm", fail)
    selection = select_trotter_step_count(
        _anchor_three_qubit(), time=0.9, error_budget=0.05, order=2
    )
    assert selection.step_count == 7
    assert selection.bound_method == "pauli_triangle"
    assert selection.bound_value_status == "structural_upper_bound"


def test_two_term_pauli_laws_zero_coefficients_and_identity_contract() -> None:
    anticommuting = SparsePauliOp.from_list([("X", 0.5), ("Z", -0.25)])
    # C1 = 2 (1/2)(1/4) gives W = 1/8. C12 = 4 (1/4)(1/4)(1/2) (the k = j triple) and
    # C24 = 4 (1/2)**2 (1/4) give W = 1/96 + 1/96 = 1/48. The outward W_up dominates each, and a lost
    # factor of the order-1 sum or of the nested C12 term (1/8 -> 1/16, 1/48 -> 1/64) falls below.
    assert Fraction(trotter_bound_coefficient(anticommuting, order=1)) >= Fraction(1, 8)
    for variant in ("exact_census", "relaxed_prefix"):
        second = trotter_bound_coefficient(anticommuting, order=2, bound_variant=variant)
        assert Fraction(second) >= Fraction(1, 48)
    with_zero = SparsePauliOp.from_list([("X", 0.5), ("Z", 0.0)])
    assert trotter_bound_coefficient(with_zero, order=1) == 0.0
    assert trotter_bound_coefficient(with_zero, order=2) == 0.0
    with pytest.raises(ValueError, match="identity term"):
        trotter_bound_coefficient(SparsePauliOp.from_list([("I", 0.0)]), order=1)


def _anticommute(left: str, right: str) -> bool:
    """Independent label test: an odd number of positions with two different non-identity symbols."""
    return sum(a != "I" and b != "I" and a != b for a, b in zip(left, right, strict=True)) % 2 == 1


def _exact_triangle(terms, order):
    """Exact rational Pauli-triangle W, relaxed W_2,rel and the pair/nested test counts.

    W_1 = sum_A a_i a_j and W_2 = sum_T a_i a_j a_k / 3 + sum_A a_i^2 a_j / 6, with A the anticommuting
    i < j and T the (i, j, k) with (i, j) in A, k > i (k = j included) and P_k anticommuting with P_i P_j,
    which by bilinearity is anticommutation with exactly one of P_i, P_j. W_2,rel replaces the nested
    indicator by one (error_budget module comment and coefficient_up).
    """
    labels = [label for label, _ in terms]
    a = [abs(Fraction(value)) for _, value in terms]
    p = len(a)
    first = relaxed = second = Fraction(0)
    pair_tests = nested_tests = 0
    for i in range(p):
        for j in range(i + 1, p):
            pair_tests += 1
            if not _anticommute(labels[i], labels[j]):
                continue
            first += a[i] * a[j]
            second += a[i] ** 2 * a[j] / 6
            relaxed += a[i] ** 2 * a[j] / 6 + a[i] * a[j] * sum(a[i + 1:], Fraction(0)) / 3
            for k in range(i + 1, p):
                nested_tests += 1
                if _anticommute(labels[k], labels[i]) != _anticommute(labels[k], labels[j]):
                    second += a[i] * a[j] * a[k] / 3
    return (first if order == 1 else second), relaxed, pair_tests, nested_tests


def _labels(rng, width, count):
    return ["".join(rng.choice(list("IXYZ"), size=width)) for _ in range(count)]


@pytest.mark.parametrize("width", [2, 5, 65, 130])
def test_outward_census_dominates_the_exact_rational_triangle(width) -> None:
    """W_up >= the exact rational W for each variant, with the same pair/nested test events."""
    rng = np.random.default_rng(4100 + width)
    draws = []
    for _ in range(4):
        labels = [label for label in _labels(rng, width, 7) if set(label) != {"I"}]
        coefficients = list(rng.uniform(-1.0, 1.0, size=len(labels)))
        draws.append(list(zip(labels, coefficients, strict=True)))
    # A duplicate term, an exact zero and a 1e-300 coefficient beside unit-scale ones, and an
    # anticommuting pair whose only nested triple is k = j.
    base = draws[0]
    draws.append(base + [base[0], (base[1][0], 0.0), (base[2][0], 1.0e-300)])
    draws.append([("X" + "I" * (width - 1), 0.5), ("Z" + "I" * (width - 1), -0.25)])
    # Every magnitude tiny, so every monomial is far below the binary64 range.
    draws.append([(label, 1.0e-300 * value) for label, value in base])
    for terms in draws:
        operator = SparsePauliOp.from_list(terms)
        for order in (1, 2):
            exact, relaxed, pair_tests, nested_tests = _exact_triangle(terms, order)
            evaluation = error_budget._bound_coefficient_evaluation(operator, order=order)
            assert evaluation.coefficient >= exact
            assert evaluation.pair_commutation_checks == pair_tests
            assert evaluation.nested_commutation_checks == (nested_tests if order == 2 else 0)
            assert Fraction(trotter_bound_coefficient(operator, order=order)) >= exact
            if order == 2:
                loose = error_budget._bound_coefficient_evaluation(
                    operator, order=2, bound_variant="relaxed_prefix"
                )
                assert relaxed >= exact and loose.coefficient >= relaxed
                assert loose.nested_commutation_checks == 0


def test_relaxed_prefix_step_relation_on_the_xyz_witness() -> None:
    """Unit X, Y, Z give W_2 = 3/2 and W_2,rel = 13/6; the relaxed count lies in [r, ceil(alpha r)]."""
    terms = [("X", 1.0), ("Y", 1.0), ("Z", 1.0)]
    operator = SparsePauliOp.from_list(terms)
    exact, relaxed, _, _ = _exact_triangle(terms, 2)
    assert (exact, relaxed) == (Fraction(3, 2), Fraction(13, 6))
    # A budget just above 3/2 at t = 1 separates the counts, 1 and 2, for the recorded upper W_up values.
    budget = 1.5 + 1.0e-9
    full = select_trotter_step_count(operator, time=1.0, error_budget=budget, order=2)
    loose = select_trotter_step_count(
        operator, time=1.0, error_budget=budget, order=2, bound_variant="relaxed_prefix"
    )
    assert (full.step_count, loose.step_count) == (1, 2)
    assert (full.bound_variant, loose.bound_variant) == ("exact_census", "relaxed_prefix")
    w_full = error_budget._bound_coefficient_evaluation(operator, order=2).coefficient
    w_loose = error_budget._bound_coefficient_evaluation(
        operator, order=2, bound_variant="relaxed_prefix"
    ).coefficient
    # r_rel <= ceil(alpha r) with alpha = sqrt(W_rel/W) is (r_rel - 1)**2 < (W_rel/W) r**2.
    assert full.step_count <= loose.step_count
    assert (loose.step_count - 1) ** 2 * w_full < w_loose * full.step_count**2


def test_common_steps_and_emitted_subtotal_on_the_worked_values() -> None:
    # Powers {1, 3}, W = 1, tau = 1 and allowance 6/5: h = 1/2, r = {1: 2, 3: 6}, bounds 1/4 and 3/4.
    h, counts = common_steps(1, 1, (1, 3), {1: Fraction(6, 5), 3: Fraction(6, 5)})
    assert (h, counts) == (Fraction(1, 2), {1: 2, 3: 6})
    assert [1 * counts[p] * h**3 for p in (1, 3)] == [Fraction(1, 4), Fraction(3, 4)]
    # Under t_p = p*val(tau), tau = h_hat = 0.1 with p = r = 3 has zero time displacement, while
    # h_hat = float(Fraction(tau)/3) with nine steps is displaced by 3/2**57 (unit kept mass).
    for step_time, steps, displacement in ((0.1, 3, Fraction(0)), (float(Fraction(0.1) / 3), 9,
                                                                    Fraction(3, 2**57))):
        _, parts = emitted_subtotal(0, 0, (1.0,), 0.0, 0.1, 3, steps, step_time,
                                    (0.5 * step_time,), (0.0,))
        assert parts["time"] == displacement
    # tau = c_I = h_hat = c_j = 0.1 and p = r = 1: the identity-phase error and the one-step
    # two-leaf angle error both equal 1080863910568919/1298074214633706907132624082305024.
    _, parts = emitted_subtotal(0, 0, (0.1,), 0.1, 0.1, 1, 1, 0.1, (0.5 * 0.1 * 0.1,),
                                (-1 * 0.1 * 0.1,))
    witness = Fraction(1080863910568919, 1298074214633706907132624082305024)
    assert parts["identity"] == parts["angles"] == witness


def _census_of(terms):
    return error_budget._pauli_bound_coefficient_from_terms(
        tuple(terms), 2, choose=lambda E, N: ("exact_census", 65536)
    )


def test_common_step_acceptance_branches() -> None:
    terms = (("XZ", 0.3), ("ZI", -0.2), ("YY", 0.1))
    evaluation = _census_of(terms)
    assert evaluation.coefficient > 0
    coefficients = [value for _, value in terms]
    selection = select_common_step(
        evaluation, coefficients=coefficients, identity=0.25, tau=0.1,
        allowances={0: 0.0, 2: 1e-4, 6: 1e-3}, dropped_mass=0.0, max_steps=10**6,
    )
    steps = dict(selection.common_prefix_steps)
    g, m = selection.common_g, selection.common_m
    assert g == 2 and steps == {0: 0, 2: m, 6: 3 * m} and selection.step_count == 3 * m
    assert selection.common_recheck_outcome == "accepted"
    assert selection.bound_variant == "exact_census"
    # Every published prefix bound is at least its exact subtotal and within its allowance.
    half = tuple(0.5 * selection.common_step_time * value for value in coefficients)
    previous, increments = 0, []
    allowances, residuals = dict(selection.common_power_allowances), []
    for p, bound in selection.common_prefix_bounds:
        increments.append(-(p - previous) * 0.1 * 0.25)
        previous = p
        total, parts = emitted_subtotal(evaluation.coefficient, 0.0, coefficients, 0.25, 0.1, p, steps[p],
                                        selection.common_step_time, half, increments)
        assert total <= Fraction(bound) <= Fraction(allowances[p])
        if p:
            residuals.append((Fraction(allowances[p]) - (total - parts["trotter"])) * steps[6] / steps[p])
    # The scalar fields of a common selection (TrotterStepSelection field descriptions): error_budget is the
    # least residual allowance rescaled to r_max steps, rounded downward, which the smaller allowance at
    # power 2 sets here, and bound_value is W_up*r_max*abs(h_hat)**3 rounded upward.
    residual = min(residuals)
    assert Fraction(selection.error_budget) <= residual < Fraction(math.nextafter(selection.error_budget, math.inf))
    formula = evaluation.coefficient * steps[6] * Fraction(selection.common_step_time) ** 3
    assert Fraction(math.nextafter(selection.bound_value, -math.inf)) < formula <= Fraction(selection.bound_value)
    assert selection.to_dict()["common_step"]["g"] == 2
    # No positive powers: no evolution and no common selection.
    assert select_common_step(evaluation, coefficients=coefficients, identity=0.25, tau=0.1,
                              allowances={0: 0.0}, dropped_mass=0.0, max_steps=10) is None
    # Pruning alone above an allowance, or exactly equal to it with W > 0, refuses.
    for mass in (1.0, 0.5):
        with pytest.raises(ValueError, match="pruning exhausts"):
            select_common_step(evaluation, coefficients=coefficients, identity=0.0, tau=0.5,
                               allowances={1: 0.25}, dropped_mass=mass, max_steps=10**6)
    # A step count above max_steps refuses before any parameter is formed.
    with pytest.raises(ValueError, match="exceeds max_steps"):
        select_common_step(evaluation, coefficients=coefficients, identity=0.0, tau=0.1,
                           allowances={2: 1e-9}, dropped_mass=0.0, max_steps=10)
    # An ideal step below half the smallest subnormal represents as zero for positive-time
    # evolution: at the single power 10**700 with tau = 1 the grid needs about p**1.5 steps.
    with pytest.raises(ValueError, match="rounds to zero"):
        select_common_step(evaluation, coefficients=coefficients, identity=0.0, tau=1.0,
                           allowances={10**700: 1.0}, dropped_mass=0.0, max_steps=10**1100)
    # W = 0 with a zero allowance still checks the emitted identity phase, which here does not fit.
    commuting = _census_of((("ZI", 0.3), ("IZ", 0.2)))
    assert commuting.coefficient == 0
    with pytest.raises(ValueError, match="does not meet the allowance 0.0 at power 1"):
        select_common_step(commuting, coefficients=(0.3, 0.2), identity=0.1, tau=0.1,
                           allowances={1: 0.0}, dropped_mass=0.0, max_steps=10)
    # The recheck compares exact binary64 values, so an allowance that would round is refused, and a
    # common step (g*tau = 2e308) or largest target time (2e308) beyond the binary64 range refuses.
    with pytest.raises(ValueError, match="finite binary64 value"):
        select_common_step(commuting, coefficients=(0.3, 0.2), identity=0.0, tau=0.1,
                           allowances={1: Fraction(1, 1000)}, dropped_mass=0.0, max_steps=10)
    for allowances in ({2: 1e300}, {1: 1e300, 2: 1e300}):
        with pytest.raises(ValueError, match="finite binary64 number|exceeds the binary64 range"):
            select_common_step(commuting, coefficients=(0.3, 0.2), identity=0.0, tau=1e308,
                               allowances=allowances, dropped_mass=0.0, max_steps=10)


def test_subnormal_common_step_counts_the_lost_half_angle_product() -> None:
    # h_hat = 2**-1074 with c = 1: 0.5*h_hat rounds to zero (ties to even), so each of the 2r leaf
    # occurrences loses h/2 and B_angles = 2 r (h/2) = r h.
    tau = 5e-324
    evaluation = _census_of((("X", 1.0),))
    selection = select_common_step(evaluation, coefficients=(1.0,), identity=0.0, tau=tau,
                                   allowances={1: 1.0}, dropped_mass=0.0, max_steps=10)
    assert selection.common_step_time == tau and 0.5 * tau == 0.0
    _, parts = emitted_subtotal(evaluation.coefficient, 0, (1.0,), 0.0, tau, 1, 1, tau, (0.0,), (-0.0,))
    assert parts["angles"] == Fraction(tau) and parts["time"] == 0
    assert Fraction(dict(selection.common_prefix_bounds)[1]) >= parts["angles"]


def _common_selection():
    terms = (("XZ", 0.3), ("ZI", -0.2), ("YY", 0.1))
    return select_common_step(
        _census_of(terms), coefficients=[value for _, value in terms], identity=0.25, tau=0.1,
        allowances={0: 0.0, 2: 1e-4, 6: 1e-3}, dropped_mass=0.0, max_steps=10**6,
    )


def test_common_selection_record_refuses_an_inconsistent_grid() -> None:
    """The record derives g from the positive powers and requires r(p) = (p/g)*m and r(p_max) steps."""
    selection = _common_selection()
    m = selection.common_m
    assert (selection.common_g, dict(selection.common_prefix_steps)) == (2, {0: 0, 2: m, 6: 3 * m})
    prefix = r"r\(p\) = \(p/g\)\*m"
    refused = (
        ("common_g must be the gcd", {"common_g": 1}),
        (prefix, {"common_m": m + 1}),
        (prefix, {"common_prefix_steps": ((0, 0), (2, m), (6, 3 * m + 1)), "step_count": 3 * m + 1}),
        ("step_count must be the common count", {"step_count": 3 * m + 1}),
        ("second-order step", {"formula_order": 1}),
        # Each case below is refused by one guard only.
        ("common_m positive", {"common_m": float(m)}),
        ("same distinct nonnegative integer powers",
         {"common_power_allowances": ((0, 1e-3), (2, 1e-3), (4, 1e-3))}),
        ("accepted recheck", {"common_recheck_outcome": "refused"}),
        ("common_tau must be finite and positive", {"common_tau": -0.1}),
        ("finite positive binary64 step", {"common_step_time": 0.0}),
        ("all-zero schedule", {"common_prefix_steps": ((0, 0),), "common_power_allowances": ((0, 1e-3),),
                               "common_prefix_bounds": ((0, 0.0),)}),
    )
    for message, fields in refused:
        with pytest.raises(ValueError, match=message):
            dataclasses.replace(selection, **fields)


@pytest.mark.parametrize("step", ["mul_up", "scaled_magnitudes", "sum_up"])
def test_each_outward_rounding_step_keeps_its_value_above_the_exact_one(step) -> None:
    """Each upward step alone must bound its exact rational value; the others do not compensate here.

    With a dominant commuting term, the scaled magnitudes of the anticommuting pair IX, IZ are tiny:
    at 1e-10 beside 1e200 their scaled product underflows to zero, and at 2**-1074 beside 1e300 the
    scaled magnitude itself underflows. The exact W_1 is the pair product in both cases. The sum
    1 + 3*2**-53 of four terms rounds to 1 in binary64, and one ulp above 1 is still below it.
    """
    if step == "sum_up":
        values = [1.0, 2.0**-53, 2.0**-53, 2.0**-53]
        assert Fraction(error_budget.sum_up(values)) >= sum(map(Fraction, values))
        return
    small, large = (1.0e-10, 1.0e200) if step == "mul_up" else (5e-324, 1.0e300)
    terms = [("ZI", large), ("IX", small), ("IZ", 1.0 if step == "scaled_magnitudes" else small)]
    exact, _, _, _ = _exact_triangle(terms, 1)
    assert exact > 0
    evaluation = error_budget._bound_coefficient_evaluation(SparsePauliOp.from_list(terms), order=1)
    assert evaluation.coefficient >= exact


def test_controlled_suzuki_step_is_the_symmetric_product_of_its_terms() -> None:
    """construct_suzuki_step realizes S_2(t) = prod_forward(t/2) then prod_reverse(t/2) on the control-1 block.

    A repeated forward sweep is a different product formula; on this non-commuting Hamiltonian at
    t = 0.5 its error exceeds the certified W_up*t**3.
    """
    from nwqlib.algorithms.qpe.powers import construct_suzuki_step

    terms = (("XZ", 0.3), ("ZI", -0.2), ("YY", 0.1), ("IX", 0.25))
    step_time = 0.5
    block = SimpleNamespace(
        _payload=(terms, step_time),
        record=SimpleNamespace(signature=SimpleNamespace(quantum=(SimpleNamespace(width=1),
                                                                  SimpleNamespace(width=2)))),
    )
    unitary = Operator(construct_suzuki_step(block, (), None)).data
    # Qubit 0 is the control; the odd basis indices are its |1> block.
    operator = SparsePauliOp.from_list(terms)
    expected = _dense_product_formula(_term_matrices(operator), step_time, 1, 2)
    np.testing.assert_allclose(unitary[1::2, 1::2], expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(unitary[0::2, 0::2], np.eye(4), rtol=0.0, atol=1e-12)
    exact = expm(-1j * step_time * operator.to_matrix())
    certified = evaluate_trotter_bound(operator, time=step_time, steps=1, order=2)
    assert np.linalg.norm(expected - exact, 2) <= certified


@pytest.mark.parametrize("value", [0.3 + 0.0j, np.complex128(0.3), "0.5", np.float32(0.1)])
def test_common_step_binary64_intake_raises_only_value_errors(value) -> None:
    """Complex coefficients and a string time unit are refused with ValueError; float32 0.1 is exact binary64."""
    terms = (("XZ", 0.3), ("ZI", -0.2), ("YY", 0.1))
    evaluation = _census_of(terms)
    arguments = dict(coefficients=[0.3, -0.2, 0.1], identity=0.25, tau=0.1,
                     allowances={2: 1e-4, 6: 1e-3}, dropped_mass=0.0, max_steps=10**6)
    if isinstance(value, (complex, np.complexfloating)):
        # A dropped mass beyond the binary64 range or without a real value is refused alike.
        for mass in (Fraction(10**400), value):
            with pytest.raises(ValueError, match="dropped mass must be finite and nonnegative"):
                select_common_step(evaluation, **{**arguments, "dropped_mass": mass})
        arguments["coefficients"] = [value, -0.2, 0.1]
        with pytest.raises(ValueError, match="each coefficient must be a finite binary64 value"):
            select_common_step(evaluation, **arguments)
        return
    if isinstance(value, str):
        with pytest.raises(ValueError, match="tau must be a finite binary64 value"):
            select_common_step(evaluation, **{**arguments, "tau": value})
        return
    reference = select_common_step(evaluation, **{**arguments, "tau": float(value)})
    assert select_common_step(evaluation, **{**arguments, "tau": value}) == reference


def _random_operator(count, width, seed):
    rng = np.random.default_rng(seed)
    labels = set()
    while len(labels) < count:
        label = "".join(rng.choice(list("IXYZ"), size=width))
        if set(label) != {"I"}:
            labels.add(label)
    return SparsePauliOp.from_list(
        list(zip(sorted(labels), rng.uniform(-1.0, 1.0, size=count), strict=True))
    )


def test_public_census_admits_its_triple_table_before_building_it(monkeypatch) -> None:
    """A byte allowance below the 24F triple table refuses the requested exact_census before any triple.

    For 400 random terms on 8 qubits the pair scan finds about 1.1e7 nested tests, reserved as triples
    before the triple scan, and about 5.3e6 surviving triples, whose 24F bytes (about 128 MB) exceed
    the 100 MB allowance. The pair stage fits it, so the refusal comes after the pair scan and names
    the relaxed alternative; a sufficient allowance returns the unadmitted kernel's value.
    """
    operator = _random_operator(400, 8, 20260929)
    limit = 100_000_000

    def forbidden(*args, **kwargs):
        pytest.fail("a triple table was built")

    monkeypatch.setattr(error_budget, "triple_structure", forbidden)
    with pytest.raises(ValueError) as caught:
        trotter_bound_coefficient(operator, order=2, max_bytes=limit)
    message = str(caught.value)
    assert "Pauli census requested expression stage needs work=" in message
    assert f"max_bytes={limit}" in message and "No checked block fits" in message
    assert 'bound_variant="relaxed_prefix"' in message
    for helper, arguments in (
        (select_trotter_step_count, {"time": 1.0, "error_budget": 0.1}),
        (evaluate_trotter_bound, {"time": 1.0, "steps": 3}),
    ):
        with pytest.raises(ValueError, match="requested expression stage needs work="):
            helper(operator, order=2, max_bytes=limit, **arguments)
    # The explicitly requested relaxation fits the same allowance with pair structure only.
    assert trotter_bound_coefficient(operator, order=2, bound_variant="relaxed_prefix", max_bytes=limit) > 0
    # The pair stage is admitted before the labels are converted.
    monkeypatch.setattr(error_budget, "_validated_terms", forbidden)
    with pytest.raises(ValueError, match="pair stage needs work=.*max_work=1000,"):
        trotter_bound_coefficient(operator, order=2, max_work=1000)
    with pytest.raises(ValueError, match="max_work"):
        trotter_bound_coefficient(operator, order=2, max_work=0)
    monkeypatch.undo()
    terms = tuple(error_budget._validated_terms(operator))
    unadmitted = error_budget._upward_float(error_budget._pauli_bound_coefficient_from_terms(
        terms, 2, choose=lambda E, N: ("exact_census", 65536)).coefficient)
    assert trotter_bound_coefficient(operator, order=2) == unadmitted


def test_public_census_pair_stage_charges_the_label_envelope_and_linear_work(monkeypatch) -> None:
    """The first stage holds 128*p*(q+1)+65536 label bytes and adds 16*p*(q+1)+p*q preparation work."""
    operator = SparsePauliOp.from_list([("X", 0.5), ("Z", -0.25)])
    p, q = 2, 1
    pairs, _ = error_budget.census_sizes(p)
    label_bytes = 128 * p * (q + 1) + 65536
    linear = 16 * p * (q + 1) + p * q
    stage = {"order": 1, "variant": "exact_census"}
    # With p = 2 the only candidate block is one, so the stage needs exactly this many bytes and work.
    needed_bytes = error_budget.census_bytes(p, q, pairs, 0, 1, held=label_bytes, **stage)
    needed_work = linear + error_budget.census_work(p, q, pairs=pairs, **stage)

    def forbidden(*args, **kwargs):
        pytest.fail("labels were converted before the pair stage was admitted")

    monkeypatch.setattr(error_budget, "_validated_terms", forbidden)
    for limits in ({"max_bytes": needed_bytes - 1}, {"max_work": needed_work - 1}):
        with pytest.raises(ValueError, match="pair stage needs work="):
            trotter_bound_coefficient(operator, order=1, **limits)
    monkeypatch.undo()
    # At order 1 the second stage, with the actual E = P = 1, needs the same bytes and work.
    assert trotter_bound_coefficient(operator, order=1, max_bytes=needed_bytes, max_work=needed_work) > 0


@pytest.mark.parametrize("variant", ["exact_census", "relaxed_prefix"])
def test_order_two_variants_admit_pair_work_before_conversion(variant, monkeypatch) -> None:
    """Both order-two variants refuse before converting labels or scanning pairs."""
    import itertools

    q = 6
    patterns = ["".join(t) for t in itertools.product("IZ", repeat=q - 1)]
    labels = sorted(["X" + s for s in patterns] + ["Z" + s for s in patterns])
    operator = SparsePauliOp(labels, np.linspace(.2, 1, len(labels)))
    events = []
    for name in ("pack_labels", "pair_structure"):
        original = getattr(error_budget, name)
        monkeypatch.setattr(error_budget, name, lambda *a, _o=original, _n=name, **k: (events.append(_n), _o(*a, **k))[1])
    to_list = SparsePauliOp.to_list
    monkeypatch.setattr(SparsePauliOp, "to_list", lambda self, *a, **k: (events.append("to_list"), to_list(self, *a, **k))[1])
    with pytest.raises(ValueError, match="Pauli census pair stage needs work="):
        trotter_bound_coefficient(operator, order=2, bound_variant=variant, max_work=1, max_bytes=10**12)
    assert events == []
