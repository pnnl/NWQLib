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

import importlib
import math
from fractions import Fraction

import numpy as np
import pytest
from qiskit.quantum_info import Operator, SparsePauliOp
from scipy.linalg import expm

from nwqlib.subroutines.trotterization import (
    build_trotter_evolution_circuit,
    dense_trotter_bound_coefficient,
    evaluate_trotter_bound,
    select_trotter_step_count,
    trotter_bound_coefficient,
)
from nwqlib.subroutines.trotterization import error_budget


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
    assert first.synthesis_method == "lie_trotter"
    assert second.synthesis_method == "suzuki_trotter"


@pytest.mark.parametrize("builder, time, budget, expected_r1, expected_r2", _ANCHORS[:1])
@pytest.mark.parametrize("order", [1, 2])
def test_built_circuit_matches_dense_product_formula(
    builder, time: float, budget: float, expected_r1: int, expected_r2: int, order: int
) -> None:
    """The delegated circuit realizes exactly the bounded product formula.

    Machine-precision identity class (atol <= 1e-12): both sides are the
    same dense arithmetic — Pauli-rotation matrices versus scipy expm of
    the same summands — on two qubits; the observed gap is <= 5e-15.
    ``decompose()`` first: Operator() on the raw PauliEvolutionGate takes
    the gate's exact-exponential matrix, not the synthesized formula.

    An independent exact-exponential comparison also checks Props. 9–10
    certificate soundness on the two-qubit anchor: actual <= selection.bound_value <= budget.
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


@pytest.mark.parametrize("order,time,budget,scale", [
    (1, 1.0, 0.125, 1.0),
    (2, 1.0, 0.03125, 1.0),
    (1, 1.0e15, 0.7, 1.0),
])
def test_step_selection_is_minimal_for_the_supplied_upper_coefficient(order, time, budget, scale) -> None:
    # For X and Z with coefficient c the Pauli triangle is W = c**2 at order 1 (C1 = 2 c**2) and W = c**3/2 at
    # order 2 (C12 = C24 = 4 c**3), an exact rational of the binary64 inputs. The recorded upper coefficient
    # W_up must dominate it (outward evaluation, error_budget.coefficient_up), and the step count must be
    # minimal for W_up: exact comparisons of B = W_up t**(order + 1) test admissibility and minimality at
    # integer and square boundaries and for a step count above 2**50, and the published bound is the least
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
    which by bilinearity is anticommutation with exactly one of P_i, P_j. W_2,rel caps its nested mass
    by min(S_i, D_i+D_j-a_i), with S_i the suffix and D_v all anticommuting-neighbor magnitudes
    (error_budget module comment and coefficient_up). Neighbor parity is evaluated from characters.
    """
    labels = [label for label, _ in terms]
    a = [abs(Fraction(value)) for _, value in terms]
    p = len(a)
    degree = [sum((a[k] for k in range(p) if _anticommute(labels[v], labels[k])), Fraction(0))
              for v in range(p)]
    first = relaxed = second = Fraction(0)
    pair_tests = nested_tests = 0
    for i in range(p):
        for j in range(i + 1, p):
            pair_tests += 1
            if not _anticommute(labels[i], labels[j]):
                continue
            first += a[i] * a[j]
            second += a[i] ** 2 * a[j] / 6
            cap = min(sum(a[i + 1:], Fraction(0)), degree[i] + degree[j] - a[i])
            relaxed += a[i] ** 2 * a[j] / 6 + a[i] * a[j] * cap / 3
            for k in range(i + 1, p):
                nested_tests += 1
                if _anticommute(labels[k], labels[i]) != _anticommute(labels[k], labels[j]):
                    second += a[i] * a[j] * a[k] / 3
    return (first if order == 1 else second), relaxed, pair_tests, nested_tests


def _labels(rng, width, count):
    return ["".join(rng.choice(list("IXYZ"), size=width)) for _ in range(count)]


@pytest.mark.parametrize("width", [2, 65])
def test_outward_census_dominates_the_exact_rational_triangle(width) -> None:
    """W_up >= the exact rational W for each variant, with the same pair/nested test events."""
    rng = np.random.default_rng(4100 + width)
    draws = []
    for _ in range(4):
        labels = [label for label in _labels(rng, width, 7) if set(label) != {"I"}]
        coefficients = list(rng.uniform(-1.0, 1.0, size=len(labels)))
        draws.append(list(zip(labels, coefficients, strict=True)))
    # A duplicate term, an exact zero and a 1e-300 coefficient beside unit-scale ones.
    base = draws[0]
    draws.append(base + [base[0], (base[1][0], 0.0), (base[2][0], 1.0e-300)])
    # XI and ZI anticommute; six later IZ commute with both. coefficient_up's degree cap
    # has U_01=1, giving W_rel=1/3+1/6=1/2, versus suffix W=7/3+1/6=5/2.
    cap_terms = [(label + "I" * (width - 2), 1.0) for label in ("XI", "ZI") + ("IZ",) * 6]
    draws.append(cap_terms)
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
                if terms is cap_terms:
                    assert loose.coefficient < Fraction(5, 2)


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
