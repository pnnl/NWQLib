"""Independent finite quadrature, PSD, source remainder and product-formula relations."""
from __future__ import annotations
from math import factorial
import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp
from _lchs_suzuki_reference import fixed_trotter_term_unitary_with_record
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS,ProviderConfig,cartesian_decomposition,duhamel_quadrature
from nwqlib.algorithms.lchs.providers import composite_gauss_grid,eq7_cbeta,eq7_truncation_range
from nwqlib.algorithms.lchs.solution_error_budget import _numerical_psd_decision
from nwqlib.algorithms.lchs.time_independent_terms import (_TrotterPauliDecomposition,_budgeted_trotter_node_record,_combined_trotter_terms,_node_bound_coefficients,_prepare_decomposition,_union_nonidentity_labels,_trotter_pauli_decomposition,generate_lchs_quadrature,lchs_quadrature_summary)
import nwqlib.algorithms.lchs.inhomogeneous_theory as inhomogeneous_theory_module

def test_cartesian_decomposition_default_reconstructs_a_equals_l_plus_i_h() -> None:
    matrix = np.array([[1.0, 0.2 + 0.3j], [-0.1 + 0.7j, 0.5]], dtype=complex)

    l_part, h_part = cartesian_decomposition(matrix, eigcheck=False)

    np.testing.assert_allclose(l_part, l_part.conjugate().T, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(h_part, h_part.conjugate().T, rtol=0.0, atol=1.0e-15)
    np.testing.assert_allclose(l_part + 1.0j * h_part, matrix, rtol=0.0, atol=1.0e-15)



def test_cartesian_decomposition_can_reject_negative_l_part() -> None:
    # The window is 1e-12*||L||_2 = 4e-13 for this L.
    with pytest.raises(ValueError, match="negative eigenvalue"):
        cartesian_decomposition(np.diag([-5.0e-13, 0.4]))

    l_part, _ = cartesian_decomposition(np.diag([-2.0e-13, 0.4]))
    assert float(np.min(np.linalg.eigvalsh(l_part))) == -2.0e-13



def test_kernel_and_quadrature_helpers_are_validated() -> None:
    with pytest.raises(ValueError, match="beta"):
        eq7_cbeta(1.0)
    with pytest.raises(ValueError, match="epsilon"):
        eq7_truncation_range(0.9, 1.0)



def test_composite_gauss_grid_preserves_each_interval_moments() -> None:
    h1 = 0.4
    node_count = 3
    nodes, weights = composite_gauss_grid(
        interval_count_each_side=2,
        h1=h1,
        node_count=node_count,
    )
    assert np.all(np.isfinite(nodes)) and np.all(np.isfinite(weights))
    assert np.all(weights > 0.0)
    for offset, interval_index in enumerate(range(-2, 2)):
        lower = interval_index * h1
        upper = lower + h1
        interval = slice(offset * node_count, (offset + 1) * node_count)
        interval_nodes = nodes[interval]
        interval_weights = weights[interval]
        assert np.all((lower < interval_nodes) & (interval_nodes < upper))
        for degree in range(2 * node_count):
            integral = (upper ** (degree + 1) - lower ** (degree + 1)) / (degree + 1)
            assert np.dot(interval_weights, interval_nodes**degree) == pytest.approx(
                integral,
                rel=0.0,
                abs=1.0e-15,
            )



def test_trotter_pauli_combination_prunes_after_combining() -> None:
    l_part = np.array([[0.0, 5.0e-13], [5.0e-13, 0.0]], dtype=complex)
    h_part = np.diag([0.2, -0.2]).astype(complex)
    trotter_paulis = _trotter_pauli_decomposition(l_part, h_part)

    small = _combined_trotter_terms(trotter_paulis, k_value=1.0)
    amplified = _combined_trotter_terms(trotter_paulis, k_value=100.0)

    assert "X" not in dict(small.traceless_terms)
    assert dict(amplified.traceless_terms)["X"] == pytest.approx(5.0e-11, rel=2e-15, abs=0)



def test_trotter_identity_coefficient_rejects_imaginary_residue() -> None:
    """The exact node phase exp(-i*T*c0) needs a real identity coefficient c0."""

    def combined(identity: complex):
        decomposition = _TrotterPauliDecomposition(
            num_qubits=1,
            identity_label="I",
            l_coefficients={},
            h_coefficients={"I": identity, "Z": 0.2},
        )
        return _combined_trotter_terms(decomposition, k_value=0.0)

    with pytest.raises(ValueError, match="identity term has coefficient"):
        combined(1.0j)
    # An imaginary residue at or below the 1e-12 coefficient cutoff is roundoff.
    assert combined(0.3 + 1.0e-13j).identity_coefficient == 0.3



def test_scale_aware_numerical_psd_policy_accepts_roundoff_and_shifts_below_window() -> None:
    # ||L||_2 = .25, so the window is 1e-12*.25 = 2.5e-13.
    prepared = _prepare_decomposition(
        matrix=np.diag([-2.0e-13, 0.25]).astype(complex),
        method=LCHS(psd_tolerance=1.0e-12),
    )
    assert prepared.conversion["psd_correction_applied"] is False
    assert prepared.lambda_min_before == -2.0e-13
    assert prepared.numerical_psd_premise_satisfied is True
    assert "psd_certification_method" not in prepared.conversion

    with pytest.raises(ValueError, match="below the numerical PSD tolerance"):
        _prepare_decomposition(
            matrix=np.diag([-5.0e-13, 0.25]).astype(complex),
            method=LCHS(make_l_psd=False),
        )
    shifted = _prepare_decomposition(
        matrix=np.diag([-5.0e-13, 0.25]).astype(complex),
        method=LCHS(make_l_psd=True),
    )
    assert shifted.conversion["psd_correction_applied"] is True
    assert shifted.conversion["psd_shift"] == 5.0e-13 + 1.0e-12*0.25
    assert shifted.numerical_psd_premise_satisfied is True



def test_numerical_psd_decision_does_not_depend_on_the_time_unit() -> None:
    # A -> s*A with T -> T/s describes the same dynamics in another time unit.
    # The Hermitian part below has eigenvalues about -0.526 and 1.026, a
    # genuinely indefinite L. At every s the decision must shift by s times
    # the s = 1 shift. A window with an absolute floor of 1e-12 (in inverse
    # time) would admit the s = 1e-12 spectrum unshifted, although
    # lambda_min*T = -0.526 there.
    base = np.array([[-0.5, 0.3], [0.1, 1.0]])
    lower, upper = np.linalg.eigvalsh(0.5*(base + base.T))
    reference = _numerical_psd_decision(lambda_min=lower, lambda_max=upper, psd_tolerance=1.0e-12,
                                        make_l_psd=True)
    assert reference.shift > -lower
    for scale in (1.0e6, 1.0e-3, 1.0e-12, 1.0e-13):
        decision = _numerical_psd_decision(lambda_min=scale*lower, lambda_max=scale*upper,
                                           psd_tolerance=1.0e-12, make_l_psd=True)
        assert decision.shift == pytest.approx(scale*reference.shift, rel=4.0e-16, abs=0.0)
        with pytest.raises(ValueError, match="below the numerical PSD tolerance"):
            _numerical_psd_decision(lambda_min=scale*lower, lambda_max=scale*upper,
                                    psd_tolerance=1.0e-12, make_l_psd=False)
    # The same scale invariance admits the rounding of an exactly PSD L at
    # every scale. The rank-one c c^T has an exact zero eigenvalue, which
    # eigvalsh returns within a few unit roundoffs times ||L||_2 of zero.
    column = np.array([[1.0], [1.0/3.0], [0.7]])
    gram = column @ column.T
    for scale in (1.0e6, 1.0, 1.0e-12, 1.0e-200):
        eigenvalues = np.linalg.eigvalsh(scale*gram)
        decision = _numerical_psd_decision(lambda_min=eigenvalues[0], lambda_max=eigenvalues[-1],
                                           psd_tolerance=1.0e-12, make_l_psd=False)
        assert decision.shift == 0.0 and decision.premise_satisfied



def test_psd_shift_restores_a_time_rescaled_indefinite_problem() -> None:
    # The classical finite sum at s = 1e-12 and T = 1/s is the s = 1 problem
    # in another time unit. It must shift L, agree with expm(-A*T) u0 within
    # its published algorithmic bound, and match the s = 1 solution.
    base = np.array([[-0.5, 0.3], [0.1, 1.0]])
    initial = np.array([1.0, 1.0])/np.sqrt(2.0)
    exact = scipy.linalg.expm(-base) @ initial
    from nwqlib import solve
    solutions = []
    for scale in (1.0, 1.0e-12):
        result = solve(LinearDynamics(A=scale*base, initial_state=initial, time=1.0/scale),
                       method=LCHS(), execution="classical")
        assert result.plan.reconstruction.psd_shift > 0.0
        bound = next(item.fact.value.value for item in result.facts
                     if item.fact.quantity == "algorithmic_approximation")
        error = np.linalg.norm(np.asarray(result.solution) - exact)
        assert error <= bound
        solutions.append(np.asarray(result.solution))
    # Both runs select the same dimensionless grid (T*||L|| agrees to
    # rounding), so the two finite sums agree to rounding amplified by the
    # recovery exp(shift*T) of about 1.7.
    np.testing.assert_allclose(solutions[1], solutions[0], rtol=0.0, atol=1.0e-12)



def test_numerical_psd_window_scales_with_the_spectral_bounds() -> None:
    # Separates the scale-aware window from an absolute one: at
    # lambda_max = 1e6 the decision window is psd_tolerance * 1e6 = 1e-6,
    # so a -1e-8 eigenvalue is roundoff-scale and must pass, while an
    # absolute 1e-12 window would reject it. A genuinely indefinite value
    # below the scaled window must still refuse.
    prepared = _prepare_decomposition(
        matrix=np.diag([1.0e6, -1.0e-8]).astype(complex),
        method=LCHS(),
    )
    assert prepared.conversion["psd_correction_applied"] is False
    assert prepared.numerical_psd_premise_satisfied is True

    with pytest.raises(ValueError, match="below the numerical PSD tolerance"):
        _prepare_decomposition(
            matrix=np.diag([1.0e6, -1.0e-3]).astype(complex),
            method=LCHS(make_l_psd=False),
        )



@pytest.mark.parametrize("dimension", [2, 4, 8, 16])
def test_node_eigensystem_branch_matches_expm_under_the_regression_contract(dimension) -> None:
    """The eigensystem branch exp(-it(kL+H)) agrees with scipy.linalg.expm at D <= 16.

    Regression contract, not a certificate: atol=2e-12, rtol=0 in operator
    norm for Hermitian fixtures with ||t(kL+H)||_2 <= 1 (the accepted
    numerical-regression policy of time_independent_terms._node_eigensystem;
    the conditional relation there, |t| beta + 2r + r**2 + (1+eta) eps_phase
    + eps_form, is the derived statement). Independent reference:
    scipy.linalg.expm of the same stored generator.
    """
    from nwqlib.algorithms.lchs.time_independent_terms import _branch_unitary, _node_eigensystem
    rng = np.random.default_rng(dimension)
    for _ in range(3):
        a = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        l_part = a @ a.conj().T
        b = rng.normal(size=(dimension, dimension)) + 1j * rng.normal(size=(dimension, dimension))
        h_part = (b + b.conj().T) / 2
        k_value = float(rng.uniform(-2, 2))
        generator = np.multiply(k_value, l_part)
        generator += h_part
        elapsed = float(rng.uniform(.1, 1.)) / np.linalg.norm(generator, 2)
        branch = _branch_unitary(_node_eigensystem(k_value, l_part, h_part), elapsed, dimension)
        reference = scipy.linalg.expm(-1j * elapsed * generator)
        assert np.linalg.norm(branch - reference, 2) <= 2e-12


def test_shared_census_admits_the_relaxed_expression_or_refuses_before_its_structures(monkeypatch) -> None:
    """The shared LCHS census builds the full triple table only when its complete storage fits.

    Forty labels on six qubits at order 2: the exact expression's census
    bytes (error_budget.census_bytes at block 1 with F = N) exceed the limit
    while the degree-capped suffix relaxation fits, so time_independent_terms.
    _node_bound_coefficients records relaxed_prefix without a triple table.
    Below the relaxed envelope it refuses before any pair or triple structure.
    """
    from nwqlib.subroutines.trotterization import error_budget as eb
    from nwqlib.algorithms.lchs.time_independent_terms import _node_bound_coefficients
    rng = np.random.default_rng(11)
    labels = sorted({"".join(rng.choice(list("IXYZ"), 6)) for _ in range(80)} - {"IIIIII"})[:40]
    h = {label: 0.01 * (i + 1) for i, label in enumerate(labels)}
    decomposition = _TrotterPauliDecomposition(num_qubits=6, identity_label="IIIIII", l_coefficients={}, h_coefficients=h)
    combined = _combined_trotter_terms(decomposition, k_value=0.5)
    x, z = eb.pack_labels(tuple(sorted(h)), 6)
    pairs, _ = eb.pair_structure(x, z)
    exact = eb.census_bytes(40, 6, len(pairs), eb.nested_test_count(pairs, 40), 1, order=2, variant="exact_census")
    relaxed = eb.choose_census_block(40, 6, len(pairs), 0, 10**12, order=2, variant="relaxed_prefix")[1]
    assert relaxed < exact
    limit = (exact + relaxed) // 2
    monkeypatch.setattr(eb, "triple_structure", lambda *a, **k: pytest.fail("unadmitted triple table"))
    evaluations, ledger = _node_bound_coefficients(_union_nonidentity_labels(decomposition), 6, {"k": combined.traceless_terms}, ("k",), order=2,
                                                   max_work=10**8, max_bytes=limit, census_held=0)
    assert evaluations["k"].bound_variant == "relaxed_prefix" and ledger["bytes"] <= limit
    monkeypatch.setattr(eb, "pair_structure", lambda *a, **k: pytest.fail("unadmitted pair structure"))
    with pytest.raises(ValueError, match="census"):
        _node_bound_coefficients(_union_nonidentity_labels(decomposition), 6, {"k": combined.traceless_terms}, ("k",), order=2,
                                 max_work=10**8, max_bytes=relaxed // 2, census_held=0)


def _anticommuting_class_labels(q):
    """Two commuting classes, Z+d and X+d for every I/Z pattern d, that anticommute across.

    Every nested test of the order-two census succeeds on these labels, so
    the triple population F equals the nested-test count N.
    """
    from itertools import product
    patterns = ["".join(bits) for bits in product("IZ", repeat=q - 1)]
    return tuple(sorted(["X" + d for d in patterns] + ["Z" + d for d in patterns]))


@pytest.mark.parametrize("reserve", [False, True], ids=["shared_law_limit", "restriction_reserve_limit"])
@pytest.mark.parametrize("omitted", [0, 1], ids=["every_label_kept", "one_label_omitted"])
def test_shared_census_node_restriction_stays_within_the_admitted_bytes(omitted, reserve) -> None:
    """Two k-nodes contracted from one F = N structure stay inside the bytes the census admits.

    q = 6, p = 64, E = 1024, F = N = 48640. The limit is the complete
    envelope at block 256 of error_budget.census_bytes, alone or with the
    node restriction allowance 9*max(E, N) of time_independent_terms.
    _lchs_census_choice held on top, which is also the admitted envelope
    at the chosen block. Under the shared law alone the full
    order-two expression does not fit with that allowance, so the census
    uses relaxed_prefix. With the allowance it admits exact_census. Nodes
    keeping every label need no restriction arrays, so their peak stays
    within the shared law at the chosen block without the allowance, which
    a chained gather-and-remap copy of the rows exceeds. With one label omitted
    per node the nodes restrict their rows (F' < F). Each W_up equals a
    separate census on the node's kept terms at the same variant and block
    (time_independent_terms._node_bound_coefficients).
    """
    import tracemalloc
    from nwqlib.subroutines.trotterization import error_budget as eb
    q = 6
    labels = _anticommuting_class_labels(q)
    p = len(labels)
    x, z = eb.pack_labels(labels, q)
    pairs, _ = eb.pair_structure(x, z)
    E, N = len(pairs), eb.nested_test_count(pairs, p)
    coefficients = np.linspace(0.2, 1.0, p)
    rows = {f"k{node}": tuple((label, value) for index, (label, value) in enumerate(zip(labels, coefficients))
                              if not (omitted and index == node * (p - 1)))
            for node in (0, 1)}
    limit = eb.census_bytes(p, q, E, N, 256, order=2, variant="exact_census",
                            held=9 * max(E, N) if reserve else 0)
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        evaluations, ledger = _node_bound_coefficients(labels, q, rows, ("k0", "k1"), order=2, max_work=10**12,
                                                       max_bytes=limit, census_held=0)
        peak = tracemalloc.get_traced_memory()[1] - start
    finally:
        tracemalloc.stop()
    variant, block = ledger["variant"], ledger["block"]
    assert peak <= ledger["bytes"] <= limit
    assert variant == ("exact_census" if reserve else "relaxed_prefix")
    if reserve:
        assert block == 256 and ledger["bytes"] == limit
    if not omitted:
        triples = N if variant == "exact_census" else 0
        assert peak <= eb.census_bytes(p, q, E, triples, block, order=2, variant=variant)
    for key, kept in rows.items():
        separate = eb._pauli_bound_coefficient_from_terms(kept, 2, choose=lambda E, N: (variant, block))
        assert evaluations[key].coefficient == separate.coefficient


def test_shared_census_prefers_the_default_contraction_block_when_it_fits() -> None:
    """With room for a larger block, the LCHS census contracts with the 65536 block of the shared helpers.

    q = 7, p = 128 F = N labels (F = 391168 > 65536) under a limit that
    admits the block max(1, E, F): time_independent_terms._lchs_census_choice
    replaces it by 65536 when the complete envelope at 65536 fits, as
    error_budget._bound_coefficient_evaluation does, and the admitted W_up
    is then the coefficient of error_budget.coefficient_up at its default
    block.
    """
    from nwqlib.subroutines.trotterization import error_budget as eb
    q = 7
    labels = _anticommuting_class_labels(q)
    terms = tuple(zip(labels, np.linspace(0.2, 1.0, len(labels))))
    evaluations, ledger = _node_bound_coefficients(labels, q, {"k": terms}, ("k",), order=2, max_work=10**12,
                                                   max_bytes=10**10, census_held=0)
    assert ledger["variant"] == "exact_census" and ledger["block"] == 65536
    assert evaluations["k"].coefficient == eb._pauli_bound_coefficient_from_terms(
        terms, 2, choose=lambda E, N: ("exact_census", 65536)).coefficient


def test_budgeted_node_reserves_the_published_pruning_bound_before_selection() -> None:
    """A budgeted node keeps |t|*d + E <= P + E <= T <= epsilon exactly.

    Terms X:0.5, Z:0.5 at order one and time one, a dropped mass of 2**-80
    and epsilon equal to the published one-step formula bound: the
    nearest-rounded remainder equals epsilon and would accept one step,
    whose published pruning and formula bounds exceed epsilon by 2**-80.
    The downward remainder selects two steps
    (time_independent_terms._budgeted_trotter_node_record,
    error_budget._remaining_budget_after_pruning).
    """
    from fractions import Fraction
    from nwqlib.algorithms.lchs.time_independent_terms import _CombinedTrotterTerms, _node_bound_coefficients
    from nwqlib.subroutines.trotterization.error_budget import _finite_bound
    node = _CombinedTrotterTerms(identity_coefficient=0.0, traceless_terms=(("X", .5+0j), ("Z", .5+0j)),
                                 pruned_labels=("I",), pruned_l1_mass=2.0**-80)
    evaluations, _ = _node_bound_coefficients(("X", "Z"), 1, {"k": node.traceless_terms}, ("k",), order=1,
                                              max_work=10**8, max_bytes=10**10, census_held=0)
    epsilon = _finite_bound(evaluations["k"].coefficient, 1.0, 1, 1)
    record = _budgeted_trotter_node_record(node, evaluations["k"], final_time=1.0, error_budget=epsilon, order=1)
    assert record.step_count == 2
    assert (Fraction(2.0**-80) + Fraction(record.pf_bound_value)
            <= Fraction(record.combined_bound_value) <= Fraction(epsilon))
    # An inexact product: the published pruning bound P = up(|t|*d_up)
    # (error_budget.upper_pruning_error) is at least the exact product where
    # the nearest-rounded product falls below it, and the published total
    # still closes |t|*d_up + E <= T <= epsilon.
    from nwqlib.subroutines.trotterization.error_budget import upper_pruning_error
    time = 0.1
    mass = next(value for value in (0.3, 0.7, 0.9, 1.1, 1.3, 1.7, 1.9, 2.3)
                if Fraction(time*value) < Fraction(time)*Fraction(value))
    assert Fraction(upper_pruning_error(time, mass)) >= Fraction(time)*Fraction(mass)
    from dataclasses import replace
    node = replace(node, pruned_l1_mass=mass)
    epsilon = 2*_finite_bound(evaluations["k"].coefficient, time, 1, 1) + upper_pruning_error(time, mass)
    record = _budgeted_trotter_node_record(node, evaluations["k"], final_time=time, error_budget=epsilon, order=1)
    assert (Fraction(time)*Fraction(mass) + Fraction(record.pf_bound_value)
            <= Fraction(record.combined_bound_value) <= Fraction(epsilon))


def test_aligned_combination_keeps_the_dictionary_coefficients_and_bounds_the_dropped_mass() -> None:
    """combine_aligned reproduces the dictionary combination's rounded coefficients and order.

    Independent reference: the dictionary algorithm (copy H, add k*l_P per L
    label, sort the union) evaluated here, compared by binary64 hexadecimal
    components, with the dropped mass checked against the exact rational sum
    of abs(real)+abs(imag) of the dropped coefficients
    (time_independent_terms.combine_aligned, error_budget.upper_dropped_mass).
    """
    from fractions import Fraction
    from nwqlib.algorithms.lchs.time_independent_terms import _PAULI_COEFFICIENT_ATOL

    def reference(l, h, k):
        values = dict(h)
        for label, value in l.items():
            values[label] = values.get(label, 0.0) + k*value
        kept, dropped, identity = [], [], 0.0
        for label in sorted(values):
            value = complex(values[label])
            if abs(value) <= _PAULI_COEFFICIENT_ATOL:
                dropped.append((label, value))
            elif label == "II":
                identity = value.real
            else:
                kept.append((label, value))
        return identity, kept, dropped

    tiny = 2.0**-41
    cases = (
        ({"XX": 1.0}, {"XX": -1.0}),                                  # cancellation at k = 1
        ({}, {"IX": .3, "ZZ": -.2}),                                  # H only
        ({"IZ": .5, "XY": -.25}, {}),                                 # L only
        ({"II": 2.0}, {"II": -.5}),                                   # identity only
        ({"YZ": .7}, {}),                                             # one term
        ({"II": tiny, "XI": 1.0, "ZZ": 2.0**-100}, {"IX": complex(tiny/4, tiny/8), "XI": .5, "ZZ": 2.0**-99}),
    )
    def hexed(terms):
        return [(label, value.real.hex(), value.imag.hex()) for label, value in terms]
    for l, h in cases:
        decomposition = _TrotterPauliDecomposition(num_qubits=2, identity_label="II",
                                                   l_coefficients=l, h_coefficients=h)
        for k in (0.0, 1.0, -1.0, 2.0**-80, 3.5):
            combined = _combined_trotter_terms(decomposition, k_value=k)
            identity, kept, dropped = reference(l, h, k)
            assert combined.identity_coefficient.hex() == float(identity).hex()
            assert hexed(combined.traceless_terms) == hexed(kept)
            assert combined.pruned_labels == tuple(label for label, _ in dropped)
            exact = sum((abs(Fraction(v.real)) + abs(Fraction(v.imag)) for _, v in dropped), Fraction(0))
            assert Fraction(combined.pruned_l1_mass) >= exact
            assert (combined.pruned_l1_mass == 0.0) == (exact == 0)


def test_pruned_mass_and_empty_branch_accounting() -> None:
    """Separate pruned coefficient mass from product-formula error, including empty and commuting
    branches.
    """
    identity = np.asarray(SparsePauliOp.from_list([("I", 1.0)]).to_matrix())
    x_pauli = np.asarray(SparsePauliOp.from_list([("X", 1.0)]).to_matrix())
    y_pauli = np.asarray(SparsePauliOp.from_list([("Y", 1.0)]).to_matrix())
    z_pauli = np.asarray(SparsePauliOp.from_list([("Z", 1.0)]).to_matrix())
    tiny_l = 5.0e-13 * x_pauli
    budgeted_cases = (
        (tiny_l, 0.2 * z_pauli, 1.0, 1.0, 5.0e-13, 5.0e-13, 0.0, 5.0e-13, 1),
        (tiny_l, np.zeros((2, 2), dtype=complex), 1.0, 1.0, 1.0e-12, 5.0e-13, 0.0, 5.0e-13, 0),
        (5.0e-13 * identity, 0.2 * z_pauli, 1.0, 1.0, 1.0e-12, 5.0e-13, 0.0, 5.0e-13, 1),
        (np.zeros((2, 2), dtype=complex), 0.2 * identity, 1.0, 1.0, 1.0e-6, 0.0, 0.0, 0.0, 0),
        (0.2 * z_pauli, 0.3 * z_pauli, 0.4, 0.7, 1.0e-6, 0.0, 0.0, 0.0, 1),
    )
    def budgeted(decomposition, combined, **fields):
        # The node's coefficient comes from the admitted shared census.
        evaluations, _ = _node_bound_coefficients(
            _union_nonidentity_labels(decomposition), decomposition.num_qubits,
            {"node": combined.traceless_terms}, ("node",) if combined.traceless_terms else (),
            order=fields["order"], max_work=10**8, max_bytes=10**10, census_held=0)
        return _budgeted_trotter_node_record(combined, evaluations.get("node"), **fields)

    for l_part, h_part, time, k_value, budget, pruned, pf, combined, steps in budgeted_cases:
        decomposition = _trotter_pauli_decomposition(l_part, h_part)
        record = budgeted(
            decomposition,
            _combined_trotter_terms(decomposition, k_value=k_value),
            final_time=time,
            error_budget=budget,
            order=2,
        )
        assert record.pruned_l1_mass == pruned
        assert record.pf_bound_value == pf
        assert record.combined_bound_value == combined
        assert record.step_count == steps

    tiny_decomposition = _trotter_pauli_decomposition(tiny_l, 0.2 * z_pauli)
    tiny = _combined_trotter_terms(tiny_decomposition, k_value=1.0)
    with pytest.raises(ValueError, match="pruned l1 mass exceeds"):
        budgeted(tiny_decomposition, tiny, final_time=1.0, error_budget=4.0e-13, order=2)

    noncommuting_l = 5.0e-13 * y_pauli
    noncommuting_h = 0.2 * x_pauli + 0.3 * z_pauli
    noncommuting = _trotter_pauli_decomposition(noncommuting_l, noncommuting_h)
    combined = _combined_trotter_terms(noncommuting, k_value=1.0)
    assert combined.pruned_l1_mass > 0.0 and len(combined.traceless_terms) == 2

    with pytest.raises(ValueError, match="kept PF theorem term is nonzero"):
        budgeted(
            noncommuting,
            combined,
            final_time=1.0,
            error_budget=combined.pruned_l1_mass,
            order=2,
        )

    options = LCHS(
        hamiltonian_evolution_backend="trotter",
        trotter_steps=2,
    )
    _, record = fixed_trotter_term_unitary_with_record(
        final_time=1.0,
        k_value=1.0,
        method=options,
        trotter_paulis=_trotter_pauli_decomposition(tiny_l, 0.2 * z_pauli),
    )
    assert record.pruned_l1_mass == 5.0e-13
    assert record.pf_bound_value is None
    assert record.combined_bound_value is None
    assert record.bound_value_status == "not_evaluated"
    assert record.step_count == options.trotter_steps



def _anchor_problem() -> LinearDynamics:
    return LinearDynamics(
        A=np.array([[0.35, 0.05j], [0.02j, 0.45]], dtype=complex),
        initial_state=np.array([1.0, 0.25], dtype=complex),
        source=np.array([0.1, -0.05j], dtype=complex),
        time=0.4,
    )



def test_inhomogeneous_duhamel_quadrature_integrates_polynomials_exactly() -> None:
    final_time = 0.7
    nodes, weights = duhamel_quadrature(final_time, 4)

    # Machine-precision identity: 4-point Gauss-Legendre integrates
    # polynomials through degree 7 exactly on [0, T].
    for degree in range(8):
        assert np.dot(weights, nodes**degree) == pytest.approx(
            final_time ** (degree + 1) / (degree + 1),
            rel=0.0,
            abs=1.0e-15,
        )



def test_duhamel_remainder_uses_physical_growth_and_exact_zero_routes() -> None:
    matrix = np.diag([-0.2, 0.3]).astype(complex)
    source = np.array([1.0, -0.5j], dtype=complex)
    final_time = 0.4
    node_count = 2
    expected = (
        final_time ** (2 * node_count + 1)
        * factorial(node_count) ** 4
        / ((2 * node_count + 1) * factorial(2 * node_count) ** 3)
        * float(np.linalg.norm(matrix, ord=2) ** (2 * node_count))
        * float(np.exp(0.2 * final_time))
        * float(np.linalg.norm(source, ord=2))
    )
    cases = (
        (matrix, source, final_time, node_count, -0.2, expected, False),
        (np.zeros((2, 2), dtype=complex), source, final_time, node_count, 0.0, 0.0, False),
        (matrix, np.zeros(2, dtype=complex), final_time, node_count, -0.2, 0.0, False),
        (
            np.diag([1.0e-200, 2.0e-200]).astype(complex),
            np.array([1.0e-200, 0.0], dtype=complex),
            final_time,
            1,
            0.0,
            0.0,
            True,
        ),
        (np.eye(2, dtype=complex), np.ones(2, dtype=complex), 1.0e200, 1, -1.0, np.inf, True),
    )
    for case_matrix, case_source, time, nodes, lower_bound, value, unusable in cases:
        evaluation = inhomogeneous_theory_module._duhamel_quadrature_error_bound(
            matrix=case_matrix,
            source_term=case_source,
            final_time=time,
            node_count=nodes,
            lambda_min_before_psd_conversion=lower_bound,
        )
        if value == 0.0:
            assert evaluation.value == 0.0
        else:
            # Direct-product reference versus the production log-domain reduction.
            assert evaluation.value == pytest.approx(value, rel=3.0e-15, abs=0.0)
        assert evaluation.unusable_nonzero is unusable



def test_quadrature_summary_projects_selected_provider_bounds() -> None:
    problem = _anchor_problem()
    options = LCHS(
        lchs_kernel=ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": 0.8}),
        approximation_tolerance=0.1,
        k_quadrature=ProviderConfig(
            implementation="composite_gauss", parameters={"truncation_multiplier": 1.1}
        ),
    )
    matrix = problem.A.dense_array()
    quadrature_data = generate_lchs_quadrature(
        matrix=matrix,
        final_time=problem.elapsed_time,
        method=options,
    )
    shared = lchs_quadrature_summary(
        quadrature_data,
        method=options,
        he_backend="dense_exact",
        final_time=problem.elapsed_time,
    )
    provider = quadrature_data.coefficient_plan.quadrature
    # The independent closed-form witnesses above establish the bound itself;
    # this relation checks that summary projects the selected provider bounds.
    assert shared['approximate_lchs_error_bound'] == provider.approximate_lchs_error_bound
    assert shared['quadrature_error_bound'] == provider.quadrature_error_bound



# exp(720) exceeds the largest binary64 number, about exp(709.78).
_GROWTH = 720.0


def _growth_problem(initial, source=None):
    return LinearDynamics(A=[[-_GROWTH, 0.0], [0.0, -_GROWTH]], initial_state=initial, time=1.0, source=source)


@pytest.mark.parametrize("backend", ("dense_exact", "trotter"))
def test_growth_factor_beyond_binary64_is_carried_to_a_representable_solution(backend) -> None:
    # exp(-A*T) u0 = exp(720)*1e-300*e_0, about 4.9e12, although the PSD
    # recovery exp(720) alone overflows. The shifted L is 7.2e-10*I, so the
    # finite sum is within the kernel and quadrature allowances,
    # approximation_tolerance = .01 relative to ||u0||*exp(shift*T), of the
    # exact solution.
    from math import exp, log
    from nwqlib import solve
    from nwqlib.algorithms.lchs import LCHSVerification
    result = solve(_growth_problem([1e-300, 0.0]), method=LCHS(hamiltonian_evolution_backend=backend),
                   execution="classical")
    exact = exp(_GROWTH + log(1e-300))
    solution = np.asarray(result.solution)
    assert abs(solution[0] - exact) <= 0.01*exact and solution[1] == 0.0
    # The weighted components need exp(shift*T) as a float, so they are
    # unknown with that reason.
    kernel = next(item.fact for item in result.facts if item.fact.quantity == "kernel_approximation")
    assert kernel.availability == "unknown" and kernel.reason == "unrepresentable_psd_recovery:applications"
    # The exact saved finite recipe, with the same carried power of two,
    # reproduces the acquired vector to rounding.
    _, facts = result.verify(checks=LCHSVerification(reference="selected_grid", metric="relative_l2"))
    assert facts[0].fact.value.value <= 1e-13


def test_growth_factor_beyond_binary64_with_source_and_quantum_planning() -> None:
    from nwqlib import NormSquared, plan, solve
    from nwqlib.algorithms.lchs import LCHSVerification
    # Each Duhamel application carries its own exp(shift*(T - s_q)) below
    # the common power of two.
    source_result = solve(_growth_problem([1e-300, 0.0], source=[0.0, 1e-300]),
                          method=LCHS(duhamel_nodes=2), execution="classical")
    _, facts = source_result.verify(checks=LCHSVerification(reference="selected_grid", metric="relative_l2"))
    assert facts[0].fact.value.value <= 1e-13
    # The quantum recovery composes the stored coefficient 1-norm, ||u0|| and
    # 2**e with e = floor(shift*T/ln 2) = 1038, and the stored coefficients
    # times 2**e are the compensated ones, exp(shift*T)*sum_j |c_j|.
    from math import ldexp, log
    selected = plan(_growth_problem([1e-300, 0.0]), method=LCHS())
    rec, data = selected.reconstruction, selected._native["native_data"]
    exponent = int(np.floor(rec.psd_shift*rec.elapsed_time/log(2.0)))
    assert exponent == 1038
    assert rec.recovery.as_float() == pytest.approx(rec.coefficient_l1_norm*ldexp(1e-300, exponent), rel=1e-15, abs=0.0)
    compensated_log = log(np.sum(np.abs(data.quadrature.coefficients))) + rec.psd_shift*rec.elapsed_time
    assert log(rec.coefficient_l1_norm) + exponent*log(2.0) == pytest.approx(compensated_log, rel=1e-14, abs=0.0)
    # An unrepresentable physical result is unavailable, not raised.
    unit_input = solve(_growth_problem([1.0, 0.0]), method=LCHS(), output=NormSquared(), execution="classical")
    assert unit_input.value is None and "not representable" in unit_input.unavailable
    unit_vector = solve(_growth_problem([1.0, 0.0]), method=LCHS(), execution="classical")
    assert unit_vector.artifact is None and "unrepresentable" in unit_vector.unavailable


def test_duhamel_remainder_admits_the_singular_values_of_its_norm(monkeypatch):
    # Without a supplied ||A||, the remainder computes it from the singular
    # values of A. LAPACK zgesdd reduces the 64-square matrix to bidiagonal
    # form, ceil(4 * 64**3 / 3) = 349,526 complex multiply-adds, and the dqds
    # algorithm then computes the singular values of the bidiagonal matrix.
    # For a random matrix of this order dqds makes about 2.4 * 64**2
    # divisions, each with five further operations, near 59,000 units. A
    # max_dense_work of 380,000 therefore lies below the actual work and must
    # be refused before the norm is computed. A charge of the reduction, the
    # 4 * 64**2 entrywise work and the scalar terms alone, 366,454, would
    # admit it. The default limit admits the remainder.
    from nwqlib.algorithms.lchs.inhomogeneous_theory import _duhamel_quadrature_error_bound
    generator = np.random.default_rng(3)
    matrix = generator.normal(size=(64, 64)) + 1j*generator.normal(size=(64, 64))
    options = dict(matrix=matrix, source_term=np.ones(64, dtype=complex), final_time=1.0, node_count=1,
                   lambda_min_before_psd_conversion=0.0)

    def norm_before_admission(*args, **kwargs):
        raise AssertionError("the spectral norm ran before its admission")

    with monkeypatch.context() as guarded:
        guarded.setattr(np.linalg, "norm", norm_before_admission)
        with pytest.raises(ValueError, match="max_dense_work"):
            _duhamel_quadrature_error_bound(**options, max_dense_work=380_000)
    assert _duhamel_quadrature_error_bound(**options).value > 0
