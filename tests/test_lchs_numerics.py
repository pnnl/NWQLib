"""Independent finite quadrature, PSD, source remainder and product-formula relations."""
from __future__ import annotations
from math import factorial
import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp
from _lchs_suzuki_reference import fixed_trotter_term_unitary_with_record
from nwqlib import LinearDynamics, plan
from nwqlib.algorithms.lchs import LCHS,Trotter,cartesian_decomposition,duhamel_quadrature
from nwqlib.algorithms.lchs.time_independent_terms import (_budgeted_trotter_node_record,_combined_trotter_terms,_node_bound_coefficients,_union_nonidentity_labels,_trotter_pauli_decomposition)
import nwqlib.algorithms.lchs.inhomogeneous_theory as inhomogeneous_theory_module

def test_cartesian_decomposition_can_reject_negative_l_part() -> None:
    # The window is 1e-12*||L||_2 = 4e-13 for this L.
    with pytest.raises(ValueError, match="negative eigenvalue"):
        cartesian_decomposition(np.diag([-5.0e-13, 0.4]))

    l_part, _ = cartesian_decomposition(np.diag([-2.0e-13, 0.4]))
    assert float(np.min(np.linalg.eigvalsh(l_part))) == -2.0e-13


def test_l_endpoints_follow_the_exact_structure_and_dtype_of_a(monkeypatch) -> None:
    solved = []
    eigvalsh = np.linalg.eigvalsh
    monkeypatch.setattr(np.linalg, "eigvalsh", lambda matrix: solved.append(matrix.dtype) or eigvalsh(matrix))

    def conversion(matrix, **changes):
        problem = LinearDynamics(A=matrix, initial_state=np.eye(len(matrix))[0], time=.5)
        quadrature = plan(problem, method=LCHS(**changes), execution="classical")._native["native_data"].quadrature
        return quadrature.conversion, quadrature.l_norm

    # Off-diagonal L = [[0, 1e-13], [1e-13, 0]] has endpoints -+1e-13, far below the window 1e-25:
    # it is solved, shifted, and refused without make_l_psd.
    tiny = np.array([[0., 1e-13], [1e-13, 0.]])
    facts, _ = conversion(tiny)
    assert facts["min_l_eigenvalue_before_psd_conversion"] == -1e-13 and facts["psd_shift"] == 1e-13 + 1e-25
    with pytest.raises(ValueError, match="negative eigenvalue"):
        cartesian_decomposition(tiny)
    assert solved == [np.float64, np.float64]
    # A diagonal L = diag(3, -2, 1) under an off-diagonal H, padded to 4: exact endpoints (-2, 3)
    # with the padding zero, s = 2 + 3e-12 and ||L + s I|| = 3 + s, without an eigensolve or d**3 work.
    solved.clear()
    hamiltonian = np.array([[0., .3, .1j], [.3, 0., .2], [-.1j, .2, 0.]])
    facts, l_norm = conversion(np.diag([3., -2., 1.]) + 1j*hamiltonian, max_spectral_work=26)
    assert facts["min_l_eigenvalue_before_psd_conversion"] == -2. and facts["psd_shift"] == 2. + 3e-12
    assert l_norm == 3. + facts["psd_shift"]
    # A zero L has endpoints (0, 0).
    facts, l_norm = conversion(1j*hamiltonian)
    assert facts["psd_shift"] == l_norm == 0. and solved == []
    # A real A is decomposed into complex parts as before, and its endpoints come from the real
    # symmetric L.
    real = np.array([[2., 1.], [0., 2.]])
    l_part, h_part = cartesian_decomposition(real)
    assert l_part.dtype == h_part.dtype == np.complex128 and solved == [np.float64]
    np.testing.assert_array_equal(l_part, (real + real.T)/2)
    np.testing.assert_array_equal(h_part, (real - real.T)/2j)



def test_cartesian_pauli_access_matches_exact_rational_bounds() -> None:
    """The Re/Im parts of a 70-qubit Pauli A equal ingesting the same terms, and each
    part bound is the outward rounding of its exact rational value. The direct bound
    sum_P |c_P| lies above a 50-digit evaluation and within a few roundings of it."""
    from decimal import Decimal, localcontext
    from fractions import Fraction
    from math import inf, nextafter
    from nwqlib.operators import ingest_pauli
    from nwqlib.operators.cartesian import cartesian_access

    def outward(value, upward):
        result = float(value)
        if upward and Fraction(result) < value:
            result = nextafter(result, inf)
        if not upward and Fraction(result) > value:
            result = nextafter(result, -inf)
        return result

    rng = np.random.default_rng(11)
    q = 70
    words = {"I" * q}
    while len(words) < 40:
        words.add("".join(rng.choice(list("IXYZ"), size=q)))
    scales = 10.0 ** rng.integers(-300, 3, size=len(words))
    values = (rng.normal(size=len(words)) + 1j * rng.normal(size=len(words))) * scales
    values[1], values[2], values[3] = 2.5, -0.75j, 1e-310 + 3e-310j
    terms = list(zip(sorted(words), values, strict=True))
    access = cartesian_access(ingest_pauli(terms, num_qubits=q))
    for part, facts, pick in ((access.dissipative, access.dissipative_facts, np.real),
                              (access.hamiltonian, access.hamiltonian_facts, np.imag)):
        kept = [(label, complex(pick(value))) for label, value in terms if pick(value) != 0]
        reference = ingest_pauli(kept, num_qubits=q)
        assert part.manifest == reference.manifest and part.structure == "hermitian"
        assert part.pauli_terms().coefficients.tobytes() == reference.pauli_terms().coefficients.tobytes()
        center = sum((Fraction(c.real) for label, c in kept if label == "I" * q), Fraction(0))
        radius = sum((abs(Fraction(c.real)) for label, c in kept if label != "I" * q), Fraction(0))
        assert facts.identity == float(center)
        assert facts.centered_alpha == outward(radius, True)
        assert facts.lower == outward(center - radius, False)
        assert facts.upper == outward(center + radius, True)
        assert facts.norm == outward(abs(center) + radius, True)
    with localcontext() as context:
        context.prec = 50
        exact = sum((Decimal(value.real) ** 2 + Decimal(value.imag) ** 2).sqrt() for value in values)
    split = Fraction(access.dissipative_facts.norm) + Fraction(access.hamiltonian_facts.norm)
    assert access.generator_norm >= exact
    assert Decimal(access.generator_norm) <= min(Decimal(split.numerator) / Decimal(split.denominator),
                                                 exact * (1 + Decimal(2) ** -49))

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


def test_node_census_counts_its_weightings_in_the_selection(monkeypatch) -> None:
    """Two LCHS nodes over one union select the census expression from N(A)*max(w, 2).

    With the threshold at 2N(A) both nodes take the full expression, and one below it both take the
    relaxation. The ledger keeps the w*N(A) word tests and the selection count apart.
    """
    from nwqlib.subroutines.trotterization import error_budget

    labels = ("XYZ", "ZZI", "YIX", "IXY", "XXX", "ZYI", "IZZ", "YXI")
    terms = {"a": tuple((label, 0.1 * (i + 1)) for i, label in enumerate(labels)),
             "b": tuple((label, 0.3) for label in labels[::2])}
    n = error_budget.census_structure(list(labels), 3, 2).nested_tests
    for threshold, variant in ((2 * n, "exact_census"), (2 * n - 1, "relaxed_prefix")):
        monkeypatch.setattr(error_budget, "EXACT_CENSUS_WORD_TESTS", threshold)
        evaluations, ledger = _node_bound_coefficients(labels, 3, terms, ("a", "b"), order=2, max_work=None,
                                                       max_bytes=None, census_held=0)
        assert (ledger["variant"], ledger["word_tests"], ledger["selection_count"]) == (variant, n, 2 * n)
        assert {evaluation.bound_variant for evaluation in evaluations.values()} == {variant}


def test_pruned_mass_and_empty_branch_accounting() -> None:
    """Separate pruned coefficient mass from product-formula error, including empty and commuting
    branches.
    """
    identity = np.asarray(SparsePauliOp.from_list([("I", 1.0)]).to_matrix())
    x_pauli = np.asarray(SparsePauliOp.from_list([("X", 1.0)]).to_matrix())
    y_pauli = np.asarray(SparsePauliOp.from_list([("Y", 1.0)]).to_matrix())
    z_pauli = np.asarray(SparsePauliOp.from_list([("Z", 1.0)]).to_matrix())
    # The node cutoff is 1e-12*max_l max(|h_l|, |k*l_l|) (combine_aligned), 2e-13 beside 0.2*Z.
    # Alone, tiny_l sets the node's own scale, so nothing is dropped.
    tiny_l = 1.0e-13 * x_pauli
    budgeted_cases = (
        (tiny_l, 0.2 * z_pauli, 1.0, 1.0, 1.0e-13, 1.0e-13, 0.0, 1.0e-13, 1),
        (tiny_l, np.zeros((2, 2), dtype=complex), 1.0, 1.0, 1.0e-12, 0.0, 0.0, 0.0, 1),
        (1.0e-13 * identity, 0.2 * z_pauli, 1.0, 1.0, 1.0e-12, 1.0e-13, 0.0, 1.0e-13, 1),
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
        budgeted(tiny_decomposition, tiny, final_time=1.0, error_budget=4.0e-14, order=2)

    # cutoff(s*H, s*L, k) = s*cutoff(H, L, k), so the kept set does not depend on the
    # unit. The cutoff grows with k: 2**-39 = 1.8e-12 survives at k=1 and leaves at k=8.
    for scale in (1.0, 2.0**-43, 2.0**20):
        mixed = _trotter_pauli_decomposition(scale*x_pauli, scale*(0.25*z_pauli + 2.0**-39*y_pauli))
        for k_value, node_scale, kept, dropped in ((0.0, 0.25, ("Y", "Z"), 0.0), (1.0, 1.0, ("X", "Y", "Z"), 0.0),
                                                   (8.0, 8.0, ("X", "Z"), 2.0**-39)):
            node = _combined_trotter_terms(mixed, k_value=k_value)
            assert node.coefficient_cutoff == 1.0e-12*node_scale*scale
            assert tuple(label for label, _ in node.traceless_terms) == kept
            assert node.pruned_l1_mass == dropped*scale

    # An exact cancellation leaves a roundoff residual far above its own scale's
    # cutoff. The operands set the cutoff, so it is cut and its size is counted.
    cancelling = _trotter_pauli_decomposition(z_pauli, (0.1 + 0.2)*z_pauli)
    residual = _combined_trotter_terms(cancelling, k_value=-0.3)
    assert (0.1 + 0.2) - 0.3 > 0.0 and residual.traceless_terms == ()
    assert residual.pruned_l1_mass == (0.1 + 0.2) - 0.3

    zero = np.zeros((2, 2), dtype=complex)
    empty = _combined_trotter_terms(_trotter_pauli_decomposition(zero, zero), k_value=1.0)
    assert (empty.coefficient_cutoff, empty.pruned_l1_mass, empty.traceless_terms) == (0.0, 0.0, ())

    # The scale covers every label, so 2**40*I sets the cutoff 1e-12*2**40 of this node. The
    # term 0.125*X falls below it and its mass is recorded, while the identity stays.
    phase_decomposition = _trotter_pauli_decomposition(0.25*x_pauli, 2.0**40*identity)
    phase = _combined_trotter_terms(phase_decomposition, k_value=0.5)
    assert phase.identity_coefficient == 2.0**40 and phase.traceless_terms == ()
    assert (phase.coefficient_cutoff, phase.pruned_l1_mass) == (1.0e-12*2.0**40, 0.125)
    with pytest.raises(ValueError, match="pruned l1 mass exceeds"):
        budgeted(phase_decomposition, phase, final_time=1.0, error_budget=0.1, order=2)

    noncommuting_l = 2.0e-13 * y_pauli
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
        ham_evo=Trotter(steps=2),
    )
    _, record = fixed_trotter_term_unitary_with_record(
        final_time=1.0,
        k_value=1.0,
        method=options,
        trotter_paulis=_trotter_pauli_decomposition(tiny_l, 0.2 * z_pauli),
    )
    assert record.pruned_l1_mass == 1.0e-13
    assert record.pf_bound_value is None
    assert record.combined_bound_value is None
    assert record.bound_value_status == "not_evaluated"
    assert record.step_count == options.ham_evo.steps


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
    from math import frexp
    from nwqlib.problems.inputs import PhysicalScale

    def scale(norm):
        mantissa, exponent = frexp(norm)
        return PhysicalScale(mantissa=mantissa, exponent=exponent if mantissa else 0)

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
    norm, source_norm = float(np.linalg.norm(matrix, ord=2)), float(np.linalg.norm(source))
    # (||A||, ||b||, T, m, lower bound of L, expected binary64 value). With
    # ||A|| = 2e-200 and ||b|| = 1e-200 the bound, near 1e-602, lies below
    # binary64, and the growth exp(T) at T = 1e200 lies above it. Both stay
    # positive scales, which the stage conversion reports as an upper bound
    # and as unknown.
    cases = (
        (norm, source_norm, final_time, node_count, -0.2, expected),
        (0.0, source_norm, final_time, node_count, 0.0, 0.0),
        (norm, 0.0, final_time, node_count, -0.2, 0.0),
        (2.0e-200, 1.0e-200, final_time, 1, 0.0, None),
        (1.0, 2.0**0.5, 1.0e200, 1, -1.0, None),
    )
    for generator_norm, b_norm, time, nodes, lower_bound, value in cases:
        bound = inhomogeneous_theory_module._duhamel_quadrature_error_bound(
            generator_norm=generator_norm,
            lower_bound=lower_bound,
            source_scale=scale(b_norm),
            final_time=time,
            node_count=nodes,
        )
        if value == 0.0:
            assert bound.mantissa == 0
        elif value is None:
            assert bound.mantissa > 0 and bound.as_float() is None
        else:
            # Direct-product reference versus the production log-domain reduction.
            assert bound.as_float() == pytest.approx(value, rel=3.0e-15, abs=0.0)


# exp(720) exceeds the largest binary64 number, about exp(709.78).
_GROWTH = 720.0


def _growth_problem(initial):
    return LinearDynamics(A=[[-_GROWTH, 0.0], [0.0, -_GROWTH]], initial_state=initial, time=1.0)


@pytest.mark.parametrize("backend", ("dense_exact",))
def test_growth_factor_beyond_binary64_is_carried_to_a_representable_solution(backend, tmp_path) -> None:
    # exp(-A*T) u0 = exp(720)*1e-300*e_0, about 4.9e12, although the PSD
    # recovery exp(720) alone overflows. The shifted L is 7.2e-10*I, so the
    # finite sum is within the kernel and quadrature allowances,
    # approx_tol = .01 relative to ||u0||*exp(shift*T), of the
    # exact solution.
    from math import exp, log
    from nwqlib import load_result, plan, solve
    from nwqlib.algorithms.lchs import LCHSVerification
    result = solve(_growth_problem([1e-300, 0.0]), method=LCHS(ham_evo=backend),
                   execution="classical")
    exact = exp(_GROWTH + log(1e-300))
    solution = np.asarray(result.solution)
    assert abs(solution[0] - exact) <= 0.01*exact and solution[1] == 0.0
    # The kernel component ||u0||*exp(shift*T)*e (solution_error_budget) is
    # about 1e13*e. Its factors combine with integer exponents before one
    # binary64 conversion, so it is reported although exp(shift*T) alone
    # overflows. Each evaluation of the product rounds with relative error of
    # order u*shift*T, about 1.6e-13.
    rec = result.plan.reconstruction
    kernel = next(item.fact for item in result.facts if item.fact.quantity == "kernel_approximation")
    expected = exp(log(1e-300) + rec.psd_shift*1.0 + log(rec.kernel_approximation_bound))
    assert kernel.availability == "concrete" and kernel.value.value == pytest.approx(expected, rel=1e-12, abs=0)
    # The exact saved finite recipe, with the same carried power of two,
    # reproduces the acquired vector to rounding.
    _, facts = result.verify(checks=LCHSVerification(reference="selected_grid", metric="relative_l2"))
    assert facts[0].fact.value.value <= 1e-13
    # Source b = (0, 2**-600) at T = 2**-600, one Duhamel node at T/2 with
    # weight T: the input weight T*||b|| = 2**-1200 is below binary64, while
    # its kernel component 2**-1200*exp(shift*T/2)*e is about 1e-213. The
    # weight and norm are composed with the recovery before conversion.
    final = 2.0**-600
    growth = 1000*log(2.)/final
    chosen = plan(LinearDynamics(A=[[-growth, 0.], [0., -growth]], initial_state=[0., 0.], source=[0., final],
                                 time=final), method=LCHS(ham_evo=backend, duhamel_nodes=1),
                  execution="classical")
    planned = {item.fact.quantity: item.fact for item in chosen.facts}
    rec = chosen.reconstruction
    for name, bound in (("kernel_approximation", rec.kernel_approximation_bound), ("k_quadrature", rec.quadrature_bound)):
        expected = exp(-1200*log(2.) + rec.psd_shift*final/2 + log(bound))
        assert planned[name].availability == "concrete"
        assert planned[name].value.value == pytest.approx(expected, rel=1e-12, abs=0)
    # The descriptive weight 2**-1200 has no binary64 value, so the record
    # omits it instead of binding a false zero. The selected grid forms the
    # same unavailable weight from the saved factors and accepts the
    # schedule, before and after save/load (selected_grid.host_applications).
    result = solve(chosen)
    assert "weight" not in {binding.parameter for binding in result.applications[0].arguments}
    for item in (result, load_result(result.save(tmp_path/"source"))):
        _, facts = item.verify(checks=LCHSVerification(reference="selected_grid", metric="relative_l2"))
        assert facts[0].fact.value.value <= 1e-13


def test_sufficient_counts_hold_any_storable_count_and_never_form_a_huge_power() -> None:
    """Counts above 2**53 are kept exactly; a recovery exponent of +-10**9 decides without forming 2**(4e)."""
    from math import ceil, log
    import tracemalloc
    from nwqlib.algorithms.lchs.analysis import sufficient_counts
    from nwqlib.problems.inputs import PhysicalScale

    def counts(exponent):
        return sufficient_counts(PhysicalScale(mantissa=0.5, exponent=exponent), kind="norm_squared", ranges=(),
                                 tolerance=1e-3, confidence=0.95)

    # N >= Gamma**4 log(2/delta)/(2 epsilon**2) with Gamma = 2**19.
    shots = counts(20)
    assert type(shots) is int and shots > 2**53
    assert shots == pytest.approx(ceil(2.0**76 * log(40) / 2e-6), rel=1e-12)
    tracemalloc.start()
    assert (counts(10**9), counts(-10**9)) == (None, 1)
    # Zero score ranges give radius zero for every N, so any recovery needs one shot.
    for exponent in (10**9, 2000, 0):
        assert sufficient_counts(PhysicalScale(mantissa=0.5, exponent=exponent), kind="quadratic_form",
                                 ranges=(0.0, 0.0), tolerance=0.05, confidence=0.95) == 1
    assert tracemalloc.get_traced_memory()[1] < 2**20
    tracemalloc.stop()
