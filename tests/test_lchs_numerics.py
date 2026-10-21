"""Independent finite quadrature, PSD, source remainder and product-formula relations."""
from __future__ import annotations
from math import factorial
import numpy as np
import pytest
import scipy.linalg
from qiskit.quantum_info import SparsePauliOp
from _lchs_suzuki_reference import fixed_trotter_term_unitary_with_record
from nwqlib import LinearDynamics
from nwqlib.algorithms.lchs import LCHS,cartesian_decomposition,duhamel_quadrature
from nwqlib.algorithms.lchs.time_independent_terms import (_budgeted_trotter_node_record,_combined_trotter_terms,_node_bound_coefficients,_union_nonidentity_labels,_trotter_pauli_decomposition)
import nwqlib.algorithms.lchs.inhomogeneous_theory as inhomogeneous_theory_module

def test_cartesian_decomposition_can_reject_negative_l_part() -> None:
    # The window is 1e-12*||L||_2 = 4e-13 for this L.
    with pytest.raises(ValueError, match="negative eigenvalue"):
        cartesian_decomposition(np.diag([-5.0e-13, 0.4]))

    l_part, _ = cartesian_decomposition(np.diag([-2.0e-13, 0.4]))
    assert float(np.min(np.linalg.eigvalsh(l_part))) == -2.0e-13


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


# exp(720) exceeds the largest binary64 number, about exp(709.78).
_GROWTH = 720.0


def _growth_problem(initial):
    return LinearDynamics(A=[[-_GROWTH, 0.0], [0.0, -_GROWTH]], initial_state=initial, time=1.0)


@pytest.mark.parametrize("backend", ("dense_exact",))
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
