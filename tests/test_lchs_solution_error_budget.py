"""Stage-bound composition contracts for LCHS solution-level error accounting."""

from __future__ import annotations

import re

import numpy as np
import pytest
from scipy.linalg import expm

import nwqlib
from nwqlib.algorithms import LCHS
from nwqlib.algorithms.lchs import QSPBlockEncoding, Trotter
from nwqlib.algorithms.lchs.solution_error_budget import (
    LCHSApplicationRecord,
    _build_lchs_application_record,
    application_stage_bounds,
    circuit_stage_bounds,
)
from nwqlib.algorithms.lchs.source_selection import select_source_branches
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import PhysicalScale, compose_recovery


RAW_BOUNDS = {
    "approximate_lchs_error_bound": 0.25,
    "quadrature_error_bound": 0.125,
}
INCOMPLETE_PROFILE = {
    **RAW_BOUNDS,
    "solution_error_certificate_status": "incomplete",
    "kernel_parameter_selection": {"profile": "paper_closed_form_f2_1"},
    "unbudgeted_error_stages": [
        "trotter_synthesis",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ],
}


def _application(
    *,
    start_time: float = 0.0,
    elapsed_time: float = 1.0,
    weight: float = 2.0,
    recovery_scale: float = 3.0,
    synthesis_error_bound: float | None = 0.125,
) -> LCHSApplicationRecord:
    return LCHSApplicationRecord(
        start_time=start_time,
        elapsed_time=elapsed_time,
        weight=weight,
        recovery_scale=recovery_scale,
        synthesis_error_bound=synthesis_error_bound,
    )


def _classical(*, raw_record=RAW_BOUNDS, **overrides):
    arguments = {
        "applications": (_application(),),
        "backend": "dense_exact",
        "psd_premise_satisfied": True,
    }
    arguments.update(overrides)
    return application_stage_bounds(raw_record, **arguments)


def _circuit(*, raw_record=RAW_BOUNDS, **overrides):
    arguments = {
        "applications": (_application(),),
        "backend": "dense_exact",
        "inhomogeneous": False,
        "psd_premise_satisfied": True,
        "gamma": 1.0,
        "delta_lcu": 0.0,
        "input_preparation_output_error_bound": 0.0,
    }
    arguments.update(overrides)
    return circuit_stage_bounds(raw_record, **arguments)


def test_stage_bound_manifest_and_composition() -> None:
    """Each stage is its own physical L2 term with an independently derived weight.

    Application a has weight w_a and PSD recovery scale r_a. Operator-level
    truncation and k-quadrature bounds e contribute sum_a w_a*r_a*e, and node
    synthesis bounds s_a contribute sum_a w_a*r_a*s_a. The prepared coefficient
    state enters PREP and PREP-dagger, so its amplitude error contributes
    2*gamma*delta_lcu, and it scales the compensated QSP bound once by gamma.
    Dyadic inputs make every expected sum exact in binary64.
    """
    applications = (
        _application(weight=2.0, recovery_scale=3.0, synthesis_error_bound=0.125),
        _application(
            start_time=0.5,
            elapsed_time=0.5,
            weight=0.5,
            recovery_scale=4.0,
            synthesis_error_bound=0.25,
        ),
    )
    truncation = 2.0 * 3.0 * 0.25 + 0.5 * 4.0 * 0.25
    k_quadrature = 2.0 * 3.0 * 0.125 + 0.5 * 4.0 * 0.125
    synthesis = 2.0 * 3.0 * 0.125 + 0.5 * 4.0 * 0.25

    kernel_stages = {"kernel_approximation": (truncation, None), "k_quadrature": (k_quadrature, None)}
    for backend in ("dense_exact", "qsp_block_encoding"):
        classical = _classical(applications=applications, backend=backend)
        assert list(classical) == ["kernel_approximation", "k_quadrature"]
        assert classical == kernel_stages
    for backend in ("trotter", "trotter_error_budgeted"):
        classical = _classical(applications=applications, backend=backend)
        assert list(classical) == ["kernel_approximation", "k_quadrature", "trotter_synthesis"]
        assert classical == {**kernel_stages, "trotter_synthesis": (synthesis, None)}

    circuit = _circuit(
        applications=applications,
        backend="trotter_error_budgeted",
        inhomogeneous=True,
        gamma=3.0,
        delta_lcu=0.125,
        input_preparation_output_error_bound=0.25,
        duhamel_quadrature_error_bound=0.5,
    )
    assert list(circuit) == [
        "kernel_approximation",
        "k_quadrature",
        "trotter_synthesis",
        "duhamel_quadrature",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert circuit == {
        **kernel_stages,
        "trotter_synthesis": (synthesis, None),
        "duhamel_quadrature": (0.5, None),
        "lcu_coefficient_preparation": (2.0 * 3.0 * 0.125, None),
        "initial_state_preparation": (0.25, None),
    }

    dense = _circuit()
    assert list(dense) == [
        "kernel_approximation",
        "k_quadrature",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert dense["kernel_approximation"] == (2.0 * 3.0 * 0.25, None)

    qsp = _circuit(backend="qsp_block_encoding", gamma=3.0, compensated_recovery_error_bound=0.125)
    assert list(qsp) == [
        "kernel_approximation",
        "k_quadrature",
        "qsp_synthesis",
        "lcu_coefficient_preparation",
        "initial_state_preparation",
    ]
    assert qsp["qsp_synthesis"] == (3.0 * 0.125, None)


def test_incomplete_profile_neither_removes_supplied_stage_values_nor_fills_a_gap() -> None:
    """The Low-Somma profile says what its kernel and k-quadrature split covers, so each stage keeps its own value or reason.

    Independent dyadic relations with two applications (w, r, s) = (2, 3, 1/8) and (1/2, 4, 1/4): synthesis
    sum_a w_a*r_a*s_a = 5/4, kernel e = 1/4 gives 2, k quadrature e = 1/8 gives 1, coefficient PREP 2*gamma*delta_lcu at
    gamma = 3, delta = 1/8 gives 3/4, and QSP gamma*(1/8) gives 3/8 (`circuit_stage_bounds`).
    """
    applications = (_application(weight=2.0, recovery_scale=3.0, synthesis_error_bound=0.125),
                    _application(start_time=.5, elapsed_time=.5, weight=.5, recovery_scale=4.0, synthesis_error_bound=.25))
    kernel = {"kernel_approximation": (2.0, None), "k_quadrature": (1.0, None)}
    profile = dict(raw_record=INCOMPLETE_PROFILE, applications=applications)
    assert _classical(**profile, backend="trotter_error_budgeted") == {**kernel, "trotter_synthesis": (1.25, None)}
    assert _circuit(**profile, backend="trotter_error_budgeted", gamma=3.0, delta_lcu=.125) == {
        **kernel, "trotter_synthesis": (1.25, None), "lcu_coefficient_preparation": (.75, None),
        "initial_state_preparation": (0.0, None)}
    assert _circuit(**profile, backend="qsp_block_encoding", gamma=3.0, compensated_recovery_error_bound=.125) == {
        **kernel, "qsp_synthesis": (.375, None), "lcu_coefficient_preparation": (0.0, None),
        "initial_state_preparation": (0.0, None)}
    assert _circuit(**profile, gamma=3.0) == {**kernel, "lcu_coefficient_preparation": (0.0, None),
                                              "initial_state_preparation": (0.0, None)}

    def stage(name, **overrides):
        return _circuit(raw_record=INCOMPLETE_PROFILE, backend="trotter", **overrides)[name]

    # A gap keeps its own reason, and a required factor stays required even when another factor is zero.
    assert stage("trotter_synthesis", applications=(_application(synthesis_error_bound=None),)) == (
        None, "missing_stage:trotter_synthesis")
    assert stage("lcu_coefficient_preparation", delta_lcu=None) == (None, "missing_stage:lcu_coefficient_preparation")
    assert stage("lcu_coefficient_preparation", gamma=None) == (None, "missing_stage:lcu_coefficient_preparation")
    assert stage("initial_state_preparation", input_preparation_output_error_bound=None) == (
        None, "missing_stage:initial_state_preparation")
    assert stage("trotter_synthesis", unusable_stages={"trotter_synthesis"}) == (None, "unusable_stage:trotter_synthesis")
    assert stage("trotter_synthesis", applications=(_application(synthesis_error_bound=np.inf),)) == (
        None, "nonfinite_stage:trotter_synthesis")
    # A failed PSD premise suppresses the application-weighted stages and no preparation stage.
    assert stage("trotter_synthesis", psd_premise_satisfied=False) == (None, "uncertified_psd_premise")
    assert stage("lcu_coefficient_preparation", psd_premise_satisfied=False) == (0.0, None)
    # An exact zero stays zero, and a positive bound below binary64 stays a positive upper bound.
    assert stage("trotter_synthesis", applications=(_application(synthesis_error_bound=0.0),)) == (0.0, None)
    smallest = np.nextafter(0.0, 1.0)
    assert stage("trotter_synthesis", applications=(_application(weight=smallest, recovery_scale=1.0,
                                                                 synthesis_error_bound=.5),)) == (smallest, None)


def test_stage_bounds_suppress_their_scope_and_tell_exact_zero_from_underflow() -> None:
    """A failed premise or missing stage suppresses its scope, and exact zero differs from
    underflowed nonzero error.
    """
    kernel_values = {"kernel_approximation": 2.0 * 3.0 * 0.25, "k_quadrature": 2.0 * 3.0 * 0.125}

    # The PSD premise is shared by every classical stage; circuit preparation
    # stages do not depend on it.
    assert _classical(backend="trotter", psd_premise_satisfied=False) == {
        name: (None, "uncertified_psd_premise")
        for name in ("kernel_approximation", "k_quadrature", "trotter_synthesis")
    }
    assert _circuit(backend="trotter", psd_premise_satisfied=False, gamma=2.0, delta_lcu=0.125) == {
        "kernel_approximation": (None, "uncertified_psd_premise"),
        "k_quadrature": (None, "uncertified_psd_premise"),
        "trotter_synthesis": (None, "uncertified_psd_premise"),
        "lcu_coefficient_preparation": (2.0 * 2.0 * 0.125, None),
        "initial_state_preparation": (0.0, None),
    }
    for applications, reason in (((), "missing_stage:applications"),):
        assert _classical(applications=applications) == {
            "kernel_approximation": (None, reason),
            "k_quadrature": (None, reason),
        }
        circuit = _circuit(applications=applications)
        assert circuit["kernel_approximation"] == (None, reason)
        assert circuit["initial_state_preparation"] == (0.0, None)

    stage_cases = (
        ({"raw_record": {"quadrature_error_bound": 0.125}}, "kernel_approximation", "missing_stage:kernel_approximation"),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": np.inf}},
            "kernel_approximation",
                "nonfinite_stage:kernel_approximation",
        ),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": -1.0}},
            "kernel_approximation",
            "unusable_stage:kernel_approximation",
        ),
        (
            {"raw_record": {**RAW_BOUNDS, "approximate_lchs_error_bound": 1.0e308}},
            "kernel_approximation",
                "nonfinite_stage:kernel_approximation",
        ),
        ({"unusable_stages": {"k_quadrature"}}, "k_quadrature", "unusable_stage:k_quadrature"),
    )
    for overrides, stage, reason in stage_cases:
        other = "k_quadrature" if stage == "kernel_approximation" else "kernel_approximation"
        assert _classical(**overrides) == {stage: (None, reason), other: (kernel_values[other], None)}
    assert _classical(
        backend="trotter",
        applications=(_application(synthesis_error_bound=None),),
    ) == {
        "kernel_approximation": (kernel_values["kernel_approximation"], None),
        "k_quadrature": (kernel_values["k_quadrature"], None),
        "trotter_synthesis": (None, "missing_stage:trotter_synthesis"),
    }
    circuit_cases = (
        ({"gamma": None}, "lcu_coefficient_preparation"),
        ({"input_preparation_output_error_bound": None}, "initial_state_preparation"),
        ({"backend": "qsp_block_encoding"}, "qsp_synthesis"),
        ({"inhomogeneous": True}, "duhamel_quadrature"),
    )
    for overrides, stage in circuit_cases:
        circuit = _circuit(**overrides)
        assert circuit[stage] == (None, f"missing_stage:{stage}")
        assert circuit["kernel_approximation"] == (kernel_values["kernel_approximation"], None)

    # 2**-1074 * 0.5 = 2**-1075 rounds to zero in binary64. The positive
    # bound becomes the smallest positive binary64 number, 2**-1074, above
    # it, while the exactly zero quadrature bound stays zero. 2**-1074 * 0.75
    # rounds down to 2**-1074 and is raised to 2**-1073, and 2**-1074 * 1 is
    # exact and stays.
    smallest = np.nextafter(0.0, 1.0)
    for raw, expected in ((0.5, smallest), (0.75, 2*smallest), (1.0, smallest)):
        assert _classical(
            applications=(_application(weight=smallest, recovery_scale=1.0),),
            raw_record={"approximate_lchs_error_bound": raw, "quadrature_error_bound": 0.0},
        ) == {
            "kernel_approximation": (expected, None),
            "k_quadrature": (0.0, None),
        }

    # A source-only layout with T = 2**-600 and ||b|| = 2**-600 has the one
    # application weight w*||b|| = 2**-1200, below binary64. Its kernel and
    # k-quadrature stages are the positive 2**-1202 and 2**-1203, so each is
    # reported as the upper bound 2**-1074, not zero.
    final = 2.0**-600
    layout = select_source_branches(coefficients=(1+0j,), k_nodes=(0.,),
                                    initial_scale=PhysicalScale(mantissa=0, exponent=0),
                                    source_scale=PhysicalScale(mantissa=.5, exponent=-599), final_time=final,
                                    shift=0., duhamel_nodes=1, compact=True)
    from_layout = tuple(_build_lchs_application_record({"psd_shift": 0.0}, final_time=final,
        start_time=item.start_time, weight=compose_recovery(item.weight, item.input_norm), input_norm=1.,
        backend="dense_exact", synthesis_error_bound=None) for item in layout.applications)
    assert _classical(applications=from_layout) == {
        "kernel_approximation": (smallest, None),
        "k_quadrature": (smallest, None),
    }
    # Factors combine before the one conversion: a weight 2**-1200 with a
    # recovery 2**1100 gives the representable 2**-100*e, and gamma = 2**1100
    # times a recovery bound .75*2**-1100 gives .75.
    tiny, huge = PhysicalScale(mantissa=.5, exponent=-1199), PhysicalScale(mantissa=.5, exponent=1101)
    assert _classical(applications=(_application(weight=tiny, recovery_scale=huge),)) == {
        "kernel_approximation": (2.0**-100*0.25, None), "k_quadrature": (2.0**-100*0.125, None)}
    # The product-formula bound s_a = sum_j |c_j|*e_j uses the stored raw
    # coefficients: four c_j = 2**-1074 with e_j = 2**-10 give 2**-1082,
    # below binary64, and the input norm 2**1074 makes it 2**-8, exactly in
    # dyadic arithmetic. It neither rounds to zero nor takes the one-norm of
    # the lifted table, 2**52 times larger (_trotter_budget_quadrature_terms).
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.algorithms.lchs.time_independent_terms import (_LCHSTrotterNodeRecord,
                                                               _trotter_budget_quadrature_terms)
    node = _LCHSTrotterNodeRecord(pf_bound_value=2.0**-10, pruned_l1_mass=0.0, combined_bound_value=2.0**-10,
                                  step_count=1, selection=None)
    raw = _trotter_budget_quadrature_terms([node]*4, coefficients=np.full(4, 2.0**-1074, dtype=complex),
        method=LCHS(ham_evo="trotter_error_budgeted"))["trotter_synthesis_error_bound"]
    weighted = _build_lchs_application_record({"psd_shift": 0.0}, final_time=1., start_time=0., weight=1.,
        input_norm=PhysicalScale(mantissa=.5, exponent=1075), backend="trotter_error_budgeted",
        synthesis_error_bound=raw)
    assert _classical(applications=(weighted,), backend="trotter_error_budgeted")["trotter_synthesis"] == (2.0**-8, None)
    composed = _circuit(backend="qsp_block_encoding", inhomogeneous=True, gamma=huge,
                        compensated_recovery_error_bound=PhysicalScale(mantissa=.75, exponent=-1100),
                        duhamel_quadrature_error_bound=PhysicalScale(mantissa=.5, exponent=-1100))
    assert composed["qsp_synthesis"] == (0.75, None)
    assert composed["duhamel_quadrature"] == (smallest, None)
    # Two representable terms 1e308*1*1 whose sum 2e308 exceeds binary64:
    # the stage is unknown with its overflow reason, not a crash.
    pair = (_application(weight=1e308, recovery_scale=1.),)*2
    assert _classical(applications=pair, raw_record={**RAW_BOUNDS, "approximate_lchs_error_bound": 1.0}) == {
        "kernel_approximation": (None, "nonfinite_stage:kernel_approximation"), "k_quadrature": (2.5e307, None)}


_PAULI = {"I": np.eye(2), "X": np.array([[0., 1.], [1., 0.]]), "Z": np.diag([1., -1.])}


def _pauli_sum(terms):
    return sum(c*np.kron(_PAULI[label[0]], _PAULI[label[1]]) for label, c in terms)


def _printed_pruning(result):
    """Return the value of the printed pruning line, which states its physical-solution frame."""
    return float(re.search(r"Pauli pruning bound: (\S+) in the physical solution;", str(result))[1])


def test_fixed_step_pruning_follows_the_unit_and_reports_its_propagated_component():
    """Fixed-step pruning is covariant under a unit change and reports its known part.

    Independent relations: exp(-A*T) of A*T is unchanged by A -> s*A, T -> T/s
    with the same 40-step formula, and Duhamel's bound for the deleted terms
    is sum_a |w_a|*|t_a|*||v_a||*sum_j |c_j|*d_j. The expected values come from
    scipy's expm and a hand-formed Gauss-Legendre sum. The total stays unknown
    because the product-formula error is unknown (time_independent_terms.combine_aligned).
    Printed values have 8 significant digits, so they are compared at 1e-7.
    """
    terms = [("ZI", .4), ("IX", .15), ("II", .6), ("XX", .3j), ("ZZ", .2j)]
    dense = _pauli_sum(terms)
    initial = np.array([1., 0., 0., 0.])
    method = LCHS(approx_tol=1e-2, ham_evo=Trotter(steps=40))
    reference = expm(-dense) @ initial

    def error(scale):
        problem = nwqlib.LinearDynamics(A=dense*scale, initial_state=initial, time=1/scale)
        return np.linalg.norm(nwqlib.solve(problem, method=method, execution="classical", seed=7).solution-reference)

    # scale=1e6 has Pauli imaginary residues near 1e-10, inside this node's own window.
    assert error(1.) < 3e-4
    for scale in (1e-12, 1e-13, 1e6):
        assert error(scale) == pytest.approx(error(1.), rel=1e-3)

    # Dyadic coefficients keep the Pauli transform exact, so every node drops exactly
    # 2**-43 < 1e-12*.25 and keeps the rest.
    tiny = 2.**-43
    l_part = _pauli_sum([("II", .5), ("ZI", .5)])
    h_kept = _pauli_sum([("XX", .25), ("IZ", .25)])
    h_part = h_kept+tiny*_pauli_sum([("ZZ", 1.)])
    time = .75
    problem = nwqlib.LinearDynamics(A=l_part+1j*h_part, initial_state=initial, time=time)
    chosen = nwqlib.plan(problem, method=method, execution="classical")
    data = chosen._native["native_data"]
    assert chosen.reconstruction.psd_shift == 0.
    nodes = data.quadrature.k_nodes
    coefficients = data.quadrature.coefficients
    expected = time*sum(abs(coefficients))*tiny
    facts = {item.fact.quantity: item.fact for item in chosen.facts}
    assert facts["trotter_pruning"].value.value == pytest.approx(expected, rel=1e-14)
    assert facts["trotter_synthesis"].value is None
    result = nwqlib.solve(chosen)
    reported = {item.fact.quantity: item.fact for item in result.facts}
    assert reported["trotter_pruning"].value == facts["trotter_pruning"].value
    assert reported["algorithmic_approximation"].value is None
    assert "accuracy not assessed" in str(result)
    assert _printed_pruning(result) == pytest.approx(expected, rel=1e-7)
    assert result.assess(absolute_tolerance=1.).status == "INCONCLUSIVE"

    def branches(generator):
        return sum(c*expm(-1j*time*(generator+k*l_part)) for k, c in zip(nodes, coefficients, strict=True)) @ initial
    # Only the pruning term: both sums use exact exponentials of their generators.
    measured = np.linalg.norm(branches(h_part)-branches(h_kept))
    assert 0 < measured <= expected and measured > .01*expected

    # A later fixed-step certificate contains the pruning bound, so the total counts it once.
    _, refined = result.verify(checks=nwqlib.algorithms.lchs.LCHSRefinement(components=("fixed_pf",)))
    refined = {item.fact.quantity: item.fact.value.value for item in refined}
    kernel = sum(reported[name].value.value for name in ("kernel_approximation", "k_quadrature"))
    assert refined["algorithmic_approximation"] == pytest.approx(kernel+refined["trotter_synthesis"], rel=1e-12)

    # Each Duhamel application has its own start and input, and a quantum Plan reports the same sum.
    source = np.array([0., 3., 0., 4.])
    inhomogeneous = nwqlib.LinearDynamics(A=l_part+1j*h_part, initial_state=initial, source=source, time=time)
    forced = LCHS(approx_tol=1e-2, ham_evo=Trotter(steps=4), duhamel_nodes=3)
    points, weights = np.polynomial.legendre.leggauss(3)
    starts, weights = (points+1)*time/2, weights*time/2
    duration_sum = time*np.linalg.norm(initial)+sum(w*(time-s)*np.linalg.norm(source) for s, w in zip(starts, weights, strict=True))
    for execution, output in (("classical", nwqlib.Solution()), ("quantum", nwqlib.NormSquared())):
        chosen = nwqlib.plan(inhomogeneous, method=forced, execution=execution, output=output)
        l1 = sum(abs(chosen._native["native_data"].quadrature.coefficients))
        facts = {item.fact.quantity: item.fact for item in chosen.facts}
        assert facts["trotter_pruning"].value.value == pytest.approx(duration_sum*l1*tiny, rel=1e-13)
    homogeneous = nwqlib.plan(problem, method=method, execution="quantum", output=nwqlib.NormSquared())
    facts = {item.fact.quantity: item.fact for item in homogeneous.facts}
    assert facts["trotter_pruning"].value.value == pytest.approx(expected, rel=1e-14)

    # A quantum Result prints the same value in the physical-solution frame when another output is
    # requested. The step count does not enter this bound, and 4 steps run faster.
    quantum = nwqlib.solve(problem, method=LCHS(approx_tol=1e-2, ham_evo=Trotter(steps=4)),
                                                output=nwqlib.NormSquared(), seed=7)
    assert _printed_pruning(quantum) == pytest.approx(expected, rel=1e-7)

    # The budgeted route already has pruning inside trotter_synthesis and refuses an exhausted allowance.
    budgeted = nwqlib.plan(problem, method=LCHS(approx_tol=1e-2,
        ham_evo="trotter_error_budgeted"), execution="classical")
    assert "trotter_pruning" not in {item.fact.quantity for item in budgeted.facts}


def _phase_coefficient_residual(solution):
    """Return the coefficient one-norm between saved Wx phases and their target, from plain 2 x 2 products.

    P(x) = <0|exp(i phi_0 Z) W(x) exp(i phi_1 Z) ... W(x) exp(i phi_d Z)|0> with
    W(x) = [[x, i s], [i s, x]] and s = sqrt(1 - x**2). Re P at d + 1 Chebyshev
    points determines the realized coefficients of degree at most d.
    """
    phases = np.asarray(solution.phases)
    nodes = np.cos(np.pi*(np.arange(len(phases))+.5)/len(phases))
    values = []
    for x in nodes:
        signal = np.array([[x, 1j*np.sqrt(1-x**2)], [1j*np.sqrt(1-x**2), x]])
        product = np.diag(np.exp(1j*phases[0]*np.array([1., -1.])))
        for phi in phases[1:]:
            product = product @ signal @ np.diag(np.exp(1j*phi*np.array([1., -1.])))
        values.append(product[0, 0].real)
    realized = np.polynomial.chebyshev.chebfit(nodes, values, len(phases)-1)
    return np.abs(realized-np.asarray(solution.target_chebyshev_coefficients)).sum()


def test_shared_route_result_prints_its_phase_target_and_residuals_and_another_route_does_not(tmp_path):
    """A shared route's Result shows the saved phase target and each residual, also after reloading.

    Independent relations: each printed residual is the coefficient one-norm of
    plain products of that parity's phases against its target, and the printed
    target is (A - tail)/zeta, where zeta = 2 s on the raw route
    (coherent_selection._shared_row). Rounding to the printed 3 significant
    digits moves a value by at most 0.5%. A Laurent route has no such pair of
    solutions, so it has no line.
    """
    A = ingest_pauli([("II", .3), ("ZI", 2j), ("IZ", 2j), ("XX", 1j)], num_qubits=2)
    problem = nwqlib.LinearDynamics(A=A, initial_state=[1., 0., 0., 0.], time=1.)
    result = nwqlib.solve(problem, method=LCHS(), output=nwqlib.NormSquared(), seed=3)
    prepared = result.plan._native["native_data"].qsp_plan
    row = prepared.candidate
    assert row.route == "shared_raw"

    def printed(item):
        line = re.search(r"Phase synthesis \(shared_raw\): residuals cos=(\S+), sin=(\S+) against target (\S+) per parity "
                         r"for the normalized polynomials; nominal numerical estimates, not total physical-output error",
                         str(item))
        return tuple(float(value) for value in line.groups())

    expected = (_phase_coefficient_residual(prepared.phases.cos_solution),
                _phase_coefficient_residual(prepared.phases.sin_solution),
                (row.allowance["algorithmic"]-row.error_forecast["tail"])/row.recovery)
    assert printed(result) == pytest.approx(expected, rel=.006)
    assert printed(nwqlib.load_result(result.save(tmp_path/"result"))) == printed(result)
    laurent = nwqlib.solve(problem, method=LCHS(ham_evo=QSPBlockEncoding(route="laurent")),
                           output=nwqlib.NormSquared(), seed=3)
    assert "Phase synthesis" not in str(laurent)
