"""Actual selected Low–Somma (arXiv:2508.19238v2) PF body and its component bounds on each route.

Independent coefficient/grid formulas live in test_lchs_providers. This file
uses the SELECT built during public preparation, not a second selection.
"""

from math import fsum, sqrt

import numpy as np
import pytest
import scipy.linalg

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS, ProviderConfig


def test_low_somma_coefficient_rounding_is_refused_at_planning():
    """A shift c whose coefficient 1-norm defeats binary64 is refused, a moderate one kept.

    At c = 50 the Low-Somma coefficients have alpha about 1.2e16, so the
    binary64 finite sum carries rounding of order u*alpha, about 1.3 of the
    input norm, far above the published 2*epsilon/3 kernel and quadrature
    components. Near the allowance 0.1*epsilon = 1e-3, u*alpha is about
    3.4e-3 at c = 42 and 7.7e-4 at c = 40, which pins the threshold within a
    factor of 4.4. At c = 30, u*alpha is about 5e-7 and the classical sum
    still agrees with expm within the published bound.
    """
    target = LinearDynamics(A=[[.4, .15], [.05, .25]], initial_state=[1., 0.], time=.1)

    def shifted(c):
        return LCHS(lchs_kernel=ProviderConfig(implementation="low_somma_f2", parameters={"c": c}),
                    k_quadrature=ProviderConfig(implementation="symmetric_uniform_trapezoid"))

    with pytest.raises(ValueError, match=r"u\*alpha = 1\.\d+ .* above 0\.1\*approximation_tolerance = 0\.001"):
        plan(target, method=shifted(50.), execution="classical")
    with pytest.raises(ValueError, match=r"u\*alpha = 0\.003\d* .* above 0\.1\*approximation_tolerance"):
        plan(target, method=shifted(42.), execution="classical")
    edge = plan(target, method=shifted(40.), execution="classical").reconstruction.coefficient_l1_norm
    assert 5e-4 < 2.0**-53*edge <= 1e-3
    result = solve(target, method=shifted(30.), execution="classical")
    alpha = result.plan.reconstruction.coefficient_l1_norm
    assert 1e9 < alpha and 2.0**-53*alpha <= 1e-3
    exact = scipy.linalg.expm(-np.array([[.4, .15], [.05, .25]])*.1) @ np.array([1., 0.])
    bound = next(item.fact.value.value for item in result.facts if item.fact.quantity == "algorithmic_approximation")
    assert np.linalg.norm(np.asarray(result.solution) - exact) <= bound


_LS = dict(lchs_kernel=ProviderConfig(implementation="low_somma_f2"),
           k_quadrature=ProviderConfig(implementation="symmetric_uniform_trapezoid"))
_A = np.array([[.4, .15], [.05, .25]])
_TARGET = dict(A=_A, initial_state=[1., 0.], time=.1)
_EXACT = scipy.linalg.expm(-_A*.1) @ np.array([1., 0.])
_TOLERANCE = .01


def _facts(chosen):
    """Return each listed component as its value, or as its reason when it is unknown."""
    return {item.fact.quantity: (item.fact.value.value if item.fact.value is not None else item.fact.reason)
            for item in chosen.facts}


def _total(result):
    return next(item.fact for item in result.facts if item.fact.quantity == "algorithmic_approximation")


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_low_somma_dense_routes_report_the_same_ideal_total(execution):
    """Direct preparation adds exact ideal zeros, so a quantum dense route has the classical route's total.

    The input has unit norm and L is PSD, so the total is the kernel and quadrature caps, 2*epsilon/3 (Theorems 2-3 with the
    default split, `providers._resolve_low_somma_pair_profile`). The classical sum and the exact-readout circuit each lie
    within it of expm. Native floating-point error stays a separate unknown term.
    """
    chosen = plan(LinearDynamics(**_TARGET), method=LCHS(**_LS, hamiltonian_evolution_backend="dense_exact"), execution=execution)
    assert chosen.reconstruction.psd_shift == 0
    if execution == "quantum":
        assert _facts(chosen)["lcu_coefficient_preparation"] == _facts(chosen)["initial_state_preparation"] == 0.
    result = solve(chosen)
    total = _total(result)
    assert total.value.value == pytest.approx(2*_TOLERANCE/3, rel=1e-12, abs=0)
    assert np.linalg.norm(np.asarray(result.solution)-_EXACT) <= total.value.value
    native = next(term.fact.fact for term in chosen.error_model.terms if term.name == "native_floating_point")
    assert native.availability == "unknown"


def test_low_somma_budgeted_product_formula_reports_its_synthesis_bound():
    """The per-node budget gives s = sum_j |c_j|*e_j with e_j <= 0.1*epsilon, so s <= 0.1*epsilon*alpha, and the total holds.

    `method.py` gives each node the allowance 0.1*epsilon. The independent check of the finite sum is expm.
    """
    method = LCHS(**_LS, hamiltonian_evolution_backend="trotter_error_budgeted")
    chosen = plan(LinearDynamics(**_TARGET), method=method, execution="classical")
    synthesis = _facts(chosen)["trotter_synthesis"]
    assert 0 < synthesis <= .1*_TOLERANCE*chosen.reconstruction.coefficient_l1_norm
    assert _facts(plan(LinearDynamics(**_TARGET), method=method, execution="quantum"))["trotter_synthesis"] == synthesis
    result = solve(chosen)
    assert _total(result).value.value == pytest.approx(2*_TOLERANCE/3+synthesis, rel=1e-12, abs=0)
    assert np.linalg.norm(np.asarray(result.solution)-_EXACT) <= _total(result).value.value


def test_low_somma_compact_qsp_reports_its_stages_and_its_scalar_total():
    """A compact generator's coherent route lists its QSP bound and ideal PREP zeros, which give a NormSquared total.

    The reference is expm of the same generator, and the frame change is `propagate_physical_error`.
    """
    from nwqlib import NormSquared
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(A=ingest_pauli([("Z", .4j), ("X", .1)], num_qubits=1), initial_state=[.6, .8], time=1.)
    chosen = plan(problem, method=LCHS(**_LS, hamiltonian_evolution_backend="qsp_block_encoding", approximation_tolerance=.2),
                  output=NormSquared(), execution="quantum")
    facts = _facts(chosen)
    assert facts["lcu_coefficient_preparation"] == facts["initial_state_preparation"] == 0.
    assert facts["qsp_synthesis"] > 0
    delta = fsum(facts[name] for name in ("kernel_approximation", "k_quadrature", "qsp_synthesis"))
    generator = .4j*np.diag([1., -1.]) + .1*np.array([[0., 1.], [1., 0.]])
    exact = np.linalg.norm(scipy.linalg.expm(-generator) @ np.array([.6, .8]))**2
    result = solve(chosen)
    assert _total(result).value.value == pytest.approx(delta*(2*sqrt(result.value)+delta), rel=1e-12, abs=0)
    assert abs(result.value-exact) <= _total(result).value.value


def test_low_somma_keeps_a_stage_unknown_where_its_route_has_no_bound():
    """Fixed steps, MPS PREP and analytic PREP keep the gap and reason that the default pair's routes give."""
    from nwqlib.algorithms.lchs.refinement import LCHSRefinement
    from nwqlib.operators import ingest_pauli
    problem = LinearDynamics(**_TARGET)
    fixed = plan(problem, method=LCHS(**_LS, hamiltonian_evolution_backend="trotter", trotter_steps=2), execution="classical")
    assert _facts(fixed)["trotter_synthesis"] == "missing_stage:trotter_synthesis"
    assert _facts(fixed)["trotter_pruning"] == 0.
    unknown = solve(fixed)
    assert _total(unknown).availability == "unknown"
    # Explicit refinement supplies the bound that planning lacks.
    _, refined = unknown.verify(checks=LCHSRefinement(components=("fixed_pf",)))
    bound = next(item.fact.value.value for item in refined if item.fact.quantity == "trotter_synthesis")
    assert np.linalg.norm(np.asarray(unknown.solution)-_EXACT) <= 2*_TOLERANCE/3+bound
    mps = plan(problem, method=LCHS(**_LS, hamiltonian_evolution_backend="dense_exact", lcu_state_preparation="mps_circuit"),
               execution="quantum")
    assert _facts(mps)["lcu_coefficient_preparation"] == "missing_stage:lcu_coefficient_preparation"
    assert _facts(mps)["initial_state_preparation"] == 0.
    compact = LinearDynamics(A=ingest_pauli([("Z", .4j), ("X", .1)], num_qubits=1), initial_state=[.6, .8], time=1.)
    analytic = plan(compact, method=LCHS(**_LS, hamiltonian_evolution_backend="qsp_block_encoding", approximation_tolerance=.2,
                                         lcu_state_preparation="analytic"), execution="quantum")
    assert _facts(analytic)["lcu_coefficient_preparation"].startswith("analytic kernel PREP: known part 2 gamma delta_k")
    assert _facts(analytic)["initial_state_preparation"] == 0.


def test_low_somma_periodic_route_reports_its_strang_bound():
    """The periodic route supplies its weighted Strang bound under the backend label trotter, and direct PREP gives zeros."""
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    problem = LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=1., diffusion=.25, potential=.3),
                             initial_state=ingest_occupation((0, 0), num_qubits=2), time=.5)
    method = LCHS(**_LS, hamiltonian_evolution_backend="trotter", trotter_steps=None, trotter_synthesis_tolerance=1e-3)
    facts = _facts(plan(problem, method=method, execution="quantum"))
    assert 0 < facts["trotter_synthesis"] <= 1e-3
    assert facts["lcu_coefficient_preparation"] == facts["initial_state_preparation"] == 0.
