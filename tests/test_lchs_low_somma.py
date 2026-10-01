"""Actual selected Low–Somma (arXiv:2508.19238v2) PF body and explicit incomplete component bounds.

Independent coefficient/grid formulas live in test_lchs_providers. This file
uses the SELECT built during public preparation, not a second selection.
"""

import numpy as np
import pytest
import scipy.linalg
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from nwqlib import LinearDynamics, plan, prepare, solve
from nwqlib.algorithms.lchs import LCHS
from nwqlib.algorithms.lchs.providers import LOW_SOMMA_PROFILE_ID, LOW_SOMMA_UNBUDGETED_STAGES


def problem():
    return LinearDynamics(A=[[0.25, 0.04j], [0.04j, 0.4]], initial_state=[1.0, 0.2], time=0.1)


def method(**changes):
    return LCHS(
        approximation_tolerance=0.8,
        lchs_kernel="low_somma_f2",
        k_quadrature="symmetric_uniform_trapezoid",
        hamiltonian_evolution_backend="trotter",
        trotter_steps=1,
        **changes,
    )


def test_low_somma_actual_multiplexor_keeps_phase_step_scaling_and_identity_padding(monkeypatch):
    from nwqlib.algorithms.lchs import select_synthesis

    selected = plan(problem(), method=method(), seed=7)
    data = selected._native["native_data"]
    pf = data.select_data.plan
    coefficients = data.coefficient_plan
    assert pf.physical_node_count == 15 and pf.padded_node_count == 16
    assert pf.branch_to_node == tuple(range(15)) + (None,)
    assert pf.branch_step_counts == (1,) * 15 + (0,)
    assert all(table[-1] == 0.0 for table in pf.occurrence_angle_tables.array)
    assert pf.structure_certificate is None and data.selected_select == "multiplexor"
    amplitudes = coefficients.prep_amplitudes()
    assert len(amplitudes) == 16 and amplitudes[-1] == 0.0
    assert np.vdot(amplitudes, amplitudes).real == pytest.approx(1.0, rel=0, abs=2e-15)
    captured = []
    build = select_synthesis.build_product_formula_select

    def actual(body, **kwargs):
        assert body is pf
        circuit = build(body, **kwargs)
        captured.append(circuit)
        return circuit

    monkeypatch.setattr(select_synthesis, "build_product_formula_select", actual)
    monkeypatch.setattr(
        np.polynomial.legendre,
        "leggauss",
        lambda *a, **k: pytest.fail("native preparation reselected grid"),
    )
    monkeypatch.setattr(
        PauliEvolutionGate,
        "to_matrix",
        lambda *a, **k: pytest.fail("ideal exponential bypassed assigned synthesis"),
    )
    prepared = prepare(selected)
    try:
        assert len(captured) == 1 and captured[0].num_qubits == 5
        assert prepared.run.trace.preparations == 1 and prepared.run.trace.events == ()
        realized = np.asarray(Operator(captured[0]).data)
    finally:
        prepared.run.close()
    x = np.array([[0.0, 1.0], [1.0, 0.0]])
    z = np.diag([1.0, -1.0])
    half_x = scipy.linalg.expm(-0.002j * x)
    expected_operator = np.zeros((32, 32), dtype=complex)
    # A=L+iH gives L=.325I-.075Z and H=.04X. One selected Suzuki2
    # step is X/2,Z,X/2. Low–Somma's coefficient phase is exp(-ik*c), c=1.
    for address, k in enumerate(coefficients.nodes):
        expected = (
            np.exp(-1j * k * (1 + 0.0325)) * half_x @ scipy.linalg.expm(0.0075j * k * z) @ half_x
        )
        indices = [address, address + 16]  # Address is least significant.
        expected_operator[np.ix_(indices, indices)] = expected
    expected_operator[np.ix_([15, 31], [15, 31])] = np.eye(2)
    # The full operator also rejects leakage between distinct addresses.
    np.testing.assert_allclose(realized, expected_operator, rtol=0, atol=1e-11)
    # O(100) native rotations give binary64 accumulation; 1e-11 is a roundoff
    # allowance, not a PF approximation allowance. Padding has no phase/action.
    np.testing.assert_allclose(realized[np.ix_([15, 31], [15, 31])], np.eye(2), rtol=0, atol=1e-12)
    k = coefficients.nodes[-1]
    fixed = half_x @ scipy.linalg.expm(0.0075j * k * z) @ half_x
    ideal = scipy.linalg.expm(-0.1j * (0.04 * x - 0.075 * k * z))
    assert np.linalg.norm(fixed - ideal) > 1e-8  # Discriminates one PF step from ideal expm.
    with pytest.raises(ValueError, match="no structure certificate"):
        plan(problem(), method=method(lcu_select_implementation="structured"), seed=7)


def test_low_somma_profile_bounds_remain_separate_from_uncomputed_pf_error():
    selected = plan(problem(), method=method(), seed=7)
    data = selected._native["native_data"]
    record = data.coefficient_plan.record()
    assert record["kernel_parameter_selection"]["profile"] == LOW_SOMMA_PROFILE_ID
    assert record["unbudgeted_error_stages"] == list(LOW_SOMMA_UNBUDGETED_STAGES)
    for name in ("approximate_lchs_error_bound", "quadrature_error_bound"):
        assert record[name] == pytest.approx(0.8 / 3, rel=0, abs=1e-15)
    assert selected.reconstruction.kernel_approximation_bound == pytest.approx(0.8 / 3, rel=0, abs=1e-15)
    assert selected.reconstruction.quadrature_bound == pytest.approx(0.8 / 3, rel=0, abs=1e-15)
    stages = {item.fact.quantity: item.fact.value for item in selected.facts}
    assert "truncation" not in stages
    assert stages["kernel_approximation"].value == pytest.approx(
        (0.8 / 3)*np.hypot(1., .2), rel=2e-14, abs=0.)
    missing = {fact.fact.quantity for fact in selected.facts if fact.fact.value is None}
    assert "trotter_synthesis" in missing
    aggregate = next(
        term for term in selected.error_model.terms if term.name == "algorithmic_approximation"
    )
    assert aggregate.fact.fact.value is None
    result = solve(selected)
    assert result.references == "not_run"
    assessment = result.assess(absolute_tolerance=0.8)
    assert (
        assessment.status == "INCONCLUSIVE" and "algorithmic_approximation" in assessment.remaining
    )


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
    from nwqlib.algorithms.lchs import ProviderConfig

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
