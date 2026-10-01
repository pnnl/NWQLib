"""Independent physical references and selected-grid/refinement continuations."""

from dataclasses import replace
from math import sqrt
from types import SimpleNamespace

import numpy as np
import pytest

import nwqlib
from nwqlib.algorithms.lchs.method import LCHS
from nwqlib.algorithms.lchs.provider_config import ProviderConfig
from nwqlib.algorithms.lchs.verification import LCHSVerification
from nwqlib.algorithms.lchs.refinement import LCHSRefinement
from nwqlib.core.records import Float64


def method(**changes):
    return LCHS(k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
        parameters={"num_qubits":2, "lsb_position":0}), **changes)


def arguments(receipt):
    return {entry.parameter:entry.value.value if isinstance(entry.value, Float64) else entry.value
            for entry in receipt.applications[0].arguments}


def raw_values(receipt):
    return {fact.fact.quantity:None if fact.fact.value is None else fact.fact.value.value
            for fact in receipt.applications[0].facts}


def patch_exponential(patcher, replacement):
    """Replace both SciPy kernels between which the LCHS references choose."""
    import scipy.linalg
    import scipy.sparse.linalg
    patcher.setattr(scipy.linalg, "expm", replacement)
    patcher.setattr(scipy.sparse.linalg, "expm", replacement)


def test_complex_source_reference_uses_original_dimension_phase_and_one_propagator(monkeypatch, tmp_path):
    import scipy.linalg
    from uuid import UUID
    from nwqlib.core.analysis import RunData
    from nwqlib.evidence.error_model import Certificate
    from nwqlib.execution import VerificationReceipt
    diagonal = np.array([.2+.1j, .3-.1j, .4+.05j])
    initial = .5*np.exp(1j*np.pi/3)*np.array([1., 1j, -.5])
    source = .25j*np.array([1., -.5j, 2.])
    problem = nwqlib.LinearDynamics(A=np.diag(diagonal), initial_state=initial, source=source, time=.2)
    result = nwqlib.solve(problem, method=method(duhamel_nodes=2), execution="classical")
    import scipy.sparse.linalg
    calls = []
    def recorded(kernel):
        def call(matrix):
            calls.append(matrix.copy())
            return kernel(matrix)
        return call
    # The augmented matrix of a diagonal A is upper triangular, so the
    # reference uses scipy.sparse.linalg.expm.
    monkeypatch.setattr(scipy.linalg, "expm", lambda *a, **k: pytest.fail("triangular input reached scipy.linalg.expm"))
    monkeypatch.setattr(scipy.sparse.linalg, "expm", recorded(scipy.sparse.linalg.expm))
    read = []
    lookup = RunData.artifact
    monkeypatch.setattr(RunData, "artifact",
        lambda data, manifest: read.append(getattr(manifest, "content_id", manifest)) or lookup(data, manifest))
    checks = LCHSVerification(reference="closed_form")
    receipt, facts = result.verify(checks=checks)
    monkeypatch.setattr(RunData, "artifact", lookup)
    expected = np.exp(-.2*diagonal)*initial + (-np.expm1(-.2*diagonal)/diagonal)*source
    assert result.solution.shape == expected.shape == (3,)
    # One augmented exponential of [[-A*T, b*T], [0, 0]] on the original
    # three coordinates (A_pad would be singular).
    assert len(calls) == 1 and calls[0].shape == (4,4)
    np.testing.assert_array_equal(calls[0][:3,:3], -.2*np.diag(diagonal))
    np.testing.assert_array_equal(calls[0][:3,3], .2*source)
    np.testing.assert_array_equal(calls[0][3], np.zeros(4))
    discrepancy = np.linalg.norm(result.solution-expected)
    assert facts[0].fact.value.value == pytest.approx(discrepancy, rel=2e-13, abs=2e-15)
    # Raw facts: the same discrepancy and the analytic reference norm. The
    # binary64 augmented expm at ||A||*T < .1 is far inside 2e-13 relative error.
    raw = raw_values(receipt)
    assert raw.keys() == {"reference_error", "reference_error.reference_norm"}
    assert raw["reference_error"] == facts[0].fact.value.value
    assert raw["reference_error.reference_norm"] == pytest.approx(np.linalg.norm(expected), rel=2e-13, abs=0)
    assert all(entry.fact.evidence.status == "declared" for entry in receipt.applications[0].facts)
    counts = arguments(receipt)
    assert counts["expm_completed"] == 1 and counts["matvec_completed"] == 1
    assert (receipt.plan_id, receipt.result_id, receipt.construction_id, receipt.options_id) == (
        result.plan.content_id, result.content_id, result.plan.construction.content_id, checks.content_id)
    assert receipt.artifact_ids == tuple(dict.fromkeys(read)) == (result.artifact.content_id,)
    assert UUID(receipt.invocation_id).version == 4
    evidence = facts[0].fact.evidence
    assert (evidence.status, evidence.artifact_kind, evidence.artifact, evidence.subject_id, evidence.witnessed_scope,
            evidence.options_id) == ("witnessed", "verification_receipt", receipt.content_id, result.content_id,
                                     facts[0].frame.scope, checks.content_id)
    from nwqlib.evidence.error_model import assemble_check
    assessed = assemble_check(checks.verification_checks(result)[0], artifact_id=result.content_id, fact=facts[0])
    assert assessed.status == "INCONCLUSIVE"  # No threshold was requested.
    # A Certificate is the consumer projection of witnessed facts; receipts
    # have no text display and persist only as their model JSON.
    certificate = Certificate(plan_id=result.plan_id, result_id=result.content_id,
        assessment=result.assess(absolute_tolerance=3.), checks=()).with_verification(result, options=checks, evidence=facts)
    check, = certificate.checks
    assert (check.check_id, check.artifact_id, check.fact) == (
        checks.verification_checks(result)[0].content_id, result.content_id, facts[0])
    # The witnessed closed-form value answers only its complete options. A new
    # threshold or another reference needs an explicit new verification, and
    # the rejected attachment itself starts no invocation or reference.
    base = Certificate(plan_id=result.plan_id, result_id=result.content_id,
        assessment=result.assess(absolute_tolerance=3.), checks=())
    rethreshold = LCHSVerification(reference="closed_form", threshold=1.)
    other = LCHSVerification(reference="ivp", rtol=1e-8, atol=1e-10, max_rhs_calls=10000)
    with monkeypatch.context() as local:
        local.setattr("nwqlib.execution.uuid4", lambda: pytest.fail("attachment created a verification invocation"))
        patch_exponential(local, lambda *a, **k: pytest.fail("attachment replayed a reference"))
        for selected in (rethreshold, other):
            with pytest.raises(ValueError, match="options identity"):
                base.with_verification(result, options=selected, evidence=facts)
    _, refreshed = result.verify(checks=rethreshold)  # The user's explicit new closed-form evaluation.
    assert refreshed[0].fact.evidence.options_id == rethreshold.content_id != checks.content_id
    assert base.with_verification(result, options=rethreshold, evidence=refreshed).checks[0].status == "PASS"
    path = result.save(tmp_path/"result")
    with monkeypatch.context() as local:
        local.setattr("nwqlib.execution.uuid4", lambda: pytest.fail("load created a verification invocation"))
        patch_exponential(local, lambda *a, **k: pytest.fail("load replayed a reference"))
        local.setattr(LCHS, "verify", lambda *a, **k: pytest.fail("load repeated verification"))
        loaded = nwqlib.load_result(path)
        for kind, record in ((VerificationReceipt, receipt), (Certificate, certificate)):
            restored = kind.model_validate_json(record.model_dump_json())
            assert restored == record and restored.content_id == record.content_id
    again, repeated = loaded.verify(checks=checks)
    # A second call on the saved Result keeps every association and value;
    # only the invocation, and hence the receipt it witnesses, is new.
    assert UUID(again.invocation_id).version == 4 and again.invocation_id != receipt.invocation_id
    assert (again.plan_id, again.result_id, again.construction_id, again.options_id, again.artifact_ids, again.applications) == (
        receipt.plan_id, receipt.result_id, receipt.construction_id, receipt.options_id, receipt.artifact_ids, receipt.applications)
    assert repeated[0].fact.value == facts[0].fact.value and repeated[0].fact.evidence.artifact == again.content_id != receipt.content_id


def test_reference_failure_and_missing_physical_access_do_not_start_another_operation(monkeypatch):
    problem = nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1,1j], time=.1)
    physical = nwqlib.solve(problem, method=method(), execution="classical")
    scalar = nwqlib.solve(problem, method=method(), output=nwqlib.NormSquared(), execution="classical")
    failure = RuntimeError("reference expm failed")
    calls = []
    def fail(*args):
        calls.append(True)
        raise failure
    patch_exponential(monkeypatch, fail)
    checks = LCHSVerification(reference="expm")
    with pytest.raises(ValueError, match="phase-faithful"):
        scalar.verify(checks=checks)
    with pytest.raises(ValueError, match="max_dense_work"):
        physical.verify(checks=checks.revise(max_dense_work=1))
    with pytest.raises(ValueError, match="bytes"):
        physical.verify(checks=checks.revise(max_bytes=16))
    with pytest.raises(ValueError, match="original configured Method"):
        method(trotter_steps=3).verify(physical.plan, physical, checks=checks)
    assert not calls
    with pytest.raises(RuntimeError) as caught:
        physical.verify(checks=checks)
    assert caught.value is failure and calls == [True]


def test_zero_relative_reference_preserves_absolute_discrepancy():
    problem = nwqlib.LinearDynamics(A=np.eye(2), initial_state=[0,0], time=.1)
    result = nwqlib.solve(problem, method=method())
    receipt, facts = result.verify(checks=LCHSVerification(reference="expm", metric="relative_l2"))
    assert facts[0].fact.availability == "unknown" and "nonzero" in facts[0].fact.reason
    raw = raw_values(receipt)
    assert raw["reference_error.absolute_l2"] == raw["reference_error.reference_norm"] == 0


@pytest.mark.parametrize("metric, threshold, expected, status", [
    ("absolute_l2", 2.6, sqrt(7.), "FAIL"), ("relative_l2", 2., sqrt(3.5), "PASS")])
def test_signed_discrepancy_threshold_is_a_scoped_check(monkeypatch, metric, threshold, expected, status):
    from nwqlib.algorithms.lchs import host
    from nwqlib.evidence.error_model import assemble_check
    monkeypatch.setattr(host, "_evolve", lambda *a: (np.array([1j,2j]), None, ()))
    result = nwqlib.solve(nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1,1], time=.5),
                          method=method(), execution="classical")
    patch_exponential(monkeypatch, lambda matrix: np.eye(2, dtype=complex))
    checks = LCHSVerification(reference="expm", metric=metric, threshold=threshold)
    _, facts = result.verify(checks=checks)
    # Physical target(i,2i), reference(1,1): squared discrepancy2+5=7,
    # reference squared norm2. Aligning phase would give another answer.
    assert facts[0].fact.value.value == pytest.approx(expected, rel=0, abs=2e-15)
    assessment = assemble_check(checks.verification_checks(result)[0], artifact_id=result.content_id, fact=facts[0])
    assert assessment.status == status
    from nwqlib.evidence.error_model import Certificate
    patch_exponential(monkeypatch, lambda *a, **k: pytest.fail("certificate replayed a reference"))
    certificate = Certificate(plan_id=result.plan_id, result_id=result.content_id,
        assessment=result.assess(absolute_tolerance=3.), checks=())
    for _ in range(2):
        certificate = certificate.with_verification(result, options=checks, evidence=facts)
        assert certificate.checks[0].status == status and len(certificate.checks)==1
        assert certificate.assessment.status == "INCONCLUSIVE"


def test_unavailable_exponential_leaves_the_reference_unknown_without_fallback(monkeypatch):
    # For A = diag(-800, 1) and T = 1 the exponential holds exp(800), above the
    # binary64 range, although u0 and b have no component on that mode. A
    # 1-norm above 2**37 is refused before any kernel runs.
    import warnings
    import scipy.integrate
    from nwqlib.algorithms.lchs.references import closed_form_reference, expm_reference
    checks = SimpleNamespace(max_bytes=10**6, max_dense_work=10**6)
    counts = dict.fromkeys(("expm_attempts", "expm_completed", "matvec_attempts", "matvec_completed"), 0)
    initial, source = np.array([0, 1j]), np.array([0, 1+0j])
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        overflow = np.diag([-800., 1.]).astype(complex)
        assert closed_form_reference(overflow, initial, source, 1., checks=checks, counts=counts) is None
        assert expm_reference(overflow, initial, 1., checks=checks, counts=counts) is None
    assert counts["expm_completed"] == 2 and counts["matvec_attempts"] == 0
    with monkeypatch.context() as local:
        patch_exponential(local, lambda *a, **k: pytest.fail("kernel ran above the 1-norm limit"))
        assert closed_form_reference(np.diag([2.**38, 1.]).astype(complex), initial, source, 1.,
                                     checks=checks, counts=counts) is None
    result = nwqlib.solve(nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1,0], source=[1,0], time=.1),
        method=method(duhamel_nodes=1), execution="classical")
    patch_exponential(monkeypatch, lambda matrix: np.full(matrix.shape, np.inf, dtype=complex))
    monkeypatch.setattr(scipy.integrate, "solve_ivp", lambda *a, **k: pytest.fail("unselected IVP fallback"))
    receipt, facts = result.verify(checks=LCHSVerification(reference="closed_form"))
    assert facts[0].fact.availability == "unknown" and "not finite" in facts[0].fact.reason
    assert raw_values(receipt) == {"reference_error": None, "reference_error.reference_norm": None}


def test_explicit_ivp_signed_rhs_consistency_and_rhs_cap(monkeypatch):
    import scipy.integrate
    result = nwqlib.solve(nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1,1], source=[1j,-1], time=.5),
        method=method(duhamel_nodes=1), execution="classical")
    calls = []
    def ivp(rhs, span, packed, **kwargs):
        assert span == (0.,.5)
        assert kwargs == dict(method="RK45", rtol=1e-10, atol=1e-12, t_eval=(.5,), dense_output=False)
        np.testing.assert_array_equal(rhs(0., np.array([1.,2.,3.,4.])), [-1.,-3.,-2.,-4.])
        calls.append("rhs")
        return SimpleNamespace(success=True, y=np.array([[.5],[0.],[.5],[0.]]), nfev=1)
    monkeypatch.setattr(scipy.integrate, "solve_ivp", ivp)
    # A fake augmented exponential: propagator .5*I and source response
    # (.5j, -.5) in its last column.
    def augmented(matrix):
        assert matrix.shape == (3, 3)
        return np.array([[.5, 0, .5j], [0, .5, -.5], [0, 0, 1]], dtype=complex)
    patch_exponential(monkeypatch, augmented)
    checks = LCHSVerification(reference="ivp_closed_form", rtol=1e-10, atol=1e-12, max_rhs_calls=2)
    receipt, facts = result.verify(checks=checks)
    assert facts[1].fact.value.value == 0  # Both references equal (.5+.5i,0).
    # The reference Source does not name the IVP tolerances, but the options identity does.
    from nwqlib.evidence.error_model import Certificate
    base = Certificate(plan_id=result.plan_id, result_id=result.content_id,
        assessment=result.assess(absolute_tolerance=1.), checks=())
    with pytest.raises(ValueError, match="options identity"):
        base.with_verification(result, options=checks.revise(rtol=1e-8), evidence=facts)
    assert {fact.fact.evidence.options_id for fact in facts} == {checks.content_id} == {receipt.options_id}
    assert arguments(receipt)["rhs_attempts"] == arguments(receipt)["rhs_completed"] == 1
    assert calls == ["rhs"]
    def twice(rhs, span, packed, **kwargs):
        rhs(0.,packed)
        calls.append("first completed")
        rhs(.1,packed)
        pytest.fail("unadmitted RHS returned")
    monkeypatch.setattr(scipy.integrate, "solve_ivp", twice)
    with pytest.raises(ValueError, match="RHS evaluation cap"):
        result.verify(checks=LCHSVerification(reference="ivp", rtol=1e-10, atol=1e-12, max_rhs_calls=1))
    assert calls[-1] == "first completed"
    with pytest.raises(ValueError, match="float64 floor"):
        LCHSVerification(reference="ivp", rtol=np.finfo(float).eps, atol=1e-12, max_rhs_calls=1)


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_selected_grid_preserves_padded_source_phase_and_archive_without_selection(tmp_path, monkeypatch, execution):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    problem = nwqlib.LinearDynamics(A=np.diag([-.1+.2j,.1-.2j,.3+.1j]),
        initial_state=.5*np.exp(1j*np.pi/3)*np.array([1,1j,.5]), source=[.25j,0,.1], time=.1)
    chosen = nwqlib.plan(problem, method=method(duhamel_nodes=2), execution=execution)
    if execution == "quantum":
        assert sum(register.width for register in chosen.construction.program.registers) <= 13
    result = nwqlib.solve(chosen)
    saved = result.save(tmp_path/execution)
    monkeypatch.setattr(terms, "generate_lchs_quadrature", lambda *a, **k: pytest.fail("quadrature reselected"))
    monkeypatch.setattr(terms, "generate_lchs_product_formula_select_plan", lambda *a, **k: pytest.fail("PF reselected"))
    restored = nwqlib.load_result(saved)
    receipt, facts = restored.verify(checks=LCHSVerification(reference="selected_grid"))
    # Same finite selected sum; dense native lowering has a looser roundoff envelope.
    assert facts[0].fact.value.value < (2e-10 if execution == "quantum" else 3e-14)
    counts = arguments(receipt)
    # One Hermitian eigensystem per k node serves all three applications.
    assert counts["applications"] == 3 and counts["expm_completed"] == 0 and counts["eigh_calls"] == 4
    assert counts["dimension"] == 3 and counts["reference_dimension"] == 4
    assert counts["application_0_source"] == 0 and counts["application_1_source"] == counts["application_2_source"] == 1
    monkeypatch.setattr(terms, "_node_eigensystem", lambda *a, **k: pytest.fail("unadmitted node reached a reference"))
    with pytest.raises(ValueError, match="max_node_evaluations"):
        restored.verify(checks=LCHSVerification(reference="selected_grid", max_node_evaluations=1))


def test_selected_grid_rejects_foreign_payload_and_corrupt_actual_source_role(monkeypatch):
    from nwqlib.algorithms.lchs.selected_grid import selected_payload
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    problem = nwqlib.LinearDynamics(A=np.eye(2), initial_state=[0,0], source=[1,1j], time=.1)
    result = nwqlib.solve(problem, method=method(duhamel_nodes=1), execution="classical")
    app = result.applications[0]
    from nwqlib.ir import Binding
    wrong = app.revise(arguments=tuple(Binding(parameter=entry.parameter, value=0) if entry.parameter=="source_application" else entry
                                      for entry in app.arguments))
    # A forged Result application cannot substitute for the completed chunk.
    changed = result.revise(applications=(wrong,))._attach(result.plan, result.data)
    monkeypatch.setattr(terms, "_node_eigensystem", lambda *a, **k: pytest.fail("bad schedule reached reference"))
    with pytest.raises(ValueError, match="actual acquisition"):
        changed.verify(checks=LCHSVerification(reference="selected_grid"))
    chosen = nwqlib.plan(problem, method=method(duhamel_nodes=1))
    payload = chosen._native["native_data"]
    chosen._native["native_data"] = replace(payload)
    with pytest.raises(ValueError, match="exact SELECT"):
        selected_payload(chosen)


# A budgeted Plan's classical reference repeats each acquired node step
# count as a fixed-step formula (selected_grid.grid_reference).
@pytest.mark.parametrize("execution,backend", [("classical", "trotter"), ("quantum", "trotter"),
                                               ("classical", "trotter_error_budgeted")])
def test_fixed_pf_selected_grid_never_reselects_steps_and_keeps_identity_phase(monkeypatch, execution, backend):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    matrix = np.array([[.5,.25j],[.25j,.2]])
    fixed = dict(trotter_steps=2) if backend == "trotter" else {}
    result = nwqlib.solve(nwqlib.LinearDynamics(A=matrix, initial_state=[0,0], source=[1,1j], time=.3),
        method=method(hamiltonian_evolution_backend=backend, duhamel_nodes=2, **fixed), execution=execution)
    if execution == "quantum":
        assert sum(register.width for register in result.plan.construction.program.registers) <= 13
    monkeypatch.setattr(terms, "generate_lchs_product_formula_select_plan", lambda *a, **k: pytest.fail("PF steps reselected"))
    monkeypatch.setattr(terms, "_budgeted_trotter_node_record", lambda *a, **k: pytest.fail("budgeted PF selection replayed"))
    from qiskit.circuit.library import PauliEvolutionGate
    monkeypatch.setattr(PauliEvolutionGate, "to_matrix", lambda *a, **k: pytest.fail("PF reference became an exact exponential"))
    receipt, facts = result.verify(checks=LCHSVerification(reference="selected_grid"))
    assert facts[0].fact.value.value < (2e-10 if execution == "quantum" else 1e-13)
    counts = arguments(receipt)
    assert counts["applications"] == 2 and counts["fixed_pf_completed"] == 8
    assert counts["application_0_source"] == counts["application_1_source"] == 1
    if backend == "trotter":
        assert all(value == 2 for name, value in counts.items() if name.endswith("_steps"))
    # The reference never decomposes L and H: the classical one reads the
    # stored node table, the quantum one evolves the selected branch circuits.
    assert counts["pauli_decomposition_attempts"] == counts["pauli_decomposition_completed"] == 0
    assert (counts["stored_node_read_work"] > 0) == (execution == "classical")


def test_refinement_propagates_all_physical_components_once_without_vector(monkeypatch):
    from nwqlib.algorithms.lchs import host, inhomogeneous_theory as theory
    from nwqlib.algorithms.lchs.analysis import fact
    from nwqlib.execution import KernelApplication
    from nwqlib.ir import Binding

    corrupt = False
    def supplied(plan, data):
        specs = [(False,0.,1.)]+[(True,node,weight) for node,weight in
            zip(plan.reconstruction.source_nodes,plan.reconstruction.source_weights,strict=True)]
        apps = tuple(KernelApplication(name=f"application_{index}", implementation=plan.construction.kernels[0].implementation,
            arguments=(Binding(parameter="start_time",value=Float64(value=start)),
                Binding(parameter="elapsed_time",value=Float64(value=plan.problem.elapsed_time-start)),
                Binding(parameter="weight",value=Float64(value=weight)),
                Binding(parameter="source_application",value=int(source) if not corrupt else 1),
                Binding(parameter="recovery_scale",value=Float64(value=1.))),
            facts=tuple(fact(name, nwqlib.Solution().frame(plan.problem), .125)
                        for name in ("kernel_approximation", "k_quadrature"))) for index,(source,start,weight) in enumerate(specs))
        return np.array([1.,2.,2.,0.], dtype=complex), None, apps
    monkeypatch.setattr(host, "_evolve", supplied)
    from nwqlib.core.records import Unit
    problem = nwqlib.LinearDynamics(A=np.eye(3), initial_state=[1,0,0], source=[1j,0,0], time=.5,
        unit=Unit(symbol="amplitude", dimension="custom"), time_unit=Unit(symbol="s", dimension="time"))
    result = nwqlib.solve(problem, method=method(duhamel_nodes=1), output=nwqlib.NormSquared(), execution="classical")
    assert result.artifact is None and result.value == 9
    calls = []
    def norm(matrix):
        calls.append(matrix.shape)
        return 1. if matrix.shape == (3,3) else 0.
    def duhamel(**kwargs):
        calls.append("duhamel")
        assert kwargs["matrix_norm"] == 1. and kwargs["node_count"] == 1
        assert kwargs["lambda_min_before_psd_conversion"] == 0.  # Actual A⊕0 PSD minimum.
        return theory._DuhamelBoundEvaluation(value=.25, unusable_nonzero=False)
    monkeypatch.setattr(theory, "_scaling_safe_spectral_norm", norm)
    monkeypatch.setattr(theory, "_duhamel_quadrature_error_bound", duhamel)
    checks = LCHSRefinement(components=("spectral_norms", "duhamel"))
    with pytest.raises(ValueError, match="max_dense_work"):
        result.verify(checks=checks.revise(max_dense_work=1))
    assert not calls
    receipt, facts = result.verify(checks=checks)
    # Two applications: 2*(1/8+1/8)+1/4=3/4 physical L2 error.
    # The NormSquared bridge uses radius3: (3/4)*(6+3/4)=81/16.
    assert {entry.fact.quantity:entry.fact.value.value for entry in facts} == {
        "duhamel_quadrature":.25, "algorithmic_approximation":81/16}
    assert calls == [(3,3),(4,4),"duhamel"]
    assert receipt.artifact_ids == ()
    assert {entry.fact.evidence.options_id for entry in facts} == {checks.content_id} == {receipt.options_id}
    assert next(entry.frame.unit.symbol for entry in receipt.applications[0].facts
                if entry.fact.quantity=="refinement.matrix_norm") == "1/(s)"
    assessment = result.assess(absolute_tolerance=6., facts=facts)
    assert assessment.status == "INCONCLUSIVE"  # Native rounding was not acquired.
    assert calls == [(3,3),(4,4),"duhamel"]
    corrupt = True
    wrong = nwqlib.solve(result.plan)
    with pytest.raises(ValueError, match="recorded physical schedule"):
        wrong.verify(checks=checks)
    assert calls == [(3,3),(4,4),"duhamel"]


def test_fixed_pf_structural_and_dense_bounds_follow_independent_pauli_commutators(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.subroutines.trotterization import error_budget
    decomposition = terms._TrotterPauliDecomposition(num_qubits=1, identity_label="I",
        l_coefficients={"Z":1.}, h_coefficients={"X":1.})
    nodes = terms._selected_node_table(decomposition, terms._distinct_combined_nodes(decomposition, [1.]), [1.])
    fields = dict(nodes=nodes, applications=((.5, (2,)),), method=LCHS(hamiltonian_evolution_backend="trotter"))
    (records,) = terms._fixed_trotter_certificate_records(**fields, dense_validation=True)
    # ||[Z,[Z,X]]||=||[X,[X,Z]]||=4, so coefficient=4/12+4/24=1/2.
    # Two steps give (1/2)*(.5)^3/(2)^2 = 1/64.
    assert records[0].combined_bound_value == pytest.approx(1/64, rel=0, abs=2e-16)
    assert records[0].dense_bound_value == pytest.approx(1/64, rel=0, abs=2e-16)
    monkeypatch.setattr(error_budget, "_bound_coefficient_evaluation", lambda *a, **k: pytest.fail("unadmitted commutator"))
    with pytest.raises(ValueError, match="max_structural_work"):
        terms._fixed_trotter_certificate_records(**fields, max_structural_work=1)
    with pytest.raises(ValueError, match="positive step count"):
        terms._fixed_trotter_certificate_records(**{**fields, "applications":((.5, (0,)),)})
    # The borrowed table is held in this phase: one byte below its payload
    # refuses before the read, and one unit below W_read = sum_k (p_k + 2)
    # refuses before any census work.
    with pytest.raises(ValueError, match="stored node table"):
        terms._fixed_trotter_certificate_records(**fields, max_bytes=terms._stored_node_table_bytes(nodes)-1)
    read = terms.stored_node_read_work(len(node["union_indices"]) for node in nodes.nodes)
    assert read == 2 + 2
    with pytest.raises(ValueError, match="before reading its node table"):
        terms._fixed_trotter_certificate_records(**fields, max_structural_work=read-1)


def test_fixed_pf_dense_commutators_admit_the_singular_values_of_their_norms(monkeypatch):
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.subroutines.trotterization import error_budget
    q, d, p = 6, 64, 2
    decomposition = terms._TrotterPauliDecomposition(num_qubits=q, identity_label="I"*q,
        l_coefficients={"Z"+"I"*(q-1):1.}, h_coefficients={"X"+"I"*(q-1):1.})
    nodes = terms._selected_node_table(decomposition, terms._distinct_combined_nodes(decomposition, [1.]), [1.])
    fields = dict(nodes=nodes, applications=((.5, (2,)),),
                  method=LCHS(hamiltonian_evolution_backend="trotter"), dense_validation=True)
    # Second order, for each of the p = 2 terms: 6 products (d**3 each), two
    # spectral norms by the SVD without vectors (ceil(4 d**3/3) +
    # 32 d (d + 1) = 482,646 units each) and 24 d**2 entrywise, 5,272,920
    # units in total.
    law = 5_272_920
    counts = {}
    (records,) = terms._fixed_trotter_certificate_records(**fields, max_dense_work=law, counts=counts)
    assert counts["dense_work"] == law and counts["dense_spectral_norms"] == 2*p
    # The identity factors leave the one-qubit commutator norms of the
    # direct test above unchanged.
    assert records[0].dense_bound_value == pytest.approx(1/64, rel=0, abs=2e-16)
    monkeypatch.setattr(error_budget, "dense_trotter_bound_coefficient",
                        lambda *a, **k: pytest.fail("unadmitted dense commutators"))
    with pytest.raises(ValueError, match="max_dense_work before dense commutators"):
        terms._fixed_trotter_certificate_records(**fields, max_dense_work=law-1)
    # The products and the bidiagonal reductions of the norms alone make
    # 6 p d**3 + 2 p ceil(4 d**3/3) = 4,543,832 multiply-adds. A charge of
    # one product per norm, 8 p d**3 + 24 p d**2 = 4,390,912, would admit this cap.
    with pytest.raises(ValueError, match="max_dense_work before dense commutators"):
        terms._fixed_trotter_certificate_records(**fields, max_dense_work=8*p*d**3+24*p*d*d)


@pytest.mark.parametrize("execution", ["classical", "quantum"])
def test_fixed_pf_refinement_keeps_source_only_applications_and_physical_weight(execution):
    # The quantum source layout names physical branches; the stored node
    # table still covers the quadrature grid that refinement evaluates.
    matrix = np.array([[.7,.3j],[.3j,.3]])
    result = nwqlib.solve(nwqlib.LinearDynamics(A=matrix, initial_state=[0,0], source=[1,1j], time=.3),
        method=method(hamiltonian_evolution_backend="trotter", trotter_steps=2, duhamel_nodes=2), execution=execution)
    receipt, facts = result.verify(checks=LCHSRefinement(components=("fixed_pf",), dense_validation=True))
    quad = result.plan._native["native_data"].quadrature
    # H=.3X, L=.5I+.2Z. The identity is exact; the two Pauli commutators
    # give C_k=.004*k²+.003*|k|. Each source term has |w|*sqrt2*T_app³/r².
    subtotal = sum(abs(c)*(.004*float(k)**2+.003*abs(float(k))) for k,c in zip(quad.k_nodes,quad.coefficients,strict=True))
    expected = sum(abs(weight)*sqrt(2)*(.3-start)**3*subtotal/4
        for start,weight in zip(result.plan.reconstruction.source_nodes,result.plan.reconstruction.source_weights,strict=True))
    value = next(entry.fact.value.value for entry in facts if entry.fact.quantity=="trotter_synthesis")
    assert value == pytest.approx(expected, rel=3e-14, abs=1e-18)
    counts = arguments(receipt)
    assert counts["applications"] == 2 and counts["application_0_source"] == counts["application_1_source"] == 1
    # The dense validation coefficient is independent of elapsed time, so each
    # of the four distinct k-nodes is evaluated once and both source
    # applications reuse it.
    assert len(set(map(float, quad.k_nodes))) == 4
    assert counts["pauli_decomposition_completed"] == 0 and counts["dense_bound_completed"] == 4
    assert next(entry for entry in facts if entry.fact.quantity=="algorithmic_approximation").fact.availability == "unknown"


def test_zero_source_refinement_keeps_exact_zero_without_spectral_reference(monkeypatch):
    from nwqlib.algorithms.lchs import inhomogeneous_theory as theory
    result = nwqlib.solve(nwqlib.LinearDynamics(A=np.eye(2), initial_state=[1,0], source=[0,0], time=.2),
        method=method(), execution="classical")
    monkeypatch.setattr(theory, "_scaling_safe_spectral_norm", lambda *a: pytest.fail("zero source acquired a norm"))
    monkeypatch.setattr(theory, "_duhamel_quadrature_error_bound", lambda **k: pytest.fail("zero source acquired a remainder"))
    receipt, facts = result.verify(checks=LCHSRefinement(components=("duhamel",)))
    assert next(entry.fact.value.value for entry in facts if entry.fact.quantity=="duhamel_quadrature") == 0
    assert all(value==0 for name,value in arguments(receipt).items() if name.endswith("_attempts"))


def test_periodic_refinement_keeps_existing_scalar_bound_and_rejects_dense_reference(monkeypatch):
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.algorithms.lchs import inhomogeneous_theory as theory
    problem = nwqlib.LinearDynamics(A=PeriodicStencil(2,.2,.1,.05),
        initial_state=ingest_occupation("00",num_qubits=2), time=.1)
    result = nwqlib.solve(problem, method=method(hamiltonian_evolution_backend="trotter", trotter_steps=2),
        output=nwqlib.NormSquared())
    monkeypatch.setattr(theory, "_scaling_safe_spectral_norm", lambda *a: pytest.fail("periodic input was densified"))
    receipt, facts = result.verify(checks=LCHSRefinement(components=("fixed_pf",)))
    original = next(entry.fact.value.value for entry in result.plan.facts if entry.fact.quantity=="trotter_synthesis")
    assert next(entry.fact.value.value for entry in facts if entry.fact.quantity=="trotter_synthesis") == original
    assert arguments(receipt)["dense_work"] == arguments(receipt)["structural_work"] == 0
    with pytest.raises(ValueError, match="no implicit densification"):
        result.verify(checks=LCHSRefinement(components=("spectral_norms",)))
    with pytest.raises(ValueError, match="no implicit densification"):
        LCHSVerification(reference="selected_grid").validate_domain(problem)


@pytest.mark.parametrize("scale", (1., 1e-6, 1e-10, 1e-13))
def test_closed_form_source_response_keeps_its_digits_for_small_norm_times_time(scale):
    # For symmetric A = V diag(lam) V^T the source response is
    # V diag(-expm1(-lam*T)/lam) V^T b, accurate to rounding for every lam*T.
    # Forming I - expm(-A*T) cancels about log10(1/(||A||*T)) digits, a
    # relative error of 1e-3 at ||A||*T = 6e-14.
    from nwqlib.algorithms.lchs.references import closed_form_reference
    base = np.array([[.6, .1], [.1, .3]])
    lam, vectors = np.linalg.eigh(scale*base)
    source = np.array([.2, -.1], dtype=complex)
    initial = np.array([.5, 1j])
    exact = (vectors @ np.diag(np.exp(-lam)) @ vectors.T) @ initial + (
        vectors @ np.diag(-np.expm1(-lam)/lam) @ vectors.T) @ source
    counts = dict.fromkeys(("expm_attempts", "expm_completed", "matvec_attempts", "matvec_completed"), 0)
    checks = SimpleNamespace(max_bytes=10**6, max_dense_work=10**6)
    reference = closed_form_reference((scale*base).astype(complex), initial, source, 1., checks=checks, counts=counts)
    # About 90 unit roundoffs (1e-14) of the expm, relative to the solution norm.
    assert np.linalg.norm(reference-exact) <= 1e-14*np.linalg.norm(exact)


@pytest.mark.parametrize("rotated", [False, True])
def test_closed_form_is_accurate_for_close_diagonal_entries_and_singular_a(rotated):
    # A = V diag(0, 40, 1e-12) V^H with T = 1. For diagonal A (V = I) the
    # augmented matrix is upper triangular, and SciPy 1.18.1's
    # scipy.linalg.expm recomputes its superdiagonal entry between -1e-12 and
    # 0 as a cancelling divided difference, a relative error of 1.6e-5 here.
    # The expected value is componentwise in the eigenbasis:
    # exp(-a T) w0 + (-expm1(-a T)/a) wb, with T wb at a = 0. The error model
    # of the exponential is kappa_exp * u, about ||M|| u = 4.4e-15, and the
    # tolerance leaves a factor of about 20 above it.
    from nwqlib.algorithms.lchs.references import closed_form_reference
    eigenvalues, elapsed = np.array([0., 40., 1e-12]), 1.
    initial, source = np.array([.5, 1j, -.25+.5j]), np.array([.2, -.1+.3j, .7])
    basis = np.linalg.qr(np.array([[1., 2, 0], [0, 1, 3], [2, 0, 1]]))[0] if rotated else np.eye(3)
    response = np.array([elapsed if a == 0 else -np.expm1(-a*elapsed)/a for a in eigenvalues])
    exact = basis @ (np.exp(-eigenvalues*elapsed)*(basis.T @ initial) + response*(basis.T @ source))
    counts = dict.fromkeys(("expm_attempts", "expm_completed", "matvec_attempts", "matvec_completed"), 0)
    reference = closed_form_reference((basis @ np.diag(eigenvalues) @ basis.T).astype(complex), initial, source,
        elapsed, checks=SimpleNamespace(max_bytes=10**6, max_dense_work=10**6), counts=counts)
    assert np.linalg.norm(reference-exact) <= 1e-13*np.linalg.norm(exact)


def test_exponential_work_is_refused_before_the_kernel(monkeypatch):
    # For M = [[-diag(0, 40, 1e-12), b], [0, 0]], eta is about 40, so
    # Algorithm 5.1 of Al-Mohy and Higham, doi:10.1137/09074721X, takes the
    # degree-13 branch with
    # ceil(log2(40/4.25)) = 4 squarings. Its five powers, three Pade products
    # and 4 squarings alone are 12 * 4**3 = 768 units. That branch skips the
    # backward-error tests of lower orders, and the test of order 13
    # multiplies |M| with a vector 27 times, 27 * 4**2 = 432 units. The law
    # allows 79 such products, the tests of all orders. A cap of 1000 must
    # refuse the kernel before it runs.
    #
    # With A = 2**20 I of order 63, M is 64-square and ||M**k||_1 = 2**(20 k),
    # so every norm estimate of the algorithm is 2**20 and it takes at least
    # ceil(log2(2**20/theta_13)) = 18 squarings, theta_13 = 5.37. With the
    # powers M**2, M**4 and M**6 and three Pade products that makes at least
    # 24 products, 24 * 64**3 = 6,291,456 units, so a cap there must refuse
    # the kernel. A law without the squarings would charge
    # 10 * 64**3 + 128 * 64**2 = 3,145,728 units and admit it.
    from nwqlib.algorithms.lchs.references import closed_form_reference

    def counts():
        return dict.fromkeys(("expm_attempts", "expm_completed", "matvec_attempts", "matvec_completed"), 0)

    large = (2.0**20*np.eye(63, dtype=complex), np.ones(63, complex), np.ones(63, complex), 1.)
    with monkeypatch.context() as guarded:
        patch_exponential(guarded, lambda *a, **k: pytest.fail("kernel ran beyond max_dense_work"))
        for problem, cap in (((np.diag([0., 40., 1e-12]).astype(complex), np.ones(3, complex), np.ones(3, complex), 1.),
                              1000), (large, 24*64**3)):
            refused = counts()
            with pytest.raises(ValueError, match="max_dense_work"):
                closed_form_reference(*problem, checks=SimpleNamespace(max_bytes=10**8, max_dense_work=cap),
                                      counts=refused)
            assert refused["expm_attempts"] == 0
    # exp(-A) vanishes in binary64, and the Duhamel term is A^{-1} b = b/2**20.
    admitted = counts()
    reference = closed_form_reference(*large, checks=SimpleNamespace(max_bytes=10**8, max_dense_work=10**7),
                                      counts=admitted)
    assert admitted["expm_completed"] == 1
    np.testing.assert_allclose(reference, np.full(63, 2.0**-20), rtol=1e-12, atol=0)


def test_selected_grid_admits_its_node_exponentials_before_the_first(monkeypatch):
    # The dense reference applies each node evolution to the application
    # vectors from one Hermitian eigensystem of k L + H per node
    # (time_independent_terms.spectral_lchs_sum). A max_dense_work of 8 d**3
    # per eigensystem lies below the admitted law and must be refused before
    # the first eigensystem. The default cap admits the reference.
    from nwqlib.algorithms.lchs import time_independent_terms as terms
    l_part = np.array([[.4, .1, 0, 0], [.1, .3, .1, 0], [0, .1, .35, .05], [0, 0, .05, .3]])
    h_part = np.diag([.1, .1, .1], 1) + np.diag([.1, .1, .1], -1)
    problem = nwqlib.LinearDynamics(A=l_part + 1j*h_part, initial_state=[1., 0., 0., 0.], time=.1)
    result = nwqlib.solve(problem, method=method(), execution="classical")
    receipt, _ = result.verify(checks=LCHSVerification(reference="selected_grid"))
    calls = arguments(receipt)["eigh_calls"]
    # One Hermitian eigensystem for each of the four k nodes, no exponential.
    assert calls == 4 and arguments(receipt)["expm_completed"] == 0
    monkeypatch.setattr(terms, "_node_eigensystem", lambda *a, **k: pytest.fail("a node eigensystem ran beyond max_dense_work"))
    # The eigensystems alone take 8 D**3 units each (time_independent_terms.
    # _spectral_host_requirements), so a cap of exactly that lies below the
    # admitted law, which also prices the generator passes and actions.
    with pytest.raises(ValueError, match="max_dense_work"):
        result.verify(checks=LCHSVerification(reference="selected_grid", max_dense_work=8*4**3*calls))


def test_triangular_reference_exponential_keeps_the_global_random_state():
    # From order 200 scipy.sparse.linalg.expm, which a triangular reference
    # matrix reaches, estimates norms with onenormest, which draws from
    # NumPy's global generator. The reference must leave the caller's state
    # unchanged and give the same vector bitwise for different caller states.
    from nwqlib.algorithms.lchs.references import expm_reference
    generator = np.random.default_rng(5)
    d = 200
    matrix = np.triu(generator.normal(size=(d, d)) + 1j*generator.normal(size=(d, d))) / d + np.eye(d)
    initial = generator.normal(size=d) + 1j*generator.normal(size=d)
    checks = SimpleNamespace(max_bytes=10**9, max_dense_work=10**9)
    def reference():
        counts = dict.fromkeys(("expm_attempts", "expm_completed", "matvec_attempts", "matvec_completed"), 0)
        return expm_reference(matrix, initial, 1., checks=checks, counts=counts)
    np.random.seed(1)
    before = np.random.get_state()
    first = reference()
    after = np.random.get_state()
    assert before[0] == after[0] and np.array_equal(before[1], after[1]) and before[2:] == after[2:]
    np.random.seed(2)
    assert np.array_equal(first, reference())


def test_refinement_spectral_norm_admits_its_singular_values():
    # The spectral norm of a nonzero 64-square matrix comes from LAPACK zgesdd.
    # It reduces the matrix to bidiagonal form, ceil(4 * 64**3 / 3) = 349,526
    # complex multiply-adds, and the dqds algorithm then computes the singular
    # values of the bidiagonal matrix. For a random matrix of this order dqds
    # makes about 2.4 * 64**2 divisions, each with five further operations,
    # near 59,000 units. A max_dense_work of 380,000 therefore lies below the
    # actual work and must be refused before the decomposition. A charge of
    # the reduction and the 4 * 64**2 entrywise scan alone, 365,910, would
    # admit it. The default cap admits the norm.
    from nwqlib.algorithms.lchs.refinement import _spectral_norm
    generator = np.random.default_rng(3)
    matrix = generator.normal(size=(64, 64)) + 1j*generator.normal(size=(64, 64))
    def counts():
        return dict(matrix_norm_attempts=0, matrix_norm_completed=0, spectral_svd_attempts=0,
                    spectral_svd_completed=0)
    refused = counts()
    with pytest.raises(ValueError, match="max_dense_work"):
        _spectral_norm(matrix, SimpleNamespace(max_bytes=10**9, max_dense_work=380_000), refused)
    assert refused["spectral_svd_attempts"] == 0
    value = _spectral_norm(matrix, SimpleNamespace(max_bytes=10**9, max_dense_work=100_000_000), counts())
    assert value == pytest.approx(np.linalg.norm(matrix, 2), rel=1e-12)


def test_selected_pf_branch_vector_evolution_matches_the_operator_product():
    """This phase-sensitive regression compares the selected branch's vector
    evolution with the full Operator product on the same once-decomposed
    circuit. The named fixtures use at most four system qubits and 5000
    elementary gates, with input norm at most one up to normalization rounding.
    The componentwise tolerance is 2e-12 with zero relative tolerance. It is
    an empirical test allowance, not a product-formula or backend error bound.
    The conditional represented-matrix bound (selected_pf_comparison_tolerance)
    is asserted as well on the same outputs.
    """
    from qiskit.quantum_info import Operator, SparsePauliOp, Statevector
    from _pf_branch_comparison import (assert_represented_vector_distance, regression_atol,
                                       selected_pf_comparison_tolerance)
    from nwqlib.algorithms.lchs.select_synthesis import _build_product_formula_branch
    from collections import defaultdict
    from nwqlib.algorithms.lchs.selected_grid import grid_reference, selected_grid
    from nwqlib.algorithms.lchs.solution_error_budget import psd_recovery_part
    q, d = 3, 8
    L = SparsePauliOp.from_list([("III", .4), ("ZII", .1)]).to_matrix()
    H = SparsePauliOp.from_list([("III", .13), ("IXY", .21), ("XZI", -.17), ("YYZ", .11), ("ZZI", .09)]).to_matrix()
    v = np.arange(1, d+1, dtype=complex)+1j*np.arange(d, 0, -1)
    v /= np.linalg.norm(v)
    method = LCHS(k_quadrature=ProviderConfig(implementation="signed_binary_uniform",
                                              parameters={"num_qubits": 2, "lsb_position": 0}),
                  hamiltonian_evolution_backend="trotter", trotter_order=2, trotter_steps=3, duhamel_nodes=2)
    plan = nwqlib.plan(nwqlib.LinearDynamics(A=L+1j*H, initial_state=v, source=np.roll(v, 1)*(.3+.2j), time=.2),
                       method=method, execution="quantum")
    checks = LCHSVerification(reference="selected_grid")
    grid = selected_grid(plan, None, checks=checks)
    selected = grid.payload.select_data.plan
    summed = np.zeros(d, dtype=complex)
    branches = 0
    for app in grid.applications:
        if not app.weight or not app.input_norm:
            continue
        for node, coefficient in enumerate(grid.coefficients):
            if coefficient == 0:
                continue
            circuit = _build_product_formula_branch(selected, app.slots[node], num_system_qubits=q)
            circuit.global_phase += selected.identity_phases[app.slots[node]]
            decomposed = circuit.decompose()
            limits = selected_pf_comparison_tolerance(decomposed, app.vector)
            actual = Statevector(app.vector).evolve(decomposed).data
            expected = Operator(decomposed).data @ app.vector
            assert_represented_vector_distance(actual, expected, limits["atol_2"])
            np.testing.assert_allclose(actual, expected, atol=regression_atol(decomposed, app.vector), rtol=0)
            summed += (app.weight*psd_recovery_part(grid.shift, app.elapsed, 0)*coefficient)*expected
            branches += 1
    assert branches == 12
    # The reference assembles the same saved branches, phases and weights.
    counts = defaultdict(int)
    assert_represented_vector_distance(grid_reference(plan, grid, checks=checks, counts=counts), summed, 2e-12)
    assert counts["fixed_pf_completed"] == branches
