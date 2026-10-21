"""Existing projected coordinates and exact count populations; no reference replay."""

from fractions import Fraction
from types import SimpleNamespace

import numpy as np
import pytest

import nwqlib
from nwqlib.algorithms.lanczos import Lanczos
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.evidence.energy_shift import EnergyShiftOptions
from nwqlib.evidence.sector import NumberSectorOptions
from nwqlib.evidence.verification import ProjectedVerificationOptions
from nwqlib.evidence.error_model import Certificate
from nwqlib.execution import VerificationReceipt
from nwqlib.operators import ingest_pauli


def ratio(fact):
    value = fact.fact.value
    return None if value is None else Fraction(value.numerator, value.denominator)


def forbidden(*args, **kwargs):
    raise AssertionError("stored verification replayed scientific work")


def certificate(result):
    return Certificate(
        plan_id=result.plan_id,
        result_id=result.content_id,
        assessment=result.assess(absolute_tolerance=0.2),
        checks=(),
    )


@pytest.mark.parametrize("family", ["lanczos", "gcim"])
def test_actual_scalar_energy_shift_and_saved_results(family, monkeypatch, tmp_path):
    """Scalar identity targets give exact energy shifts without eigensolves, while an unproved
    shifted relation stays conditional.
    """
    method = (
        Lanczos(initial_state=[1, 0], krylov_dimension=1)
        if family == "lanczos"
        else FixedGCIM(basis=([1, 0],))
    )
    plans = tuple(
        nwqlib.plan(
            nwqlib.Eigenproblem(A=ingest_pauli((("I", value),), num_qubits=1)),
            method=method,
            seed=7,
        )
        for value in (2.0, 4.0)
    )
    import scipy.linalg

    for owner in (np.linalg, scipy.linalg):
        for name in ("eigh", "eigvalsh", "svd"):
            monkeypatch.setattr(owner, name, forbidden)
    first, second = (nwqlib.solve(plan) for plan in plans)
    assert (first.eigenvalue, second.eigenvalue) == (2.0, 4.0)
    assert first.data.trace.events == second.data.trace.events == ()
    option = EnergyShiftOptions.for_result(first, name="shift", shift=2.0, tolerance=0.0)
    receipt, facts = second.verify(checks=option)
    assert ratio(facts[0]) == 0
    assert (
        option.baseline.plan_id == first.plan_id and option.baseline.result_id == first.content_id
    )
    assert option.baseline.operator_id != second.plan.problem.A.reference.identity
    again, repeated = second.verify(checks=option)
    assert receipt.invocation_id != again.invocation_id and receipt.content_id != again.content_id
    assert ratio(repeated[0]) == ratio(facts[0])
    projection = ProjectedVerificationOptions(
        name="projected", comparisons=("gram_psd_deficit",), tolerance=0.0
    )
    projected, unknown = second.verify(checks=projection)
    assert ratio(unknown[0]) is None  # Scalar identities did not run a projected solve.
    cert = certificate(second).with_verification(second, options=option, evidence=facts)
    cert = cert.with_verification(second, options=projection, evidence=unknown)
    assert tuple(check.status for check in cert.checks) == ("PASS", "INCONCLUSIVE")
    assert cert.assessment.status != "PASS"  # Ground identification remains separate.
    # A witnessed value answers only the complete options that produced it.
    # For the shift and baseline changes below, |4-2-1| = |4-3-2| = 1 and verify
    # rejects both selections, so the stored value 0 must not answer them.
    third = nwqlib.solve(nwqlib.plan(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 3.0),), num_qubits=1)), method=method, seed=7))
    other_baseline = EnergyShiftOptions.for_result(third, name="shift", shift=2.0, tolerance=0.0)
    with pytest.raises(ValueError, match="Pauli tables"):
        second.verify(checks=other_baseline)
    base = certificate(second)
    for changed in (option.revise(shift=1.0), other_baseline, option.revise(tolerance=1.0)):
        with pytest.raises(ValueError, match="options identity"):
            base.with_verification(second, options=changed, evidence=facts)
    evidence = facts[0].fact.evidence
    assert evidence.options_id == option.content_id == receipt.options_id
    for unbound in (dict(options_id=None), dict(artifact_kind=None)):
        with pytest.raises(ValueError, match="options identity"):
            evidence.revise(**unbound)
    from nwqlib.evidence import Evidence
    from nwqlib.evidence.error_model import assemble_check

    selected = option.verification_checks(second)[0]
    with pytest.raises(ValueError, match="options identity"):
        assemble_check(selected.revise(options_id=None), artifact_id=second.content_id, fact=facts[0])
    # The fact also names its exact CheckSpec, so the direct form rejects a criterion
    # revised after verification. The stored value 0 would pass the looser threshold.
    assert evidence.check_id == selected.content_id
    assert assemble_check(selected, artifact_id=second.content_id, fact=facts[0]).fact == facts[0]
    with pytest.raises(ValueError, match="answers another check"):
        assemble_check(selected.revise(threshold=1.0), artifact_id=second.content_id, fact=facts[0])
    with pytest.raises(ValueError, match="check identity belongs"):
        Evidence(kind="numerical_estimate", source=option.source, check_id=selected.content_id)
    path = second.save(tmp_path / family)
    with monkeypatch.context() as local:
        local.setattr(type(method), "plan", forbidden)
        local.setattr(type(method), "analyze", forbidden)
        local.setattr("nwqlib.execution.uuid4", forbidden)
        loaded = nwqlib.load_result(path)
        assert loaded.report() == second.report()
        assert Certificate.model_validate_json(cert.model_dump_json()) == cert
        for item in (receipt, projected):
            assert VerificationReceipt.model_validate_json(item.model_dump_json()) == item
        # Saved options and facts keep their identities, values and receipt
        # provenance. Attaching them to the loaded Result performs no work.
        restored = EnergyShiftOptions.model_validate_json(option.model_dump_json())
        saved = tuple(type(fact).model_validate_json(fact.model_dump_json()) for fact in facts)
        assert restored is not option and restored.content_id == option.content_id and saved == facts
        attached = base.with_verification(loaded, options=restored, evidence=saved)
        assert attached.checks == cert.checks[:1]
    _, saved_facts = loaded.verify(checks=option)
    assert ratio(saved_facts[0]) == 0
    with pytest.raises(ValueError, match="Pauli tables"):
        second.verify(checks=option.revise(shift=1.0))
    asserted = option.revise(shift=1.0, relation="asserted", assertion="external H2=H1+I premise")
    _, disagreement = second.verify(checks=asserted)
    assert ratio(disagreement[0]) == 1
    conditional = certificate(second).with_verification(
        second, options=asserted, evidence=disagreement
    )
    assert conditional.checks[0].status == "INCONCLUSIVE"
    changed_seed = nwqlib.solve(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 4.0),), num_qubits=1)),
        method=(
            Lanczos(initial_state=[0, 1], krylov_dimension=1)
            if family == "lanczos"
            else FixedGCIM(basis=([0, 1],))
        ),
    )
    with pytest.raises(ValueError, match="preparation/subspace"):
        changed_seed.verify(checks=option)


def test_observed_sector_leakage_is_not_mean_particle_number(monkeypatch, prepared_stubs, tmp_path):
    from nwqlib.algorithms.qhd import method as owner
    from nwqlib.backends import qiskit_aer as aer
    from test_qhd_workflow import make_plan

    plan = make_plan(shots=10)
    option = NumberSectorOptions(name="sector", particles=1, tolerance=0.0)
    from nwqlib.evidence.sector import verify_number_sector

    # Capability admission is shared by direct CheckSpec and wrapper consumers.
    # A partial producer cannot trigger its data accessor before rejection.
    unsupported = SimpleNamespace(number_sector_observations=forbidden)
    with pytest.raises(ValueError, match="complete computational-register counts"):
        option.verification_checks(unsupported)
    with pytest.raises(ValueError, match="complete computational-register counts"):
        verify_number_sector(unsupported, options=option)
    results = []
    for index, bins in enumerate(((("000", 5), ("011", 5)), (("001", 5), ("100", 5)))):
        # These distinct populations both have empirical mean N=1.
        assert sum(count * bits.count("1") for bits, count in bins) == 10
        monkeypatch.setattr(
            aer,
            "_submit_aer_execution",
            lambda *a, **k: SimpleNamespace(
                raw_output={"counts": dict(bins)}, metadata={"native_job_id": "supplied"}
            ),
        )
        result = nwqlib.solve(plan)
        results.append(result)
        with monkeypatch.context() as local:
            local.setattr(owner, "_evolve_restricted", forbidden)
            local.setattr(owner, "_host_summary", forbidden)
            receipt, facts = result.verify(checks=option)
        expected = 1 - index
        assert ratio(facts[0]) == expected
        cert = certificate(result).with_verification(result, options=option, evidence=facts)
        assert cert.checks[0].status == ("PASS" if expected == 0 else "FAIL")
        assert "empirical" in facts[0].frame.conditioning and receipt.artifact_ids == ()
        # The selected particle number is part of the frame conditioning, so the
        # frame check rejects this mismatch.
        with pytest.raises(ValueError, match="frame"):
            certificate(result).with_verification(
                result, options=option.revise(particles=2), evidence=facts)
        path = result.save(tmp_path / f"sector-{index}")
        with monkeypatch.context() as local:
            local.setattr(type(plan.method), "plan", forbidden)
            local.setattr(type(plan.method), "analyze", forbidden)
            loaded = nwqlib.load_result(path)
            assert loaded.content_id == result.content_id
        _, saved = loaded.verify(checks=option)
        assert ratio(saved[0]) == expected
    assert results[0].contribution_ids != results[1].contribution_ids
    from nwqlib.core.analysis import RunData

    foreign = RunData(results[1].data.observations, results[0].data.trace, results[0].data.receipts)
    with pytest.raises(ValueError, match="observations"):
        results[0].model_copy()._attach(plan, foreign)
    assert len(prepared_stubs.prepared) == len(prepared_stubs.lowered) == 2


def _qls_case():
    from nwqlib.algorithms.qls import QLS, QLSVerification

    result = nwqlib.solve(nwqlib.LinearSystem(A=[[1.1, 0.1], [0.1, 0.9]], b=[1.0, 0.25]),
                          method=QLS(), seed=7, execution="classical")
    return result, lambda tolerance: QLSVerification(
        comparisons=("inverse_relative_error",), relative_tolerance=tolerance)


def _qpe_case():
    from nwqlib.algorithms.qpe import QCELS, QPEVerification

    # The prepared state has weight .7 on the eigenvalue .7 and .3 on .2.
    result = nwqlib.solve(nwqlib.Eigenproblem(A=np.diag([0.2, 0.7])),
                          method=QCELS(initial_state=[np.sqrt(0.3), np.sqrt(0.7)]), seed=7)
    return result, lambda overlap: QPEVerification(tolerance=1.0, minimum_overlap=overlap)


def _qhd_case():
    import sympy as sp
    from nwqlib.algorithms.qhd import QHD, QHDVerification, QuadraticSchedule

    x = sp.Symbol("x")
    # The stored state comes from the IR product, so the Schrodinger reference
    # differs from it by a small positive infidelity.
    result = nwqlib.solve(
        nwqlib.Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),)),
        method=QHD(num_grid_points=3, num_steps=2, total_time=0.17, schedule=QuadraticSchedule(gamma=0.3),
                   theory_flavor="ir_product", keep_state=True),
        execution="classical", seed=7)
    return result, lambda tolerance: QHDVerification(
        comparisons=("schrodinger_fidelity",), schrodinger_infidelity_tolerance=tolerance)


@pytest.mark.parametrize("case,passing,failing", [
    (_qls_case, 0.01, 0.001),  # Relative L2 inverse error .0051.
    (_qpe_case, 0.6, 0.9),  # Largest prepared cluster weight .7.
    (_qhd_case, 1.0, 0.0),  # Nonzero cross-model infidelity.
])
def test_method_tolerances_give_check_verdicts(case, passing, failing):
    """QLS, QPE and QHD tolerances reach a PASS or FAIL through Certificate.with_verification."""
    result, options = case()
    statuses = []
    for tolerance in (passing, failing):
        selected = options(tolerance)
        _, facts = result.verify(checks=selected)
        attached = certificate(result).with_verification(result, options=selected, evidence=facts)
        assert [check.check_id for check in attached.checks] == [
            spec.content_id for spec in selected.verification_checks(result)]
        statuses.append(attached.checks[-1].status)
    assert statuses == ["PASS", "FAIL"]
