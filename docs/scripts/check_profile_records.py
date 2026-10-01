#!/usr/bin/env python3
"""Fresh-process assessment/interchange import-attempt and no-work witness."""

import argparse

import builtins
import importlib.abc
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]


def calibrated_inputs():
    """Plan one actual selected HZH preparation and a synthetic calibrated profile.

    The declarations and coefficients are synthetic arithmetic, not measured
    calibration. Imports stay inside this function so the audit guard sees them.
    """
    from datetime import datetime, timezone
    from nwqlib import Expectation, plan as select_plan
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AER_STATEVECTOR_TARGET, InstructionSupport
    from nwqlib.backends.profiles import (
        Allocation, CalibrationReference, DeviceConfiguration, DeviceProfile, ModelDomain,
        ModelUncertainty, TimeCoefficient, TimeModel,
    )
    from nwqlib.core import Limit, Source, Unit
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib.evidence import Evidence
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems.inputs import ingest_occupation
    from nwqlib.resources import ResourceContext

    source = Source(name="synthetic stored-profile audit", version="1",
        domain="arithmetic fixtures only; not observed hardware calibration",
        reference="docs/scripts/check_profile_records.py")
    recorded = datetime(2026, 1, 1, tzinfo=timezone.utc)
    expires = datetime(2026, 2, 1, tzinfo=timezone.utc)
    problem = Expectation(state=ingest_occupation("1", num_qubits=1),
                          observable=ingest_pauli((("Z", 1.0),), num_qubits=1))
    plan = select_plan(problem, method=ExpectationMethod(preparation_choice="hzh"), seed=7)
    # Declare only the H/Z operations and readout that the supplied profile covers.
    target = AER_STATEVECTOR_TARGET.revise(
        description="supplied one-qubit HZH/Pauli subset", artifacts=("selected_construction",),
        readouts=("pauli_expectation",), max_qubits=1,
        instructions=tuple(InstructionSupport(primitive=gate, max_qubits=1) for gate in ("h", "z")),
        program_nodes=("sequence", "allocate", "block_call"))
    configuration = DeviceConfiguration(name="synthetic single-device audit", version="1", target=target,
        hardware=source, runtime=Source(name=target.name, version="audit-runtime-1", domain=target.description,
            reference="synthetic runtime declaration; no installed package queried"),
        build=source, compiler=source, precision="complex128", representation="statevector")
    specification = Evidence(kind="external_specification", source=source)
    allocation = Allocation(name="audit grant", configuration_id=configuration.content_id,
        locations=("logical_device",), topology="single_device",
        limits=(Limit(stage="execution", metric="memory", unit=Unit(symbol="byte", dimension="bytes"),
                      kind="capacity_stock", value=1024, scope="logical_device"),),
        recorded_at=recorded, valid_until=expires, evidence=specification)
    context = ResourceContext()
    domain = ModelDomain(configuration_id=configuration.content_id, allocation_id=allocation.content_id,
        basis=context.basis, batch_schedule=context.batch_schedule, runtime="seed_independent",
        acquisition="direct_observation", population="unconditional", readouts=("pauli_expectation",),
        selection_ids=(), primitive_gates=("h", "z"), min_qubits=1, max_qubits=1, min_operations=3,
        max_operations=3, max_shots=0, max_exact_evaluations=1, max_readout_items=1, max_resets=0,
        max_measurements=0, max_classical_work=0, max_adaptive_rounds=0)
    # Hand arithmetic: one invocation * 1 s + one exact group * 2 s
    # + three logical operations * 0.5 s = 4.5 s, interval [4.25, 5.0].
    model = TimeModel(name="audit supplied calibration", kind="calibrated", scope="selected_acquisition",
        coefficients=tuple(TimeCoefficient(feature=feature, seconds_per_unit=value, unit=unit)
            for feature, value, unit in zip(("invocations", "exact_evaluations", "logical_operations"),
                (1.0, 2.0, 0.5), ("s/invocation", "s/exact_evaluation", "s/logical_operation"), strict=True)),
        domain=domain, recorded_at=recorded, valid_until=expires,
        evidence=Evidence(kind="empirical_prediction", source=source),
        assumptions=("synthetic one-qubit HZH reduced-Pauli acquisition coefficients; no hardware certification",
                     "fixed runtime/build/precision and single declared acquisition; seed-independent relation"),
        calibration=CalibrationReference(source=source, training=source, validation=source,
                                         validation_errors=source),
        uncertainty=ModelUncertainty(kind="future_run_prediction", lower_residual_seconds=-0.25,
                                     upper_residual_seconds=0.5, coverage=0.9, source=source))
    profile = DeviceProfile(configuration=configuration, recorded_at=recorded, valid_until=expires,
                            evidence=specification, models=(model,))
    return (plan, plan.resolve("expectation"), profile, allocation, context, RuntimeOptions(seed=7),
            datetime(2026, 1, 2, tzinfo=timezone.utc))


def child(poison=False):
    forbidden_prefixes = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_ionq", "pytket", "qulacs",
        "nwqlib._prepared_execution", "nwqlib.backends.qiskit_aer",
        "nwqlib.backends.ibm_runtime", "nwqlib.backends.ionq",
        "nwqlib.backends.nexus", "nwqlib.backends.nwqsim", "nwqlib.backends.nwqec",
    )
    attempts = []
    def forbidden(name):
        return any(name == prefix or name.startswith(prefix + ".") for prefix in forbidden_prefixes)
    def record(name):
        if forbidden(name):
            attempts.append(name)
            raise ImportError("blocked SDK/provider import: " + name)
    original = builtins.__import__
    def guarded(name, globals=None, locals=None, fromlist=(), level=0):
        record(name)
        for item in fromlist or ():
            record(name + "." + item)
        return original(name, globals, locals, fromlist, level)
    class Blocker(importlib.abc.MetaPathFinder):
        def find_spec(self, fullname, path=None, target=None):
            record(fullname)
            return None
    builtins.__import__ = guarded
    sys.meta_path.insert(0, Blocker())
    if poison:
        try:
            __import__("qiskit")
        except ImportError:
            pass
    from nwqlib.backends import ProfileAssessment, assess
    from nwqlib.backends import targets
    from nwqlib.backends.capabilities import BackendTarget
    from nwqlib.backends.profiles import Allocation, DeviceProfile
    # Every builtin stored declaration is inspected without loading its adapter.
    for target in vars(targets).values():
        if isinstance(target, BackendTarget):
            assert BackendTarget.model_validate_json(target.model_dump_json()) == target
    plan, realization, profile, allocation, context, runtime, at = calibrated_inputs()
    from nwqlib import compare, estimate
    from nwqlib.algorithms.expectation import ExpectationMethod
    comparison = compare(plan.problem, methods=(ExpectationMethod(), ExpectationMethod(preparation_choice="hzh")), seed=7)
    assert [row.estimate.quantity("operations").fact.value.numerator for row in comparison.rows] == [1, 3]
    events = []
    def blocked(*args, **kwargs):
        events.append("extra input/native/network work")
        raise AssertionError(events[-1])
    import socket
    import numpy as np
    import nwqlib.blocks as blocks
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators.inputs import OperatorInput
    # Actual planner/input admission above is allowed once. The assessment and
    # interchange below must not repeat it or acquire missing machine evidence.
    ExpectationMethod.plan = blocked
    OperatorInput.pauli_terms = blocked
    blocks.lower_qiskit = blocked
    socket.socket = blocked
    np.empty = np.zeros = np.array = blocked
    public = estimate(plan, profile=profile, allocation=allocation)
    assert public.plan_id == plan.content_id and public.allocation == allocation
    for stored in (profile, allocation):
        assert type(stored).model_validate_json(stored.model_dump_json()) == stored
        type(stored).model_json_schema()
    result = assess(plan, realization, profile=profile, allocation=allocation, context=context,
                    runtime=runtime, assessed_at=at)
    restored = ProfileAssessment.model_validate_json(result.model_dump_json())
    restored.validate_context(plan, profile, allocation, runtime)
    assert restored.predictions[0].seconds.value == 4.5
    assert restored.error is None and restored.accuracy.status == "unknown"
    assert restored.applicability.status == "unknown"
    ProfileAssessment.model_json_schema()
    DeviceProfile.model_json_schema()
    Allocation.model_json_schema()
    from nwqlib.backends import PredictionLedger, align_telemetry
    from nwqlib.execution import ConsumptionEvent, ExecutionTrace, TimingObservation, SubmissionItem, SubmissionRecord
    import nwqlib.backends.assessment as assessment_owner
    assessment_owner.assess = assessment_owner._select_workload = blocked
    timing = TimingObservation(scope="native_call_wall", seconds=2., censoring="right_censored",
        reason="supplied continuing target at a two-second cutoff; no native call", source=profile.models[0].evidence.source)
    event = ConsumptionEvent(attempt="offline", prepared_id=plan.content_id, status="uncertain",
        submission="offline", shots=0, evaluations=1, started="supplied fixture", assessment_id=restored.content_id, timing=timing)
    trace = ExecutionTrace(run_id="offline", plan_id=plan.content_id,
        preparations=0, construction_work_reserved=0, events=(event,),
        submissions=(SubmissionRecord(submission_id="offline", run_id="offline", backend=timing.source,
            items=(SubmissionItem(attempt="offline", prepared_id=plan.content_id, item=0),),
            status="uncertain", started="supplied fixture"),))
    ledger = align_telemetry(trace=trace, assessments=(restored,))
    loaded_ledger = PredictionLedger.model_validate_json(ledger.model_dump_json())
    loaded_ledger.validate_context(trace=trace, assessments=(restored,))
    assert loaded_ledger.rows[0].timings == (timing,)
    assert loaded_ledger.rows[0].comparisons[0].residual_seconds is None
    PredictionLedger.model_json_schema()
    loaded = [name for name in sys.modules if forbidden(name)]
    if attempts or loaded or events:
        raise AssertionError(f"assessment isolation violation: attempts={attempts}, loaded={loaded}, events={events}")
    print("Actual profile assessment and telemetry/interchange: zero SDK/provider imports or extra input/native/network work.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--child", action="store_true", help="run the isolated audit directly")
    parser.add_argument("--poison", action="store_true", help="inject a forbidden import; requires --child")
    args = parser.parse_args(argv)
    if args.poison and not args.child:
        parser.error("--poison requires --child")
    if args.child:
        child(args.poison)
        return
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    command = [sys.executable, str(Path(__file__).resolve()), "--child"]
    clean = subprocess.run(command, cwd=ROOT, env=env, text=True, capture_output=True)
    if clean.returncode:
        raise RuntimeError(clean.stdout + clean.stderr)
    print(clean.stdout, end="")
    poisoned = subprocess.run(command + ["--poison"], cwd=ROOT, env=env, text=True, capture_output=True)
    if poisoned.returncode == 0 or "attempts=['qiskit']" not in poisoned.stderr:
        raise AssertionError("swallowed-import negative control failed:\n" + poisoned.stderr)
    print("Swallowed SDK import negative control rejected as required.")


if __name__ == "__main__":
    main()
