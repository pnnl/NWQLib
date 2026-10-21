"""Small independent metadata comparisons; no native preparation or simulation."""

from datetime import timedelta
import json

import pytest

from _profile_fixtures import AT, SYNTHETIC_SOURCE, stored_inputs
from nwqlib.backends import BackendCapability
from nwqlib.backends.assessment import (
    _assessment_inputs,
    _assess_capability, _assess_capacity, _assess_time, _select_workload,
)
from nwqlib.blocks import SelectedConstruction
from nwqlib.core import Limit, Unit
from nwqlib.evidence import Evidence, Fact, FramedFact
from nwqlib.ir import (
    Allocate, Binding, BlockCall, Definition, ExprRef, Expression, Parallel, Parameter, ParameterRef, PortMap, Program, Register,
    Repeat, Sequence,
)
from nwqlib.resources import ResourceContext, Workspace
from nwqlib.problems.records import Accuracy


def _workload(inputs):
    plan, realization, profile, allocation, context, runtime, at = inputs
    point, workload = _assessment_inputs(plan, realization, profile, allocation, context, runtime, at)
    return point, workload


def _memory_limit(value, location):
    return Limit(stage="execution", metric="memory", unit=Unit(symbol="byte", dimension="bytes"),
                 kind="capacity_stock", value=value, scope=location)


def _capacity_fixture(*, parallel=False, workspace_bytes=5 * 2**30, unknown=False):
    plan, _, profile, allocation, *_ = stored_inputs()
    # Two one-qubit calls to the same actual HZH recipe. Declared workspace is
    # simultaneous per invocation: 5 GiB serial reuse, 10 GiB parallel demand.
    selected = plan.construction.selections[0].revise(workspace=(
        Workspace(location="gpu0", purpose="workspace", bytes=workspace_bytes, source=SYNTHETIC_SOURCE),
    ) + ((Workspace(location="gpu0", purpose="workspace", bytes=None, source=SYNTHETIC_SOURCE),) if unknown else ()))
    definitions = (
        Definition(id="aa", node=Allocate(wire="a")), Definition(id="ab", node=Allocate(wire="b")),
        Definition(id="a", node=BlockCall(signature="state", ports=(PortMap(port="system", wire="a"),))),
        Definition(id="b", node=BlockCall(signature="state", ports=(PortMap(port="system", wire="b"),))),
        Definition(id="body", node=(Parallel if parallel else Sequence)(children=("a", "b"))),
        Definition(id="root", node=Sequence(children=("aa", "ab", "body"))),
    )
    construction = SelectedConstruction(program=Program(root="root", definitions=definitions,
        registers=(Register(name="a", width=1), Register(name="b", width=1)),
        signatures=(selected.signature,)), selections=(selected,))
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id))
    experiment = plan.experiments[0]
    plan = plan.revise(experiments=(experiment.revise(observation=experiment.observation.revise(labels=("ZZ",))),))
    profile = profile.revise(configuration=profile.configuration.revise(
        target=profile.configuration.target.revise(max_qubits=2)))
    allocation = allocation.revise(locations=("gpu0", "gpu1"), topology="independent_devices",
                                  configuration_id=profile.configuration.content_id,
                                  limits=tuple(_memory_limit(8 * 2**30, loc) for loc in ("gpu0", "gpu1")))
    return _select_workload(plan, plan.resolve("expectation"), ResourceContext()), profile, allocation


def test_calibration_number_interval_and_dependent_only_freshness_domain():
    inputs = stored_inputs(calibrated=True)
    point, workload = _workload(inputs)
    profile, allocation, at = inputs[2], inputs[3], inputs[6]
    _, predictions = _assess_time(workload, profile, allocation, at, runtime=inputs[5], point_id=point.content_id)
    prediction = predictions[0]
    # Independent scalar reference, no replay of the evaluator formula.
    assert (prediction.seconds.value, prediction.lower_seconds.value, prediction.upper_seconds.value) == (4.5, 4.25, 5.)
    assert prediction.uncertainty.kind == "future_run_prediction"
    assert prediction.uncertainty.coverage == 0.9
    mean_model = profile.models[0].revise(uncertainty=profile.models[0].uncertainty.revise(kind="sample_mean_confidence"))
    _, mean_predictions = _assess_time(workload, profile.revise(models=(mean_model,)), allocation, at, runtime=inputs[5], point_id=point.content_id)
    assert mean_predictions[0].uncertainty.kind == "sample_mean_confidence"
    capacity = _assess_capacity(workload, profile, allocation, at)
    capability = _assess_capability(workload, profile, at)
    for model in (profile.models[0].revise(valid_until=at - timedelta(hours=1)),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(max_operations=2, min_operations=0)),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(configuration_id="sha256:" + "0" * 64)),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(acquisition="measurement_batch")),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(runtime=inputs[5].revise(seed=8))),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(population="native_conditioned")),
                  profile.models[0].revise(domain=profile.models[0].domain.revise(bindings=(Binding(parameter="n", value=4),)))):
        changed = profile.revise(models=(model,))
        _, missing = _assess_time(workload, changed, allocation, at, runtime=inputs[5], point_id=point.content_id)
        assert missing[0].seconds is None and missing[0].reasons
        assert _assess_capacity(workload, changed, allocation, at) == capacity
        assert _assess_capability(workload, changed, at) == capability
    # A stale model cannot launder its old interval into a fresh prediction.
    assert missing[0].lower_seconds is None and missing[0].upper_seconds is None


def test_per_location_parallel_failure_and_serial_legal_peak():
    serial, profile, allocation = _capacity_fixture()
    parallel, _, _ = _capacity_fixture(parallel=True)
    serial_axis = _assess_capacity(serial, profile, allocation, AT)
    parallel_axis = _assess_capacity(parallel, profile, allocation, AT)
    serial_peak = next(item for item in serial_axis.details if item.quantity == "memory")
    parallel_peak = next(item for item in parallel_axis.details if item.quantity == "memory")
    assert serial_peak.fact.value.numerator == 5 * 2**30 and serial_peak.status == "feasible"
    assert parallel_peak.fact.value.numerator == 10 * 2**30 and parallel_peak.status == "infeasible"
    # Neither case pools gpu0 and gpu1 into a spurious 16-GiB device.
    assert serial_peak.limit.scope == parallel_peak.limit.scope == "gpu0"
    assert parallel_axis.status == "infeasible"


def test_large_state_body_rejects_by_bounded_integer_comparison():
    inputs = stored_inputs()
    plan, _, profile, allocation, *_ = inputs
    # Width metadata only: 100000 qubits. No vector/matrix/power-sized buffer is made.
    program = Program(root="allocate", definitions=(Definition(id="allocate", node=Allocate(wire="q")),),
                      registers=(Register(name="q", width=100_000),))
    construction = SelectedConstruction(program=program, selections=())
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id))
    experiment = plan.experiments[0]
    plan = plan.revise(experiments=(experiment.revise(observation=experiment.observation.revise(
        kind="probabilities", labels=(), qubits=(0,))),))
    workload = _select_workload(plan, plan.resolve("expectation"), ResourceContext())
    detail = next(d for d in _assess_capacity(workload, profile, allocation, AT).details if d.quantity == "native state body")
    assert detail.status == "infeasible" and detail.fact.availability == "unknown"
    assert "integer-output" in detail.fact.reason
    assert len(detail.model_dump_json()) < 10_000
    # A one-qubit body needs 32 bytes, while a density-matrix body needs 64 bytes.
    _, small = _workload(inputs)
    roomy = allocation.revise(limits=(_memory_limit(48, "logical_device"),))
    state = next(d for d in _assess_capacity(small, profile, roomy, AT).details if d.quantity == "native state body")
    density_profile = profile.revise(configuration=profile.configuration.revise(representation="density_matrix"))
    density_allocation = roomy.revise(configuration_id=density_profile.configuration.content_id)
    density = next(d for d in _assess_capacity(small, density_profile, density_allocation, AT).details if d.quantity == "native state body")
    assert state.status == "conditional" and state.fact.value.numerator == 32
    assert density.status == "infeasible" and density.fact.value.numerator == 64
    # Unknown extra ancillas cannot erase the admitted one-qubit system body.
    plan = inputs[0]
    selected = plan.construction.selections[0]
    construction = plan.construction.revise(selections=(selected.revise(semantics=selected.semantics.revise(workspace=None)),))
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id))
    extra = _select_workload(plan, plan.resolve("expectation"), ResourceContext())
    assert extra.estimate.quantity("logical_width", location="logical_device").fact.availability == "unknown"
    lower = next(d for d in _assess_capacity(extra, profile, allocation, AT).details if d.quantity == "native state body")
    assert lower.status == "infeasible" and lower.fact.value.numerator == 32


def test_coarse_capability_never_certifies_missing_instruction_or_readout():
    inputs = stored_inputs()
    point, workload = _workload(inputs)
    profile = inputs[2]
    assert _assess_capability(workload, profile, AT).status == "feasible"
    missing = profile.revise(configuration=profile.configuration.revise(
        target=profile.configuration.target.revise(instructions=())))
    assert _assess_capability(workload, missing, AT).status == "infeasible"
    unknown = profile.revise(configuration=profile.configuration.revise(
        target=profile.configuration.target.revise(instructions=None)))
    assert _assess_capability(workload, unknown, AT).status == "unknown"
    readout = profile.revise(configuration=profile.configuration.revise(
        target=profile.configuration.target.revise(readouts=())))
    assert _assess_capability(workload, readout, AT).status == "infeasible"
    assert BackendCapability.STATEVECTOR in readout.configuration.target.capabilities
    expectations = profile.revise(configuration=profile.configuration.revise(
        target=profile.configuration.target.revise(capabilities=(BackendCapability.EXPECTATION,))))
    assert _assess_capability(workload, expectations, AT).status == "feasible"


def _scalar_plan(plan):
    """Same actual one-qubit acquisition, supplied bounds 1/8 + 1/n."""
    from nwqlib.evidence import ErrorFrame, ErrorModel, ErrorTerm
    from nwqlib.core import Rational
    program = plan.construction.program.revise(parameters=(Parameter(name="n", domain="integer", lower=1),))
    construction = plan.construction.revise(program=program)
    frame = ErrorFrame.from_output(plan.problem, plan.output)
    evidence = Evidence(kind="proved_relation", source=SYNTHETIC_SOURCE, status="witnessed",
        artifact=plan.problem.content_id, subject_id=construction.content_id, witnessed_scope=frame.scope)
    fixed = FramedFact(frame=frame, bindings=(), fact=Fact(quantity="fixed", unit=frame.unit,
        scope=frame.scope, availability="concrete", value=Rational(numerator=1, denominator=8), evidence=evidence))
    unknown = FramedFact(frame=frame, bindings=(), fact=Fact(quantity="approximation", unit=frame.unit,
        scope=frame.scope, availability="unknown", reason="point bound must be supplied"))
    model = ErrorModel(output_id=plan.output.content_id, subject_id=plan.problem.content_id,
        construction_id=construction.content_id, frame=frame, required_sources=("fixed", "approximation"),
        terms=tuple(ErrorTerm(name=f.fact.quantity, stage="analysis", source=SYNTHETIC_SOURCE,
            formula="independent supplied bound", fact=f, coverage=f.fact.quantity,
            failure_probability=0.) for f in (fixed, unknown)), source=SYNTHETIC_SOURCE)
    plan = plan.revise(construction=construction, error_model=model,
        experiments=(plan.experiments[0].revise(name="scalar"),))
    point = plan.resolve("scalar", bindings=(Binding(parameter="n", value=4),))
    bound = unknown.revise(failure_probability=0., bindings=point.bindings, fact=unknown.fact.revise(
        availability="concrete", value=Rational(numerator=1, denominator=4), evidence=evidence, reason=None))
    return plan, point, (bound,)


def test_joined_accuracy_keeps_real_point_restrictions_and_target_reference(guarded_profile, monkeypatch):
    from nwqlib.backends import ProfileAssessment, assess
    import nwqlib.backends.assessment as owner
    from nwqlib.core import Rational
    from nwqlib.core.planning import Realization
    from nwqlib.evidence import ErrorModel
    from nwqlib.evidence import TargetReference
    inputs, _ = guarded_profile
    plan, point, facts = _scalar_plan(inputs[0])
    model = plan.error_model
    _, _, profile, allocation, context, runtime, at = inputs
    # Preserve the independent capacity failure: a one-qubit complex128 body
    # needs 32 bytes even when the mathematical accuracy criterion passes.
    allocation = allocation.revise(limits=(_memory_limit(16, "logical_device"),))
    reference = TargetReference(relation="exact_target", failure_probability=0.,
        fact=FramedFact(frame=model.frame, bindings=(), fact=Fact(quantity=model.frame.quantity, unit=model.frame.unit, scope=model.frame.scope,
                  availability="concrete", value=Rational(numerator=2, denominator=1),
                  evidence=model.terms[0].fact.fact.evidence)))
    result = assess(plan, point, profile=profile, allocation=allocation, context=context, runtime=runtime,
                    assessed_at=at, accuracy=Accuracy(absolute_tolerance=.375), facts=facts, reference=reference)
    assert result.accuracy.status == "feasible" and result.error.status == "PASS"
    # Hand law: e_4 = 1/8 + 1/4 = 3/8. Other axes cannot erase this scoped fact.
    assert (result.error.covered_subtotal.numerator, result.error.covered_subtotal.denominator) == (3, 8)
    assert result.error.reference == reference
    assert result.error.context.bindings == point.bindings
    assert result.capacity.status == "infeasible"
    wrong = plan.resolve("scalar", bindings=(Binding(parameter="n", value=1),))
    # The current framed contract rejects a contradictory supplied n=4 fact at
    # n=1. Absence and an insufficient valid bound remain distinct from that.
    with pytest.raises(ValueError, match="conflicting evidence parameter restriction"):
        assess(plan, wrong, profile=profile, allocation=allocation, context=context, runtime=runtime,
               assessed_at=at, accuracy=Accuracy(absolute_tolerance=.375), facts=facts, reference=reference)
    # Independent law at n=1: 1/n=1, so the total sufficient bound is 9/8.
    # Exceeding the 3/8 target does not refute actual accuracy.
    at_one = FramedFact(failure_probability=0., frame=model.frame, bindings=wrong.bindings, fact=Fact(
        quantity="approximation", unit=model.frame.unit, scope=model.frame.scope,
        availability="concrete", value=Rational(numerator=1, denominator=1), evidence=model.terms[0].fact.fact.evidence))
    insufficient = assess(plan, wrong, profile=profile, allocation=allocation, context=context, runtime=runtime,
                          assessed_at=at, accuracy=Accuracy(absolute_tolerance=.375), facts=(at_one,), reference=reference)
    assert (insufficient.error.covered_subtotal.numerator, insufficient.error.covered_subtotal.denominator) == (9, 8)
    assert insufficient.error.status == "INCONCLUSIVE" and insufficient.accuracy.status == "conditional"
    assert insufficient.error.context.bindings == wrong.bindings
    missing = assess(plan, point, profile=profile, allocation=allocation, context=context, runtime=runtime,
                     assessed_at=at, accuracy=Accuracy(absolute_tolerance=.375), reference=reference)
    assert missing.point == result.point
    assert missing.error.remaining == ("approximation",)
    assert missing.error.status == "INCONCLUSIVE" and missing.accuracy.status == "unknown"

    def no_replay(*args, **kwargs):
        pytest.fail("accuracy interchange must not replay evaluation or construct a replacement axis")
    for obj, name in ((Realization, "_selected_construction"), (ErrorModel, "assess"),
                      (owner, "estimate"), (owner, "_predict_time")):
        monkeypatch.setattr(obj, name, no_replay)
    for actual in (result, missing, insufficient):
        restored = ProfileAssessment.model_validate_json(actual.model_dump_json())
        assert restored.validate_context(plan, profile, allocation, runtime) == actual
    # Same point/model, different evidence: the PASS summary cannot be paired
    # with the actual missing-contribution assessment, in either direction.
    for original, foreign in ((result, missing), (missing, result)):
        with pytest.raises(ValueError, match="accuracy summary"):
            original.revise(error=foreign.error)
        data = original.model_dump(mode="json")
        data.pop("content_id")
        data["error"] = foreign.error.model_dump(mode="json")
        with pytest.raises(ValueError, match="accuracy summary"):
            ProfileAssessment.model_validate_json(json.dumps(data))
    # Correct identities alone cannot validate a changed scientific conclusion.
    for change in ({"status": "unknown"}, {"quantity": "another quantity"}, {"scope": "another scope"},
                   {"assumptions": ("unresolved premise",)}, {"reason": "another conclusion"},
                   {"evidence_ids": (result.error.content_id, "sha256:" + "0" * 64)}):
        detail = result.accuracy.details[0].revise(**change)
        axis = result.accuracy.revise(status=detail.status, details=(detail,))
        with pytest.raises(ValueError, match="accuracy summary"):
            result.revise(accuracy=axis)


def test_upper_memory_envelope_is_not_a_proved_failure():
    from nwqlib.ir import Branch, ClassicalStage, ClassicalValue
    workload, profile, allocation = _capacity_fixture(workspace_bytes=10 * 2**30)
    original = workload.construction
    # An exclusive unresolved branch can choose the 10-GiB call or an empty
    # path. The maximum is an upper envelope; the lower requirement is not 10.
    definitions = tuple(d for d in original.program.definitions if d.id not in {"body", "root"}) + (
        Definition(id="coin", node=ClassicalStage(implementation=SYNTHETIC_SOURCE, outputs=("coin",))),
        Definition(id="empty", node=Sequence()),
        Definition(id="branch", node=Branch(condition="coin", when_true="a", when_false="empty")),
        Definition(id="root", node=Sequence(children=("aa", "ab", "coin", "branch"))),
    )
    construction = original.revise(program=original.program.revise(definitions=definitions,
                                       classical=(ClassicalValue(name="coin", dtype="bool"),)))
    plan = stored_inputs()[0]
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id))
    experiment = plan.experiments[0]
    plan = plan.revise(experiments=(experiment.revise(observation=experiment.observation.revise(labels=("ZZ",))),))
    branch = _select_workload(plan, plan.resolve("expectation"), ResourceContext())
    result = _assess_capacity(branch, profile, allocation, AT)
    peak = next(d for d in result.details if d.quantity == "memory" and d.limit and d.limit.scope == "gpu0")
    assert peak.fact.value.numerator == 10 * 2**30 and peak.status == "conditional"
    assert result.status != "infeasible"


@pytest.fixture
def guarded_profile(monkeypatch):
    """Actual planner first; public assessment cannot acquire more evidence."""
    import builtins
    import socket
    import numpy as np
    import nwqlib.blocks as blocks
    import nwqlib._prepared_execution as execution
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators.inputs import OperatorInput
    from nwqlib.backends import assess
    inputs = stored_inputs(calibrated=True)
    events = []
    def forbidden(*args, **kwargs):
        events.append("input/native/network")
        raise AssertionError(events[-1])
    for obj, name in ((ExpectationMethod, "plan"), (OperatorInput, "pauli_terms"),
                      (blocks, "lower_qiskit"), (execution, "prepare_experiment"), (execution, "submit_experiment"),
                      (socket, "socket"), (np, "empty"), (np, "zeros"), (np, "array")):
        monkeypatch.setattr(obj, name, forbidden)
    prefixes = ("qiskit", "qiskit_aer", "qiskit_ibm_runtime", "qiskit_ionq", "nwqlib._prepared_execution",
                "nwqlib.backends.qiskit_aer",
                "nwqlib.backends.ibm_runtime", "nwqlib.backends.ionq", "nwqlib.backends.nexus")
    original_import = builtins.__import__
    def guarded_import(name, globals=None, locals=None, fromlist=(), level=0):
        names = (name, *(name + "." + member for member in fromlist or ()))
        if any(candidate == prefix or candidate.startswith(prefix + ".") for candidate in names for prefix in prefixes):
            events.append("SDK/native import " + name)
            raise ImportError(events[-1])
        return original_import(name, globals, locals, fromlist, level)
    monkeypatch.setattr(builtins, "__import__", guarded_import)
    plan, _, profile, allocation, context, runtime, at = inputs
    def evaluate(p=plan, pr=profile, grant=allocation, ctx=context, point=None):
        return assess(p, p.resolve(p.experiments[0].name) if point is None else point,
                      profile=pr, allocation=grant, context=ctx, runtime=runtime, assessed_at=at)
    yield inputs, evaluate
    assert events == []


def test_predictions_keep_actual_profile_and_realization_association(guarded_profile):
    from nwqlib.backends import ProfileAssessment
    inputs, evaluate = guarded_profile
    first = evaluate()
    model = inputs[2].models[0]
    second_profile = inputs[2].revise(models=(model.revise(coefficients=tuple(
        c.revise(seconds_per_unit=2 * c.seconds_per_unit) for c in model.coefficients)),))
    second = evaluate(pr=second_profile)
    assert (first.predictions[0].seconds.value, second.predictions[0].seconds.value) == (4.5, 9.)
    for result, profile in ((first, inputs[2]), (second, second_profile)):
        loaded = ProfileAssessment.model_validate_json(result.model_dump_json())
        assert loaded.validate_context(inputs[0], profile, inputs[3], inputs[5]) == result
        assert loaded.time.details[0].prediction_ids == (loaded.predictions[0].content_id,)
    with pytest.raises(ValueError, match="another assessment point"):
        first.revise(predictions=second.predictions)
    # A single profile covers n=1 and n=2. The second real producer output is
    # therefore a stronger association control than a foreign model ID alone.
    old = inputs[0].construction
    program = old.program.revise(parameters=(Parameter(name="n", domain="integer", lower=1, upper=2),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),),
        definitions=tuple(d for d in old.program.definitions if d.id != "root") + (
            Definition(id="repeat", node=Repeat(body="prepare", count=ExprRef(expression="n"))),
            Definition(id="root", node=Sequence(children=("allocate", "repeat"))),))
    construction = old.revise(program=program)
    plan = inputs[0].revise(construction=construction,
                            error_model=inputs[0].error_model.revise(construction_id=construction.content_id))
    profile = inputs[2].revise(models=(model.revise(domain=model.domain.revise(max_operations=6)),))
    a = evaluate(p=plan, pr=profile, point=plan.resolve("expectation", bindings=(Binding(parameter="n", value=1),)))
    b = evaluate(p=plan, pr=profile, point=plan.resolve("expectation", bindings=(Binding(parameter="n", value=2),)))
    assert a.predictions[0].model_id == b.predictions[0].model_id
    assert (a.predictions[0].seconds.value, b.predictions[0].seconds.value) == (4.5, 6.)
    assert a.predictions[0].point_id != b.predictions[0].point_id
    with pytest.raises(ValueError, match="another assessment point"):
        a.revise(predictions=b.predictions)
    with pytest.raises(ValueError, match="another prediction"):
        a.revise(time=b.time)
    # Context validation checks known model provenance without evaluating it.
    changed = a.predictions[0].revise(evidence=a.predictions[0].evidence.revise(source=SYNTHETIC_SOURCE.revise(version="foreign")))
    details = tuple(d.revise(prediction_ids=(changed.content_id,)) if d.prediction_ids else d for d in a.time.details)
    mixed = a.revise(predictions=(changed,), time=a.time.revise(details=details))
    with pytest.raises(ValueError, match="evidence/provenance"):
        mixed.validate_context(plan, profile, inputs[3], inputs[5])
