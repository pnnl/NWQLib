"""Small independent metadata comparisons; no native preparation or simulation."""

from datetime import timedelta
import json

import pytest

from _profile_fixtures import AT, SYNTHETIC_SOURCE, stored_inputs
from nwqlib.backends import BackendCapability, BackendTarget, InstructionSupport
from nwqlib.backends.assessment import (
    _assessment_inputs,
    _assess_capability, _assess_capacity, _assess_time, _select_workload,
)
from nwqlib.backends.profiles import ModelUncertainty, TimeCoefficient, TimeModel
from nwqlib.blocks import SelectedConstruction
from nwqlib.core import Limit, Unit
from nwqlib.core.planning import Experiment, ReadoutDetails
from nwqlib.evidence import Evidence, Fact, FramedFact
from nwqlib.ir import (
    Allocate, Binding, BlockCall, ClassicalValue, Definition, ExprRef, Expression, Measure, MeasurementBatch,
    MetadataRef, Parallel, Parameter, ParameterRef, PortMap, Program, Register,
    Release, Repeat, Sequence, Setting,
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


def _batch_fixture(*, schedule="unspecified", repetitions=3, resident=()):
    plan, _, profile, allocation, *_ = stored_inputs()
    original = plan.construction
    selected = original.selections[0].revise(workspace=(
        Workspace(location="logical_device", purpose="workspace", bytes=5, source=SYNTHETIC_SOURCE),))
    definitions = tuple(item for item in original.program.definitions if item.id != "root") + (
        Definition(id="measure", node=Measure(wire="system", result="out")),
        Definition(id="release", node=Release(wire="system")),
        Definition(id="body", node=Sequence(children=("allocate", "prepare", "measure", "release"))),
        Definition(id="batch", node=MeasurementBatch(body="body", repetitions=repetitions, observation_kind="counts",
            settings=(Setting(label="one", metadata=MetadataRef(format=SYNTHETIC_SOURCE,
                data=plan.problem.observable.reference)),))),
    )
    construction = original.revise(program=original.program.revise(root="batch", definitions=definitions,
        classical=(ClassicalValue(name="out", dtype="bits", width=1),)), selections=(selected,))
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id),
                       experiments=(Experiment(name="batch", batch="batch", setting_index=0, readout=ReadoutDetails()),))
    context = ResourceContext(batch_schedule=schedule, resident=resident)
    allocation = allocation.revise(limits=(_memory_limit(8, "logical_device"),))
    return _select_workload(plan, plan.resolve("batch"), context), profile, allocation


def test_current_target_profile_allocation_identity_and_interchange():
    inputs = stored_inputs(calibrated=True)
    for record in inputs[2:4]:
        restored = type(record).model_validate_json(record.model_dump_json())
        assert restored == record and restored.content_id == record.content_id
        assert type(record).model_json_schema()["additionalProperties"] is False
    target = inputs[2].configuration.target
    # Canonical set ordering is independent of the caller's iteration order.
    reordered = BackendTarget.model_validate({**target.model_dump(exclude_computed_fields=True),
                                              "program_nodes": tuple(reversed(target.program_nodes))})
    assert reordered.content_id == target.content_id
    unknown = BackendTarget(name="unknown", provider="stored")
    unsupported = BackendTarget(name="unsupported", provider="stored", capabilities=())
    assert unknown.model_dump(mode="json")["capabilities"] is None
    assert unsupported.model_dump(mode="json")["capabilities"] == []
    with pytest.raises(ValueError, match="exactly one"):
        InstructionSupport(max_qubits=1)
    with pytest.raises(ValueError):
        inputs[3].revise(locations=("gpu0", "gpu0"))
    with pytest.raises(ValueError, match="capacity stocks"):
        inputs[3].revise(limits=(_memory_limit(8, "not_granted"),))




def test_actual_nonbatch_acquisition_is_not_the_zero_batch_fold_population():
    inputs = stored_inputs()
    point, workload = _workload(inputs)
    assert workload.estimate.quantity("settings").fact.value.numerator == 0
    assert workload.estimate.quantity("exact_evaluations").fact.value.numerator == 0
    features = {fact.quantity: fact.value.numerator for fact in workload.features}
    assert features == {"invocations": 1, "sampled_shots": 0, "exact_evaluations": 1, "logical_operations": 3}
    # One H, one Z, one H plus one grouped readout: 1/4 + 1/2 + 3/8.
    axis, predictions = _assess_time(workload, inputs[2], inputs[3], inputs[6], runtime=inputs[5], point_id=point.content_id)
    assert axis.status == "conditional"
    assert predictions[0].seconds.value == 1.125  # exact dyadic arithmetic
    assert point.construction_id == workload.estimate.construction_id
    assert point.resource_estimate_id == workload.estimate.content_id
    assert point.resource_context_id == inputs[4].content_id
    assert point.plan_id == inputs[0].content_id and point.realization_id == inputs[1].content_id
    for record in (axis, predictions[0], point):
        assert type(record).model_validate_json(record.model_dump_json()) == record
    experiment = inputs[0].experiments[0]
    two = inputs[0].revise(experiments=(experiment.revise(observation=experiment.observation.revise(labels=("Z", "X"))),))
    grouped = _select_workload(two, two.resolve("expectation"), inputs[4])
    grouped_features = {fact.quantity: fact.value.numerator for fact in grouped.features}
    assert grouped.readout_items == 2
    assert grouped_features["invocations"] == grouped_features["exact_evaluations"] == 1
    _, excluded = _assess_time(grouped, inputs[2], inputs[3], inputs[6], runtime=inputs[5], point_id=point.content_id)
    assert excluded[0].seconds is None  # one-label calibration domain does not cover two labels


def test_kept_amplitude_batch_is_assessed_as_the_amplitude_readout_it_returns():
    """A kept-amplitude QHD batch reads amplitudes, as preparation resolves it, in one invocation."""
    import sympy as sp

    from nwqlib import Optimization, estimate, plan
    from nwqlib.algorithms.qhd import QHD
    from nwqlib.backends import AER_STATEVECTOR_TARGET

    x = sp.Symbol("x")
    problem = Optimization(objective=x, variables=(x,), bounds=((-1.0, 1.0),))
    method = QHD(num_grid_points=2, num_steps=1, keep_state=True)
    selected = plan(problem, method=method, execution="quantum", seed=7)
    construction = selected.construction
    # Declare every Program node and exact selected kernel of this Plan, so the
    # readout is the only requirement the target can fail.
    target = AER_STATEVECTOR_TARGET.revise(
        readouts=("amplitudes",), artifacts=("selected_construction",),
        program_nodes=tuple(sorted({item.node.kind for item in construction.program.definitions})),
        instructions=tuple(
            InstructionSupport(implementation=s.implementation, controlled=s.controlled, adjoint=s.adjoint,
                               max_qubits=sum(port.width for port in s.signature.quantum))
            for s in construction.selections))
    _, _, profile, allocation, *_ = stored_inputs()
    configuration = profile.configuration.revise(
        target=target, runtime=profile.configuration.runtime.revise(name=target.name))
    grant = allocation.revise(configuration_id=configuration.content_id, limits=(
        _memory_limit(2**30, "logical_device"),))
    profile = profile.revise(configuration=configuration, models=())
    (assessment,) = estimate(selected, profile=profile, allocation=grant, assessed_at=AT).assessments
    assert assessment.capability.status == "feasible"
    workload = _select_workload(selected, selected.resolve("qhd"), ResourceContext())
    features = {fact.quantity: fact.value for fact in workload.features}
    assert workload.readout == "amplitudes"
    assert features["invocations"].numerator == features["exact_evaluations"].numerator == 1


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


def test_model_schema_rejects_code_wrong_units_and_confidence_meaning():
    model = stored_inputs(calibrated=True)[2].models[0]
    with pytest.raises(ValueError):
        model.revise(form="eval/python")
    with pytest.raises(ValueError):
        TimeModel.model_validate({**model.model_dump(exclude_computed_fields=True), "callable": "os.system"})
    with pytest.raises(ValueError, match="feature/unit"):
        TimeCoefficient(feature="sampled_shots", seconds_per_unit=1., unit="s/exact_evaluation")
    with pytest.raises(ValueError, match="coverage"):
        ModelUncertainty(kind="sample_mean_confidence", lower_residual_seconds=-1., upper_residual_seconds=1., source=SYNTHETIC_SOURCE)
    with pytest.raises(ValueError, match="duplicate"):
        model.revise(coefficients=(model.coefficients[0], model.coefficients[0]))


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


def test_missing_workspace_preserves_known_lower_requirement():
    small, profile, allocation = _capacity_fixture(unknown=True)
    large, _, _ = _capacity_fixture(workspace_bytes=10 * 2**30, unknown=True)
    for workload, expected in ((small, "conditional"), (large, "infeasible")):
        axis = _assess_capacity(workload, profile, allocation, AT)
        assert next(d for d in axis.details if d.quantity == "memory").status == "unknown"
        assert next(d for d in axis.details if d.quantity == "known_memory").status == expected


def test_batch_schedule_and_resident_failure_remain_independent():
    unspecified, profile, allocation = _batch_fixture()
    serial, _, _ = _batch_fixture(schedule="serial")
    first = next(d for d in _assess_capacity(unspecified, profile, allocation, AT).details if d.quantity == "memory")
    second = next(d for d in _assess_capacity(serial, profile, allocation, AT).details if d.quantity == "memory")
    assert first.status == "conditional" and second.status == "feasible"
    assert unspecified.estimate.quantity("memory", location="logical_device").required_schedule == "serial_acquisitions"
    resident = (Workspace(location="logical_device", purpose="input", bytes=9, source=SYNTHETIC_SOURCE),)
    crowded, _, _ = _batch_fixture(resident=resident)
    result = _assess_capacity(crowded, profile, allocation, AT)
    assert result.status == "infeasible"
    assert any(d.quantity == "input_bytes" and d.status == "infeasible" for d in result.details)
    generic_residents = tuple(Workspace(location="logical_device", purpose="workspace", bytes=5, source=SYNTHETIC_SOURCE)
                              for _ in range(2))
    crowded, _, _ = _batch_fixture(resident=generic_residents)
    generic = _assess_capacity(crowded, profile, allocation, AT)
    assert any(d.quantity == "resident memory lower requirement" and d.status == "infeasible" for d in generic.details)
    features = {fact.quantity: fact.value.numerator for fact in unspecified.features}
    # Three sampled acquisitions of HZH, no exact evaluations and one native job.
    assert features == {"invocations": 1, "sampled_shots": 3, "exact_evaluations": 0, "logical_operations": 9}


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


def test_no_generic_applicability_claim_is_inferred_from_method_admission(guarded_profile):
    inputs, evaluate = guarded_profile
    assert inputs[0].facts == ()
    result = evaluate()
    assert result.applicability.status == "unknown"
    assert result.capability.status == "feasible"
    unresolved = evaluate(p=inputs[0].revise(requirements=("unresolved executable requirement",)))
    assert unresolved.applicability.status == "conditional"


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


def test_consumption_and_deadline_scope_are_not_pooled_into_a_score():
    inputs = stored_inputs()
    point, workload = _workload(inputs)
    limits = (Limit(stage="execution", metric="jobs", unit=Unit(symbol="count", dimension="count"),
                    kind="consumption", value=0, scope="selected_acquisition"),
              Limit(stage="execution", metric="wall_time", unit=Unit(symbol="s", dimension="time"),
                    kind="deadline", value=100., scope="whole_run"))
    profile = inputs[2].revise(limits=limits)
    axis, predictions = _assess_time(workload, profile, inputs[3], AT, runtime=inputs[5], point_id=point.content_id)
    assert axis.status == "infeasible"
    assert next(d for d in axis.details if d.quantity == "jobs").status == "infeasible"
    assert next(d for d in axis.details if d.scope == "whole_run").status == "unknown"
    assert predictions[0].seconds.value == 1.125


def test_selected_effective_point_drives_fold_and_identity():
    inputs = stored_inputs()
    plan, *_ = inputs
    original = plan.construction
    program = original.program.revise(parameters=(Parameter(name="r", domain="integer", lower=1, upper=4),),
        expressions=(Expression(id="r_value", value=ParameterRef(parameter="r")),),
        definitions=tuple(d for d in original.program.definitions if d.id != "root") + (
            Definition(id="repeat", node=Repeat(body="prepare", count=ExprRef(expression="r_value"))),
            Definition(id="root", node=Sequence(children=("allocate", "repeat"))),))
    construction = original.revise(program=program)
    plan = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id))
    one = plan.resolve("expectation", bindings=(Binding(parameter="r", value=1),))
    four = plan.resolve("expectation", bindings=(Binding(parameter="r", value=4),))
    a = _assessment_inputs(plan, one, *inputs[2:])[0:2]
    b = _assessment_inputs(plan, four, *inputs[2:])[0:2]
    assert a[1].estimate.quantity("operations").fact.value.numerator == 3
    assert b[1].estimate.quantity("operations").fact.value.numerator == 12
    assert a[0].construction_id != b[0].construction_id
    assert a[0].resource_estimate_id != b[0].resource_estimate_id
    assert a[0].realization_id != b[0].realization_id
    assert a[0].plan_id == b[0].plan_id == plan.content_id




def test_joined_five_axes_use_one_selection_fold_and_common_assessment(monkeypatch):
    """Five assessment axes must share one selection/fold and preserve unknown accuracy despite
    other feasible axes.
    """
    from nwqlib.backends import ProfileAssessment, assess
    from nwqlib.core.planning import Realization
    from nwqlib.evidence import ErrorModel
    import nwqlib.backends.assessment as owner
    inputs = stored_inputs()
    plan, realization, profile, allocation, context, runtime, at = inputs
    calls = {"selection": 0, "fold": 0, "accuracy": 0}
    select = Realization._selected_construction
    fold = owner.estimate
    common = ErrorModel.assess
    def selected(self, supplied_plan, **kwargs):
        calls["selection"] += 1
        return select(self, supplied_plan, **kwargs)
    def folded(*args, **kwargs):
        calls["fold"] += 1
        return fold(*args, **kwargs)
    def assessed(self, accuracy, **kwargs):
        calls["accuracy"] += 1
        assert kwargs["context"].bindings == realization.bindings
        return common(self, accuracy, **kwargs)
    monkeypatch.setattr(Realization, "_selected_construction", selected)
    monkeypatch.setattr(owner, "estimate", folded)
    monkeypatch.setattr(ErrorModel, "assess", assessed)
    result = assess(plan, realization, profile=profile, allocation=allocation, context=context,
                    runtime=runtime, assessed_at=at, accuracy=Accuracy(absolute_tolerance=1e-8))
    assert calls == {"selection": 1, "fold": 1, "accuracy": 1}
    assert result.capability.status == "feasible"
    assert result.capacity.status == "infeasible"
    assert result.time.status == "conditional" and result.accuracy.status == "unknown"
    assert {"native_preparation", "native_simulation", "physical_model"} <= set(result.error.remaining)
    assert result.error.context.plan_id == plan.content_id
    assert result.error.context.selected_construction_id == result.resources.construction_id
    assert result.error.context.bindings == result.realization.bindings
    assert result.error.context.problem_id == plan.problem.content_id
    assert result.error.context.base_construction_id == plan.construction.content_id
    assert result.error.context.admitted
    assert result.error.context.observation_id is result.error.context.result_id is None
    assert result.error.context.contribution_ids == ()
    restored = ProfileAssessment.model_validate_json(result.model_dump_json())
    assert restored.validate_context(plan, profile, allocation, runtime) == result
    assert calls == {"selection": 1, "fold": 1, "accuracy": 1}  # interchange is not replay
    with pytest.raises(ValueError, match="another"):
        restored.validate_context(plan, profile, allocation, runtime.revise(seed=8))
    with pytest.raises(ValueError, match="another scientific point"):
        result.revise(error=result.error.revise(context=result.error.context.revise(selected_construction_id="sha256:" + "0" * 64)))


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


def test_stored_stock_failure_survives_unknown_total_memory(guarded_profile):
    inputs, evaluate = guarded_profile
    resident = Workspace(location="logical_device", purpose="stored", bytes=5, source=SYNTHETIC_SOURCE)
    context = inputs[4].revise(resident=(resident,))
    stored = Limit(stage="execution", metric="stored", unit=Unit(symbol="byte", dimension="bytes"),
                     kind="capacity_stock", value=4, scope="logical_device")
    for stock, expected in ((4, "infeasible"), (5, "conditional")):
        limit = stored.revise(value=stock)
        result = evaluate(grant=inputs[3].revise(limits=(*inputs[3].limits, limit)), ctx=context)
        detail, = (d for d in result.capacity.details if d.limit == limit)
        assert detail.fact.value.numerator == 5 and detail.status == expected
        assert result.capacity.status == expected
        assert any(d.quantity == "memory" and d.status == "unknown" for d in result.capacity.details)
    foreign = stored.revise(scope="other_device")
    grant = inputs[3].revise(locations=("logical_device", "other_device"), topology="independent_devices",
                             limits=(*inputs[3].limits, foreign))
    result = evaluate(grant=grant, ctx=context)
    assert next(d for d in result.capacity.details if d.limit == foreign).status == "unknown"


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


def test_finite_time_domain_covers_non_gate_populations(guarded_profile):
    from nwqlib.backends import AER_COUNTS_TARGET
    from nwqlib.ir import Reset
    inputs, evaluate = guarded_profile
    plan, _, profile, grant, context, *_ = inputs
    model = profile.models[0]
    target = AER_COUNTS_TARGET.revise(artifacts=("selected_construction",), readouts=("counts",), max_qubits=1,
        instructions=profile.configuration.target.instructions,
        program_nodes=("sequence", "allocate", "block_call", "repeat", "reset", "measure", "release", "measurement_batch"))
    config = profile.configuration.revise(target=target, runtime=profile.configuration.runtime.revise(name=target.name))
    grant = grant.revise(configuration_id=config.content_id)
    domain = model.domain.revise(configuration_id=config.content_id, allocation_id=grant.content_id,
        acquisition="measurement_batch", batch_schedule="serial", readouts=("counts",),
        min_operations=9, max_operations=9, max_shots=3, max_exact_evaluations=0, max_readout_items=2,
        max_resets=0, max_measurements=3, max_classical_work=0, max_adaptive_rounds=0)
    profile = profile.revise(configuration=config, models=(model.revise(domain=domain,
        assumptions=("synthetic three-shot HZH/counts model without reset; no hardware timing evidence",)),))
    for repetitions in (0, 8):
        old = plan.construction
        program = old.program.revise(root="batch", classical=(ClassicalValue(name="out", dtype="bits", width=1),),
            definitions=tuple(d for d in old.program.definitions if d.id != "root") + (
                Definition(id="reset", node=Reset(wire="system")),
                Definition(id="resets", node=Repeat(body="reset", count=repetitions)),
                Definition(id="measure", node=Measure(wire="system", result="out")),
                Definition(id="release", node=Release(wire="system")),
                Definition(id="body", node=Sequence(children=("allocate", "prepare", "resets", "measure", "release"))),
                Definition(id="batch", node=MeasurementBatch(body="body", repetitions=3, observation_kind="counts",
                    settings=(Setting(label="one", metadata=MetadataRef(format=plan.method.descriptor.source,
                        data=plan.problem.observable.reference)),))),))
        construction = old.revise(program=program)
        candidate = plan.revise(construction=construction, error_model=plan.error_model.revise(construction_id=construction.content_id),
            experiments=(Experiment(name="counts", batch="batch", setting_index=0, readout=ReadoutDetails()),))
        assert construction.program.check_readiness().ready
        _, observation = candidate.resolve("counts").resolved_observation(candidate)
        assert observation.kind == "counts" and observation.shots == 3
        result = evaluate(p=candidate, pr=profile, grant=grant, ctx=context.revise(batch_schedule="serial"))
        assert result.resources.quantity("operations").fact.value.numerator == 9
        assert result.resources.quantity("resets").fact.value.numerator == 3 * repetitions
        assert result.capability.status == "feasible"
        prediction = result.predictions[0]
        if repetitions:
            assert prediction.seconds is prediction.uncertainty is None
            assert prediction.lower_seconds is prediction.upper_seconds is None
            assert any("resets" in reason for reason in prediction.reasons)
        else:
            assert (prediction.seconds.value, prediction.lower_seconds.value, prediction.upper_seconds.value) == (5.5, 5.25, 6.)
            assert prediction.uncertainty.coverage == 0.9 and prediction.reasons == ()
            experiment = candidate.experiments[0]
            malformed = candidate.revise(experiments=(experiment.revise(
                readout=experiment.readout.revise(position=0)),))
            # Kind and field already conflict without native instruction-count
            # knowledge; this pair must never obtain the legal 5.5 s prediction.
            with pytest.raises(ValueError, match="instruction position"):
                evaluate(p=malformed, pr=profile, grant=grant, ctx=context.revise(batch_schedule="serial"))




def test_selected_readout_dimensions_and_symbolic_preservation(guarded_profile):
    inputs, evaluate = guarded_profile
    plan = inputs[0]
    baseline = evaluate()
    assert baseline.capability.status == "feasible" and baseline.predictions[0].seconds.value == 4.5
    experiment = plan.experiments[0]
    for observation, message in (
        (experiment.observation.revise(labels=("ZZ",)), "entire logical circuit width"),
        (experiment.observation.revise(kind="probabilities", labels=(), qubits=(1,)), "belong to the circuit"),
        (experiment.observation.revise(kind="counts", labels=(), shots=3), "measurement registers"),
    ):
        candidate = plan.revise(experiments=(experiment.revise(observation=observation),))
        with pytest.raises(ValueError, match=message):
            evaluate(p=candidate)
    # Early Pauli instruction position is passed through the shared dimensional
    # helper; it is not replaced by the end position or a counts-style guard.
    early = plan.revise(experiments=(experiment.revise(observation=experiment.observation.revise(position=0)),))
    early_result = evaluate(p=early)
    assert early_result.point.plan_id == early.content_id != baseline.point.plan_id
    assert early_result.predictions[0].seconds.value == 4.5
    # These two declared registers are live sequentially. Native circuit width
    # is two while the resource fold's simultaneous logical peak is only one.
    selected = plan.construction.selections[0]
    program = Program(root="root", registers=(Register(name="a", width=1), Register(name="b", width=1)),
        signatures=(selected.signature,), definitions=(
            Definition(id="aa", node=Allocate(wire="a")), Definition(id="ab", node=Allocate(wire="b")),
            Definition(id="pa", node=BlockCall(signature="state", ports=(PortMap(port="system", wire="a"),))),
            Definition(id="pb", node=BlockCall(signature="state", ports=(PortMap(port="system", wire="b"),))),
            Definition(id="ra", node=Release(wire="a")), Definition(id="rb", node=Release(wire="b")),
            Definition(id="root", node=Sequence(children=("aa", "pa", "ra", "ab", "pb", "rb"))),))
    construction = SelectedConstruction(program=program, selections=(selected,))
    candidate = plan.revise(construction=construction,
                            error_model=plan.error_model.revise(construction_id=construction.content_id))
    with pytest.raises(ValueError, match="entire logical circuit width"):
        evaluate(p=candidate)
    two = candidate.revise(experiments=(experiment.revise(observation=experiment.observation.revise(labels=("ZZ",))),))
    legal = evaluate(p=two)
    assert legal.resources.quantity("logical_width", location="logical_device").fact.value.numerator == 1
    # A genuinely unresolved width remains useful metadata. It neither passes
    # concrete readout capability nor acquires an in-domain numeric prediction.
    symbolic = Program(root="allocate", parameters=(Parameter(name="n", domain="integer", lower=1, upper=2),),
        expressions=(Expression(id="n", value=ParameterRef(parameter="n")),),
        registers=(Register(name="q", width=ExprRef(expression="n")),),
        definitions=(Definition(id="allocate", node=Allocate(wire="q")),))
    construction = SelectedConstruction(program=symbolic, selections=())
    unresolved = plan.revise(construction=construction,
                             error_model=plan.error_model.revise(construction_id=construction.content_id))
    result = evaluate(p=unresolved)
    assert result.capability.status == "unknown" and result.predictions[0].seconds is None
    assert result.error is None and result.accuracy.status == "unknown"
    assert result.resources.quantity("logical_width", location="logical_device").fact.availability == "symbolic"
