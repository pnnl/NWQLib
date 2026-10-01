"""Canonical selected points, compact IR and actual one-qubit native readouts."""

from types import SimpleNamespace
import pytest
from qiskit import QuantumCircuit
from nwqlib.backends import qiskit_aer as aer
from nwqlib.blocks import lowering
from nwqlib.core import InputRef, Source
from nwqlib.core.planning import Experiment, ObservationSpec, ReadoutDetails, Realization, RuntimeOptions
from nwqlib._prepared_execution import Run, prepare_experiment as prepare
from nwqlib.ir import (
    Binary,
    Binding,
    ClassicalValue,
    Definition,
    ExprRef,
    Expression,
    Measure,
    MeasurementBatch,
    MetadataRef,
    Parameter,
    ParameterRef,
    RangeAxis,
    Release,
    Reset,
    Sequence,
    Setting,
)
from test_prepared_execution import base_plan


def _batch_plan(base, *, fixed, settings, axes=(), selected_index=0, root_selected=False):
    """Create competing fixed/setting bindings and optional axes to exercise exact
    effective-point selection.
    """
    source = Source(name="selected setting", version="1", domain="test", reference="test:setting")
    metadata = MetadataRef(
        format=source, data=InputRef(identity="test:setting", representation="scalar", source=source)
    )
    program = base.construction.program
    batch = MeasurementBatch(
        body="batch-body",
        settings=tuple(
            (
                Setting(
                    label=f"setting-{index}",
                    metadata=metadata,
                    bindings=(Binding(parameter="theta", value=value),),
                )
                for index, value in enumerate(settings)
            )
        ),
        axes=axes,
        repetitions=1,
        observation_kind="pauli_expectation",
    )
    program = program.revise(
        parameters=(
            Parameter(name="theta", domain="integer", lower=0, upper=2),
            Parameter(name="k", domain="integer", lower=0),
        ),
        bindings=() if fixed is None else (Binding(parameter="theta", value=fixed),),
        definitions=program.definitions
        + (
            Definition(id="release", node=Release(wire="system")),
            Definition(id="batch-body", node=Sequence(children=(program.root, "release"))),
            Definition(id="batch", node=batch),
        ),
        root="batch" if root_selected else program.root,
    )
    experiment = Experiment(
        name="expectation", batch="batch", setting_index=selected_index, readout=ReadoutDetails(labels=("Z",))
    )
    return base.revise(
        construction=base.construction.revise(program=program), experiments=(experiment,)
    )._bind(blocks=base.blocks)


def test_effective_setting_binding_is_fixed_for_every_realization_route(monkeypatch):
    """Conflicting bindings must fail on every entry path before lowering, while legal axis
    points remain compact.
    """
    base = base_plan()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(1)
        raise AssertionError("invalid effective point reached native lowering")

    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    for root_selected in (False, True):
        plan = _batch_plan(base, fixed=0, settings=(1,), root_selected=root_selected)
        assert plan.construction.program.check_readiness().ready
        with pytest.raises(ValueError, match="new Plan revision"):
            plan.resolve("expectation")
        direct = Realization(plan_id=plan.content_id, experiment="expectation")
        with pytest.raises(ValueError, match="new Plan revision"):
            direct.selected_construction(plan)
        with pytest.raises(ValueError, match="new Plan revision"):
            prepare(direct, run=Run(plan), runtime=RuntimeOptions(seed=7))
        with pytest.raises(ValueError, match="new Plan revision"):
            plan.resolve("expectation", bindings=(Binding(parameter="theta", value=2),))
        for fixed, setting in ((0, 0), (None, 1)):
            legal = _batch_plan(base, fixed=fixed, settings=(setting,), root_selected=root_selected)
            actual = legal.resolve("expectation").selected_construction(legal)
            assert {item.parameter: item.value for item in actual.program.bindings} == {"theta": setting}
    unbound = _batch_plan(base, fixed=None, settings=(1,))
    for bindings in ((Binding(parameter="theta", value=2),),):
        with pytest.raises(ValueError, match="conflicting"):
            unbound.resolve("expectation", bindings=bindings)
        direct = Realization(plan_id=unbound.content_id, experiment="expectation", bindings=bindings)
        loaded = Realization.model_validate_json(direct.model_dump_json())
        for operation in (loaded.validate_plan, loaded.selected_construction, loaded.resolved_observation):
            with pytest.raises(ValueError, match="conflicting"):
                operation(unbound)
    legal = unbound.resolve("expectation", bindings=(Binding(parameter="theta", value=1),))
    assert legal.bindings == (Binding(parameter="theta", value=1),)
    legal.validate_plan(unbound)
    with pytest.raises(ValueError, match="canonical effective bindings"):
        Realization(plan_id=unbound.content_id, experiment="expectation").validate_plan(unbound)
    implicit = _batch_plan(base, fixed=None, settings=(1,), root_selected=True).revise(
        experiments=base.experiments
    )
    with pytest.raises(ValueError, match="explicit batch selector"):
        implicit.resolve("expectation")
    indexed = _batch_plan(base, fixed=0, settings=(0, 1), selected_index=1)
    with pytest.raises(ValueError, match="new Plan revision"):
        indexed.resolve("expectation")
    legal_index = _batch_plan(base, fixed=None, settings=(0, 1), selected_index=1)
    assert legal_index.resolve("expectation").resolved_observation(legal_index)[0] == "setting-1"
    axis = _batch_plan(base, fixed=0, settings=(0,), axes=(RangeAxis(parameter="k", start=1, stop=10**12),))
    selected = axis.resolve("expectation", bindings=(Binding(parameter="k", value=3),)).selected_construction(
        axis
    )
    assert {item.parameter: item.value for item in selected.program.bindings} == {"theta": 0, "k": 3}
    assert len(selected.program.definitions) == len(axis.construction.program.definitions)
    coupled = _batch_plan(
        base, fixed=None, settings=(1,), axes=(RangeAxis(parameter="k", start=0, stop=10**12),)
    )
    program = coupled.construction.program.revise(
        expressions=(
            Expression(id="theta", value=ParameterRef(parameter="theta")),
            Expression(id="k", value=ParameterRef(parameter="k")),
            Expression(id="ordered", value=Binary(op="le", left="theta", right="k")),
        ),
        constraints=(ExprRef(expression="ordered"),),
    )
    coupled = coupled.revise(construction=coupled.construction.revise(program=program))
    assert (
        coupled.resolve("expectation", bindings=(Binding(parameter="k", value=2),))
        .selected_construction(coupled)
        .program.check_readiness()
        .ready
    )
    with pytest.raises(ValueError, match="constraint is false"):
        coupled.resolve("expectation", bindings=(Binding(parameter="k", value=0),))
    assert calls == []


def test_direct_root_keeps_unreachable_batch_alternatives():
    """An unreachable alternative batch must not change direct-root semantics or trigger
    batch-selector requirements.
    """
    base = base_plan()
    original = base.construction.program
    method_source = base.method.descriptor.source
    metadata = MetadataRef(
        format=method_source,
        data=InputRef(identity="unused", representation="scalar", source=method_source),
    )
    alternative = original.revise(
        definitions=original.definitions
        + (
            Definition(id="unused_body", node=Sequence()),
            Definition(
                id="unused_batch",
                node=MeasurementBatch(
                    body="unused_body",
                    settings=(Setting(label="unused", metadata=metadata),),
                    repetitions=1,
                    observation_kind="pauli_expectation",
                ),
            ),
        )
    )
    plan = base.revise(construction=base.construction.revise(program=alternative))._bind(blocks=base.blocks)
    realized = plan.resolve("expectation")
    loaded = Realization.model_validate_json(realized.model_dump_json())
    assert loaded.validate_plan(plan) == base.experiments[0]
    assert loaded.selected_construction(plan).program == alternative
    assert loaded.resolved_observation(plan) == (base.experiments[0].setting, base.experiments[0].observation)
    for root in ("unused_batch", "reachable"):
        program = alternative.revise(
            root=root,
            definitions=alternative.definitions
            + (Definition(id="reachable", node=Sequence(children=("unused_batch",))),),
        )
        batch = plan.revise(construction=plan.construction.revise(program=program))
        with pytest.raises(ValueError, match="explicit batch selector"):
            batch.resolve("expectation")


def test_actual_early_pauli_population_ignores_later_measurement_and_reset(monkeypatch, tmp_path):
    """A pre-measurement Pauli observation is unconditional even when later circuit operations
    measure or reset.
    """
    base = base_plan()
    program = base.construction.program
    program = program.revise(
        classical=(ClassicalValue(name="readout", dtype="bits", width=1),),
        definitions=tuple((item for item in program.definitions if item.id != "root"))
        + (
            Definition(id="measure", node=Measure(wire="system", result="readout")),
            Definition(id="reset", node=Reset(wire="system")),
            Definition(id="root", node=Sequence(children=("allocate", "prepare", "measure", "reset"))),
        ),
    )
    plan = base.revise(
        construction=base.construction.revise(program=program),
        experiments=(
            Experiment(
                name="expectation",
                setting="X before measurement",
                observation=ObservationSpec(kind="pauli_expectation", position=1, labels=("X",)),
            ),
        ),
    )._bind(blocks=base.blocks)

    def forbidden(*args, **kwargs):
        raise AssertionError("prepare-only witness must not submit")

    monkeypatch.setattr(aer, "_submit_aer_execution", forbidden)
    handle = prepare(plan.resolve("expectation"), run=Run(plan), runtime=RuntimeOptions(seed=7))
    operations = [item.operation.name for item in handle._native.circuit.data]
    assert operations.index("save_expval") < operations.index("measure") < operations.index("reset")
    assert handle._native.metadata["statevector_is_measurement_conditioned"] is True
    assert handle.record.population == "unconditional"
    # Exclusions list every renormalizing operation in the native circuit, even after the readout.
    assert handle.record.probability_window_exclusions == ("measure", "reset")
    from nwqlib.backends import NWQSimBackend

    backend = NWQSimBackend(
        executable="/not-called/nwqsim",
        spool="/not-created/spool",
        max_input_bytes=65536,
        max_output_bytes=65536,
        max_buffer_bytes=65536,
    )
    native_run = Run(plan, backend=backend, directory=tmp_path / "prefix")
    native_run._state["backend_context"]["build"] = {
        "nwqsim_revision": "supplied-build",
        "compiler": "supplied C++17",
    }
    prefix = prepare(plan.resolve("expectation"), run=native_run, runtime=RuntimeOptions(seed=7))
    assert {item.operation.name for item in prefix._native.circuit.data} <= {"u", "cx"}
    assert prefix.record.population == "unconditional"
    # A description without the roundoff fields that build_runner compiles in is
    # outside the derivation (nwqsim._ROUNDOFF_BASE_REVISION states the rule).
    assert prefix.record.probability_window_exclusions == ("unchecked NWQ-Sim revision",)
    build = native_run._state["backend_context"]["build"]
    build.update(roundoff_base_revision="efd02262ff9c5def4f2eac1df6416c1ee2023d5b", roundoff_sources_identical=False)
    changed = prepare(plan.resolve("expectation"), run=native_run, runtime=RuntimeOptions(seed=8))
    assert changed.record.probability_window_exclusions == ("changed NWQ-Sim roundoff sources",)
    build["roundoff_sources_identical"] = True
    checked = prepare(plan.resolve("expectation"), run=native_run, runtime=RuntimeOptions(seed=9))
    assert checked.record.probability_window_exclusions == ()
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    circuit.measure(0, 0)
    circuit.reset(0)
    for boundary, error in ((2, "follow a measurement"), (3, "selected observation prefix")):
        with pytest.raises(ValueError, match=error):
            backend.prepare(
                circuit,
                observation=plan.experiments[0].observation.revise(position=boundary),
                runtime=RuntimeOptions(seed=7),
                position=boundary,
                source_definitions=(),
                snapshot="prefix-refusal",
                run=native_run,
            )
    conditioned = aer._prepare_aer_execution(
        aer.AER_STATEVECTOR_TARGET, circuit, shots=None, seed=7, pauli_expectation_readout=(2, ("X",))
    )
    assert conditioned.metadata["readout_population"] == "native_conditioned"
    probabilities = aer._prepare_aer_execution(
        aer.AER_STATEVECTOR_TARGET, circuit, shots=None, seed=7, probability_qubits=(0,)
    )
    assert probabilities.metadata["readout_population"] == "native_conditioned"
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(0)
    unknown = aer._prepare_aer_execution(
        aer.AER_STATEVECTOR_TARGET,
        circuit,
        shots=None,
        seed=7,
        pauli_expectation_readout=(len(circuit.data), ("X",)),
    )
    assert unknown.metadata["readout_population"] == "unknown"
    assert "if_else" in unknown.metadata["probability_window_exclusions"]
    native_run.close()


def test_batch_selection_indexes_once_and_copies_only_the_reachable_graph(monkeypatch):
    """Count source indexing/copies and inspect reachable definitions to detect repeated
    whole-graph work.
    """
    from nwqlib.core import Record
    from nwqlib.ir.validation import _Admission

    experiments = 2

    base = base_plan()
    original = base.construction.program
    method_source = base.method.descriptor.source
    metadata = MetadataRef(
        format=method_source,
        data=InputRef(identity="shared-readout", representation="scalar", source=method_source),
    )
    names = tuple((f"batch-{index}" for index in range(experiments)))
    program = original.revise(
        root="all-experiments",
        definitions=original.definitions
        + (
            Definition(id="release", node=Release(wire="system")),
            Definition(id="batch-body", node=Sequence(children=(original.root, "release"))),
            *(
                Definition(
                    id=name,
                    node=MeasurementBatch(
                        body="batch-body",
                        settings=(Setting(label=name, metadata=metadata),),
                        repetitions=1,
                        observation_kind="pauli_expectation",
                    ),
                )
                for name in names
            ),
            Definition(id="all-experiments", node=Sequence(children=names)),
        ),
    )
    plan = base.revise(
        construction=base.construction.revise(program=program),
        experiments=tuple(
            (
                Experiment(name=name, batch=name, setting_index=0, readout=ReadoutDetails(labels=("Z",)))
                for name in names
            )
        ),
    )._bind(blocks=base.blocks)
    source = plan.construction.program
    source_id = source.content_id
    full_copies, admissions = ([], [])
    original_dump, original_admission = (Record.model_dump, _Admission.__init__)

    def dump(record, *args, **kwargs):
        if record is source or record is plan.construction:
            full_copies.append(type(record).__name__)
        return original_dump(record, *args, **kwargs)

    def admit(owner, program):
        admissions.append(len(program.definitions))
        original_admission(owner, program)

    monkeypatch.setattr(Record, "model_dump", dump)
    monkeypatch.setattr(_Admission, "__init__", admit)
    body = {definition.id for definition in original.definitions} | {"batch-body", "release"}
    for name in names:
        point = plan.resolve(name)
        selected = point.selected_construction(plan)
        assert {d.id for d in selected.program.definitions} == body | {name}
        assert selected.program.root == name and selected.program.parent_id == source_id
        assert selected.selections == base.construction.selections
        loaded = Realization.model_validate_json(point.model_dump_json())
        assert loaded.validate_plan(plan).name == name
        assert loaded.selected_construction(plan).content_id == selected.content_id
    # The source identity was read into source_id before counting and the record stores it, so
    # selecting both points dumps the whole source Program zero times.
    assert full_copies.count("Program") == 0
    assert full_copies.count("SelectedConstruction") == 1
    assert admissions.count(len(source.definitions)) == 1
    assert set(admissions) == {len(source.definitions), len(body) + 1}


def test_method_selected_host_declaration_binds_actual_point_and_rejects_foreign_output():
    """Specialize scalar labels at each point, then return a foreign kernel identity to test the
    execution boundary.
    """
    from nwqlib.algorithms.protocol import Method, AlgorithmDescriptor
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.blocks.kernels import BoundKernel, KernelOutput
    from nwqlib.blocks.records import SelectedKernel
    from nwqlib.core.planning import Plan, RandomStreams
    from nwqlib.execution import ScalarValue
    from nwqlib.ir import ClassicalStage, Program
    from nwqlib.problems import Eigenproblem
    from nwqlib.problems.inputs import compose_recovery
    from nwqlib._prepared_execution import submit_experiment

    source = Source(
        name="selected statistics", version="1", domain="two bounded scalar labels", reference="test:host"
    )
    template = SelectedKernel(
        name="statistics",
        implementation=source,
        inputs=(),
        scalars=("first", "second"),
        scalar_frames=("physical", "physical"),
        resource_laws=(),
        workspace=(),
        construction_work=1,
        invocation_work=3,
    )
    calls, bindings, wrong = [], [], [False]

    class HostMethod(Method):
        @property
        def descriptor(self):
            return AlgorithmDescriptor(method="selected-statistics", version="1")

        def specialize_experiment(self, plan, experiment, values):
            return experiment.revise(
                observation=experiment.observation.revise(labels=template.scalars[values["removed"] :])
            )

        def selected_kernels(self, plan, experiment, program):
            labels = experiment.observation.labels
            return (
                template.revise(
                    scalars=labels, scalar_frames=("physical",) * len(labels), invocation_work=1 + len(labels)
                ),
            )

    method = HostMethod()
    problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
    program = Program(
        root="statistics",
        definitions=(
            Definition(
                id="statistics",
                node=ClassicalStage(implementation=source, boundary="host", kernel="statistics"),
            ),
        ),
        parameters=(Parameter(name="removed", domain="integer", lower=0, upper=1),),
    )
    plan = Plan(
        problem=problem,
        method=method,
        output=problem.default_output(),
        execution="classical",
        randomness=RandomStreams(7).snapshot(),
        construction=SelectedConstruction(program=program, selections=(), kernels=(template,)),
        experiments=(
            Experiment(
                name="statistics",
                setting="statistics",
                observation=ObservationSpec(kind="host_scalars", labels=template.scalars),
            ),
        ),
    )

    def binder(point, declaration, run):
        bindings.append(point.content_id)
        labels = declaration.scalars
        output_id = template.content_id if wrong[0] else declaration.content_id
        plan_id = plan.content_id

        def invoke():
            calls.append(labels)
            return KernelOutput(
                plan_id=plan_id,
                selected_kernel_id=output_id,
                physical_scale=compose_recovery(1.0),
                scalars=tuple(ScalarValue(label=label, value=float(i + 1)) for i, label in enumerate(labels)),
            )

        return invoke

    native = BoundKernel._bind_pointwise(plan, template, binder)
    plan._bind(blocks=(native,))
    with Run(plan) as run:
        for removed, expected in ((0, ("first", "second")), (1, ("second",))):
            point = plan.resolve("statistics", bindings=(Binding(parameter="removed", value=removed),))
            handle = prepare(point, run=run)
            assert len(calls) == removed and handle.record.observation.labels == expected
            chunk = submit_experiment(handle, run=run)
            assert tuple(value.label for value in chunk.values) == expected
            assert chunk.selected_kernel_id == handle.record.selected_kernel_id != template.content_id
        assert run.trace.host_work_reserved == 5 and run.trace.jobs == 0
        wrong[0] = True
        with pytest.raises(ValueError, match="another Plan or selected kernel"):
            submit_experiment(prepare(point, run=run), run=run)
        assert run.trace.host_invocations == 3 and run.trace.events[-1].status == "failed"
        foreign = Run(plan.revise(assumptions=("foreign exact Plan",)))
        with pytest.raises(ValueError, match="another exact Plan"):
            native._at(point, point.selected_construction(plan).kernels[0], foreign)
        assert len(bindings) == 3
        foreign.close()


def test_physical_amplitude_readout_requires_positive_recovery():
    """A physical amplitude output needs its positive composed recovery, while a unit output does not."""
    import numpy as np
    from nwqlib.amplitudes import AmplitudeReadout
    from nwqlib.artifacts import ArrayOutput
    from nwqlib.core import Basis
    from nwqlib.problems.inputs import PhysicalScale, compose_recovery

    fields = dict(
        construction_id="sha256:" + "0" * 64,
        source=Source(name="readout", version="1", domain="test", reference="unresolved recovery"),
        width=1,
        coordinates=(0,),
        output=ArrayOutput(
            name="vector",
            kind="vector",
            basis=Basis(identity="two", dimension=2, ordering="LSB"),
            frame="physical",
            global_phase="physical",
        ),
        recovery=None,
    )
    with pytest.raises(ValueError, match="selected recovery"):
        AmplitudeReadout(**fields)
    with pytest.raises(ValueError, match="positive composed"):
        AmplitudeReadout(**dict(fields, recovery=PhysicalScale(mantissa=0.0, exponent=0)))
    executable = AmplitudeReadout(**dict(fields, recovery=compose_recovery(2.0)))
    assert executable.select(np.array([1j, 0.0], dtype=np.complex128)) is not None
    unit = AmplitudeReadout(**dict(fields, output=fields["output"].revise(frame="unit")))
    np.testing.assert_array_equal(unit.select(np.array([1j, 0.0], dtype=np.complex128)), [1j, 0.0])


def test_actual_joint_counts_and_probability_coordinates_keep_their_populations(monkeypatch):
    """Keep joint classical-bit layout and separate acquisition identities, then test reversed
    probability-coordinate order.
    """
    from nwqlib._prepared_execution import submit_experiment
    from nwqlib.execution import ExecutionLimits
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.ir import Allocate, Program, Register
    from test_run_lifecycle import selected

    base = selected(shots=4)
    original = base.construction.program
    program = original.revise(
        classical=(ClassicalValue(name="unused", dtype="bits", width=1), *original.classical)
    )
    plan = base.revise(construction=base.construction.revise(program=program))._bind()
    calls = []

    def counts(native):
        calls.append(native)
        return SimpleNamespace(
            raw_output={"counts": {"0 0": 1, "1 0": 3}}, metadata={"native_job_id": "supplied-counts"}
        )

    monkeypatch.setattr(aer, "_submit_aer_execution", counts)
    with Run(plan, limits=ExecutionLimits(max_total_shots=8)) as run:
        handle = prepare(plan.resolve("counts"), run=run)
        assert [(r.name, r.bits) for r in handle.record.classical_layout] == [("unused", (0,)), ("c", (1,))]
        first, second = submit_experiment(handle, run=run), submit_experiment(handle, run=run)
        assert first.histogram().width == 2 and set(first.histogram().index_list()) == {0b00, 0b10}
        assert (
            first.returned_shots == second.returned_shots == 4
            and first.acquisition_key != second.acquisition_key
        )
        assert all((event.shots, event.evaluations) == (4, 0) for event in run.trace.events)
        with pytest.raises(ValueError, match="max_total_shots"):
            submit_experiment(handle, run=run)
        assert len(calls) == 2
    program = Program(
        root="body",
        registers=(Register(name="q", width=2),),
        definitions=(
            Definition(id="allocate", node=Allocate(wire="q")),
            Definition(id="release", node=Release(wire="q")),
            Definition(id="body", node=Sequence(children=("allocate", "release"))),
        ),
    )
    plan = base.revise(
        construction=SelectedConstruction(program=program, selections=()),
        shots=None,
        experiments=(
            Experiment(
                name="probability",
                setting="ordered bits",
                observation=ObservationSpec(kind="probabilities", qubits=(1, 0)),
            ),
        ),
    )._bind()
    monkeypatch.setattr(
        aer,
        "_submit_aer_execution",
        lambda native: SimpleNamespace(
            raw_output={"probabilities": {"01": 0.25, "10": 0.75}},
            metadata={"native_job_id": "supplied-probabilities"},
        ),
    )
    with Run(plan) as run:
        handle = prepare(plan.resolve("probability"), run=run)
        chunk = submit_experiment(handle, run=run)
        assert handle._items == 4 and handle.record.observation.qubits == (1, 0)
        assert chunk.returned_shots is None and run.trace.events[0].evaluations == 1
    from nwqlib.core.planning import readout_shape

    with pytest.raises(ValueError):
        readout_shape(
            kind="probabilities",
            details=plan.experiments[0].observation,
            width=2,
            classical_width=0,
            max_items=3,
        )
