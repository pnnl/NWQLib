"""Tiny external Method through the shared selected-experiment lifecycle."""

from types import SimpleNamespace
from typing import ClassVar, Literal

import numpy as np
import pytest

from nwqlib.algorithms.protocol import AlgorithmDescriptor, Method
from nwqlib.blocks.records import SelectedConstruction
from nwqlib.core.analysis import Result, capture_analysis_origin
from nwqlib.core.planning import Experiment, ObservationSpec, Plan, RandomStreams
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, ExecutionLimits
from nwqlib.ir import Allocate, ClassicalValue, Definition, Measure, Program, Register, Sequence
from nwqlib.scientist import plan, prepare, solve, submit


class ScalarOutput(Record):
    quantity: str = "Z expectation"


class ToyProblem(Record):
    width: int = 1

    def default_output(self):
        return ScalarOutput()


class ToyResult(Result):
    value: float


class ToyMethod(Method):
    scale: float = 2.
    calls: ClassVar[list] = []

    @property
    def descriptor(self):
        return AlgorithmDescriptor(method="external.toy", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        self.calls.append("plan")
        rng.method.integers(100)  # Must not consume the separate backend stream.
        program = Program(root="body", registers=(Register(name="q", width=problem.width),),
            classical=(ClassicalValue(name="c", dtype="bits", width=problem.width),),
            definitions=(Definition(id="body", node=Sequence(children=("allocate", "measure"))),
                Definition(id="allocate", node=Allocate(wire="q")),
                Definition(id="measure", node=Measure(wire="q", result="c"))))
        return Plan(problem=problem, method=self, output=output, execution=execution,
            shots=shots, randomness=rng.snapshot(), construction=SelectedConstruction(program=program, selections=()),
            experiments=(Experiment(name="z", setting="computational", observation=ObservationSpec(kind="counts", shots=shots)),))

    def analyze(self, selected, data, *, settings):
        self.calls.append("analyze")
        chunk, = data.observations.chunks
        histogram = chunk.histogram()
        counts = dict(zip(histogram.index_list(), histogram.weights.tolist()))
        mean = (counts.get(0, 0)-counts.get(1, 0))/chunk.returned_shots
        return ToyResult(plan_id=selected.content_id, construction_id=selected.construction.content_id,
            observation_id=data.observations.content_id, contribution_ids=(chunk.content_id,),
            value=self.scale*mean,
            origin=capture_analysis_origin(analyzer=self.descriptor.source, method_id=self.content_id))


class StubBackend(Record):
    kind: Literal["stub"] = "stub"
    calls: ClassVar[list] = []
    fail: ClassVar[bool] = False

    def target_for(self, observation):
        return SimpleNamespace(readouts=("counts",))

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        from nwqlib.backends.connection import NativePreparation
        self.calls.append(("prepare", runtime.seed))
        source = Source(name="supplied", version="1", domain="stubbed native boundary", reference="test")
        native = SimpleNamespace(circuit=circuit, shots=observation.shots, seed=runtime.seed)
        return NativePreparation(native=native, target=source, compiler=source, native_basis=("measure",),
            environment=(), quantum_layout=(), classical_layout=(), logical_to_native=(0,), operations=1,
            population="unconditional", transformation="supplied", counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed))

    def submit(self, native, *, submission_id, run):
        self.calls.append(("submit", native.shots, native.seed))
        assert run.trace.events[-1].status == "reserved"  # Intent is visible before launch.
        assert run._state["workflow_items"]["z"].attempt == run.trace.events[-1].attempt
        if self.fail:
            raise OSError("supplied acquisition failure")
        return SimpleNamespace(raw_output={"counts": {"0": 3, "1": 1}}, metadata={"native_job_id": submission_id})


@pytest.fixture
def stub(monkeypatch):
    from nwqlib.blocks import lowering
    ToyMethod.calls.clear()
    StubBackend.calls.clear()
    StubBackend.fail = False

    class Bits(tuple):
        def __new__(cls, name, bits):
            obj = super().__new__(cls, bits)
            obj.name = name
            return obj

    def lower(construction, **kwargs):
        width = construction.program.registers[0].width
        q, c = Bits("q", range(width)), Bits("c", range(width))
        circuit = SimpleNamespace(data=("measure",), qregs=(q,), cregs=(c,), num_qubits=width,
                                  find_bit=lambda bit: SimpleNamespace(index=bit))
        return lowering.LogicalCircuit(construction.content_id, circuit, (("q", tuple(q)),),
                                       (("c", tuple(c)),), 3, 1, ())
    monkeypatch.setattr(lowering, "_lower_qiskit", lower)
    return StubBackend()


def test_selected_plan_executes_without_replanning_and_reanalysis_shares_data(stub):
    problem, method = ToyProblem(), ToyMethod()
    selected = plan(problem, method=method, shots=4, seed=19)
    prepared = prepare(selected, backend=stub)
    assert ToyMethod.calls == ["plan"] and len(StubBackend.calls) == 1
    run = submit(prepared)
    result = run.wait()
    assert result.value == 1.  # Supplied 3:1 histogram gives <Z>=1/2, scale=2.
    assert result.plan is selected and selected.problem is problem and selected.method is method
    expected_backend = np.random.Generator(np.random.PCG64(np.random.SeedSequence(19).spawn(2)[1]))
    expected_seed = int(expected_backend.integers(0, 2**32, dtype="uint32"))
    assert StubBackend.calls == [("prepare", expected_seed), ("submit", 4, expected_seed)]
    before = tuple(StubBackend.calls)
    changed = result.analyze()
    assert changed is not result and changed.data is result.data and changed.value == result.value
    assert changed.origin.invocation_id != result.origin.invocation_id
    result.report()
    assert tuple(StubBackend.calls) == before and ToyMethod.calls.count("plan") == 1
    with pytest.raises(ValueError, match="cannot be overridden"):
        solve(selected, backend=stub, shots=4)
    assert tuple(StubBackend.calls) == before


def test_a_run_fills_circuit_and_shot_budgets_only_for_a_backend_that_may_charge(stub):
    """Aer leaves both budgets unset. The stub backend may charge, so its Run fills the unset one and keeps the set one."""
    from nwqlib._prepared_execution import Run

    selected = plan(ToyProblem(), method=ToyMethod(), shots=4, seed=7)
    with Run(selected) as local:
        assert (local.limits.max_total_circuits, local.limits.max_total_shots) == (None, None)
        local.check_capacity(preparations=10**4, circuits=10**4, shots=10**9)
    with Run(selected, backend=StubBackend(), limits=ExecutionLimits(max_total_shots=64)) as paid:
        assert (paid.limits.max_total_circuits, paid.limits.max_total_shots) == (512, 64)
        paid.extend_limits(max_total_shots=None)
        assert paid.limits.max_total_shots is None and paid.limit_amendments[0].new.max_total_shots is None
        paid.check_capacity(shots=10**9)


def test_data_and_complete_workload_limits_stop_before_native_work(stub):
    selected = plan(ToyProblem(), method=ToyMethod(), shots=4, seed=7)
    with pytest.raises(ValueError, match=r"ExecutionLimits\(max_total_shots="):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_total_shots=3))
    assert StubBackend.calls == []
    with pytest.raises(ValueError, match=r"ExecutionLimits\(max_data_bytes="):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_data_bytes=100))
    assert StubBackend.calls == []


def test_random_stream_snapshot_preserves_both_draw_positions():
    original = RandomStreams(17)
    original.method.integers(100, size=3)
    original.next_seed()
    restored = RandomStreams.restore(original.snapshot())
    assert restored.next_seed() == original.next_seed()
    assert restored.method.integers(100) == original.method.integers(100)


@pytest.mark.parametrize("target", ["statevector", "counts"])
def test_aer_memory_reaches_native_configuration(monkeypatch, target):
    from qiskit import QuantumCircuit
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.backends.targets import AER_COUNTS_TARGET, AER_STATEVECTOR_TARGET

    measured = QuantumCircuit(1, 1)
    measured.measure(0, 0)
    received = []
    class StopAtConfiguration(Exception):
        pass
    def simulator(**options):
        received.append(options)
        raise StopAtConfiguration
    monkeypatch.setattr(aer, "AerSimulator", simulator)
    with pytest.raises(StopAtConfiguration):
        aer._prepare_aer_execution(AER_STATEVECTOR_TARGET if target == "statevector" else AER_COUNTS_TARGET,
                                   SimpleNamespace() if target == "statevector" else measured,
                                   shots=None if target == "statevector" else 4, seed=7,
                                   simulator_memory_mb=37)
    assert received[0]["max_memory_mb"] == 37


class AxisToy(ToyMethod):
    """ToyMethod whose analysis accepts one data collection per static point, and whose Plan a Run folder saves."""

    result_type: ClassVar[type] = ToyResult

    def analyze(self, selected, data, *, settings):
        chunks = data.observations.chunks
        return ToyResult(plan_id=selected.content_id, construction_id=selected.construction.content_id,
            observation_id=data.observations.content_id, contribution_ids=tuple(chunk.content_id for chunk in chunks),
            value=float(len(chunks)), origin=capture_analysis_origin(analyzer=self.descriptor.source,
                                                                     method_id=self.content_id))

    def save_archive(self, plan, files):
        return dict(plan=plan.to_record(), problem=plan.problem.model_dump(mode="json"),
                    output=plan.output.model_dump(mode="json"))

    @classmethod
    def load_archive(cls, saved, files):
        return files.read_plan(saved["plan"], problem=ToyProblem.model_validate(saved["problem"]), method=cls(),
                               output=ScalarOutput.model_validate(saved["output"]))._bind()


def axis_plan(rows, *, flat=None, method=None):
    """A ToyMethod Plan of one batch experiment ``series`` with a ``row`` axis of ``rows`` points, 3 shots each.

    With ``flat``, a second batch experiment of that name has no axis and binds ``row=0`` in its setting.
    """
    from nwqlib.core.planning import ReadoutDetails
    from nwqlib.ir import Binding, MeasurementBatch, Parameter, RangeAxis, Release, Setting
    from test_ir import metadata
    base = plan(ToyProblem(), method=method or ToyMethod(), shots=3, seed=14)
    setting = Setting(label="Z", metadata=metadata("axis_counts"))
    batch = MeasurementBatch(body="body", settings=(setting,), axes=(RangeAxis(parameter="row", start=0, stop=rows),),
                             repetitions=3, observation_kind="counts")
    program = base.construction.program
    definitions = [item.revise(node=item.node.revise(children=(*item.node.children, "release"))) if item.id == "body"
                   else item for item in program.definitions]
    definitions += [Definition(id="release", node=Release(wire="q")), Definition(id="axis", node=batch)]
    experiments = [Experiment(name="series", batch="axis", setting_index=0, readout=ReadoutDetails())]
    root = "axis"
    if flat is not None:
        bound = setting.revise(bindings=(Binding(parameter="row", value=0),))
        definitions += [Definition(id="flat", node=batch.revise(axes=(), settings=(bound,))),
                        Definition(id="root", node=Sequence(children=("axis", "flat")))]
        experiments.append(experiments[0].revise(name=flat, batch="flat"))
        root = "root"
    program = program.revise(root=root, parameters=(Parameter(name="row", domain="integer", lower=0),),
                             definitions=tuple(definitions))
    return base.revise(construction=base.construction.revise(program=program), experiments=tuple(experiments))


def test_an_experiment_named_like_an_axis_point_keeps_its_own_item_seed_and_attempt(tmp_path):
    """The axis points of ``series`` and an axis-free experiment named ``series[row=0]`` have distinct keys.

    The experiment name's ``[`` is escaped in its key, so the three points each get a workflow item with
    its own preparation, seed and attempt, and a reopened Run keeps that mapping without new work.
    """
    import nwqlib

    selected = axis_plan(2, flat="series[row=0]", method=AxisToy())
    keys = [point.key for point in selected._static_points()]
    assert keys == ["series[row=0]", "series[row=1]", "series\\[row=0]"]
    folder = tmp_path / "run"
    with prepare(selected, settings="all", directory=folder, progress=False).run as run:
        assert list(run._state["workflow_items"]) == keys and run.trace.preparations == 3
        result = run.resume().result
        items ={key: (item.realization.experiment, [(b.parameter, b.value) for b in item.realization.bindings],
                       item.runtime.seed, item.prepared_id, item.attempt)
                 for key, item in run._state["workflow_items"].items()}
    assert result.value == 3. and len(result.data.trace.events) == 3
    assert [(name, bindings) for name, bindings, *_ in items.values()] == [
        ("series", [("row", 0)]), ("series", [("row", 1)]), ("series[row=0]", [("row", 0)])]
    for column in (2, 3, 4):
        assert len({row[column] for row in items.values()}) == 3 and None not in {row[column] for row in items.values()}
    with nwqlib.load_run(folder, method=AxisToy) as reopened:
        assert {key: (item.realization.experiment, [(b.parameter, b.value) for b in item.realization.bindings],
                      item.runtime.seed, item.prepared_id, item.attempt)
                for key, item in reopened._state["workflow_items"].items()} == items
        assert reopened.resume().result.value == 3. and len(reopened.trace.events) == 3


@pytest.mark.parametrize("stop", ("before the charge", "after the charge", "after the outcome"))
def test_a_process_stopped_at_the_second_point_keeps_its_seed_and_collected_prefix(tmp_path, stop):
    """A child process exits at a commit of the second of three axis points, and the parent reopens its folder.

    Before the point's preparation charge nothing of the point is saved, and resuming draws its seed again
    from the saved RNG position. After the charge its workflow row holds that seed, and the Run refuses
    to rebuild the preparation. After its outcome the observation is already collected. The seeds match an
    uninterrupted Run, and no completed attempt runs again.
    """
    import subprocess
    import sys
    from pathlib import Path
    import nwqlib
    from nwqlib._prepared_execution import PreparationNotRebuilt

    code = '''
import os, sys
sys.path.insert(0, sys.argv[3])
from nwqlib._run_journal import LocalJournal
from nwqlib.scientist import prepare
from test_selected_runtime import AxisToy, axis_plan
stop, commit, seen = sys.argv[2], LocalJournal.commit, []
def stopping(self, records=(), **kwargs):
    kinds = {kind for kind, _, _ in records}
    # The charge is the only commit with a preparation row and the RNG position.
    marked = "chunk" in kinds if stop == "after the outcome" else {"preparation", "rng"} <= kinds
    seen.append(None) if marked else None
    if marked and len(seen) == 2 and stop == "before the charge":
        os._exit(0)
    commit(self, records, **kwargs)
    if marked and len(seen) == 2:
        os._exit(0)
LocalJournal.commit = stopping
prepare(axis_plan(3, method=AxisToy()), directory=sys.argv[1], progress=False).run.resume()
raise AssertionError("the second point did not stop")
'''
    subprocess.run([sys.executable, "-c", code, str(tmp_path / "stopped"), stop, str(Path(__file__).parent)],
                   check=True, timeout=120)
    with prepare(axis_plan(3, method=AxisToy()), directory=tmp_path / "whole", progress=False).run as whole:
        whole.resume()
        keys = list(whole._state["workflow_items"])
        seeds = [item.runtime.seed for item in whole._state["workflow_items"].values()]
    with nwqlib.load_run(tmp_path / "stopped", method=AxisToy, progress=False) as run:
        items = run._state["workflow_items"]
        assert list(items) == keys[:1 if stop == "before the charge" else 2]
        assert run.trace.preparations == 1 + (stop != "before the charge")
        assert len(run.observations.chunks) == 1 + (stop == "after the outcome")
        if stop == "after the charge":
            assert items[keys[1]].runtime.seed == seeds[1] and items[keys[1]].prepared_id is None
            with pytest.raises(PreparationNotRebuilt):
                run.resume()
            assert len(run.trace.events) == 1
            return
        assert run.resume().result.value == 3. and len(run.trace.events) == 3
        assert [item.runtime.seed for item in run._state["workflow_items"].values()] == seeds


def test_an_interrupt_in_the_collect_progress_leaves_its_point_completed(tmp_path):
    """Ctrl-C in a progress observer that reports the first collected point reaches the caller unchanged.

    The point's outcome and collection have committed, so the reopened Run shows it completed and
    finishes the other two points without acquiring it again.
    """
    import nwqlib

    def observe(stage, done, total):
        if stage == "collect":
            raise KeyboardInterrupt

    with prepare(axis_plan(3, method=AxisToy()), directory=tmp_path / "run", progress=observe).run as run:
        with pytest.raises(KeyboardInterrupt):
            run.resume()
    with nwqlib.load_run(tmp_path / "run", method=AxisToy, progress=False) as reopened:
        assert [event.status for event in reopened.trace.events] == ["completed"]
        assert len(reopened.observations.chunks) == 1
        assert reopened.resume().result.value == 3. and len(reopened.trace.events) == 3


def test_a_trillion_point_axis_is_read_only_as_far_as_needed(stub, monkeypatch):
    """A Plan whose axis has 10**12 points generates, counts and refuses its points without reading the axis.

    A guard replaces the axis ranges of `Plan._static_points` and fails if more values are read than the
    caller needs. The circuit budget refuses from the axis bounds before any point is generated or prepared,
    the profile forecast refuses after its first point, and a static Run refuses from its first point.
    """
    from nwqlib import _prepared_execution as execution
    from nwqlib.core import planning

    selected = axis_plan(10**12)
    reads = []

    def guarded(*args):
        def values():
            for value in range(*args):
                if len(reads) == 3:
                    raise AssertionError("the axis was read beyond the values needed")
                reads.append(value)
                yield value
        return values()
    monkeypatch.setattr(planning, "range", guarded, raising=False)
    assert selected._point_count() == 10**12
    assert next(selected._static_points()).key == "series[row=0]" and reads == [0]
    reads.clear()
    with pytest.raises(ValueError, match=r"max_total_circuits=1 cannot admit the requested circuit preparations "
                                         r"\(0 counted, 1000000000000 requested"):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_total_circuits=1), progress=False)
    assert reads == [] and StubBackend.calls == []
    # Each point is a counts readout with 3 repetitions, so the shots follow from the axis bounds too.
    with pytest.raises(ValueError, match=r"max_total_shots=1 cannot admit the requested raw shots "
                                         r"\(0 counted, 3000000000000 requested"):
        prepare(selected, backend=stub, limits=ExecutionLimits(max_total_circuits=10**12, max_total_shots=1),
                progress=False)
    assert reads == [] and StubBackend.calls == []

    # A Run reached without prepare meets the same circuit budget before it spends any of it.
    with execution.Run(selected, backend=stub) as run:
        with pytest.raises(ValueError, match=r"max_total_circuits=512 cannot admit .* 1000000000000 requested"):
            execution.execute_static(selected, run=run)
        assert run.trace.preparations == 0 and StubBackend.calls == []
    # With those budgets raised, the circuits that 10**12 handles would keep do not fit.
    # The loop's first point and the check's own first point and the axis bounds establish
    # that before anything is prepared.
    reads.clear()
    ample = ExecutionLimits(max_total_circuits=10**12, max_total_shots=3 * 10**12)
    with execution.Run(selected, backend=stub, limits=ample) as run:
        with pytest.raises(ValueError, match="running the remaining circuits of this Plan needs about"):
            execution.execute_static(selected, run=run)
        assert reads == [0, 0] and run.trace.preparations == 0 and not run._state["workflow_items"]


def test_the_points_of_a_fixed_shot_batch_share_their_first_readout_before_preparation(stub, monkeypatch):
    """Before its first preparation, a Run on a backend other than Aer resolves only the first point of ``series``.

    Its Program fixes the 3 shots of every point, so the five points share the first point's readout. The
    check resolves the first point's construction, which its preparation reuses, and counts the circuits and
    shots of the other four from the axis bounds without generating them. A circuit limit of 4 still refuses
    the five points.
    """
    from nwqlib.core import planning

    calls, consumed = [], []
    resolve, points = planning.Plan.resolve, planning.Plan._static_points
    monkeypatch.setattr(planning.Plan, "resolve", lambda self, *args, **kwargs: (calls.append(1),
                                                                                resolve(self, *args, **kwargs))[1])

    def generated(self, *args, **kwargs):
        consumed.append(0)
        for point in points(self, *args, **kwargs):
            consumed[-1] += 1
            yield point
    monkeypatch.setattr(planning.Plan, "_static_points", generated)
    with prepare(axis_plan(5), backend=stub, progress=False).run as run:
        # One point for the check and one for the first preparation.
        assert calls == [] and consumed == [1, 1] and run.trace.preparations == 1
    with pytest.raises(ValueError, match=r"max_total_circuits=4 cannot admit"):
        prepare(axis_plan(5), backend=stub, limits=ExecutionLimits(max_total_circuits=4), progress=False)


def revised_batch(selected, *, axes=None, repetitions=None, expressions=()):
    """Revise the batch of an `axis_plan` Plan with other range axes or repetitions, declaring each axis parameter."""
    from nwqlib.ir import Parameter
    program = selected.construction.program
    (batch,) = [item.node for item in program.definitions if item.id == "axis"]
    batch = batch.revise(axes=batch.axes if axes is None else tuple(axes),
                         repetitions=batch.repetitions if repetitions is None else repetitions)
    program = program.revise(
        parameters=tuple(Parameter(name=axis.parameter, domain="integer", lower=0) for axis in batch.axes),
        expressions=program.expressions + tuple(expressions),
        definitions=tuple(item.revise(node=batch) if item.id == "axis" else item for item in program.definitions))
    return selected.revise(construction=selected.construction.revise(program=program))


def test_a_narrowed_axis_is_a_new_workload_that_estimate_and_run_count_alike():
    """Rewriting the row axis of a 2 x 3 grid to [1, 2) keeps the three points of row 1, 3 shots each.

    The estimate and a run on Aer count the same points and shots for the full and the narrowed grid.
    """
    import nwqlib
    from nwqlib.ir import RangeAxis

    def value(estimate, name):
        fact = estimate.quantity(name).fact.value
        return fact.numerator // fact.denominator
    for start in (0, 1):
        chosen = revised_batch(axis_plan(1, method=AxisToy()), axes=(
            RangeAxis(parameter="row", start=start, stop=2), RangeAxis(parameter="col", start=0, stop=3)))
        points = 3 * (2 - start)
        folded = nwqlib.estimate(chosen)
        with prepare(chosen, settings="all", progress=False).run as run:
            run.resume()
            keys = list(run._state["workflow_items"])
            attempts = len(run.trace.events)
            shots = sum(chunk.returned_shots for chunk in run.observations.chunks)
        assert keys == [f"series[row={row},col={col}]" for row in range(start, 2) for col in range(3)]
        assert (value(folded, "settings"), value(folded, "shots")) == (attempts, shots) == (points, 3 * points)


def test_stepped_axes_count_order_and_match_their_points_and_keys_stay_distinct():
    """Three axes with steps 1, 3 and 4 give 18 points, the last axis fastest, each formed again from its values.

    A missing, off-grid or boolean value forms no point. Escaping both the backslash and the bracket keeps
    an experiment ``s\\`` with a row axis apart from an axis-free experiment named ``s[row=0]``.
    """
    from itertools import product
    from nwqlib.core.planning import point_key
    from nwqlib.ir import RangeAxis

    chosen = revised_batch(axis_plan(1), axes=(RangeAxis(parameter="a", start=0, stop=2),
                                               RangeAxis(parameter="b", start=1, stop=8, step=3),
                                               RangeAxis(parameter="c", start=2, stop=11, step=4)))
    points = list(chosen._static_points())
    expected = list(product(range(0, 2), range(1, 8, 3), range(2, 11, 4)))
    assert chosen._point_count() == len(points) == len(expected) == 18
    assert [tuple(binding.value for binding in point.bindings) for point in points] == expected
    for point in points:
        assert chosen._static_point("series", {item.parameter: item.value for item in point.bindings}) == point
    for values in ({"a": 0, "b": 2, "c": 2}, {"a": 0, "b": 1}, {"a": True, "b": 1, "c": 2}, {"a": 2, "b": 1, "c": 2}):
        assert chosen._static_point("series", values) is None
    assert point_key("s\\", (("row", 0),)) == "s\\\\[row=0]" != point_key("s[row=0]", ()) == "s\\[row=0]"


def test_fixed_shots_refuse_from_the_axis_bounds_and_point_dependent_shots_point_by_point(stub, monkeypatch):
    """A counts batch with constant repetitions requests them at every point, so its shots follow from the axes.

    Two axes of 2 and 3 values with 3 repetitions request 18 shots, refused at 17 before any point is checked
    against the backend target and accepted at 18. Repetitions of row + 1 depend on the point, so the six
    points are checked one by one and their 21 shots are refused at 20 after the scan.
    """
    import sys
    from nwqlib.ir import Binary, Constant, Expression, ExprRef, ParameterRef, RangeAxis

    checks = []
    target_for = StubBackend.target_for
    monkeypatch.setattr(StubBackend, "target_for", lambda self, observation: (checks.append(1),
                                                                           target_for(self, observation))[1])
    grid = revised_batch(axis_plan(1), axes=(RangeAxis(parameter="row", start=0, stop=2),
                                             RangeAxis(parameter="col", start=0, stop=3)))
    assert [point.key for point in grid._static_points()] == [
        f"series[row={row},col={col}]" for row in range(2) for col in range(3)]
    assert (grid._point_count(), grid._point_shots(grid.experiments[0])) == (6, 3)
    with pytest.raises(ValueError, match=r"max_total_shots=17 .* \(0 counted, 18 requested, at least 18 needed\)"):
        prepare(grid, backend=stub, limits=ExecutionLimits(max_total_shots=17), progress=False)
    assert checks == [] and StubBackend.calls == []
    prepare(grid, backend=stub, limits=ExecutionLimits(max_total_shots=18), progress=False).run.close()
    # One check in the scan, whose readout the six points share, and one more when the first point is prepared.
    assert len(checks) == 2 and len(StubBackend.calls) == 1

    checks.clear(), StubBackend.calls.clear()
    rising = revised_batch(axis_plan(6), repetitions=ExprRef(expression="reps"), expressions=(
        Expression(id="row", value=ParameterRef(parameter="row")), Expression(id="one", value=Constant(value=1)),
        Expression(id="reps", value=Binary(op="add", left="row", right="one"))))
    assert rising._point_shots(rising.experiments[0]) is None
    with pytest.raises(ValueError, match=r"max_total_shots=20 .* \(0 counted, 21 requested, at least 21 needed\)"):
        prepare(rising, backend=stub, limits=ExecutionLimits(max_total_shots=20), progress=False)
    assert len(checks) == 6 and StubBackend.calls == []
    prepare(rising, backend=stub, limits=ExecutionLimits(max_total_shots=21), progress=False).run.close()
    assert len(StubBackend.calls) == 1

    # A direct readout counts the shots its Method specializes for the Program's bindings, here 3 instead
    # of the 10 that the template declares, so a budget of 3 admits it and 2 refuses it.
    class Calibrated(ToyMethod):
        def specialize_experiment(self, plan, experiment, values):
            return experiment.revise(observation=experiment.observation.revise(shots=values["rep"]))

    from nwqlib.ir import Parameter
    base = plan(ToyProblem(), method=Calibrated(), shots=10, seed=19)
    program = base.construction.program.revise(parameters=(Parameter(name="rep", domain="integer", lower=1),))
    calibrated = base.revise(construction=base.construction.revise(program=program.bind(rep=3)))
    assert calibrated._point_shots(calibrated.experiments[0]) is None
    StubBackend.calls.clear()
    with pytest.raises(ValueError, match=r"max_total_shots=2 .* \(0 counted, 3 requested, at least 3 needed\)"):
        prepare(calibrated, backend=stub, limits=ExecutionLimits(max_total_shots=2), progress=False)
    prepare(calibrated, backend=stub, limits=ExecutionLimits(max_total_shots=3), progress=False).run.close()
    assert len(StubBackend.calls) == 1

    # More single-value axes than the recursion limit still give their one point.
    many = sys.getrecursionlimit() + 2
    single = revised_batch(axis_plan(1), axes=tuple(RangeAxis(parameter=f"a{index}", start=0, stop=1)
                                                     for index in range(many)))
    (point,) = single._static_points()
    assert single._point_count() == 1 and len(point.bindings) == many
