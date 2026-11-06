"""Static preparation keeps its original backend seed, and seed-specific calibrations keep their domains."""

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.core.planning import RandomStreams, RuntimeOptions
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def backend_draws(entropy, count):
    """Independent uint32 oracle: low then high halves of PCG64 raw words."""
    _, backend = np.random.SeedSequence(entropy).spawn(2)
    words = np.random.PCG64(backend).random_raw((count + 1) // 2)
    return [part for word in words for part in (int(word) & 0xFFFFFFFF, int(word) >> 32)][:count]


def forbidden(*args, **kwargs):
    raise AssertionError("static read/resume replayed native or numerical work")


def test_failed_static_preparation_keeps_original_seed_and_rejects_implicit_rebuild(
    tmp_path, monkeypatch, prepared_stubs
):
    plan = selected()
    expected = backend_draws(31, 1)[0]
    calls = []

    def fail(self, circuit, **kwargs):
        calls.append(kwargs["runtime"].seed)
        raise RuntimeError("interrupted native preparation")

    monkeypatch.setattr(InjectedBackend, "prepare", fail)
    path = tmp_path / "failed-preparation"
    with Run(plan, backend=InjectedBackend(), directory=path) as run:
        with pytest.raises(RuntimeError, match="interrupted native"):
            plan.method.prepare(plan, run=run)
        state = run.rng.snapshot()
        assert calls == [expected] and run.trace.preparations == 1
        assert run.prepared_artifacts == () and run.trace.events == ()
        with pytest.raises(RuntimeError, match="no replacement"):
            run.resume()
        assert calls == [expected] and run.rng.snapshot() == state
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.rng.snapshot() == state and restored.trace.preparations == 1
        with pytest.raises(RuntimeError, match="no replacement"):
            restored.resume()
        assert calls == [expected] and not restored.trace.events


@pytest.mark.parametrize("reopen", (False, True))
@pytest.mark.parametrize("templates", (True, False), ids=("row-templates", "one-by-one"))
def test_a_charge_refused_by_the_data_budget_keeps_its_seed_for_a_retry(tmp_path, reopen, templates, monkeypatch):
    """Eight sampled LCHS rows on local Aer: max_data_bytes refuses the first point's preparation charge.

    The budget admits the readout checks but not the charge, so nothing is prepared or acquired. After
    ``run.extend_limits``, on the same Run or after reopening it, every row has the seed of a Run whose
    budget was ample from the start, and the Result is the same. Row templates and preparing each point
    in turn restore the random stream at different places, so both routes run.
    """
    from nwqlib.execution import ExecutionLimits
    from test_ibm_backend import sampled_lchs_plan

    plan = sampled_lchs_plan()
    if not templates:
        monkeypatch.setattr(type(plan.method), "has_row_templates", lambda self, selected: False)
    with Run(plan, directory=tmp_path / "size", progress=False) as sizing:
        created = sum(sizing._state[key] for key in ("data_bytes", "array_bytes", "payload_bytes", "external_bytes"))

    def finished(folder, budget):
        run = Run(plan, directory=folder, limits=ExecutionLimits(max_data_bytes=budget), progress=False)
        try:
            if budget < 2_000_000:
                with pytest.raises(ValueError, match="max_data_bytes"):
                    run.resume()
                assert run.trace.preparations == len(run.trace.events) == 0
                if reopen:
                    run.close()
                    run = nwqlib.load_run(folder, progress=False)
                run.extend_limits(max_data_bytes=2_000_000)
            run.resume()
            return [item.runtime.seed for item in run._state["workflow_items"].values()], run.result.value
        finally:
            run.close()

    seeds, value = finished(tmp_path / "ample", 2_000_000)
    assert len(seeds) == 8 and finished(tmp_path / "retry", created + 256) == (seeds, value)


class _Stop(BaseException):
    """Interrupts a Run from its progress callback after three collected rows."""


def test_aer_binds_sampled_pair_rows_to_their_templates_with_each_rows_seed(tmp_path, monkeypatch):
    """Aer lowers and translates each sampled-pair template once and runs each row with its own seed.

    Eight rows share five templates, so five lowerings serve them. Every row is its own attempt that
    binds its values to the shared translated circuit and runs with its workflow row's seed, and the
    rows keep separate counts sources. Reopened after three rows, the Run completes the rest from the
    saved template circuit without lowering or translating again, with the same records.
    """
    from qiskit_aer import AerSimulator
    from nwqlib._counts import CountsSources
    from nwqlib._prepared_execution import _fixes_footprint
    from nwqlib.backends import qiskit_aer
    from nwqlib.blocks import lowering
    from test_ibm_backend import sampled_lchs_plan

    plan = sampled_lchs_plan(lcu_select_implementation="structured", pair_control_copies=2)
    # This built-in route's memory check keeps to the first point of an axis.
    assert all(_fixes_footprint(plan, experiment) for experiment in plan.experiments)
    lowered, translated, seeds = [], [], []
    lower, translate, simulate = lowering._lower_qiskit, qiskit_aer._lower_aer_circuit, AerSimulator.run
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *args, **kwargs: lowered.append(1) or lower(*args, **kwargs))
    monkeypatch.setattr(qiskit_aer, "_lower_aer_circuit",
                        lambda *args, **kwargs: translated.append(1) or translate(*args, **kwargs))
    monkeypatch.setattr(AerSimulator, "run", lambda self, circuit, **kwargs: seeds.append(
        kwargs.get("seed_simulator")) or simulate(self, circuit, **kwargs))

    def stop(stage, done, total):
        if stage == "collect" and done == 3:
            raise _Stop

    def records(run):
        items = run._state["workflow_items"]
        return {key: run.prepared_artifact(item.prepared_id) for key, item in items.items()}

    with pytest.raises(_Stop):
        with Run(plan, directory=tmp_path / "run", progress=stop) as run:
            run.resume()
    assert len(lowered) == 5 and len(seeds) == 3
    before = len(translated)
    with nwqlib.load_run(tmp_path / "run", progress=False) as run:
        run.resume()
        items = run._state["workflow_items"]
        assert run.result is not None and len(run.trace.submissions) == len(items) == 8
        assert (len(lowered), len(translated)) == (5, before)
        assert seeds == [item.runtime.seed for item in items.values()] and len(set(seeds)) == 8
        saved = records(run)
        sources = CountsSources(run.prepared_artifact, run._acquisition)
        assert len({sources._preparation(record.content_id).stream for record in saved.values()}) == 8
        # After reopening, the rows of a template restore one circuit, the one their estimate counts.
        restored = {}
        for handle in run._state["handles"].values():
            restored.setdefault(handle.record.template.key, set()).add(id(handle._restore_native().circuit))
        assert [len(circuits) for circuits in restored.values()] == 5 * [1]
        assert run._state["native_held"] == sum(handle._native_bytes for handle in run._state["handles"].values())
    assert all(record.template is not None and record.runtime == items[key].runtime
               and record.counts_sampling.seed == items[key].runtime.seed
               and record.probability_window_exclusions is None for key, record in saved.items())
    assert len({record.template.key for record in saved.values()}) == 5


def test_aer_prepares_the_rows_of_templates_it_cannot_bind_one_by_one(monkeypatch):
    """Aer prepares the rows of a template that needs its values, or that has none, on their own.

    The first template replaces a parameterized RZ by the equal HamiltonianGate(Z/2, t), whose
    translation needs t, and the second is split into templates without parameters. Their rows are
    prepared one by one, while the other three templates are still bound. Every row is charged and
    run once, with the seed that preparing all eight rows one by one draws (``backend_draws``).
    """
    from dataclasses import replace
    from qiskit.circuit.library import HamiltonianGate
    from nwqlib.blocks import lowering
    from test_ibm_backend import sampled_lchs_plan

    plan = sampled_lchs_plan(lcu_select_implementation="structured", pair_control_copies=2)
    ordinary = type(plan.method).row_templates

    def timed(template):
        circuit = template.circuit.copy()
        index = next(index for index, item in enumerate(circuit.data)
                     if item.operation.name == "rz" and item.operation.is_parameterized())
        item = circuit.data[index]
        circuit.data[index] = item.replace(operation=HamiltonianGate(np.diag([.5, -.5]), item.operation.params[0]))
        return replace(template, circuit=circuit)

    def concrete(template):
        return tuple(replace(template, key=f"{template.key}:{key}", parameters=(), rows=((key, ()),),
                             circuit=template.circuit.assign_parameters(dict(zip(template.parameters, values))))
                     for key, values in template.rows)

    split = {}

    def row_templates(self, selected, points):
        first, second, *rest = ordinary(self, selected, points)
        split.update(timed=[key for key, _ in first.rows], concrete=[key for key, _ in second.rows])
        return (timed(first), *concrete(second), *rest)

    lowered, lower = [], lowering._lower_qiskit
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *args, **kwargs: lowered.append(1) or lower(*args, **kwargs))
    monkeypatch.setattr(type(plan.method), "row_templates", row_templates)
    with Run(plan, progress=False) as run:
        run.resume()
        items = run._state["workflow_items"]
        records = {key: run.prepared_artifact(item.prepared_id) for key, item in items.items()}
        assert run.result is not None and len(run._state["preparation_charges"]) == len(run.trace.submissions) == 8
    alone = split["timed"] + split["concrete"]
    assert {key for key, record in records.items() if record.template is None} == set(alone)
    assert len(lowered) == 3 + len(alone)
    assert [item.runtime.seed for item in items.values()] == [record.runtime.seed for record in records.values()] \
        == backend_draws(11, 8)


def test_aer_asks_the_method_only_about_counts_points_and_streams_a_plan_without_templates(monkeypatch):
    """Aer groups only counts points, and a Plan whose Method has no templates runs its points as they come.

    An exact probability readout is never asked about or bound: alone, the Method is not asked, and
    next to a counts experiment it is asked about the counts points only. Each record matches its
    point's own preparation, with the seed drawn in Plan order. With the inherited
    ``Method.row_templates``, a Run of 48 range-axis points generates its first point twice before
    its first result, for the check of the known work and for its preparation, and so does
    ``prepare(settings="all")`` before that preparation.
    """
    from nwqlib.core.planning import Plan, ReadoutDetails
    from test_selected_runtime import AxisToy, axis_plan

    def probabilities(selected, batch):
        program = selected.construction.program
        definitions = tuple(item.revise(node=item.node.revise(observation_kind="probabilities", repetitions=1))
                            if item.id == batch else item for item in program.definitions)
        experiments = tuple(item.revise(readout=ReadoutDetails(qubits=(0,))) if item.batch == batch else item
                            for item in selected.experiments)
        return selected.revise(construction=selected.construction.revise(program=program.revise(
            definitions=definitions)), experiments=experiments)

    asked = []
    with monkeypatch.context() as patch:
        patch.setattr(AxisToy, "row_templates", lambda self, selected, points: asked.append(
            [point.key for point in points]))
        for selected, expected, kinds in ((probabilities(axis_plan(2, method=AxisToy()), "axis"), [], 2 * ["probabilities"]),
                                          (probabilities(axis_plan(2, flat="flat", method=AxisToy()), "flat"),
                                           [["series[row=0]", "series[row=1]"]], 2 * ["counts"] + ["probabilities"])):
            asked.clear()
            with Run(selected, progress=False) as run:
                run.resume()
                records = [run.prepared_artifact(item.prepared_id) for item in run._state["workflow_items"].values()]
            assert asked == expected and [record.observation.kind for record in records] == kinds
            assert all(record.template is None for record in records)
            assert [record.runtime.seed for record in records] == backend_draws(14, len(records))

    generated, points = [], Plan._static_points

    def counted(self, *args, **kwargs):
        for point in points(self, *args, **kwargs):
            generated.append(point.key)
            yield point

    def stop(stage, done, total):
        if stage in ("collect", "prepare") and done:
            raise _Stop

    monkeypatch.setattr(Plan, "_static_points", counted)
    with pytest.raises(_Stop):
        with Run(axis_plan(48, method=AxisToy()), progress=stop) as run:
            run.resume()
    assert generated == ["series[row=0]", "series[row=0]"]
    generated.clear()
    with pytest.raises(_Stop):
        nwqlib.prepare(axis_plan(48, method=AxisToy()), settings="all", progress=stop)
    assert generated == ["series[row=0]", "series[row=0]"]


def test_memory_of_a_static_plan_is_compared_per_circuit_for_the_first_and_as_a_whole_before_paid_work():
    """Five rows on Aer, each keeping one measured bit (200 bytes) and using one instruction (256 bytes) in passing.

    The first circuit needs 200 + 256 bytes and all five 5*200 + 256, the instruction list being temporary.
    A limit between the two lets the first circuit be prepared and inspected, while preparing everything or
    running the Plan refuses before any further preparation or any attempt. At the exact total the whole Plan
    runs, with a prepared circuit counted once, as held, and not again as remaining.

    Fixed shots do not fix the circuit. When a Repeat count reads the axis, the rows hold (100 + 200) and
    (200 + 200) bytes in either order, with a 2000-byte definition and three instructions at the largest,
    3468 in all. A definition with a parameter that reads the axis stays live while the next one is built,
    so two rows with 300 bytes each and two instructions need 2*300 + 2*256 + 2*1000. Two such calls whose
    arguments are row and 3*row share one definition at row 0, but rows 1 and 2 each build two new ones beside
    two of the row before, so three rows of (2*100 + 200) bytes and three instructions need
    3*400 + 3*256 + 2*1000 + 2*1000, though the first row alone suggests one definition less.
    """
    from qiskit import QuantumCircuit
    from nwqlib._prepared_execution import prepare_static
    from nwqlib.blocks import BlockSemantics, SelectedDefinition, select_zero_reflection
    from nwqlib.blocks.lowering import INSTRUCTION_BYTES
    from nwqlib.blocks.selection import NATIVE_POSITION_BYTES, SelectedBlock
    from nwqlib.core import Basis, Source
    from nwqlib.ir import (
        Argument, Binary, BlockCall, BlockSignature, Constant, Definition, ExprRef, Expression, Parameter,
        ParameterRef, PortMap, QuantumPort, Repeat,
    )
    from test_selected_runtime import AxisToy, axis_plan

    class Limited(AxisToy):
        max_bytes: int | None = None

    def rows_from_one(chosen, *, added, nodes, expressions, block, start=1):
        """Start the axis at ``start`` and insert the nodes ``added`` before the measurement of the Plan's one experiment."""
        program = chosen.construction.program
        definitions = []
        for item in program.definitions:
            if item.id == "body":
                item = item.revise(node=item.node.revise(children=("allocate", *added, "measure", "release")))
            if item.id == "axis":
                item = item.revise(node=item.node.revise(axes=(item.node.axes[0].revise(start=start, stop=3),)))
            definitions.append(item)
        program = program.revise(definitions=(*definitions, *nodes), signatures=(block.record.signature,),
                                 expressions=expressions)
        return chosen.revise(construction=chosen.construction.revise(program=program, selections=(block.record,)))._bind(
            blocks=(block,))

    def repeated(limit, *expressions):
        raw = select_zero_reflection("known", 1)
        block = SelectedBlock.bind(raw.record, payload=raw._payload, constructor=raw._constructor,
                                   native_bytes=100, definition_bytes=2000)
        nodes = (Definition(id="call", node=BlockCall(signature="known", ports=(PortMap(port="system", wire="q"),))),
                 Definition(id="repeat", node=Repeat(body="call", count=ExprRef(expression="count"))))
        return rows_from_one(axis_plan(2, method=Limited(max_bytes=limit)), added=("repeat",), nodes=nodes,
                             expressions=expressions, block=block)

    def counted_rows(limit):
        """The Repeat count is the row: 1, then 2."""
        return repeated(limit, Expression(id="count", value=ParameterRef(parameter="row")))

    def reversed_rows(limit):
        """The Repeat count is 3 - row: 2, then 1."""
        return repeated(limit, Expression(id="row", value=ParameterRef(parameter="row")),
                        Expression(id="three", value=Constant(value=3)), Expression(id="minus", value=Constant(value=-1)),
                        Expression(id="negative", value=Binary(op="multiply", left="row", right="minus")),
                        Expression(id="count", value=Binary(op="add", left="three", right="negative")))

    def rotation():
        source = Source(name="test.ry", version="1", domain="one-qubit rotation", reference="Ry(theta)")
        record = SelectedDefinition(signature=BlockSignature(name="rotation", target=source,
            quantum=(QuantumPort(name="q", width=1),), parameters=(Parameter(name="theta", domain="integer"),)),
            semantics=BlockSemantics(kind="unknown", input=None,
                basis=Basis(identity="computational", dimension=2, ordering="q0 rightmost"),
                relation="Ry(theta)", input_projector="identity", output_projector="identity",
                success="deterministic", workspace=0, restoration="none", epsilon=None,
                approximation_metric="operator norm", approximation_evidence="native roundoff unknown",
                inverse_legal=False, control_legal=False, phase="none"),
            implementation=source, choice="Ry", decomposition=None, cost_law=None, cost_context="one angle",
            construction_work=1)

        def build(selected, arguments, method_context):
            circuit = QuantumCircuit(1)
            circuit.ry(dict(arguments)["theta"], 0)
            return circuit

        return SelectedBlock.bind(record, constructor=build, native_bytes=100, definition_bytes=1000)

    def rotations(*arguments):
        return tuple(Definition(id=f"call-{argument}", node=BlockCall(signature="rotation", ports=(PortMap(port="q", wire="q"),),
            arguments=(Argument(parameter="theta", value=ExprRef(expression=argument)),))) for argument in arguments)

    def frontier_rows(limit):
        return rows_from_one(axis_plan(2, method=Limited(max_bytes=limit)), added=("call-angle",), nodes=rotations("angle"),
                             expressions=(Expression(id="angle", value=ParameterRef(parameter="row")),), block=rotation())

    def merged_rows(limit):
        """The two calls have theta = row and 3*row, so one key at row 0, then {1, 3} and {2, 6}."""
        return rows_from_one(axis_plan(3, method=Limited(max_bytes=limit)), added=("call-row", "call-tripled"), start=0,
                             nodes=rotations("row", "tripled"), block=rotation(),
                             expressions=(Expression(id="row", value=ParameterRef(parameter="row")),
                                          Expression(id="three", value=Constant(value=3)),
                                          Expression(id="tripled", value=Binary(op="multiply", left="row", right="three"))))

    for build, total, count in ((counted_rows, 3468, 2), (reversed_rows, 3468, 2), (frontier_rows, 3112, 2),
                                (merged_rows, 5968, 3)):
        if build is not reversed_rows:
            below = build(total - 1)
            with Run(below, progress=False) as run:
                with pytest.raises(ValueError, match=f"preparing every circuit of this Plan needs about {total} bytes"):
                    prepare_static(below, run=run, settings="all")
                assert run.trace.preparations == 0
        fits = build(total)
        with Run(fits, progress=False) as run:
            prepare_static(fits, run=run, settings="all")
            assert run.trace.preparations == count

    # With only the second row left, the Run holds the first one's definition and a new one comes beside it.
    with nwqlib.prepare(frontier_rows(3111), settings="first", progress=False).run as run:
        with pytest.raises(ValueError, match="running the remaining circuits of this Plan needs about 3112 bytes"):
            run.resume()
        assert run.trace.preparations == 1 and len(run.trace.events) == 0

    one, whole = NATIVE_POSITION_BYTES + INSTRUCTION_BYTES, 5 * NATIVE_POSITION_BYTES + INSTRUCTION_BYTES
    between = axis_plan(5, method=Limited(max_bytes=whole - 1))
    with nwqlib.prepare(between, settings="first", progress=False).run as run:
        assert run.trace.preparations == 1
        with pytest.raises(ValueError, match=f"running the remaining circuits of this Plan needs about {whole} bytes"
                                             rf".*max_bytes={whole - 1}"):
            run.resume()
        assert run.trace.preparations == 1 and len(run.trace.events) == 0
    with pytest.raises(ValueError, match=f"preparing every circuit of this Plan needs about {whole} bytes"):
        nwqlib.prepare(between, settings="all", progress=False)
    with pytest.raises(ValueError, match=f"preparing this circuit needs about {one} bytes"):
        nwqlib.prepare(axis_plan(5, method=Limited(max_bytes=one - 1)), settings="first", progress=False)
    exact = axis_plan(5, method=Limited(max_bytes=whole))
    prepared = nwqlib.prepare(exact, settings="first", progress=False)
    with prepared.run as run:
        assert run.wait().value == 5.
        assert run.trace.preparations == len(run.trace.events) == 5


def test_template_rows_are_compared_before_their_charge_and_share_a_circuit_until_their_last_row_goes(tmp_path, monkeypatch):
    """Each row keeps 32768 bytes and each template one circuit of 100000 + 200 bytes, with one definition of 2000.

    Eight rows of two templates need 2*100200 + 8*32768 + 2*256 + 2000 before anything is charged, one
    circuit more than the first check of the whole Plan can know. A row that reuses a compiled template
    builds no circuit, so it needs only its 32768 bytes. The circuit stays until the last of its rows is
    released, also when that row did not compile it, and a restored row reuses a circuit still in memory.

    A template that states its size, here 5000 bytes, is built only after that comparison. Its temporary
    adds three of its own size and two of the template before it (the live containers counted on
    Qiskit 2.6.0 and Aer 0.18.0), so two templates add 3*5000 + 2*5000 at the second. A new template is built
    once, a compiled one never, and each root is gone before the next one is built. When the backend cannot
    bind the template that was built, its rows are prepared on their own with the seeds that preparing them
    one by one draws.
    """
    import dataclasses
    import gc
    import weakref
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter as QiskitParameter
    from qiskit.circuit.library import HamiltonianGate
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib._prepared_execution import _row_templates, prepare_static, prepare_template_rows, submit_experiment
    from nwqlib.algorithms.protocol import RowTemplate, _DeferredRowTemplate
    from nwqlib.blocks import select_zero_reflection
    from nwqlib.blocks.selection import SelectedBlock
    from nwqlib.ir import BlockCall, Definition, PortMap
    from test_selected_runtime import AxisToy, axis_plan

    roots = []

    class Templated(AxisToy):
        max_bytes: int | None = None
        groups: int = 1
        declared: int | None = None
        unbindable: bool = False

        def row_templates(self, selected, points):
            theta = QiskitParameter("theta")

            def build():
                gc.collect()
                assert all(root() is None for root in roots)
                circuit = QuantumCircuit(1)
                if self.unbindable:
                    circuit.append(HamiltonianGate(np.diag([.5, -.5]), theta), [0])
                else:
                    circuit.rz(theta, 0)
                roots.append(weakref.ref(circuit))
                return circuit

            common = [dict(key=f"group-{group}", block_id=selected.blocks[0].record.content_id, parameters=(theta,),
                           rows=tuple((point.key, (.1 * (index + 1),)) for index, point in enumerate(points)
                                      if index % self.groups == group))
                      for group in range(self.groups) if group < len(points)]
            if self.declared is not None:
                return tuple(_DeferredRowTemplate(**fields, build=build, retained_bytes=self.declared) for fields in common)
            circuit = QuantumCircuit(1)
            circuit.rz(theta, 0)
            return tuple(RowTemplate(**fields, circuit=circuit) for fields in common)

    def rows(count, *, groups=1, limit=None, declared=None, unbindable=False):
        raw = select_zero_reflection("known", 1)
        block = SelectedBlock.bind(raw.record, payload=raw._payload, constructor=raw._constructor,
                                   native_bytes=100_000, definition_bytes=2000)
        chosen = axis_plan(count, method=Templated(max_bytes=limit, groups=groups, declared=declared, unbindable=unbindable))
        program = chosen.construction.program
        definitions = [item.revise(node=item.node.revise(children=("allocate", "call", "measure", "release")))
                       if item.id == "body" else item for item in program.definitions]
        definitions.append(Definition(id="call", node=BlockCall(signature="known", ports=(PortMap(port="system", wire="q"),))))
        return chosen.revise(construction=chosen.construction.revise(
            program=program.revise(definitions=tuple(definitions), signatures=(block.record.signature,)),
            selections=(block.record,)))._bind(blocks=(block,))

    row, circuit, instructions, definition, size = 32768, 100_200, 2 * 256, 2000, 5000
    known = 2 * circuit + 8 * row + instructions + definition
    for declared, total, builds in ((None, known, 0), (size, known + 5 * size, 2)):
        roots.clear()
        below = rows(8, groups=2, limit=total - 1, declared=declared)
        with Run(below, progress=False) as run:
            with pytest.raises(ValueError, match=f"preparing the rows of these templates needs about {total} bytes"):
                prepare_static(below, run=run, settings="all")
            assert run.trace.preparations == 0 and not roots
        fits = rows(8, groups=2, limit=total, declared=declared)
        with Run(fits, progress=False) as run:
            prepare_static(fits, run=run, settings="all")
            assert run.trace.preparations == 8 and run._state["native_held"] == 2 * circuit + 8 * row
            assert len(roots) == builds

    # Below the 100200 bytes of a circuit, a second row of a compiled template still fits, and a third does not.
    limit = circuit + 2 * row
    for declared, builds in ((None, 0), (size, 1)):
        roots.clear()
        tight = rows(8, limit=limit, declared=declared)
        points = tuple(tight._static_points())
        with Run(tight, progress=False) as run:
            prepare_static(tight, run=run, settings="first")
            assert run._state["native_held"] == circuit + row
            prepare_template_rows(tight, run=run, points=points[:2], templates=_row_templates(tight, run, points[:2]))
            assert run.trace.preparations == 2 and run._state["native_held"] == limit and len(roots) == builds
            with pytest.raises(ValueError, match=f"preparing the rows of these templates needs about {limit + row} bytes"):
                prepare_template_rows(tight, run=run, points=points[:3], templates=_row_templates(tight, run, points[:3]))
            assert run.trace.preparations == 2 and len(roots) == builds

    roots.clear()
    timed = rows(4, declared=size, unbindable=True)
    with Run(timed, progress=False) as run:
        run.resume()
        records = [run.prepared_artifact(item.prepared_id) for item in run._state["workflow_items"].values()]
        assert run.result is not None and run.trace.preparations == len(run.trace.submissions) == 4
    assert len(roots) == 1 and all(record.template is None for record in records)
    assert [record.runtime.seed for record in records] == backend_draws(14, 4)

    def never():
        raise AssertionError("built a circuit")

    deferred = _DeferredRowTemplate(key="k", block_id="b", parameters=(), rows=(), build=never, retained_bytes=size)
    assert deferred == deferred and deferred != RowTemplate("k", "b", None, (), ())
    assert repr(deferred) and hash(deferred) == hash(deferred)
    shared = QuantumCircuit(1)
    given = dataclasses.replace(deferred, circuit=shared)
    assert given.circuit is shared and given._retained_bytes is None

    four = rows(4)
    points = tuple(four._static_points())
    with Run(four, directory=tmp_path / "rows", progress=False) as run:
        prepared = prepare_template_rows(four, run=run, points=points, templates=_row_templates(four, run, points))
        first, second, third, fourth = (prepared[point.key] for point in points)
        shared = third._native.circuit
        assert first._native.circuit is shared and run._state["native_held"] == circuit + 4 * row
        for handle in (first, second):
            run.collect(submit_experiment(handle, run=run))
        # The first row compiled the circuit, and the other two rows still run it.
        assert len(run.release_native()["released"]) == 2
        assert run._state["native_held"] == circuit + 2 * row
        with monkeypatch.context() as patch:
            patch.setattr(ArchiveFiles, "read_circuit", lambda *a: pytest.fail("decoded a circuit in memory"))
            first.inspect_circuit()
        assert first._native.circuit is shared and run._state["native_held"] == circuit + 3 * row
        for handle in (third, fourth):
            run.collect(submit_experiment(handle, run=run))
        assert len(run.release_native()["released"]) == 3
        assert run._state["native_held"] == 0
        second.inspect_circuit()
        fourth.inspect_circuit()
        assert second._native.circuit is fourth._native.circuit and second._native.circuit is not shared
        assert run._state["native_held"] == circuit + 2 * row


def test_fixed_seed_calibration_and_seed_independent_forecast_keep_their_domains(monkeypatch):
    from _profile_fixtures import stored_inputs
    from nwqlib.backends.assessment import assess, estimate_plan

    plan, point, profile, allocation, context, bare, at = stored_inputs(calibrated=True)
    state = plan.randomness
    monkeypatch.setattr(RandomStreams, "next_seed", forbidden)
    independent = profile.models[0]
    seven, eight = RuntimeOptions(seed=7), RuntimeOptions(seed=8)
    models = tuple(
        independent.revise(name=name, domain=independent.domain.revise(runtime=runtime))
        for name, runtime in (("seed seven", seven), ("seed eight", eight))
    ) + (independent,)
    profile = profile.revise(models=models)
    for runtime, expected in (
        (seven, (True, False, True)),
        (eight, (False, True, True)),
        (None, (False, False, True)),
    ):
        assessment = assess(
            plan,
            point,
            profile=profile,
            allocation=allocation,
            context=context,
            runtime=runtime,
            assessed_at=at,
        )
        assert assessment.point.runtime_options_id == (
            None if runtime is None else runtime.content_id
        )
        assert (
            tuple(prediction.seconds is not None for prediction in assessment.predictions)
            == expected
        )
        # Supplied model arithmetic: 1 invocation +2 exact evaluation +3*.5 gates.
        assert all(
            prediction.seconds.value == 4.5
            for prediction in assessment.predictions
            if prediction.seconds is not None
        )
    forecast = estimate_plan(
        plan, profile=profile, allocation=allocation, context=context, assessed_at=at
    )
    assert forecast.assessments[0].point.runtime_options_id is None
    assert tuple(p.seconds is not None for p in forecast.assessments[0].predictions) == (
        False,
        False,
        True,
    )
    assert tuple(model.domain.runtime for model in profile.models) == (
        seven,
        eight,
        "seed_independent",
    )
    assert plan.randomness == state
