"""Observation-point schedule: required relations, the items/bytes law and adapter rejection.

The relations and the law are those stated in ``core/planning.py``. Expected
counts and bytes below are written by hand from that law.
"""

from types import SimpleNamespace

import pytest
from qiskit import QuantumCircuit

from nwqlib.core import planning
from nwqlib.core.planning import (
    Experiment, ObservationPoint, ObservationSpec, ReadoutView, Reducer, ReducerOutput, RuntimeOptions,
    readout_shape,
)
from nwqlib._prepared_execution import Run, prepare_experiment as prepare
from nwqlib.execution import ObservationView


def pauli(point_id, position, *labels, view=None):
    return ObservationPoint(id=point_id, position=position, kind="pauli_expectation", labels=labels, view=view)


def trajectory(*points, **fields):
    return ObservationSpec(kind="trajectory", positions=points, **fields)


@pytest.fixture
def two_complex_scalars(monkeypatch):
    """A test-local reducer publishing two complex128 scalars; its private workspace is not output."""
    monkeypatch.setitem(planning.READOUT_REDUCERS, "two_complex",
                        Reducer(shape=lambda parameters: (ReducerOutput("complex128", (2,)),)))


def test_items_law_on_the_separating_example(two_complex_scalars):
    """Three real Pauli values, an eight-outcome marginal and two complex output scalars."""
    spec = trajectory(
        pauli("p", 0, "XII", "IZI", "IIY"),
        ObservationPoint(id="m", position=1, kind="probabilities", qubits=(0, 1, 2)),
        ObservationPoint(id="r", position=2, kind="reduction", reducer="two_complex", parameters={"k": 1}),
    )
    # N_items = 3 + 2**3 + 2 = 13 logical items.
    assert readout_shape(kind="trajectory", details=spec, width=3, classical_width=0) == 13
    # The last point exceeds the one item left after 3 + 8 of 12.
    with pytest.raises(ValueError, match="point 'r' can have 2 items, more than max_items=1"):
        readout_shape(kind="trajectory", details=spec, width=3, classical_width=0, max_items=12)


def _fresh_process_json(source):
    """Run ``source`` in a new interpreter of this test's environment and parse its last output line."""
    import json
    import os
    import subprocess
    import sys
    completed = subprocess.run([sys.executable, "-c", source], capture_output=True, text=True, env=dict(os.environ),
                               timeout=300)
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.splitlines()[-1])


def test_builtin_reducers_resolve_by_name_in_a_process_that_imported_only_the_registry():
    """A saved record names its reducer, and the process reading it may not have imported the reducer's module."""
    import nwqlib
    names = _fresh_process_json(
        "import importlib, json\n"
        "from nwqlib.core import planning\n"
        "for module in planning.BUILTIN_REDUCER_MODULES:\n"
        "    importlib.import_module(module)\n"
        "print(json.dumps(sorted(planning.READOUT_REDUCERS)))")
    assert names
    for name in names:
        # The child checks that no built-in reducer module is imported before
        # the lookup, so the lookup itself must resolve the name.
        origin, unimported, resolved = _fresh_process_json(
            "import json, sys\n"
            "import nwqlib.core.planning as planning\n"
            "import nwqlib\n"
            "unimported = not any(module in sys.modules for module in planning.BUILTIN_REDUCER_MODULES)\n"
            f"reducer = planning.registered_reducer({name!r})\n"
            "print(json.dumps([nwqlib.__file__, unimported, type(reducer) is planning.Reducer and reducer.executable]))")
        assert origin == nwqlib.__file__ and unimported and resolved, name


@pytest.mark.parametrize("case", ["variant reduction before the end", "invariant reduction before the end"])
def test_nwqsim_serves_only_phase_invariant_reductions_before_the_last_boundary(case, state_reducer, tmp_path,
                                                                               monkeypatch):
    """NWQ-Sim CPU/SV has no per-segment phase ledger. A reduction before the last boundary is refused
    before the runner or the transpiler is called unless its reducer is registered phase invariant; a
    phase-invariant reducer is served.
    """
    import json
    import qiskit
    from nwqlib.backends import NWQSimBackend, nwqsim

    calls = []
    described = dict(format=nwqsim.NWQSIM_FORMAT, backends=["CPU"], methods=["SV"], distributed=False,
                     readouts=["counts", "pauli", "probabilities", "amplitudes", "trajectory"],
                     nwqsim_revision="efd02262ff9c5def4f2eac1df6416c1ee2023d5b", compiler="supplied")
    monkeypatch.setattr(nwqsim.subprocess, "run",
                        lambda *args, **kwargs: calls.append("runner") or SimpleNamespace(stdout=json.dumps(described)))
    transpile = qiskit.transpile
    monkeypatch.setattr(qiskit, "transpile", lambda *args, **kwargs: calls.append("transpile") or transpile(*args, **kwargs))
    invariant = case.startswith("invariant")
    monkeypatch.setitem(planning.READOUT_REDUCERS, "norm_first",
                        planning.READOUT_REDUCERS["norm_first"].__class__(
                            shape=planning.READOUT_REDUCERS["norm_first"].shape,
                            work=planning.READOUT_REDUCERS["norm_first"].work,
                            execute=planning.READOUT_REDUCERS["norm_first"].execute, phase_invariant=invariant))
    schedule = trajectory(reduction("state", 1), pauli("end", 3, "ZZ"))
    plan = three_call_plan(schedule)
    backend = NWQSimBackend(executable="/supplied/runner", spool=str(tmp_path), max_input_bytes=1 << 20,
                            max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)
    circuit = QuantumCircuit(2)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(1)
    with Run(plan) as run:
        arguments = dict(observation=schedule, runtime=RuntimeOptions(seed=7), position=(1, 3), source_definitions=(),
                         run=run, snapshot="points")
        if invariant:
            prepared = backend.prepare(circuit, **arguments)
            assert prepared.native.metadata["points"] == (("state", "state", None), ("end", "pauli", None))
            assert calls.count("transpile") == 1
            return
        match = ("no phase ledger for the reduction point 'state' before the last boundary.*move the point to the last "
                 "boundary, or register the reducer with phase_invariant=True")
        with pytest.raises(ValueError, match=match):
            backend.prepare(circuit, **arguments)
    assert "transpile" not in calls
    # The same refusal comes from target_for, before a Run charges a preparation.
    with Run(plan, backend=backend, directory=tmp_path / "run") as run:
        with pytest.raises(ValueError, match=match):
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_probability_total_is_the_correctly_rounded_sum_of_the_saved_values():
    """The saved mass is fsum of all values: above fl(1+w) at the endpoint, and not a sum of per-block sums."""
    from fractions import Fraction
    from math import fsum, ulp

    import numpy as np

    from nwqlib._validation import exact_probability_window
    from nwqlib.artifacts import _FINITE_CHECK_ENTRIES, probability_readout

    # Exact total T + 3*ulp(T)/4 rounds up to the next float after the
    # endpoint T = fl(1+w); a left-to-right float sum stays at T.
    endpoint = 1 + exact_probability_window(3, 2)
    small = 3 * ulp(endpoint) / 8
    _, _, summary = probability_readout(np.array([endpoint, small, small, 0.0]), width=2)
    assert summary["mass"] == np.nextafter(endpoint, 2) > endpoint

    # Each block's residual is below half an ulp of its own sum, so
    # per-block fsums added give 0.75 + c, while the exact total exceeds it
    # by 9*2**-57, more than half an ulp of 0.75.
    step = _FINITE_CHECK_ENTRIES
    values = np.full(2 * step, 2.0**-30)
    values[[0, 1, step, step + 1]] = 0.5, 3 * 2.0**-56, 0.25, 3 * 2.0**-57
    _, stored, summary = probability_readout(values, width=(2 * step).bit_length() - 1)
    assert len(stored) == 2 * step
    exact = float(sum(map(Fraction, values.tolist())))
    blockwise = fsum(values[:step].tolist()) + fsum(values[step:].tolist())
    assert summary["mass"] == fsum(values.tolist()) == exact == np.nextafter(blockwise, 1)


def test_probability_points_reserve_eight_bytes_per_possible_outcome():
    """Dense or adaptively sparse probability arrays reserve ``8 * 2**q_k`` bytes per point (``V_k``)."""
    from nwqlib._prepared_execution import _trajectory_value_bounds

    schedule = trajectory(ObservationPoint(id="first", position=0, kind="probabilities", qubits=(0,)),
                          ObservationPoint(id="end", position=None, kind="probabilities", qubits=(0,)))
    assert _trajectory_value_bounds(schedule, width=1, capacity=None, max_bytes=1 << 40) == [16, 16]


def test_restoring_a_trajectory_checks_its_reservation_against_the_cap_not_the_remaining_bytes():
    """A Run at its data limit still restores a saved trajectory preparation, admitted when it was prepared."""
    from nwqlib.execution import ExecutionLimits
    from nwqlib._prepared_execution import _admit_readout

    schedule = trajectory(ObservationPoint(id="first", position=0, kind="probabilities", qubits=(0,)),
                          ObservationPoint(id="end", position=None, kind="probabilities", qubits=(0,)))
    full = SimpleNamespace(limits=ExecutionLimits(max_data_bytes=4096), _available_data_bytes=lambda: 0)
    assert _admit_readout(schedule, width=1, classical_width=0, run=full) == 4


def test_max_data_bytes_of_only_the_probability_arrays_refuses_before_native_preparation(monkeypatch):
    """Both probability arrays and the fixed bytes are reserved, so a budget of exactly the arrays refuses.

    On 12 qubits the two full-width points need 2 * 8 * 2**12 = 65,536 bytes
    of arrays besides their declarations and headers. Charging only one
    array, or neither the declarations nor the headers, would fit that
    budget, so the refusal shows that all of them are counted. It comes
    before native preparation, and a budget of 131,072 bytes reaches native
    preparation.
    """
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_product

    width = 12
    base = nwqlib.plan(Expectation(state=ingest_product([(.6, .8)] * width),
                                   observable=ingest_pauli([("Z" * width, 1.)], num_qubits=width)),
                       method=ExpectationMethod(), seed=23)
    marginals = tuple(ObservationPoint(id=name, position=position, kind="probabilities", qubits=tuple(range(width)))
                      for name, position in (("early", 0), ("late", None)))
    chosen = base.revise(experiments=(Experiment(name="read", setting="two", observation=trajectory(*marginals)),))
    chosen = chosen._bind(blocks=base.blocks)
    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native preparation"))
    with Run(chosen, limits=ExecutionLimits(max_data_bytes=2 * 8 * 2**12)) as run:
        with pytest.raises(ValueError, match="storage exceeds remaining max_data_bytes"):
            prepare(chosen.resolve("read"), run=run, runtime=RuntimeOptions(seed=23))
        assert run.trace.preparations == 0

    class Reached(Exception):
        pass

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: (_ for _ in ()).throw(Reached()))
    with Run(chosen, limits=ExecutionLimits(max_data_bytes=2 * 8 * 2**12 + 65_536)) as run:
        with pytest.raises(Reached):
            prepare(chosen.resolve("read"), run=run, runtime=RuntimeOptions(seed=23))


def three_call_plan(schedule, *others, state=(1., 2., 3., 4.)):
    """A two-qubit Expectation Plan whose body applies its state preparation three times.

    The bound body has three operations, so points can observe after the
    first, second and third of them; ``others`` are more experiments.
    """
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.ir import Definition, Sequence
    from nwqlib.operators import ingest_pauli

    base = nwqlib.plan(Expectation(state=list(state), observable=ingest_pauli((("ZZ", 1.),), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    program = base.construction.program
    program = program.revise(definitions=tuple(item for item in program.definitions if item.id != "root") + (
        Definition(id="root", node=Sequence(children=("allocate", "prepare", "prepare", "prepare"))),))
    experiments = (Experiment(name="trajectory", setting="points", observation=schedule),) + others
    return base.revise(construction=base.construction.revise(program=program), experiments=experiments)._bind(
        blocks=base.blocks)


def prefix_values(plan, chunk, boundary):
    """Point values of an independently evaluated logical prefix (qiskit.quantum_info)."""
    from qiskit.quantum_info import SparsePauliOp, Statevector
    from nwqlib.blocks.lowering import lower_qiskit

    _, construction = plan.resolve("trajectory")._selected_construction(plan)
    selected = {record.content_id for record in construction.selections}
    logical = lower_qiskit(construction, blocks=tuple(block for block in plan.blocks
                                                      if block.record.content_id in selected)).circuit
    prefix = logical.copy_empty_like()
    prefix.data = logical.data[:boundary]
    state = Statevector(prefix)
    if chunk.observation.kind == "pauli_expectation":
        return {item.label: state.expectation_value(SparsePauliOp(item.label)).real for item in chunk.values}
    probabilities = state.probabilities(list(chunk.observation.qubits))
    return {key: probabilities[int(key, 2)] for key in _chunk_values(chunk)}


def _chunk_values(chunk):
    """Point values by Pauli label or, for a marginal, by bit-string key (qubits[0] rightmost)."""
    if chunk.observation.kind == "pauli_expectation":
        return {item.label: item.value for item in chunk.values}
    histogram = chunk.histogram()
    return {format(index, f"0{histogram.width}b"): value
            for index, value in zip(histogram.index_list(), histogram.weights.tolist())}


def receipt_comparison_allowance(receipts):
    """Compare readouts through their common ideal prefix by adding both state-error contributions and both
    absolute scalar-evaluation allowances. An unavailable state error leaves this derived comparison unavailable.

    ``T = sum_i [2 delta_i + delta_i**2 + e_i (1+delta_i)**2]`` with
    ``e_i = exact_readout_roundoff(w_i) * u``, ``w_i`` the full native width;
    a first-order qualified relation, not an all-orders certificate.
    """
    from nwqlib._validation import exact_readout_roundoff, UNIT_ROUNDOFF
    total = 0.0
    for receipt in receipts:
        delta, reason = receipt.state_error()
        if delta is None:
            return None, reason
        e = exact_readout_roundoff(len(receipt.logical_to_native)) * UNIT_ROUNDOFF
        total += 2 * delta + delta * delta + e * (1 + delta) * (1 + delta)
    return total, None


def test_aer_trajectory_observes_every_point_of_one_simulation(monkeypatch):
    """Pauli and marginal points of one coherent state, in one simulation, against separate prefixes.

    The independently evaluated host prefix has no native receipt, so it uses
    the predeclared regression threshold 2e-12 (rtol=0) at at most three
    qubits. The separately acquired Aer prefix has its own receipt, so that
    comparison uses the receipt allowance ``T`` above.
    """
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.core.planning import ObservationSpec
    from nwqlib._prepared_execution import submit_experiment

    schedule = trajectory(
        pauli("first", 1, "XI", "ZZ", "IY"),
        ObservationPoint(id="marginal", position=2, kind="probabilities", qubits=(1, 0)),
        pauli("second", 2, "XI", "YX"),
        # The same label at the same boundary shares its saved datum.
        pauli("again", 2, "XI"),
        ObservationPoint(id="end", kind="probabilities", qubits=(0,)),
    )
    separate = Experiment(name="prefix", setting="first alone",
                          observation=ObservationSpec(kind="pauli_expectation", position=1, labels=("XI", "ZZ", "IY")))
    early = Experiment(name="early", setting="before the end", observation=trajectory(pauli("only", 2, "XI")))
    plan = three_call_plan(schedule, separate, early)
    simulations = []
    submit = aer._submit_aer_execution
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: simulations.append(native) or submit(native))
    with Run(plan) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        receipt = handle.record
        # The end shorthand resolves to the bound body length before native preparation.
        assert (receipt.body_length, receipt.boundaries) == (3, (1, 2, 2, 2, 3))
        saves = [item.operation.name for item in handle._native.circuit.data if item.operation.name.startswith("save")]
        assert saves.count("save_expval") == 5 and saves.count("save_probabilities") == 2
        # G counts the native evolution operations and no save instruction.
        assert receipt.native_operations == len(handle._native.circuit.data) - len(saves)
        # Nothing runs after the last save.
        assert handle._native.circuit.data[-1].operation.name == "save_probabilities"
        chunks = submit_experiment(handle, run=run)
        assert len(simulations) == 1 and run.trace.jobs == 1 and len(run.trace.events) == 1
        event = run.trace.events[0]
        # The one completed event binds the ordered point collection.
        assert event.evaluations == 1 and event.observation_id == ObservationView(chunks=chunks).content_id
        assert [(chunk.point, chunk.boundary) for chunk in chunks] == list(zip(
            ("first", "marginal", "second", "again", "end"), receipt.boundaries))
        assert len({chunk.acquisition_key for chunk in chunks}) == 5 and len({chunk.attempt for chunk in chunks}) == 1
        for chunk in chunks:
            reference = prefix_values(plan, chunk, chunk.boundary)
            values = _chunk_values(chunk)
            assert values.keys() == reference.keys()
            for key, value in values.items():
                assert abs(value - reference[key]) <= 2e-12
        # A chunk joins its receipt only at the point's resolved boundary.
        with pytest.raises(ValueError, match="resolved point boundary"):
            chunks[-1].revise(boundary=2).validate_unit_bound(receipt)
        with pytest.raises(ValueError, match="receipt.s trajectory declaration"):
            chunks[-1].revise(trajectory_id="sha256:" + "1" * 64).validate_unit_bound(receipt)
        # Nothing after the last observation is executed.
        stopped = prepare(plan.resolve("early"), run=run, runtime=RuntimeOptions(seed=7))
        assert stopped._native.circuit.data[-1].operation.name == "save_expval"
        single = prepare(plan.resolve("prefix"), run=run, runtime=RuntimeOptions(seed=7))
        # A single-endpoint coherent readout also leaves its three saves out of G.
        assert single.record.native_operations == len(single._native.circuit.data) - 3
        alone = submit_experiment(single, run=run)
        allowance, reason = receipt_comparison_allowance((receipt, run.prepared_artifact(alone.prepared_id)))
        assert reason is None
        first = {item.label: item.value for item in chunks[0].values}
        for item in alone.values:
            assert abs(first[item.label] - item.value) <= allowance


def test_reopened_trajectory_keeps_every_point_association_without_replay(tmp_path, monkeypatch):
    """Reopening restores all point chunks and the native saves of every kind; nothing runs again."""
    import nwqlib
    from nwqlib.backends import AerBackend, qiskit_aer as aer
    from nwqlib.blocks import lowering
    from nwqlib._prepared_execution import submit_experiment

    schedule = trajectory(pauli("first", 1, "XI", "ZZ"),
                          ObservationPoint(id="marginal", position=2, kind="probabilities", qubits=(1, 0)),
                          pauli("end", None, "YX", "XI"))
    plan = three_call_plan(schedule)
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        chunks = submit_experiment(handle, run=run)
        # One point is collected; the others stay completed and uncollected.
        run.collect(chunks[1])
    calls = []
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: calls.append(native))
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *args, **kwargs: pytest.fail("reopening lowered again"))
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        assert reopened.completed_observation(chunks[0].attempt) == chunks
        assert reopened.observations.chunks == (chunks[1],)
        receipt = reopened.prepared_artifact(handle.record.content_id)
        for chunk in chunks:
            reopened.validate_observation(chunk)
            chunk.validate_unit_bound(receipt)
        assert reopened.trace.events[0].observation_id == ObservationView(chunks=chunks).content_id
        # Telemetry joins the event to its ordered point collection once every point is collected.
        from nwqlib.backends.telemetry import align_telemetry
        assert align_telemetry(reopened).rows[0].collected_observation_ids == ()
        for chunk in chunks:
            reopened.collect(chunk)
        assert align_telemetry(reopened).rows[0].collected_observation_ids == (reopened.trace.events[0].observation_id,)
        native = reopened._state["handles"][receipt.content_id]._restore_native()
    assert calls == []
    monkeypatch.undo()
    # The QPY-restored circuit keeps every save, of both kinds, at its native position.
    assert [item.operation.name for item in native.circuit.data] == [
        item.operation.name for item in handle._native.circuit.data]
    returned = aer._submit_aer_execution(native).raw_output["trajectory"]
    for chunk in chunks:
        values = _chunk_values(chunk)
        stored = returned[chunk.point].get("pauli_expectations")
        if stored is None:
            marginal = returned[chunk.point]["probabilities"]
            stored = {format(index, f"0{len(chunk.observation.qubits)}b"): float(value)
                      for index, value in enumerate(marginal)}
        assert stored == values


def test_trajectory_refuses_a_resetting_selected_body_before_native_work(monkeypatch):
    """Zero raw shots admit no reset along the executed prefix or in a view, refused before the charge.

    A reset that starts before the last point's boundary, or in a view's
    tail, is refused by the selected IR with nothing charged. A reset that
    starts at the last boundary is outside the executed prefix, so that body
    is admitted and prepared without the reset, as is the body without it.
    """
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.ir import Definition, Reset, Sequence

    def body(schedule, *children):
        base = three_call_plan(schedule)
        program = base.construction.program
        program = program.revise(definitions=tuple(item for item in program.definitions if item.id != "root") + (
            Definition(id="reset", node=Reset(wire="system")),
            Definition(id="root", node=Sequence(children=children)),))
        return base.revise(construction=base.construction.revise(program=program))._bind(blocks=base.blocks)

    resetting = ("allocate", "prepare", "reset", "prepare")
    viewed = pauli("viewed", 1, "ZZ", view=ReadoutView(tail="reset", inverse="prepare", wires=(0, 1)))
    cases = (
        (trajectory(pauli("first", 1, "ZZ"), pauli("end", None, "XI")), resetting, 0,
         "selected IR trajectory cannot observe the selected pure prefix: reset of register 'system' in "
         "definition 'reset'"),
        (trajectory(pauli("first", 1, "ZZ"), pauli("next", 2, "XI")), resetting, 0,
         "selected IR trajectory cannot observe the selected pure prefix"),
        (trajectory(viewed, pauli("end", None, "XI")), ("allocate", "prepare", "prepare", "prepare"), 0,
         "trajectory point 'viewed' has an unsupported coherent view: selected IR reset of register 'system'"),
    )
    for schedule, children, charged, message in cases:
        plan = body(schedule, *children)
        with Run(plan) as run, monkeypatch.context() as patch:
            patch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
            with pytest.raises(ValueError, match=message):
                prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
            assert run.trace.preparations == charged
    admitted = (
        (trajectory(pauli("first", 1, "ZZ")), resetting, (1,)),
        (trajectory(pauli("first", 1, "ZZ"), pauli("end", None, "XI")), ("allocate", "prepare", "prepare"), (1, 2)),
    )
    for schedule, children, boundaries in admitted:
        plan = body(schedule, *children)
        with Run(plan) as run:
            handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
            assert run.trace.preparations == 1 and handle.record.boundaries == boundaries
            assert "reset" not in handle._native.circuit.count_ops()


@pytest.mark.parametrize("body", ["initialize", "unitary composite"])
def test_aer_trajectory_admits_only_a_coherent_body_including_composite_definitions(body, monkeypatch):
    """Initialize, whose definition resets, is refused before lowering; a composite of unitary
    gates is lowered.
    """
    from qiskit.circuit.library import Initialize
    from nwqlib.backends import AerBackend, qiskit_aer as aer

    circuit = QuantumCircuit(2)
    circuit.h(0)
    if body == "initialize":
        circuit.append(Initialize([0, 1, 0, 0]), (0, 1))
    else:
        inner = QuantumCircuit(2, name="unitary_composite")
        inner.cx(0, 1)
        circuit.append(inner.to_instruction(), (0, 1))
    circuit.x(1)
    schedule = trajectory(pauli("first", 1, "ZZ"), pauli("end", 3, "XI"))
    lowered = []
    prepare_native = aer._prepare_aer_execution
    monkeypatch.setattr(aer, "_prepare_aer_execution",
                        lambda *args, **kwargs: lowered.append(1) or prepare_native(*args, **kwargs))
    plan = three_call_plan(schedule)
    with Run(plan) as run:
        arguments = dict(observation=schedule, runtime=RuntimeOptions(seed=7), position=(1, 3), source_definitions=(),
                         run=run, snapshot="body")
        if body == "unitary composite":
            prepared = AerBackend().prepare(circuit, **arguments)
            assert lowered == [1] and "reset" not in prepared.native.circuit.count_ops()
            return
        expected = "operation 'initialize' contains 'reset', which measures, resets or branches"
        with pytest.raises(ValueError, match=f"coherent body; {expected}"):
            AerBackend().prepare(circuit, **arguments)
    assert lowered == []


@pytest.fixture
def state_reducer(monkeypatch):
    """A test-local executable reducer on Aer: the squared norm and the first amplitude of the saved state."""
    import numpy as np
    from nwqlib.backends import targets

    calls = []

    def execute(state, parameters, bindings):
        # The saved state can be shared with other points at its boundary,
        # so a reducer cannot make it writable again.
        writable = state.flags.writeable
        try:
            state.flags.writeable = True
            writable = True
        except ValueError:
            pass
        outputs = np.array([np.vdot(state, state).real]), np.array([state[0]])
        calls.append((state.shape, writable, dict(parameters), bindings, outputs))
        return outputs

    monkeypatch.setitem(planning.READOUT_REDUCERS, "norm_first", Reducer(
        shape=lambda parameters: (ReducerOutput("float64", (1,)), ReducerOutput("complex128", (1,))),
        work=lambda parameters, width: 2 << width, execute=execute))
    # The shipped Aer statevector target executes every registered reducer.
    assert targets.AER_STATEVECTOR_TARGET.reducers == "registered"
    return calls


def reduction(point_id, position):
    return ObservationPoint(id=point_id, position=position, kind="reduction", reducer="norm_first",
                            parameters={"scale": 1})


def test_registered_reduction_runs_once_per_point_on_its_saved_state(state_reducer, tmp_path, monkeypatch):
    """The reducer receives the read-only state saved at its point; its outputs stay with that point."""
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import submit_experiment

    monkeypatch.setattr(ExpectationMethod, "reduction_allowance",
                        lambda self, plan, point, *, observation, width, run: 2 * (2 << 2), raising=False)
    plan = three_call_plan(trajectory(pauli("first", 1, "ZZ"), reduction("middle", 2), reduction("end", None)))
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        chunks = submit_experiment(handle, run=run)
        for chunk in chunks:
            run.collect(chunk)
    assert [(shape, writeable, parameters) for shape, writeable, parameters, *_ in state_reducer] == [
        ((4,), False, {"scale": 1})] * 2
    assert all(call[3] == chunks[0].bindings for call in state_reducer)
    # Each point chunk stores exactly the outputs its reduction returned.
    for chunk, call in zip(chunks[1:], state_reducer, strict=True):
        norm, first = call[4]
        assert (chunk.values[0].real, chunk.values[1].real, chunk.values[1].imaginary) == (
            (norm[0],), (first[0].real,), (first[0].imag,))
    middle, end = chunks[1:]
    assert (middle.point, middle.boundary, end.point, end.boundary) == ("middle", 2, "end", 3)
    assert middle.observation.positions == (plan.experiments[0].observation.point("middle"),)
    # The two saved states differ, so each reduction saw its own point's state.
    assert middle.values[1].real != end.values[1].real
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        assert reopened.observations.chunks == chunks
    assert len(state_reducer) == 2


def test_reduction_work_and_saved_states_are_admitted_before_native_preparation(state_reducer, monkeypatch):
    """Work beyond a finite allowance, a finite allowance for a reducer without registered work and over-budget saved states refuse first.

    Without the hook the Method sets no work limit, and preparation reaches native work.
    """
    import dataclasses
    import numpy as np
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
    plan = three_call_plan(trajectory(reduction("middle", 2), reduction("end", None)))
    workless = dataclasses.replace(planning.READOUT_REDUCERS["norm_first"], work=None)
    with Run(plan) as run:
        for allowance, reducer, message in ((2 * (2 << 2) - 1, None, "more than the remaining allowance 15"),
                                            (1 << 40, workless, "registers no work")):
            with monkeypatch.context() as patch, pytest.raises(ValueError, match=message):
                if reducer is not None:
                    patch.setitem(planning.READOUT_REDUCERS, "norm_first", reducer)
                patch.setattr(ExpectationMethod, "reduction_allowance",
                              lambda self, plan, point, *, observation, width, run: allowance, raising=False)
                prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0

        class Reached(Exception):
            pass

        with monkeypatch.context() as patch, pytest.raises(Reached):
            patch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: (_ for _ in ()).throw(Reached()))
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
    # A reducer without execution, a reducer the target
    # does not declare and outputs whose stored JSON exceeds an explicit
    # max_data_bytes (10 GB here) are refused before native preparation too.
    registered = planning.READOUT_REDUCERS["norm_first"]
    variants = (
        ("shape_only", Reducer(shape=registered.shape), "has no registered execution"),
        ("undeclared", registered, "cannot execute reduction .undeclared."),
        ("wide", Reducer(shape=lambda parameters: (ReducerOutput("float64", (10**9,)),), work=registered.work,
                         execute=registered.execute), "point .r. storage exceeds remaining max_data_bytes"),
    )
    # ExpectationMethod gives no work allowance, so these refusals do not depend on one.
    from nwqlib.backends import targets
    for name, reducer, message in variants:
        monkeypatch.setitem(planning.READOUT_REDUCERS, name, reducer)
        # A target that names its reducers executes only those names.
        monkeypatch.setattr(targets, "AER_STATEVECTOR_TARGET", targets.AER_STATEVECTOR_TARGET.revise(
            reducers=("norm_first",) + (() if name == "undeclared" else (name,))))
        point = ObservationPoint(id="r", position=None, kind="reduction", reducer=name)
        variant = three_call_plan(trajectory(point))
        with Run(variant, limits=ExecutionLimits(max_data_bytes=10**10)) as run:
            with pytest.raises(ValueError, match=message):
                prepare(variant.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
            assert run.trace.preparations == 0
    # Sixteen qubits: the live state and one saved state need 2*16*2**16 bytes,
    # two MiB, beyond a one-MiB simulator memory limit; the live state uses it all.
    wide = nwqlib.plan(Expectation(state=np.ones(1 << 16), observable=ingest_pauli((("Z" * 16, 1.),), num_qubits=16)),
                       method=ExpectationMethod(), seed=7)
    wide = wide.revise(experiments=(Experiment(name="trajectory", setting="wide", observation=trajectory(
        reduction("end", None))),))._bind(blocks=wide.blocks)
    with Run(wide, limits=ExecutionLimits(simulator_memory_mb=1)) as run:
        with pytest.raises(ValueError, match=r"saved state for point .end.: counted=1048576, requested=16\*2\*\*16, remaining=0"):
            prepare(wide.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_projected_reduction_workspace_is_admitted_with_the_saved_state(monkeypatch):
    """The saved state and the projected reduction's workspace must fit simulator_memory_mb before native work.

    A two-qubit trajectory reduces its end state with ``projected_moments`` over M rows. Its saved state
    takes 16*2**2 bytes and its workspace 49*2**2 + 320*M + 2**17 bytes, so at 1 MiB the largest M whose
    total fits, 2866, reaches native preparation. One more row is refused before it, and the refusal states
    the total with the saved state, which the boundary alone cannot show because the state is smaller
    than one row.
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_parameters
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits

    class Reached(Exception):
        pass

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: (_ for _ in ()).throw(Reached()))
    largest = ((1 << 20) - 16 * 4 - 49 * 4 - (1 << 17)) // 320
    workspace = 49 * 4 + 320 * (largest + 1) + (1 << 17)
    assert largest == 2866 and 16 * 4 + workspace > 1 << 20
    refused = (rf"its saves, 64 bytes with 1 saved simulator state\(s\) of 2 qubits, are held, and their largest "
               rf"reduction needs {workspace} bytes of arrays besides them, {16 * 4 + workspace} bytes together")
    for rows, outcome, message in ((largest, Reached, None), (largest + 1, ValueError, refused)):
        point = ObservationPoint(id="end", position=None, kind="reduction", reducer=PROJECTED_MOMENTS,
                                 parameters=projected_parameters(coordinates=(0, 1), success=(), dimension=4,
                                                                 terms=(("ZZ", 1.),) * rows))
        plan = three_call_plan(trajectory(point))
        with Run(plan, limits=ExecutionLimits(simulator_memory_mb=1)) as run:
            with pytest.raises(outcome, match=message):
                prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))


def test_a_projected_coefficient_beyond_binary64_is_refused_before_native_preparation(monkeypatch):
    """A term value that is not a finite binary64 number is refused when the reducer is admitted.

    Canonical point parameters refuse NaN and infinity, so the value is an integer beyond the binary64
    range. Nothing is prepared or counted.
    """
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_parameters
    from nwqlib.backends import qiskit_aer as aer

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native preparation"))
    parameters = projected_parameters(coordinates=(0, 1), success=(), dimension=4, terms=(("ZZ", 1.),))
    parameters["terms"][0][1] = 10**400
    point = ObservationPoint(id="end", position=None, kind="reduction", reducer=PROJECTED_MOMENTS,
                             parameters=parameters)
    plan = three_call_plan(trajectory(point))
    with Run(plan) as run:
        with pytest.raises(ValueError, match="finite binary64 coefficients"):
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_reduction_workspace_is_admitted_with_the_kept_probabilities(monkeypatch):
    """Aer keeps a trajectory's probabilities until the job completes, so they count with the reduction workspace.

    On 14 qubits one ``projected_moments`` reduction over 13 coordinates and one Pauli term saves a
    262,144-byte state and needs 532,800 bytes of workspace, which fit 1 MiB. Two full-width probability
    points add 262,144 kept bytes, so that trajectory is refused at 1 MiB before native preparation and
    reaches it at 2 MiB. The simulation itself, with the live state, fits 1 MiB in both cases.
    """
    import nwqlib
    from nwqlib import Expectation
    from nwqlib._quantum_readout import PROJECTED_MOMENTS, projected_parameters
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import ingest_product

    class Reached(Exception):
        pass

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: (_ for _ in ()).throw(Reached()))
    width, n = 14, 13
    base = nwqlib.plan(Expectation(state=ingest_product([(.6, .8)] * width),
                                   observable=ingest_pauli([("Z" * width, 1.)], num_qubits=width)),
                       method=ExpectationMethod(), seed=23)
    reduction = ObservationPoint(id="reduction", position=None, kind="reduction", reducer=PROJECTED_MOMENTS,
                                 parameters=projected_parameters(coordinates=tuple(range(n)), success=((n, 0),),
                                                                 dimension=1 << n, terms=(("Z" * n, 1.),)))
    marginals = tuple(ObservationPoint(id=name, position=position, kind="probabilities", qubits=tuple(range(width)))
                      for name, position in (("early", 0), ("late", None)))
    workspace = 49 * (1 << n) + 320 + (1 << 17)
    assert workspace == 532_800 and (16 << width) + workspace <= 1 << 20 < 2 * (16 << width) + workspace
    for points, refused in (((reduction,), False), ((*marginals, reduction), True)):
        chosen = base.revise(experiments=(Experiment(name="read", setting="one", observation=trajectory(*points)),))
        chosen = chosen._bind(blocks=base.blocks)
        for cap in (1, 2):
            with Run(chosen, limits=ExecutionLimits(simulator_memory_mb=cap)) as run:
                if refused and cap == 1:
                    with pytest.raises(ValueError, match=rf"its saves, 524288 bytes with 1 saved simulator state\(s\) "
                                                         rf"of 14 qubits, are held.*{524288 + workspace} bytes together"):
                        prepare(chosen.resolve("read"), run=run, runtime=RuntimeOptions(seed=23))
                    assert run.trace.preparations == 0
                else:
                    with pytest.raises(Reached):
                        prepare(chosen.resolve("read"), run=run, runtime=RuntimeOptions(seed=23))


def test_saved_trajectory_result_reloads_and_reanalyzes_without_decoding(tmp_path, monkeypatch):
    """A Result over a ten-point trajectory saves, loads and reanalyzes; nothing is decoded or run again.

    The saved-Result join groups the point chunks by attempt, orders them by
    the receipt's points and checks the event's ObservationView identity.
    """
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.blocks import lowering
    from nwqlib.core.analysis import Result
    from nwqlib import _prepared_execution as execution
    import dataclasses
    from nwqlib.saved_evidence import load_result

    def analyze(self, plan, data, settings=None):
        chunks = data.observations.chunks
        return Result(plan_id=plan.content_id, construction_id=plan.construction.content_id,
                      observation_id=data.observations.content_id, contribution_ids=tuple(c.content_id for c in chunks))

    monkeypatch.setattr(ExpectationMethod, "result_type", Result)
    monkeypatch.setattr(ExpectationMethod, "analyze", analyze)
    plan = three_call_plan(trajectory(*(pauli(f"p{k}", min(3, (4 * k) // 10), "ZZ", "XI") for k in range(10))))
    with Run(plan, progress=False) as run:
        chunks = execution.submit_experiment(prepare(plan.resolve("trajectory"), run=run,
                                                     runtime=RuntimeOptions(seed=7)), run=run)
        for chunk in reversed(chunks):
            run.collect(chunk)
        data = run.data
    result = analyze(None, plan, data)._attach(plan, data)
    for owner, name in ((execution, "_trajectory_chunks"), (execution, "_decoded_chunk"),
                        (aer, "_submit_aer_execution"), (lowering, "_lower_qiskit")):
        monkeypatch.setattr(owner, name, lambda *args, _name=name, **kwargs: pytest.fail(f"{_name} ran again"))
    loaded = load_result(result.save(tmp_path / "saved"))
    assert loaded == result and loaded.content_id == result.content_id
    assert loaded.data.observations == data.observations and loaded.data.trace.events == data.trace.events
    assert loaded.analyze() == result
    # A saved acquisition missing one of its points is refused.
    partial = dataclasses.replace(data, observations=ObservationView(chunks=data.observations.chunks[1:]))
    with pytest.raises(ValueError, match="missing a declared point"):
        analyze(None, plan, partial)._attach(plan, partial).save(tmp_path / "partial")


@pytest.mark.parametrize("observation,expected", [
    (trajectory(pauli("at0", 0, "IZ")), [1.0]),
    (trajectory(pauli("at1", 1, "IZ")), [-1.0]),
    (trajectory(pauli("at0", 0, "IZ"), pauli("at1", 1, "IZ")), [1.0, -1.0]),
    (trajectory(pauli("at0", 0, "IZ"), pauli("at2", 2, "IZ")), [1.0, 1.0]),
    (ObservationSpec(kind="pauli_expectation", position=0, labels=("IZ",)), [1.0]),
    (ObservationSpec(kind="pauli_expectation", position=1, labels=("IZ",)), [-1.0]),
])
def test_readouts_around_an_opening_direct_preparation_keep_their_order(observation, expected, monkeypatch):
    """Each readout sees its own prefix of three preparations of |01>: <IZ> is +1 at 0 and 2 and -1 at 1.

    Aer sets the first preparation as the initial state only when a readout follows it, so a readout
    at position 0 alone sees |00> and its record has no initial_state_error. Nothing after the last
    readout is built or executed, so up to position 1 no preparation circuit is built.
    """
    from nwqlib._prepared_execution import submit_experiment
    from nwqlib.subroutines.state_preparation import direct

    built = []
    original = direct._build_normalized_state_preparation
    monkeypatch.setattr(direct, "_build_normalized_state_preparation",
                        lambda *args, **kwargs: built.append(1) or original(*args, **kwargs))
    plan = three_call_plan(observation, state=(0., 1., 0., 0.))
    last = (max(point.position for point in observation.positions) if observation.kind == "trajectory"
            else observation.position)
    with Run(plan) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert (handle.record.initial_state_error is None) == (last == 0)
        names = [item.operation.name for item in handle._native.circuit.data]
        assert names.count("set_statevector") == (last > 0)
        assert (built == []) == (last <= 1)
        if last == 0:
            assert set(names) == {"save_expval"}
        chunks = submit_experiment(handle, run=run)
    chunks = chunks if isinstance(chunks, tuple) else (chunks,)
    assert [_chunk_values(chunk)["IZ"] for chunk in chunks] == expected


def test_intermediate_amplitude_points_carry_their_own_prefix_phase():
    """Amplitudes saved before the last observation equal the separately evolved prefix state, phase included.

    The body applies a complex state preparation three times, which lowers
    with a nonzero global phase per application, so a saved state that
    carried the whole circuit's phase would be off by an O(1) angle. The
    reference is ``Statevector`` of the logical prefix. This small
    phase-sensitive regression compares each saved amplitude with an
    independently evolved logical prefix at absolute threshold ``2e-12``,
    with zero relative tolerance. It checks the boundary's represented phase
    directly. The threshold is a fixture regression allowance, not a
    production state-error bound.
    """
    import numpy as np
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.ir import Definition, Sequence
    from nwqlib.operators import ingest_pauli
    from nwqlib._prepared_execution import submit_experiment
    from qiskit.quantum_info import Statevector
    from test_amplitude_masses import readout

    rng = np.random.default_rng(1)
    state = rng.normal(size=4) + 1j * rng.normal(size=4)
    state /= np.linalg.norm(state)
    base = nwqlib.plan(Expectation(state=list(state), observable=ingest_pauli((("ZZ", 1.),), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    program = base.construction.program
    program = program.revise(definitions=tuple(item for item in program.definitions if item.id != "root") + (
        Definition(id="root", node=Sequence(children=("allocate", "prepare", "prepare", "prepare"))),))
    body = base.revise(construction=base.construction.revise(program=program))._bind(blocks=base.blocks)

    def with_schedule(schedule):
        return body.revise(experiments=(Experiment(name="t", setting="t", observation=schedule),))._bind(
            blocks=body.blocks)

    probe = with_schedule(trajectory(pauli("x", 0, "ZZ")))
    _, construction = probe.resolve("t")._selected_construction(probe)
    declaration = readout(width=2, coordinates=(0, 1), success=(), conditions=()).revise(
        construction_id=construction.content_id)
    plan = with_schedule(trajectory(*(ObservationPoint(id=f"a{b}", position=b, kind="amplitudes",
                                                       amplitudes=declaration) for b in range(4))))
    selected = {record.content_id for record in construction.selections}
    logical = lower_qiskit(construction, blocks=tuple(block for block in plan.blocks
                                                      if block.record.content_id in selected)).circuit
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))
        assert handle._native.circuit.global_phase != 0
        chunks = submit_experiment(handle, run=run)
        for chunk, boundary in zip(chunks, range(4), strict=True):
            prefix = logical.copy_empty_like()
            prefix.data = logical.data[:boundary]
            saved = np.asarray(run.artifacts.get(chunk.artifacts[0]).array)
            np.testing.assert_allclose(saved, Statevector(prefix).data, atol=2e-12, rtol=0, err_msg=str(boundary))


def view_plan(schedule):
    """The three-call body with a selected two-qubit readout tail and its selected adjoint as Program definitions."""
    from nwqlib.blocks.selection import select_preparation, transform_block
    from nwqlib.ir import BlockCall, Definition, PortMap
    from nwqlib.problems.inputs import ingest_vector

    base = three_call_plan(schedule)
    tail = select_preparation("tail", ingest_vector((1., 1., 1., -1.)))
    inverse = transform_block("tail_inverse", tail, adjoint=True)
    program = base.construction.program
    program = program.revise(
        definitions=program.definitions + tuple(
            Definition(id=name, node=BlockCall(signature=name, ports=(PortMap(port="system", wire="system"),)))
            for name in ("tail", "tail_inverse")),
        signatures=program.signatures + (tail.record.signature, inverse.record.signature))
    construction = base.construction.revise(program=program,
                                            selections=base.construction.selections + (tail.record, inverse.record))
    return base.revise(construction=construction)._bind(blocks=base.blocks + (tail, inverse))


def test_aer_view_reads_the_rotated_state_and_restores_the_continuation(monkeypatch):
    """A viewed marginal equals the tail applied separately to its saved state; the state continues unchanged.

    One simulation: at boundary 1 an amplitude point saves the state, and a
    probability point reads the marginal after the view's tail, which the
    exact inverse then undoes. At boundary 2 an amplitude and a Pauli point
    read the continuing state. The oracle applies the lowered tail to the
    saved state with ``qiskit.quantum_info``, and the continuation is
    compared with the separately evolved prefix. The marginal and Pauli
    values use the predeclared small-witness regression threshold 2e-12,
    rtol 0 (two qubits), not a production bound. The saved amplitudes are a
    small phase-sensitive regression against the independently evolved
    logical prefix at absolute threshold ``2e-12`` with zero relative
    tolerance, which checks the boundary's represented phase directly: a
    fixture regression allowance, not a production state-error bound.
    """
    import io
    import numpy as np
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.blocks.lowering import _lower_qiskit
    from qiskit import qpy
    from nwqlib._prepared_execution import submit_experiment
    from qiskit.quantum_info import Pauli, Statevector
    from test_amplitude_masses import readout

    simulations = []
    submit = aer._submit_aer_execution
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: simulations.append(1) or submit(native))
    probe = view_plan(trajectory(pauli("x", 0, "ZZ")))
    _, construction = probe.resolve("trajectory")._selected_construction(probe)
    declaration = readout(width=2, coordinates=(0, 1), success=(), conditions=()).revise(
        construction_id=construction.content_id)
    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0, 1))
    plan = view_plan(trajectory(
        ObservationPoint(id="saved", position=1, kind="amplitudes", amplitudes=declaration),
        ObservationPoint(id="viewed", position=1, kind="probabilities", qubits=(0, 1), view=view),
        ObservationPoint(id="continued", position=2, kind="amplitudes", amplitudes=declaration),
        pauli("after", 2, "ZZ", "XI")))
    _, construction = plan.resolve("trajectory")._selected_construction(plan)
    selected = {record.content_id for record in construction.selections}
    blocks = tuple(block for block in plan.blocks if block.record.content_id in selected)
    body = _lower_qiskit(construction, definition=plan.construction.program.root, blocks=blocks).circuit
    tail = _lower_qiskit(construction, definition="tail", blocks=blocks).circuit
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        names = [item.operation.name for item in handle._native.circuit.data]
        # QPY restoration identifies every save of the viewed circuit.
        buffer = io.BytesIO()
        qpy.dump(handle._native.circuit, buffer)
        buffer.seek(0)
        restored = qpy.load(buffer)[0]
        aer._restore_aer_readout(restored, handle.record.observation, handle.record.boundaries)
        assert [item.operation.name for item in restored.data] == names
        chunks = {chunk.point: chunk for chunk in submit_experiment(handle, run=run)}
        saved = np.asarray(run.artifacts.get(chunks["saved"].artifacts[0]).array)
        continued = np.asarray(run.artifacts.get(chunks["continued"].artifacts[0]).array)
    assert simulations == [1] and names.count("save_probabilities") == 1
    # The viewed marginal against the tail applied separately to the saved state.
    rotated = Statevector(saved).evolve(tail).probabilities(qargs=[0, 1])
    viewed = {int(key, 2): value for key, value in _chunk_values(chunks["viewed"]).items()}
    assert max(abs(viewed.get(index, 0.) - value) for index, value in enumerate(rotated)) <= 2e-12
    # The continuation after tail and inverse against the separately evolved prefix.
    prefix = body.copy_empty_like()
    prefix.data = body.data[:2]
    reference = Statevector(prefix)
    np.testing.assert_allclose(continued, reference.data, atol=2e-12, rtol=0)
    after = {item.label: item.value for item in chunks["after"].values}
    assert all(abs(after[label] - reference.expectation_value(Pauli(label)).real) <= 2e-12 for label in after)
    # A view whose declared wire order differs from its tail's ports is refused before native preparation.
    reordered = view_plan(trajectory(ObservationPoint(
        id="viewed", position=1, kind="probabilities", qubits=(0, 1),
        view=ReadoutView(tail="tail", inverse="tail_inverse", wires=(1, 0)))))
    with Run(reordered, progress=False) as run:
        with pytest.raises(ValueError, match="declared wires in their declared order"):
            prepare(reordered.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))


def test_a_shared_phase_corrected_save_gives_each_reducer_its_qualified_state_budget(monkeypatch):
    """A phase-sensitive reducer receives no finite state error; a phase-invariant one the modulo-phase budget.

    The receipt's state-error budget bounds the computed state only up to one
    common global phase (``PreparedArtifact.state_error``). A one-qubit body
    of scalar-phase blocks makes Aer multiply each saved state that serves a
    phase-sensitive point by its prefix phase factor, in place
    (``qiskit_aer._submit_aer_execution``), and the receipt records that
    product's envelope (``PreparedArtifact.statevector_roundoff``). At each
    boundary a phase-sensitive and a phase-invariant reducer read the same
    corrected array. The phase-sensitive one receives no finite
    ``state_error``, the phase-defined reason and, separately, the
    modulo-phase budget ``saved_state_error``; the phase-invariant one
    receives that modulo-phase budget, product charge included, as its
    ``state_error``. The owner of both budgets is
    ``_prepared_execution._trajectory_chunks``.
    """
    import numpy as np
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.ir import Definition, Sequence
    from nwqlib._phase_product import phase_product_envelope
    from nwqlib._prepared_execution import submit_experiment

    seen = []

    def register(name, invariant):
        def execute(state, parameters, bindings, *, context):
            seen.append((name, state.__array_interface__["data"][0], context))
            return (np.array([state[0]]),)
        monkeypatch.setitem(planning.READOUT_REDUCERS, name, Reducer(
            shape=lambda parameters: (ReducerOutput("complex128", (1,)),), work=lambda parameters, width: 1,
            execute=execute, phase_invariant=invariant, receives_context=True,
            state_error_resolutions=("amplitude-derived masses",)))

    register("sensitive", False)
    register("invariant", True)
    native = QuantumCircuit(1)
    native.global_phase = .7312345678901234
    base = nwqlib.plan(Expectation(state=native, observable=[[1, 0], [0, -1]]), method=ExpectationMethod(), seed=29)
    program = base.construction.program
    program = program.revise(definitions=tuple(item for item in program.definitions if item.id != "root") + (
        Definition(id="root", node=Sequence(children=("allocate",) + ("prepare",) * 4)),))
    body = base.revise(construction=base.construction.revise(program=program))._bind(blocks=base.blocks)
    points = tuple(ObservationPoint(id=f"{name}{boundary}", position=boundary, kind="reduction", reducer=name)
                   for boundary in (2, 4) for name in ("sensitive", "invariant"))
    plan = body.revise(experiments=(Experiment(name="t", setting="t", observation=trajectory(*points)),))._bind(
        blocks=body.blocks)
    monkeypatch.setattr(ExpectationMethod, "reduction_allowance",
                        lambda self, plan, point, *, observation, width, run: 4, raising=False)
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=29))
        submit_experiment(handle, run=run)
    receipt = handle.record
    resolved = ("amplitude-derived masses",)
    native_delta = receipt.state_error(resolved)[0]
    saved_delta = receipt.saved_state_error(resolved)[0]
    assert native_delta is not None and receipt.statevector_roundoff != (0.0, 0.0) and saved_delta > native_delta
    # One conservative receipt-wide pair: the componentwise maxima of the
    # stored factors' envelopes (qiskit_aer._statevector_roundoff).
    envelopes = [phase_product_envelope(complex(*factor), 2)
                 for factor in handle._native.metadata["statevector_phase_factors"].values()]
    assert len(envelopes) == 2 and receipt.statevector_roundoff == tuple(map(max, zip(*envelopes)))
    assert [name for name, _, _ in seen] == ["sensitive", "invariant"] * 2
    for (_, sensitive_data, sensitive), (_, invariant_data, invariant) in zip(seen[::2], seen[1::2], strict=True):
        assert sensitive_data == invariant_data
        assert (sensitive.state_error, sensitive.modulo_phase_state_error) == (None, saved_delta)
        assert sensitive.state_error_reason == (
            "phase-defined state error is unavailable: native and prefix global-phase errors have no bound")
        assert (invariant.state_error, invariant.modulo_phase_state_error, invariant.state_error_reason) == (
            saved_delta, saved_delta, None)
        assert sensitive.probability_window == invariant.probability_window == (
            receipt.saved_state_probability_window)


def _projected_mass_checkpoints(labels, state):
    """Acquisition and publication of one projected point on an Aer receipt.

    The receipt is a validated synthetic Aer receipt with G = 0 native
    operations on three qubits, the given exclusion labels and no host phase
    correction (``statevector_roundoff`` of ``(0, 0)``), so its
    ``state_error`` and ``saved_state_error`` are ``native_state_error(0, c)
    = 5u`` when every label is resolved. The two returned callables run the actual
    ``_trajectory_chunks`` on ``state`` and the actual LCHS
    ``projected_moments`` on a stored chunk of ``state``. The stored chunk
    is built under the window convention (an added "unitary" label), then
    bound to this receipt's identity, as a chunk that another receipt
    branch accepted. Exposure has no role in these mass checks, so the
    trace's observation check is a stub; the chunk, its point declaration
    and the receipt use their real validators.
    """
    import nwqlib._prepared_execution as execution
    from nwqlib import _quantum_readout as readout
    from nwqlib.algorithms.lchs import analysis as lchs
    from nwqlib.core import Source
    from nwqlib.core.planning import Realization
    from nwqlib.execution import LogicalPreparationReceipt, PreparedArtifact, RegisterMap

    identity = "sha256:" + "1" * 64
    parameters = readout.projected_parameters(coordinates=(0, 1), success=((2, 0),), dimension=4,
                                              terms=(), moment=False)
    point = ObservationPoint(id="end", kind="reduction", position=0, reducer=readout.PROJECTED_MOMENTS,
                             parameters=parameters)
    realization = Realization(plan_id=identity, experiment="e")
    source = Source(name="aer_statevector", version="0.17.2", domain="synthetic receipt",
                    reference="eight-entry mass checkpoints")
    layout = (RegisterMap(name="q", bits=(0, 1, 2)),)
    receipt = PreparedArtifact(
        plan_id=identity, realization_id=realization.content_id, realization=realization,
        construction_id=identity, snapshot="synthetic", target=source, compiler=source,
        runtime=RuntimeOptions(seed=7), observation=trajectory(point), quantum_layout=layout,
        classical_layout=(), native_basis=("u", "cx"), environment=(),
        preparation_time="2026-09-30T00:00:00Z", construction_work_reserved=0,
        logical=LogicalPreparationReceipt(dynamic_visits=0, reserved_work=0, defined_selections=()),
        native_operations=0, transformation="supplied native amplitudes", population="unconditional",
        native_quantum_layout=layout, logical_to_native=(0, 1, 2),
        probability_window_exclusions=labels, body_length=0, boundaries=(0,), statevector_roundoff=(0.0, 0.0))
    run = SimpleNamespace(run_id="r", backend=SimpleNamespace(kind="qiskit_aer"))
    publish = lchs.projected_moments

    def acquire(record, amplitudes):
        prepared = SimpleNamespace(record=record, realization=realization, _setting="s", _bindings=())
        result = SimpleNamespace(raw_output={"trajectory": {"end": {"statevector": amplitudes.copy()}}},
                                 metadata={"native_job_id": "j"})
        return execution._trajectory_chunks(prepared, result, run=run, attempt="a", result_key="0")[0]

    def data_for(record, chunk):
        trace = SimpleNamespace(plan_id=identity, run_id="r", validate_observation=lambda chunk: None)
        return SimpleNamespace(receipts=(record,), observations=ObservationView(chunks=(chunk,)), trace=trace)

    fallback = receipt.revise(probability_window_exclusions=("amplitude-derived masses", "unitary"))
    chunk = acquire(fallback, state).revise(prepared_id=receipt.content_id)
    data = data_for(receipt, chunk)
    return receipt, (lambda: acquire(receipt, state), lambda: publish(data, chunk)), SimpleNamespace(
        acquire=acquire, data_for=data_for, publish=publish)


def test_projected_reducer_resolves_only_the_amplitude_mass_label_at_every_mass_checkpoint(monkeypatch):
    """The projected kernel's qualified state budget decides its two mass checks on Aer.

    Every Aer receipt of a saved state lists "amplitude-derived masses". The
    projected kernel bounds the masses it forms from saved amplitudes with
    its scaled two-pass allowance (``_quantum_readout.validate_saved_masses``
    and ``saved_mass_allowance``), so it resolves that label and only that
    label (``PROJECTED_MASS_EXCLUSIONS``). On eight amplitudes with G = 0
    the qualified budget is delta = 5u, and a state whose squared norm is
    about 1 - 5e-13 lies outside the complete-mass window
    ``2*delta + delta**2 + a_F*(1+delta)**2 + U_F`` while the window
    convention ``omega + a_F*(1 + omega) + U_F`` accepts it. Acquisition
    and publication must both refuse it. An exact unit
    vector passes. With another native exclusion ("unitary") delta stays
    unavailable and the window convention accepts the same state, and an
    empty exclusion list selects delta directly. A registered reducer that
    runs the same kernel but keeps the default empty resolution list receives
    the unresolved budget, so the window convention accepts the same state,
    and the projected publication refuses its point.

    A check that replaced ``PreparedArtifact.state_error`` would not observe
    which labels are resolved, which is how the delta branch stayed
    unreachable on Aer; this fixture uses the receipt's own method.
    """
    from dataclasses import replace

    import numpy as np
    from nwqlib._quantum_readout import PROJECTED_MASS_EXCLUSIONS, PROJECTED_MOMENTS

    near = np.zeros(8, dtype=np.complex128)
    near[0] = np.sqrt(1 - 5e-13)
    unit = np.zeros(8, dtype=np.complex128)
    unit[0] = 1
    receipt, (acquire_deficient_state, publish_deficient_chunk), steps = (
        _projected_mass_checkpoints(PROJECTED_MASS_EXCLUSIONS, near))
    assert receipt.probability_window_exclusions == ("amplitude-derived masses",)
    assert receipt.state_error()[0] is None
    assert receipt.state_error(PROJECTED_MASS_EXCLUSIONS)[0] == 5 * 2**-53
    for checkpoint in (acquire_deficient_state, publish_deficient_chunk):
        with pytest.raises(ValueError, match="outside the producing window"):
            checkpoint()
    for checkpoint in _projected_mass_checkpoints(PROJECTED_MASS_EXCLUSIONS, unit)[1]:
        checkpoint()

    monkeypatch.setitem(planning.READOUT_REDUCERS, "same_kernel", replace(
        planning.registered_reducer(PROJECTED_MOMENTS), state_error_resolutions=()))
    point = receipt.observation.positions[0]
    custom = receipt.revise(observation=trajectory(point.revise(reducer="same_kernel")))
    chunk = steps.acquire(custom, near)
    with pytest.raises(ValueError, match="require the projected_moments reducer"):
        steps.publish(steps.data_for(custom, chunk), chunk)

    receipt, checkpoints, _ = _projected_mass_checkpoints((*PROJECTED_MASS_EXCLUSIONS, "unitary"), near)
    assert receipt.state_error(PROJECTED_MASS_EXCLUSIONS)[0] is None
    for checkpoint in checkpoints:
        checkpoint()
    receipt, checkpoints, _ = _projected_mass_checkpoints((), near)
    assert receipt.state_error()[0] == 5 * 2**-53
    for checkpoint in checkpoints:
        with pytest.raises(ValueError, match="outside the producing window"):
            checkpoint()


def test_an_admitted_aer_acquisition_is_saved_beyond_the_data_limit(monkeypatch, tmp_path):
    """Records of admitted work are saved beyond ``max_data_bytes``, and new work waits for ``extend_limits``.

    The limit admits the preparation and the output reservation of the one
    Aer acquisition but not the unreserved records of its completion, such
    as its job and timing (``Run._write``). The observation is saved, and
    the folder Run, closed beyond its limit, reopens with it. The derived
    Result, a new preparation and a copy to a new folder are
    refused with the ``extend_limits`` remedy. On that Run a write of a
    lifecycle revision alone is saved, while the same write with a cache row
    is refused. After ``extend_limits`` the Result completes from the saved
    observation without another backend call.
    """
    import nwqlib
    import nwqlib._prepared_execution as owner
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.execution import ExecutionLimits
    from nwqlib.problems import Expectation

    def counted(run):
        state = run._state
        return (state["data_bytes"] + state["array_bytes"] + state["payload_bytes"] + state["external_bytes"]
                + sum(state["pending_output"].values()))

    monkeypatch.setattr(owner, "_now", lambda: "2026-09-29T12:34:56.123456+00:00")
    totals = []
    submit = AerBackend.submit
    monkeypatch.setattr(AerBackend, "submit", lambda self, native, **kwargs: totals.append(counted(kwargs["run"]))
                        or submit(self, native, **kwargs))
    plan = nwqlib.plan(Expectation(state=[1, 1], observable=[[1, 0], [0, -.5]]), method=ExpectationMethod(),
                       shots=200, seed=7)
    with Run(plan, directory=tmp_path / "measured") as run:
        expected = run.resume().result
    # The total at the backend call, with the reservation, admits the same
    # work in the next Run. The margin covers float and limit text whose length varies.
    limit = totals[0] + 256
    with Run(plan, limits=ExecutionLimits(max_data_bytes=limit, max_total_shots=10**6),
             directory=tmp_path / "run") as run:
        with pytest.raises(ValueError, match=r"extend_limits\(max_data_bytes=\.\.\.\)"):
            run.resume()
        assert len(totals) == 2 and counted(run) > limit and run.result is None
        saved = [chunk.content_id for chunk in run.observations.chunks]
    with nwqlib.load_run(tmp_path / "run", progress=False) as run:
        assert counted(run) > limit and [chunk.content_id for chunk in run.observations.chunks] == saved
        assert [event.status for event in run.trace.events] == ["completed"] and len(saved) == 1
        with pytest.raises(ValueError, match=r"extend_limits\(max_data_bytes=\.\.\.\)"):
            prepare(plan.resolve(plan.experiments[0].name), run=run, runtime=RuntimeOptions(seed=8))
        with pytest.raises(ValueError, match=r"extend_limits\(max_data_bytes=\.\.\.\)"):
            run.save(tmp_path / "copy")
        assert not (tmp_path / "copy").exists()
        submission = run.trace.submissions[0]
        before = counted(run)
        run._write((("submission", submission.submission_id, submission.revise(provider_status="done " * 40)),),
                   release_output=())
        assert counted(run) > before
        submission = run.trace.submissions[0]
        with pytest.raises(ValueError, match="max_data_bytes"):
            run._write((("submission", submission.submission_id, submission.revise(provider_status="done " * 50)),
                        ("cache", "probe", {"note": "x"})), release_output=())
        # Raising another limit keeps max_data_bytes, which no longer covers the counted data.
        with pytest.raises(ValueError, match=r"the proposed max_data_bytes=\d+ cannot admit the requested data "
                                             r"bytes already counted .*run\.extend_limits\(max_data_bytes=\.\.\.\)"):
            run.extend_limits(max_total_shots=2 * 10**6)
        # The copy's file limit also covers the SQLite pages of its run log.
        run.extend_limits(max_data_bytes=1_000_000)
        assert run.resume().result.facts == expected.facts and len(totals) == 2
        run.save(tmp_path / "copy")
        # Removing the data limit records one amendment, and a repeated removal
        # records none. The folder and a copy saved without the limit reopen
        # without it, with the same observation and no further backend call.
        run.extend_limits(max_data_bytes=None)
        run.extend_limits(max_data_bytes=None)
        lifted = run.limit_amendments[-1]
        assert run.limits.max_data_bytes is None and len(run.limit_amendments) == 2
        assert (lifted.old.max_data_bytes, lifted.new.max_data_bytes) == (1_000_000, None)
        run.save(tmp_path / "lifted")
    for folder in ("run", "lifted"):
        with nwqlib.load_run(tmp_path / folder, progress=False) as run:
            assert run.limits.max_data_bytes is None and run.limit_amendments[-1] == lifted
            assert run.resume().result.facts == expected.facts and len(totals) == 2


def test_aer_translates_a_view_inverse_only_when_it_composes_it(monkeypatch):
    """A view read only in the schedule's last group applies its tail once and never its inverse.

    The tail definition is translated for Aer and the inverse definition is not, so a direct
    preparation in it would never build its tree (``qiskit_aer._view_applications``).
    """
    from nwqlib.backends import qiskit_aer as aer

    translated = []
    original = aer._AerPreparation.prepare

    def recorded(self, circuit, decompose):
        translated.append(circuit)
        return original(self, circuit, decompose)

    monkeypatch.setattr(aer._AerPreparation, "prepare", recorded)
    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0, 1))
    plan = view_plan(trajectory(pauli("plain", 1, "ZZ"),
                                ObservationPoint(id="viewed", position=2, kind="probabilities", qubits=(0, 1), view=view)))
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
    saves, _ = aer._trajectory_saves(handle.record.observation, handle.record.boundaries)
    assert aer._view_applications(saves) == {"tail": 1}
    # The body and the tail are translated, the inverse is not.
    assert len(translated) == 2
