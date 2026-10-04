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
    point_array_reservation, readout_shape,
)
from nwqlib._prepared_execution import Run, prepare_experiment as prepare
from nwqlib.execution import ObservationView
from test_prepared_execution import base_plan


def pauli(point_id, position, *labels, view=None):
    return ObservationPoint(id=point_id, position=position, kind="pauli_expectation", labels=labels, view=view)


def trajectory(*points, **fields):
    return ObservationSpec(kind="trajectory", positions=points, **fields)


@pytest.fixture
def two_complex_scalars(monkeypatch):
    """A test-local reducer publishing two complex128 scalars; its private workspace is not output."""
    monkeypatch.setitem(planning.READOUT_REDUCERS, "two_complex",
                        Reducer(shape=lambda parameters: (ReducerOutput("complex128", (2,)),)))


def test_point_boundary_lies_in_the_selected_body():
    spec = trajectory(pauli("start", 0, "X"), pauli("mid", 2, "Z"), pauli("end", None, "Y"))
    assert spec.boundaries(3) == (0, 2, 3)
    assert spec.boundaries(2) == (0, 2, 2)
    with pytest.raises(ValueError, match="beyond the selected body of 1"):
        spec.boundaries(1)
    with pytest.raises(ValueError):
        pauli("negative", -1, "X")


def test_point_boundaries_are_nondecreasing():
    trajectory(pauli("a", 1, "X"), pauli("b", 1, "Z"))  # Equal boundaries are legal.
    # Two points at the end shorthand share the final boundary.
    assert trajectory(pauli("a", 1, "X"), pauli("b", None, "Z"), pauli("c", None, "Y")).boundaries(3) == (1, 3, 3)
    for points in ((pauli("a", 2, "X"), pauli("b", 1, "Z")), (pauli("a", None, "X"), pauli("b", 3, "Z"))):
        with pytest.raises(ValueError, match="nondecreasing"):
            trajectory(*points)


def test_point_ids_are_unique():
    with pytest.raises(ValueError, match="unique"):
        trajectory(pauli("a", 0, "X"), pauli("a", 1, "Z"))


def test_pauli_labels_are_full_width_and_distinct_within_a_point():
    # The same label at another point is another value.
    spec = trajectory(pauli("a", 0, "XI", "IZ"), pauli("b", 1, "XI"))
    assert readout_shape(kind="trajectory", details=spec, width=2, classical_width=0) == 3
    with pytest.raises(ValueError, match="distinct labels"):
        pauli("a", 0, "XI", "XI")
    with pytest.raises(ValueError, match="entire logical circuit width"):
        readout_shape(kind="trajectory", details=spec, width=3, classical_width=0)


def test_marginal_wires_are_valid_ordered_and_nonempty():
    point = ObservationPoint(id="m", position=0, kind="probabilities", qubits=(2, 0))
    assert readout_shape(kind="trajectory", details=trajectory(point), width=3, classical_width=0) == 4
    for qubits in ((), (1, 1)):
        with pytest.raises(ValueError, match="q_k >= 1"):
            ObservationPoint(id="m", position=0, kind="probabilities", qubits=qubits)
    with pytest.raises(ValueError, match="belong to the circuit"):
        readout_shape(kind="trajectory", details=trajectory(point), width=2, classical_width=0)


def test_readout_view_carries_its_inverse_and_wire_map():
    view = ReadoutView(tail="basis_tail", inverse="basis_tail_inverse", wires=(1, 0))
    spec = trajectory(pauli("a", 1, "ZZ", view=view))
    assert readout_shape(kind="trajectory", details=spec, width=2, classical_width=0) == 1
    with pytest.raises(ValueError):
        ReadoutView(tail="basis_tail", wires=(0,))
    for wires in ((), (0, 0)):
        with pytest.raises(ValueError, match="wire mapping"):
            ReadoutView(tail="basis_tail", inverse="basis_tail_inverse", wires=wires)
    with pytest.raises(ValueError, match="view wires"):
        readout_shape(kind="trajectory", details=trajectory(pauli("a", 1, "Z", view=view)), width=1,
                      classical_width=0)


def test_trajectory_has_zero_raw_shots_and_a_deterministic_population():
    point = pauli("a", 0, "X")
    with pytest.raises(ValueError, match="zero raw shots"):
        trajectory(point, shots=1)
    with pytest.raises(ValueError, match="measurement-conditioned"):
        trajectory(point, population="native_conditioned")
    with pytest.raises(ValueError, match="without measurement registers"):
        readout_shape(kind="trajectory", details=trajectory(point), width=1, classical_width=1)
    with pytest.raises(ValueError, match="empty request"):
        trajectory()


def test_one_schedule_per_readout():
    point = pauli("a", 0, "X")
    with pytest.raises(ValueError, match="cannot both be populated"):
        trajectory(point, position=0)
    with pytest.raises(ValueError, match="cannot both be populated"):
        ObservationSpec(kind="pauli_expectation", labels=("X",), positions=(point,))
    with pytest.raises(ValueError, match="belong only to trajectory"):
        ObservationSpec(kind="amplitudes", amplitudes=None, positions=(point,))


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
    # Dense array payload: 8*3 + 8*2**3 + 16*2 = 24 + 64 + 32 = 120 bytes, the
    # pre-acquisition reservation whatever the marginal's eventual storage.
    assert sum(point_array_reservation(point, width=3) for point in spec.positions) == 120


def test_reduction_reserves_its_output_not_its_workspace_and_unknown_reducers_fail(two_complex_scalars):
    # Even when the reduction's private state on 20 qubits would take 16*2**20
    # bytes, its registered output is two complex128 scalars: 32 bytes, 2 items.
    point = ObservationPoint(id="r", position=0, kind="reduction", reducer="two_complex")
    assert point_array_reservation(point, width=20) == 32
    assert readout_shape(kind="trajectory", details=trajectory(point), width=20, classical_width=0) == 2
    unknown = ObservationPoint(id="u", position=0, kind="reduction", reducer="unregistered")
    with pytest.raises(ValueError, match="no registered output shape"):
        readout_shape(kind="trajectory", details=trajectory(unknown), width=1, classical_width=0)


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


def test_one_endpoint_readout_is_the_one_point_case():
    base = base_plan()
    experiment = base.experiments[0]
    observation = experiment.observation
    single = readout_shape(kind=observation.kind, details=observation, width=1, classical_width=0)
    point = ObservationPoint(id="end", kind=observation.kind, labels=observation.labels, position=observation.position)
    assert readout_shape(kind="trajectory", details=trajectory(point), width=1, classical_width=0) == single
    # Every readout record carries the positions field.
    record = base.to_record()
    assert record["experiments"][0]["observation"]["positions"] == []


def two_point_plan():
    base = base_plan()
    schedule = trajectory(pauli("start", 0, "X"), pauli("end", None, "Z"))
    return base.revise(experiments=(Experiment(name="trajectory", setting="two points", observation=schedule),))._bind(
        blocks=base.blocks), schedule


def test_aer_refuses_a_view_without_its_program_definitions_before_native_preparation(monkeypatch):
    from nwqlib.blocks import lowering
    base = base_plan()
    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0,))
    schedule = trajectory(pauli("start", 0, "X"), pauli("end", None, "Z", view=view))
    plan = base.revise(experiments=(Experiment(name="trajectory", setting="view", observation=schedule),))._bind(
        blocks=base.blocks)
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *args, **kwargs: pytest.fail("native lowering started"))
    with Run(plan) as run:
        with pytest.raises(ValueError, match="readout view.*tail or inverse"):
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_every_adapter_without_a_trajectory_target_rejects_a_two_point_schedule_before_native_work(tmp_path):
    """NWQ-Sim CPU/DM, MPI and GPU targets, Slurm on MPI, IBM, IonQ and Nexus refuse before any native call."""
    from nwqlib.backends import IBMRuntimeBackend, IonQBackend, NexusBackend, NWQSimBackend
    from nwqlib.backends.slurm import NWQSimSlurmBackend, SlurmProfile

    _, schedule = two_point_plan()
    run = SimpleNamespace(_state={"backend_context": {}})
    arguments = dict(observation=schedule, runtime=RuntimeOptions(seed=7), position=1, source_definitions=(),
                     run=run, snapshot="rejected")
    profile = SlurmProfile(cluster="perlmutter", account="allocation", walltime="00:05:00", nodes=1,
                           ranks_per_node=2, threads=1, executable="/not-called/runner", spool=str(tmp_path),
                           build_identity="supplied build", backend="MPI")
    bounds = dict(max_input_bytes=65536, max_output_bytes=65536, max_buffer_bytes=65536)
    slurm = NWQSimSlurmBackend(profile=profile, accounting_since="2026-09-10T00:00:00",
                               accounting_until="2026-09-11T00:00:00", max_response_bytes=4096, timeout_seconds=2,
                               **bounds)

    def nwqsim(backend, method):
        return NWQSimBackend(executable="/not-called/nwqsim", spool=str(tmp_path), backend=backend, method=method,
                             ranks=2 if backend == "MPI" else 1, **bounds).prepare

    adapters = (
        *((f"NWQ-Sim {backend}/{method}", nwqsim(backend, method))
          for backend, method in (("CPU", "DM"), ("MPI", "SV"), ("NVGPU", "SV"), ("AMDGPU", "DM"))),
        ("Slurm NWQ-Sim", slurm.prepare),
        ("IBM Runtime", IBMRuntimeBackend(device="device", instance="instance", max_input_bytes=65536).prepare),
        ("IonQ qpu.forte-1", IonQBackend(device="qpu.forte-1", max_input_bytes=65536,
                                         max_response_bytes=65536).prepare),
        ("Nexus H2-1", NexusBackend(project="project", device="H2-1", max_input_bytes=65536).prepare_local),
    )
    circuit = QuantumCircuit(1)
    for name, prepare_native in adapters:
        with pytest.raises(ValueError, match=f"{name} does not execute a 2-point trajectory schedule"):
            prepare_native(circuit, **arguments)


@pytest.mark.parametrize("case", ["variant reduction before the end", "amplitudes", "invariant reduction before the end"])
def test_nwqsim_serves_only_phase_invariant_reductions_before_the_last_boundary(case, state_reducer, tmp_path,
                                                                               monkeypatch):
    """NWQ-Sim CPU/SV has no per-segment phase ledger. A reduction before the last boundary is refused
    before the runner or the transpiler is called unless its reducer is registered phase invariant, and an
    amplitude point is refused; a phase-invariant reducer is served.
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
    if case == "amplitudes":
        from test_amplitude_masses import readout
        probe = three_call_plan(trajectory(pauli("x", 0, "ZZ")))
        _, construction = probe.resolve("trajectory")._selected_construction(probe)
        declaration = readout(width=2, coordinates=(0, 1), success=(), conditions=()).revise(
            construction_id=construction.content_id)
        early = ObservationPoint(id="state", position=1, kind="amplitudes", amplitudes=declaration)
    else:
        early = reduction("state", 1)
    schedule = trajectory(early, pauli("end", 3, "ZZ"))
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
        match = ("does not publish the amplitude point 'state' of a detached trajectory; use AerBackend" if case == "amplitudes"
                 else "no phase ledger for the reduction point 'state' before the last boundary.*move the point to the last "
                      "boundary, or register the reducer with phase_invariant=True")
        with pytest.raises(ValueError, match=match):
            backend.prepare(circuit, **arguments)
    assert "transpile" not in calls
    # The same refusal comes from target_for, before a Run charges a preparation.
    with Run(plan, backend=backend, directory=tmp_path / "run") as run:
        with pytest.raises(ValueError, match=match):
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_payload_admission_charges_each_point_against_the_remaining_bytes():
    """The trajectory law R = J_T + sum_k (D_k + H_k + V_k) + M and M_min(K) <= M, before native work.

    The terms and the 23-byte minimal metadata envelope per point chunk are
    those of ``_prepared_execution.trajectory_reservation`` and
    ``_trajectory_fixed_bounds``; restoration supplies no header context, so
    its point headers are not charged, and a detached NWQ-Sim or Slurm
    restoration is not held to the early minimum (``_admit_readout``).
    """
    from nwqlib.execution import ExecutionLimits
    from nwqlib._prepared_execution import _admit_readout, _statistic_stride
    from nwqlib._run_journal import _json_bound

    _, schedule = two_point_plan()
    metadata = 4096

    def admit(max_data_bytes, metadata_bytes=metadata):
        run = SimpleNamespace(limits=ExecutionLimits(max_data_bytes=max_data_bytes,
                                                     max_completion_metadata_bytes=metadata_bytes))
        return _admit_readout(schedule, width=1, classical_width=0, run=run)

    receipt = _json_bound("observation", 1 << 40) + 2 + _json_bound(schedule, 1 << 40)
    declarations = sum(_json_bound({"observation": ObservationSpec.point_readout_fields(point)}, 1 << 40)
                       for point in schedule.positions)
    values = 2 * _statistic_stride("pauli_expectation", key_width=1)
    total = receipt + metadata + declarations + values
    assert admit(total) == 2
    # One byte less: the last point's value no longer fits after the receipt
    # declaration, the allowance and every earlier term.
    with pytest.raises(ValueError, match="trajectory point 'end' storage exceeds remaining max_data_bytes"):
        admit(total - 1)
    # Two point chunks need at least 2*23 bytes of variable metadata.
    with pytest.raises(ValueError, match="need at least 46 bytes, more than max_completion_metadata_bytes=45"):
        admit(1 << 30, metadata_bytes=45)
    assert admit(1 << 30, metadata_bytes=46) == 2
    # A detached restoration fetches an existing acquisition, whose outcome
    # is checked against its actual stored rows when published.
    for kind in ("nwqsim", "slurm"):
        run = SimpleNamespace(backend=SimpleNamespace(kind=kind), forecast=None,
                              limits=ExecutionLimits(max_data_bytes=1 << 30, max_completion_metadata_bytes=45))
        assert _admit_readout(schedule, width=1, classical_width=0, run=run) == 2
    # Preparation checks the Run's remaining stored-data capacity, not the nominal cap.
    run = SimpleNamespace(run_id="r", limits=ExecutionLimits(max_data_bytes=1 << 30, max_completion_metadata_bytes=metadata),
                          _available_data_bytes=lambda: total - 1)
    header = dict(experiment="e", setting="s", bindings=(), plan_id="p", realization_id="q", quantum_layout=(),
                  classical_layout=())
    with pytest.raises(ValueError, match="remaining max_data_bytes"):
        _admit_readout(schedule, width=1, classical_width=0, run=run, header=header)


def test_probability_points_reserve_eight_bytes_per_possible_outcome():
    """Dense or adaptively sparse probability arrays reserve ``8 * 2**q_k`` bytes per point (``V_k``).

    With the receipt declaration, the allowance and the point declarations,
    that total is admitted and one byte less refuses the last point before
    native work (``_admit_readout``, which ``prepare_experiment`` calls before
    lowering). Restoration charges no point headers.
    """
    from nwqlib.execution import ExecutionLimits
    from nwqlib._prepared_execution import _admit_readout, _trajectory_value_bounds
    from nwqlib._run_journal import _json_bound

    schedule = trajectory(ObservationPoint(id="first", position=0, kind="probabilities", qubits=(0,)),
                          ObservationPoint(id="end", position=None, kind="probabilities", qubits=(0,)))
    assert _trajectory_value_bounds(schedule, width=1, capacity=None, max_bytes=1 << 40) == [16, 16]
    metadata = 4096

    def admit(max_data_bytes):
        run = SimpleNamespace(limits=ExecutionLimits(max_data_bytes=max_data_bytes,
                                                     max_completion_metadata_bytes=metadata))
        return _admit_readout(schedule, width=1, classical_width=0, run=run)

    receipt = _json_bound("observation", 1 << 40) + 2 + _json_bound(schedule, 1 << 40)
    declarations = sum(_json_bound({"observation": ObservationSpec.point_readout_fields(point)}, 1 << 40)
                       for point in schedule.positions)
    total = receipt + metadata + declarations + 2 * 8 * 2
    assert admit(total) == 4
    with pytest.raises(ValueError, match="trajectory point 'end' storage exceeds remaining max_data_bytes"):
        admit(total - 1)


def test_point_chunk_keeps_its_point_and_boundary_after_reload():
    """A point chunk holds only its point's one-point readout and names its trajectory declaration."""
    from nwqlib.core import Source
    from nwqlib.execution import ObservationChunk

    marginal = ObservationPoint(id="m", position=None, kind="probabilities", qubits=(0,))
    spec = trajectory(pauli("p", 1, "Z"), marginal)
    identity = "sha256:" + "0" * 64
    fields = dict(run_id="r", plan_id=identity, realization_id=identity, prepared_id=identity, experiment="e",
                  setting="s", bindings=(), quantum_layout=(), classical_layout=(), attempt="a", job="j",
                  chunk="0/m", observation=spec.point_observation("m"), population="unconditional",
                  returned_shots=None, trajectories=1, trajectory_id=spec.content_id,
                  source=Source(name="supplied", version="1", domain="test", reference="supplied outcomes"))
    chunk = ObservationChunk.from_histogram({"0": .25, "1": .75}, point="m", boundary=4, **fields)
    reloaded = ObservationChunk.model_validate_json(chunk.model_dump_json())
    assert (reloaded.point, reloaded.boundary, reloaded.content_id) == ("m", 4, chunk.content_id)
    assert reloaded.values == chunk.values and chunk.histogram().weights.tolist() == [.25, .75]
    assert reloaded.readout() is reloaded.observation
    # Rebuilt from its JSON alone, the chunk has no payload source; its Run or Result supplies one.
    with pytest.raises(ValueError, match="load_run or nwqlib.load_result"):
        reloaded.histogram()
    # The stored readout is the point's own, not the trajectory declaration.
    assert reloaded.observation == ObservationSpec(kind="probabilities", qubits=(0,))
    pauli_fields = dict(fields, chunk="0/p", observation=spec.point_observation("p"))
    with pytest.raises(ValueError, match="differs from its declared trajectory point"):
        ObservationChunk(values=({"kind": "pauli", "label": "Z", "value": 0.},), point="p", boundary=2, **pauli_fields)
    # A point chunk names its trajectory declaration and holds no trajectory
    # declaration itself; other chunks name neither.
    for changed, message in ((dict(trajectory_id=None), "point, boundary, and trajectory identity together"),
                             (dict(observation=spec), "can carry only its singleton reduction declaration")):
        with pytest.raises(ValueError, match=message):
            ObservationChunk(values=chunk.values, point="m", boundary=4, **dict(fields, **changed))
    with pytest.raises(ValueError, match="point, boundary, and trajectory identity together"):
        ObservationChunk(values=chunk.values, **fields)


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
    if chunk.readout().kind == "pauli_expectation":
        return {item.label: state.expectation_value(SparsePauliOp(item.label)).real for item in chunk.values}
    probabilities = state.probabilities(list(chunk.readout().qubits))
    return {key: probabilities[int(key, 2)] for key in _chunk_values(chunk)}


def _chunk_values(chunk):
    """Point values by Pauli label or, for a marginal, by bit-string key (qubits[0] rightmost)."""
    if chunk.readout().kind == "pauli_expectation":
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
            stored = {format(index, f"0{len(chunk.readout().qubits)}b"): float(value)
                      for index, value in enumerate(marginal)}
        assert stored == values


def test_trajectory_refuses_a_resetting_selected_body_before_native_work(monkeypatch):
    """Zero raw shots admit no reset along the executed prefix or in a view, refused before the charge.

    A reset that starts before the last point's boundary, or in a view's
    tail, is refused by the selected IR with nothing charged. A reset that
    starts at the last boundary is outside the executed prefix, so that body
    is admitted and prepared without the reset, as is the body without it.
    A reset inside the prefix that the selected IR cannot see, hidden in a
    native composite definition, is refused by the adapter's check in
    ``test_aer_trajectory_admits_only_a_coherent_body_including_composite_definitions``.
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


def _hidden_reset():
    inner = QuantumCircuit(2, name="hidden_reset")
    inner.h(0)
    inner.reset(1)
    return inner.to_instruction()


@pytest.mark.parametrize("body", ["composite reset", "initialize", "opaque instruction", "unitary composite",
                                  "reset after the last point"])
def test_aer_trajectory_admits_only_a_coherent_body_including_composite_definitions(body, monkeypatch):
    """A reset hidden in a composite, Initialize (whose definition resets) and an opaque
    instruction are refused before lowering; a composite of unitary gates is lowered, and so is a
    body whose reset comes after the last point, since only the executed prefix is checked.
    """
    from qiskit.circuit import Instruction
    from qiskit.circuit.library import Initialize
    from nwqlib.backends import AerBackend, qiskit_aer as aer

    circuit = QuantumCircuit(2)
    circuit.h(0)
    if body == "composite reset":
        circuit.append(_hidden_reset(), (0, 1))
    elif body == "initialize":
        circuit.append(Initialize([0, 1, 0, 0]), (0, 1))
    elif body == "opaque instruction":
        circuit.append(Instruction("opaque", 1, 0, []), (1,))
    elif body == "reset after the last point":
        circuit.h(1)
    else:
        inner = QuantumCircuit(2, name="unitary_composite")
        inner.cx(0, 1)
        circuit.append(inner.to_instruction(), (0, 1))
    circuit.x(1)
    if body == "reset after the last point":
        circuit.reset(0)
    schedule = trajectory(pauli("first", 1, "ZZ"), pauli("end", 3, "XI"))
    lowered = []
    prepare_native = aer._prepare_aer_execution
    monkeypatch.setattr(aer, "_prepare_aer_execution",
                        lambda *args, **kwargs: lowered.append(1) or prepare_native(*args, **kwargs))
    plan = three_call_plan(schedule)
    with Run(plan) as run:
        arguments = dict(observation=schedule, runtime=RuntimeOptions(seed=7), position=(1, 3), source_definitions=(),
                         run=run, snapshot="body")
        if body in {"unitary composite", "reset after the last point"}:
            prepared = AerBackend().prepare(circuit, **arguments)
            assert lowered == [1] and "reset" not in prepared.native.circuit.count_ops()
            return
        expected = {"composite reset": "operation 'hidden_reset' contains 'reset', which measures, resets or branches",
                    "initialize": "operation 'initialize' contains 'reset', which measures, resets or branches",
                    "opaque instruction": "operation 'opaque' is an opaque instruction without a definition"}[body]
        with pytest.raises(ValueError, match=f"coherent body; {expected}"):
            AerBackend().prepare(circuit, **arguments)
    assert lowered == []


def test_a_bound_noise_model_refuses_exact_readouts_and_trajectories_and_admits_counts():
    """Even an empty bound NoiseModel keeps Aer on its sampled-counts route."""
    from qiskit_aer.noise import NoiseModel
    from nwqlib.backends import AerBackend

    noisy = AerBackend.from_noise_model(NoiseModel())
    for observation in (trajectory(pauli("end", None, "Z")), ObservationSpec(kind="probabilities", qubits=(0,))):
        with pytest.raises(ValueError, match="explicit Aer noise supports counts"):
            noisy.target_for(observation)
    assert noisy.target_for(ObservationSpec(kind="counts", shots=4)).readouts == ("counts",)


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


def test_registered_reduction_runs_once_per_point_on_its_saved_state(state_reducer, tmp_path):
    """The reducer receives the read-only state saved at its point; its outputs stay with that point."""
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import submit_experiment

    plan = three_call_plan(trajectory(pauli("first", 1, "ZZ"), reduction("middle", 2), reduction("end", None)))
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7),
                         reduction_allowance=2 * (2 << 2))
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
    assert middle.readout().positions == (plan.experiments[0].observation.point("middle"),)
    # The two saved states differ, so each reduction saw its own point's state.
    assert middle.values[1].real != end.values[1].real
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        assert reopened.observations.chunks == chunks
    assert len(state_reducer) == 2


def test_reduction_work_and_saved_states_are_admitted_before_native_preparation(state_reducer, monkeypatch):
    """Work beyond the Method's allowance, a missing allowance and over-budget saved states refuse first."""
    import numpy as np
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
    plan = three_call_plan(trajectory(reduction("middle", 2), reduction("end", None)))
    with Run(plan) as run:
        for allowance, message in ((2 * (2 << 2) - 1, "more than the remaining allowance 15"),
                                   (None, "needs the calling Method's remaining work allowance")):
            with pytest.raises(ValueError, match=message):
                prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7),
                        reduction_allowance=allowance)
        assert run.trace.preparations == 0
    # A reducer without execution, an indexed output, a reducer the target
    # does not declare and outputs whose stored JSON exceeds max_data_bytes
    # are refused before native preparation too.
    registered = planning.READOUT_REDUCERS["norm_first"]
    variants = (
        ("shape_only", Reducer(shape=registered.shape), "has no registered execution"),
        ("indexed", Reducer(shape=lambda parameters: (ReducerOutput("float64", (2,), index_width=2),),
                            work=registered.work, execute=registered.execute), "indexed output component"),
        ("undeclared", registered, "cannot execute reduction .undeclared."),
        ("wide", Reducer(shape=lambda parameters: (ReducerOutput("float64", (10**9,)),), work=registered.work,
                         execute=registered.execute), "point .r. storage exceeds remaining max_data_bytes"),
    )
    from nwqlib.backends import targets
    for name, reducer, message in variants:
        monkeypatch.setitem(planning.READOUT_REDUCERS, name, reducer)
        # A target that names its reducers executes only those names.
        monkeypatch.setattr(targets, "AER_STATEVECTOR_TARGET", targets.AER_STATEVECTOR_TARGET.revise(
            reducers=("norm_first",) + (() if name == "undeclared" else (name,))))
        point = ObservationPoint(id="r", position=None, kind="reduction", reducer=name)
        variant = three_call_plan(trajectory(point))
        with Run(variant) as run:
            with pytest.raises(ValueError, match=message):
                prepare(variant.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7),
                        reduction_allowance=1 << 40)
            assert run.trace.preparations == 0
    # Sixteen qubits: the live state and one saved state need 2*16*2**16 bytes,
    # two MiB, beyond a one-MiB simulator memory limit; the live state uses it all.
    wide = nwqlib.plan(Expectation(state=np.ones(1 << 16), observable=ingest_pauli((("Z" * 16, 1.),), num_qubits=16)),
                       method=ExpectationMethod(), seed=7)
    wide = wide.revise(experiments=(Experiment(name="trajectory", setting="wide", observation=trajectory(
        reduction("end", None))),))._bind(blocks=wide.blocks)
    with Run(wide, limits=ExecutionLimits(simulator_memory_mb=1)) as run:
        with pytest.raises(ValueError, match=r"saved state for point .end.: counted=1048576, requested=16\*2\*\*16, remaining=0"):
            prepare(wide.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7), reduction_allowance=1 << 40)
        assert run.trace.preparations == 0


def test_amplitude_points_publish_their_arrays_at_their_points_and_reopen(tmp_path, monkeypatch):
    """Two amplitude points share one saved state at a boundary; a third reads the final state.

    A published amplitude chunk whose header differs from the reserved one is refused.
    """
    import numpy as np
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import submit_experiment
    from test_amplitude_masses import readout

    probe = three_call_plan(trajectory(pauli("x", 1, "ZZ")))
    _, construction = probe.resolve("trajectory")._selected_construction(probe)
    amplitudes = readout(width=2, coordinates=(0, 1), success=(), conditions=()).revise(
        construction_id=construction.content_id)
    plan = three_call_plan(trajectory(
        ObservationPoint(id="a", position=1, kind="amplitudes", amplitudes=amplitudes),
        ObservationPoint(id="b", position=1, kind="amplitudes", amplitudes=amplitudes),
        ObservationPoint(id="end", kind="amplitudes", amplitudes=amplitudes)))
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        saves = [item.operation.name for item in handle._native.circuit.data if item.operation.name.startswith("save")]
        assert saves == ["save_statevector", "save_statevector"]
        chunks = submit_experiment(handle, run=run)
        for chunk in chunks:
            run.collect(chunk)
        arrays = [run.artifacts.get(chunk.artifacts[0]).array for chunk in chunks]
    assert [(chunk.point, chunk.boundary, chunk.chunk) for chunk in chunks] == [
        ("a", 1, "0/a"), ("b", 1, "0/b"), ("end", 3, "0/end")]
    assert all(chunk.artifacts[0].acquisition[3] == chunk.chunk for chunk in chunks)
    # The shared save gives both points the same state, and the final point
    # reads another one.
    assert np.array_equal(arrays[0], arrays[1]) and not np.array_equal(arrays[0], arrays[2])
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        assert reopened.observations.chunks == chunks
    from nwqlib import _prepared_execution as execution
    publish = execution._publish_amplitudes
    monkeypatch.setattr(execution, "_publish_amplitudes",
                        lambda *args, **kwargs: publish(*args, **kwargs).revise(setting="another setting"))
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        with pytest.raises(ValueError, match="differs from its reserved header"):
            submit_experiment(handle, run=run)


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


def test_a_local_output_refused_after_it_returned_releases_its_reservation(tmp_path, monkeypatch):
    """Aer returned in process and publication refused its output: nothing can retrieve it, so nothing stays pending."""
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib import _prepared_execution as execution

    def refuse(*args, **kwargs):
        raise ValueError("publication refused")

    plan = three_call_plan(trajectory(pauli("first", 1, "ZZ"), pauli("end", None, "XI")))
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        monkeypatch.setattr(execution, "_trajectory_chunks", refuse)
        with pytest.raises(ValueError, match="publication refused"):
            execution.submit_experiment(handle, run=run)
        assert run._state["pending_output"] == {}
        assert [event.status for event in run.trace.events] == ["uncertain"]
    monkeypatch.undo()
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        assert reopened._state["pending_output"] == {}


def test_saved_probability_marginals_are_admitted_in_simulator_memory(monkeypatch):
    """Aer keeps every distinct saved marginal until completion; their 8*2**k bytes count before native work."""
    import numpy as np
    import nwqlib
    from nwqlib import Expectation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits
    from nwqlib.operators import ingest_pauli

    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
    # Live state 16*2**16 bytes (1 MiB) and two full-width sorted marginals of
    # 8*2**16 bytes each fill 2 MiB; a third, distinct wire order does not fit.
    points = (ObservationPoint(id="a", position=0, kind="probabilities", qubits=tuple(range(16))),
              ObservationPoint(id="b", kind="probabilities", qubits=tuple(range(16))),
              ObservationPoint(id="c", kind="probabilities", qubits=tuple(reversed(range(16)))))
    plan = nwqlib.plan(Expectation(state=np.ones(1 << 16), observable=ingest_pauli((("Z" * 16, 1.),), num_qubits=16)),
                       method=ExpectationMethod(), seed=7)
    for schedule, admitted in ((trajectory(*points[:2]), True), (trajectory(*points), False)):
        variant = plan.revise(experiments=(Experiment(name="t", setting="m", observation=schedule),))._bind(
            blocks=plan.blocks)
        with Run(variant, limits=ExecutionLimits(simulator_memory_mb=2)) as run:
            if admitted:
                with pytest.raises(pytest.fail.Exception, match="native lowering started"):
                    prepare(variant.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))
            else:
                with pytest.raises(ValueError, match=r"saved probabilities for point 'c': counted=2097152"):
                    prepare(variant.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))


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


def test_point_chunk_fields_are_priced_once_and_completion_stays_within_the_reservation(monkeypatch):
    """Each point-chunk field is the declaration, a reserved header field or a variable field, exactly once.

    ``readout_bytes`` reserves ``sum_k (D_k + H_k + V_k) + M``; the growth that
    completion charges must stay within that one pending reservation.
    The pre-lowering gate prices the same point headers as the submission
    gate, the layouts and the acquisition source included.
    """
    from nwqlib import _prepared_execution as execution
    from nwqlib.execution import ObservationChunk
    from test_amplitude_masses import readout

    probe = three_call_plan(trajectory(pauli("x", 0, "ZZ")))
    _, construction = probe.resolve("trajectory")._selected_construction(probe)
    amplitudes = readout(width=2, coordinates=(0, 1), success=(), conditions=()).revise(
        construction_id=construction.content_id)
    plan = three_call_plan(trajectory(
        pauli("p", 1, "ZZ", "XI"), ObservationPoint(id="m", position=2, kind="probabilities", qubits=(1, 0)),
        ObservationPoint(id="a", position=2, kind="amplitudes", amplitudes=amplitudes), pauli("end", None, "XI")))
    completions = []
    finish = execution.Run._finish_attempt

    def measured(self, index, event, chunk, **kwargs):
        state = self._state
        before = state["data_bytes"] + state["array_bytes"]
        reserved = state["pending_output"][event.attempt]
        finish(self, index, event, chunk, **kwargs)
        completions.append((state["data_bytes"] + state["array_bytes"] - before, reserved,
                            event.attempt in state["pending_output"]))

    monkeypatch.setattr(execution.Run, "_finish_attempt", measured)
    contexts = []
    admit = execution._admit_readout
    monkeypatch.setattr(execution, "_admit_readout",
                        lambda observation, **kwargs: contexts.append(kwargs["header"]) or admit(observation, **kwargs))
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        common, per_point = execution._point_chunk_headers(handle, run_id=run.run_id, attempt="0" * 36,
                                                           backend_kind=run.backend.kind)
        spec = handle.record.observation
        early = execution._prospective_point_headers(spec, contexts[0], run)
        assert [execution._trajectory_fixed_bounds(spec, common=maps[0], per_point=maps[1], max_bytes=1 << 40)[1]
                for maps in (early, (common, per_point))] == [
            execution._trajectory_fixed_bounds(spec, common=common, per_point=per_point, max_bytes=1 << 40)[1]] * 2
        for point, own in zip(handle.record.observation.positions, per_point, strict=True):
            header, variable = set(common) | set(own), set(execution._point_variable_fields(point.kind))
            assert not header & variable and "observation" not in header | variable
            assert header | variable | {"observation"} == set(ObservationChunk.model_fields)
        execution.submit_experiment(handle, run=run)
    [(growth, reserved, pending)] = completions
    assert growth <= reserved and not pending


def test_a_context_reducer_reads_its_producing_receipts_window(state_reducer, monkeypatch):
    """A reducer registered with ``receives_context`` reads its receipt's saved-state window and exclusions.

    The window is the receipt's ``saved_state_probability_window``, its
    probability window propagated through the host phase correction of the
    saved state.

    A reducer that keeps the default empty ``state_error_resolutions``
    receives the unresolved state budget: the Aer receipt of a trajectory
    that saves a state lists "amplitude-derived masses", so its context has
    no modulo-phase state error, and, since the reducer is not registered
    phase invariant, no state error. Only a reducer that declares the label, such as
    ``projected_moments``, has it resolved. The three-argument reducer of
    the same trajectory runs as before.
    """
    import numpy as np
    from nwqlib._prepared_execution import submit_experiment

    contexts = []

    def execute(state, parameters, bindings, *, context):
        contexts.append(context)
        return (np.array([context.probability_window]),)

    monkeypatch.setitem(planning.READOUT_REDUCERS, "window", Reducer(
        shape=lambda parameters: (ReducerOutput("float64", (1,)),), work=lambda parameters, width: 1,
        execute=execute, receives_context=True))
    plan = three_call_plan(trajectory(
        ObservationPoint(id="w", position=1, kind="reduction", reducer="window", parameters={}),
        reduction("end", None)))
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7),
                         reduction_allowance=1 + (2 << 2))
        chunks = submit_experiment(handle, run=run)
    receipt = handle.record
    [context] = contexts
    assert (context.prepared_id, context.probability_window, context.probability_window_exclusions) == (
        receipt.content_id, receipt.saved_state_probability_window, receipt.probability_window_exclusions)
    assert chunks[0].values[0].real == (receipt.saved_state_probability_window,)
    assert "amplitude-derived masses" in receipt.probability_window_exclusions
    assert context.modulo_phase_state_error is None and receipt.saved_state_error()[0] is None
    assert context.state_error is None and receipt.state_error()[0] is None
    assert len(state_reducer) == 1
    with pytest.raises(ValueError, match="state-error resolutions must be a tuple of nonempty labels"):
        Reducer(shape=lambda parameters: (), state_error_resolutions=["amplitude-derived masses"])
    with pytest.raises(ValueError, match="needs the ReductionContext of its producing receipt"):
        planning.execute_reduction(receipt.observation.positions[0], np.zeros(4, complex), bindings=())


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
    with Run(plan, progress=False) as run:
        handle = prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=29), reduction_allowance=4)
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


def _projected_mass_checkpoints(labels, family, state):
    """Acquisition and publication of one projected point on an Aer receipt.

    The receipt is a validated synthetic Aer receipt with G = 0 native
    operations on three qubits, the given exclusion labels and no host phase
    correction (``statevector_roundoff`` of ``(0, 0)``), so its
    ``state_error`` and ``saved_state_error`` are ``native_state_error(0, c)
    = 5u`` when every label is resolved. The two returned callables run the actual
    ``_trajectory_chunks`` on ``state`` and the family's actual
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
    from nwqlib.algorithms.qls import quantum as qls
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
    publish = (lchs if family == "lchs" else qls).projected_moments

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


@pytest.mark.parametrize("family", ["lchs", "qls"])
def test_projected_reducer_resolves_only_the_amplitude_mass_label_at_every_mass_checkpoint(
        family, monkeypatch):
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
        _projected_mass_checkpoints(PROJECTED_MASS_EXCLUSIONS, family, near))
    assert receipt.probability_window_exclusions == ("amplitude-derived masses",)
    assert receipt.state_error()[0] is None
    assert receipt.state_error(PROJECTED_MASS_EXCLUSIONS)[0] == 5 * 2**-53
    for checkpoint in (acquire_deficient_state, publish_deficient_chunk):
        with pytest.raises(ValueError, match="outside the producing window"):
            checkpoint()
    for checkpoint in _projected_mass_checkpoints(PROJECTED_MASS_EXCLUSIONS, family, unit)[1]:
        checkpoint()

    monkeypatch.setitem(planning.READOUT_REDUCERS, "same_kernel", replace(
        planning.registered_reducer(PROJECTED_MOMENTS), state_error_resolutions=()))
    point = receipt.observation.positions[0]
    custom = receipt.revise(observation=trajectory(point.revise(reducer="same_kernel")))
    chunk = steps.acquire(custom, near)
    with pytest.raises(ValueError, match="require the projected_moments reducer"):
        steps.publish(steps.data_for(custom, chunk), chunk)

    receipt, checkpoints, _ = _projected_mass_checkpoints((*PROJECTED_MASS_EXCLUSIONS, "unitary"), family, near)
    assert receipt.state_error(PROJECTED_MASS_EXCLUSIONS)[0] is None
    for checkpoint in checkpoints:
        checkpoint()
    receipt, checkpoints, _ = _projected_mass_checkpoints((), family, near)
    assert receipt.state_error()[0] == 5 * 2**-53
    for checkpoint in checkpoints:
        with pytest.raises(ValueError, match="outside the producing window"):
            checkpoint()


def test_trajectory_metadata_minimum_includes_the_completion_rows_before_native_work(monkeypatch):
    """``M_min(K) = 58*K + G_max`` on Aer, with ``G_max = 1113`` from the completion rows.

    Each Pauli or probability point spends its 58-byte job and empty values
    envelope, and the completion rows their largest growth
    (``_prepared_execution._trajectory_fixed_bounds``,
    ``_probability_point_metadata`` and ``synchronous_completion_growth``);
    a probability point's manifests, summaries and publication rows are
    funded in its header. With the default
    ``max_completion_metadata_bytes = 65536``, ``58*K + 1113 <= 65536``
    admits K = 1110 and refuses K = 1111 before native preparation, for
    Pauli and for one-qubit probability points. A fresh detached NWQ-Sim or
    Slurm outcome is priced at the maximum of ``detached_completion_growth``,
    168 bytes, or 99 bytes with a forecast.
    """
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.backends import AerBackend
    from nwqlib._prepared_execution import (_completion_minimum, detached_completion_growth,
                                            synchronous_completion_growth)
    from test_prepared_execution import base_plan

    # A forecast stores the event already revised, so its parent adds nothing.
    for revised, expected in ((False, (1099, 1113)), (True, (1030, 1044))):
        assert synchronous_completion_growth(
            provider="qiskit_aer", timing_reference="nwqlib.backends.connection.AerBackend.submit", job_length=36,
            max_bytes=1 << 40, event_already_revised=revised) == expected
        run = SimpleNamespace(backend=AerBackend(), forecast=object() if revised else None)
        assert _completion_minimum(run, 1 << 40) == expected[1]
    # A fresh detached NWQ-Sim or Slurm outcome grows by (161, 168), or
    # (92, 99) with a forecast (detached_completion_growth); admission uses
    # the maximum. A backend without an established prototype adds nothing.
    for revised, expected in ((False, (161, 168)), (True, (92, 99))):
        assert detached_completion_growth(max_bytes=1 << 40, event_already_revised=revised) == expected
        for kind in ("nwqsim", "slurm"):
            run = SimpleNamespace(backend=SimpleNamespace(kind=kind), forecast=object() if revised else None)
            assert _completion_minimum(run, 1 << 40) == expected[1]
    assert _completion_minimum(SimpleNamespace(backend=SimpleNamespace(kind="ionq"), forecast=None), 1 << 40) == 0
    monkeypatch.setattr(aer, "_prepare_aer_execution", lambda *args, **kwargs: pytest.fail("native lowering started"))
    base = base_plan()
    for kind in ("pauli", "probabilities"):
        for count, admitted in ((1110, True), (1111, False)):
            if kind == "pauli":
                plan = three_call_plan(trajectory(*(pauli(f"p{index}", min(3, index // 371), "ZZ")
                                                    for index in range(count))))
            else:
                plan = base.revise(experiments=(Experiment(name="trajectory", setting=f"{count} marginals",
                    observation=trajectory(*(ObservationPoint(id=f"p{index}", position=None, kind="probabilities",
                                                              qubits=(0,)) for index in range(count)))),))._bind(
                    blocks=base.blocks)
            with Run(plan, progress=False) as run:
                if admitted:
                    with pytest.raises(pytest.fail.Exception, match="native lowering started"):
                        prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
                else:
                    with pytest.raises(ValueError, match="need at least 65551 bytes, more than "
                                                         "max_completion_metadata_bytes=65536"):
                        prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
                assert run.trace.preparations == int(admitted)
    plan = three_call_plan(trajectory(*(pauli(f"p{index}", min(3, index // 371), "ZZ") for index in range(1111))))
    # A backend that cannot execute trajectories refuses the same schedule
    # naming the capability, not the metadata allowance, at the default limit.
    from qiskit_aer.noise import NoiseModel
    with Run(plan, backend=AerBackend.from_noise_model(NoiseModel()), progress=False) as run:
        with pytest.raises(ValueError, match="exact readouts require a noiseless backend"):
            prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        assert run.trace.preparations == 0


def test_the_aer_target_executes_every_registered_reducer_and_views_only_pauli_or_probability_points(
        two_complex_scalars):
    """``reducers="registered"`` admits any registered reducer; a name tuple admits only its names.

    A readout view is refused on a reduction point as an unsupported capability.
    """
    from nwqlib.backends.capabilities import unsupported_readout
    from nwqlib.backends.targets import AER_STATEVECTOR_TARGET

    point = ObservationPoint(id="r", position=1, kind="reduction", reducer="two_complex")
    schedule = trajectory(point)
    assert AER_STATEVECTOR_TARGET.reducers == "registered"
    assert unsupported_readout(AER_STATEVECTOR_TARGET, schedule) is None
    assert unsupported_readout(AER_STATEVECTOR_TARGET.revise(reducers=()), schedule) == (
        "reduction 'two_complex' at point 'r'")
    assert unsupported_readout(AER_STATEVECTOR_TARGET.revise(reducers=("two_complex",)), schedule) is None
    unknown = trajectory(point.revise(reducer="absent"))
    assert unsupported_readout(AER_STATEVECTOR_TARGET, unknown) == "unregistered reduction 'absent' at point 'r'"
    viewed = trajectory(point.revise(view=ReadoutView(tail="t", inverse="t_inverse", wires=(0,))))
    assert unsupported_readout(AER_STATEVECTOR_TARGET, viewed) == (
        "the readout view of reduction point 'r'; remove view= from that point to observe it without a view")


def test_the_trajectory_memory_check_and_the_simulator_share_one_thread_cap(monkeypatch, tmp_path):
    """The saved-buffer admission charges the thread cap that Aer is given; restoration reapplies it.

    The cap is resolved once per open Run: a reopened Run resolves it again.
    """
    import io
    from itertools import chain, repeat
    from qiskit import qpy
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib.backends import qiskit_aer as aer

    seen = []
    admit = aer._admit_aer_saved_buffers
    caps = chain([3], repeat(5))
    monkeypatch.setattr(aer, "_aer_state_threads", lambda: next(caps))
    monkeypatch.setattr(aer, "_admit_aer_saved_buffers",
                        lambda observation, **kwargs: seen.append(kwargs["state_threads"]) or admit(observation,
                                                                                                    **kwargs))
    plan = three_call_plan(trajectory(ObservationPoint(id="m", position=1, kind="probabilities", qubits=(1,)),
                                      pauli("end", None, "ZZ")))
    with Run(plan, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
    native = handle._native
    assert seen == [3]
    assert native.simulator.options.max_parallel_threads == native.metadata["simulator_max_parallel_threads"] == 3
    stream = io.BytesIO()
    qpy.dump(native.circuit, stream)
    stream.seek(0)
    saved, = qpy.load(stream)
    restored = AerBackend().restore_native_data(handle.record, saved, native.metadata, run=None)
    assert restored.simulator.options.max_parallel_threads == 3
    metadata = {key: value for key, value in native.metadata.items() if key != "simulator_max_parallel_threads"}
    with pytest.raises(ValueError, match="simulator_max_parallel_threads"):
        AerBackend().restore_native_data(handle.record, saved, metadata, run=None)
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as reopened:
        again = prepare(reopened.plan.resolve("trajectory"), run=reopened, runtime=RuntimeOptions(seed=7))
    assert seen == [3, 5] and again._native.simulator.options.max_parallel_threads == 5


def _zero_state_plan(width, *points):
    """A trajectory over an allocated ``width``-qubit zero state with the given points."""
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.ir import Allocate, Definition, Program, Register, Release, Sequence
    from test_prepared_execution import base_plan

    program = Program(root="body", registers=(Register(name="q", width=width),), definitions=(
        Definition(id="allocate", node=Allocate(wire="q")), Definition(id="release", node=Release(wire="q")),
        Definition(id="body", node=Sequence(children=("allocate", "release")))))
    base = base_plan().revise(construction=SelectedConstruction(program=program, selections=()), shots=None)._bind()
    return base.revise(experiments=(Experiment(name="t", setting=f"{len(points)} marginals",
                                               observation=trajectory(*points)),))._bind(blocks=base.blocks)


@pytest.mark.parametrize("kind,count", [("probabilities", 1), ("pauli_expectation", 1), ("probabilities", 16)])
def test_an_admitted_trajectory_completes_at_its_required_metadata_allowance(monkeypatch, kind, count):
    """``58*K + 1113`` bytes admit a trajectory that completes; one byte less refuses before lowering.

    Two-qubit points on |00> store a probability marginal sparse (two
    manifests and four publication rows), the largest encoding at this
    width, whose record and publication rows the point header funds
    (``_prepared_execution._probability_point_metadata``). The completion
    timestamps carry microseconds, the largest growth of the completion rows
    (``synchronous_completion_growth``), so the allowance is exactly
    sufficient: K = 1 needs 1171 bytes and K = 16 needs 2041.
    """
    import nwqlib._prepared_execution as owner
    from nwqlib.blocks import lowering
    from nwqlib.execution import ExecutionLimits, submit_experiment

    fields = (dict(kind="probabilities", qubits=(0, 1)) if kind == "probabilities"
              else dict(kind="pauli_expectation", labels=("ZZ",)))
    plan = _zero_state_plan(2, *(ObservationPoint(id=f"p{index}", position=None, **fields) for index in range(count)))
    monkeypatch.setattr(owner, "_now", lambda: "2026-09-29T12:34:56.123456+00:00")
    lowered = []
    original = lowering._lower_qiskit
    monkeypatch.setattr(lowering, "_lower_qiskit", lambda *a, **k: lowered.append(1) or original(*a, **k))
    required = 58 * count + 1113
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=required - 1)) as run:
        with pytest.raises(ValueError, match=f"need at least {required} bytes, more than "
                                             f"max_completion_metadata_bytes={required - 1}, before native work"):
            prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))
    assert lowered == []
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=required)) as run:
        chunks = submit_experiment(prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=7)), run=run)
        assert len(chunks) == count and run.trace.events[-1].status == "completed"
        if kind == "probabilities":
            assert {chunk.values[0].encoding for chunk in chunks} == {"sparse"}


def test_probability_header_credit_covers_each_funded_byte_once(monkeypatch):
    """Completion credits a probability point's funded header bytes once and leaves the rest in the allowance.

    The metadata a completion spends, after replaced rows and the probability
    credit (``_prepared_execution._probability_header_credit``), is the
    job envelope and completion-row growth, 1171 bytes for one Aer point with
    microsecond timestamps: for a repeated acquisition whose payload row is
    already stored, for an empty sparse marginal (two manifests of zero
    entries) and, with 120 more bytes, for a 56-character job text whose six
    occurrences exceed the 36-character prototype by 20 each. A probability
    endpoint funds its whole chunk remainder from the allowance
    (``_probability_endpoint_minimum``): with an empty sparse one-qubit
    marginal it needs 6902 bytes, and 6901 refuses before any event.
    """
    import numpy as np
    from types import SimpleNamespace
    import nwqlib._prepared_execution as owner
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits, submit_experiment
    from test_prepared_execution import base_plan

    monkeypatch.setattr(owner, "_now", lambda: "2026-09-29T12:34:56.123456+00:00")
    spent = []
    write = owner.Run._write

    def measured(self, records=(), **kwargs):
        if kwargs.get("release_output") is not None and any(kind == "chunk" for kind, _, _ in records):
            _, raw = owner._completion_metadata_bound(records, self.limits.max_data_bytes)
            replaced = sum(self._state["data_sizes"].get((kind, key), 0) for kind, key, _ in records)
            spent.append(raw - replaced - owner._probability_header_credit(records, self.limits.max_data_bytes,
                                                                           run=self))
        return write(self, records, **kwargs)

    monkeypatch.setattr(owner.Run, "_write", measured)
    base = base_plan()
    point = ObservationPoint(id="p0", position=None, kind="probabilities", qubits=(0,))
    plan = base.revise(experiments=(Experiment(name="t", setting="1 marginals", observation=trajectory(point)),))._bind(
        blocks=base.blocks)
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=1171)) as run:
        handle = prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))
        for _ in range(2):
            submit_experiment(handle, run=run)
        # The second acquisition's payload row is already stored.
        assert spent == [1171, 1171]

    def supplied(job):
        return lambda native: SimpleNamespace(raw_output={"trajectory": {"p0": {"probabilities": np.zeros(2)}}},
                                              metadata={"native_job_id": job})

    for job, allowance in (("x" * 36, 1171), ("x" * 56, 2000)):
        spent.clear()
        monkeypatch.setattr(aer, "_submit_aer_execution", supplied(job))
        with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=allowance)) as run:
            (chunk,) = submit_experiment(prepare(plan.resolve("t"), run=run, runtime=RuntimeOptions(seed=7)), run=run)
            assert chunk.values[0].encoding == "sparse" and chunk.values[0].entries == 0
            assert spent == [1171 + 6 * (len(job) - 36)]
    single = base.revise(experiments=(Experiment(name="t", setting="1 marginals", observation=ObservationSpec(
        kind="probabilities", qubits=(0,))),))._bind(blocks=base.blocks)
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: SimpleNamespace(
        raw_output={"probabilities": np.zeros(2)}, metadata={"native_job_id": "x" * 36}))
    with Run(single, limits=ExecutionLimits(max_completion_metadata_bytes=6901)) as run:
        with pytest.raises(ValueError, match="need at least 6902 bytes, more than max_completion_metadata_bytes=6901"):
            prepare(single.resolve("t"), run=run, runtime=RuntimeOptions(seed=7))
        assert not run.trace.events and run.trace.preparations == 0
    spent.clear()
    with Run(single, limits=ExecutionLimits(max_completion_metadata_bytes=6902)) as run:
        submit_experiment(prepare(single.resolve("t"), run=run, runtime=RuntimeOptions(seed=7)), run=run)
        assert spent == [6902] and run.trace.events[-1].status == "completed"


@pytest.mark.parametrize("shots", [None, 200])
def test_counts_and_exact_scalar_metadata_refuse_before_the_backend_call(monkeypatch, shots):
    """Counts and exact Pauli scalar readouts check their completion metadata before native preparation.

    ``prepare_static`` checks the largest known requirement of the Plan's
    settings (``_prepared_execution._scalar_endpoint_metadata``) before the
    first preparation, so an allowance one byte below it refuses with no
    ``AerBackend.submit`` call, no event and no preparation, and the value the
    refusal names completes on the first retry.
    """
    import re
    import nwqlib
    from nwqlib.algorithms import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.execution import ExecutionLimits
    from nwqlib.problems import Expectation

    calls = []
    submit = AerBackend.submit
    monkeypatch.setattr(AerBackend, "submit", lambda self, *args, **kwargs: calls.append(1) or submit(
        self, *args, **kwargs))
    plan = nwqlib.plan(Expectation(state=[1, 1], observable=[[1, 0], [0, -.5]]), method=ExpectationMethod(),
                       shots=shots, seed=7)
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=1000)) as run:
        with pytest.raises(ValueError, match="need max_completion_metadata_bytes of at least") as caught:
            plan.method.prepare(plan, run=run)
        assert not run.trace.events and run.trace.preparations == 0
    required = int(re.search(r"at least (\d+) per completion", str(caught.value)).group(1))
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=required - 1)) as run:
        with pytest.raises(ValueError, match=f"at least {required} per completion, exceeding {required - 1}"):
            plan.method.prepare(plan, run=run)
    assert calls == []
    prepared = nwqlib.prepare(plan, limits=ExecutionLimits(max_completion_metadata_bytes=required), progress=False)
    with prepared.run:
        nwqlib.submit(prepared).wait()
        assert [event.status for event in prepared.run.trace.events] == ["completed"]
    assert calls == [1]


@pytest.mark.parametrize("case", ["rfe", "qls"])
def test_static_metadata_check_names_the_largest_setting_requirement(case):
    """A static Plan's metadata refusal names its largest setting requirement before the first preparation.

    The sampled RFE Plan's settings need 2,079 or 2,081 bytes by the decimal
    width of their pooled shots (``_prepared_execution._scalar_endpoint_metadata``),
    and its first settings need the smaller value. ``prepare_static`` checks
    the maximum over all settings, so an allowance of 2,079 refuses naming
    2,081 with no preparation. The exact QLS amplitude endpoint needs 5,316
    bytes (``_prepared_execution._amplitude_endpoint_metadata``).
    """
    import nwqlib
    from nwqlib import Eigenproblem, LinearSystem
    from nwqlib.algorithms import QLS, RFE
    from nwqlib.execution import ExecutionLimits

    if case == "rfe":
        plan = nwqlib.plan(Eigenproblem(A=[[.2, 0], [0, .7]]), method=RFE(initial_state=[1, 0]), shots=200, seed=7)
        required, allowance = 2081, 2079
    else:
        plan = nwqlib.plan(LinearSystem(A=[[1.1, .1], [.1, .9]], b=[1, .25]), method=QLS(), seed=7)
        required, allowance = 5316, 5315
    with Run(plan, limits=ExecutionLimits(max_completion_metadata_bytes=allowance)) as run:
        with pytest.raises(ValueError, match=f"at least {required} per completion, exceeding {allowance}"):
            plan.method.prepare(plan, run=run)
        assert not run.trace.events and run.trace.preparations == 0


def test_static_scan_admits_each_setting_once_and_stops_pricing_at_a_known_cap_excess(monkeypatch):
    """The Aer static scan admits each setting's Program once and prices no setting past a known cap excess.

    One readout resolution per setting (``_selected_readout``) gives both the
    observation and the header of the completion-metadata law, so the scan
    calls ``_Admission.admitted`` once per setting. With
    ``max_total_circuits=1`` the second setting already exceeds the cap, so
    only the first setting is priced, and the refusal names the complete
    population of the Plan.
    """
    import nwqlib
    from nwqlib import Eigenproblem, _prepared_execution
    from nwqlib.algorithms import RFE
    from nwqlib.execution import ExecutionLimits
    from nwqlib.ir.validation import _Admission

    plan = nwqlib.plan(Eigenproblem(A=[[.2, 0], [0, .7]]), method=RFE(initial_state=[1, 0]), shots=200, seed=7)
    calls = {"admitted": 0, "priced": 0}
    admitted, priced = _Admission.admitted, _prepared_execution._known_endpoint_metadata
    monkeypatch.setattr(_Admission, "admitted", lambda self: calls.__setitem__(
        "admitted", calls["admitted"] + 1) or admitted(self))
    monkeypatch.setattr(_prepared_execution, "_known_endpoint_metadata", lambda *args, **kwargs: calls.__setitem__(
        "priced", calls["priced"] + 1) or priced(*args, **kwargs))

    class ScanDone(Exception):
        pass

    def stop(*args, **kwargs):
        raise ScanDone

    monkeypatch.setattr(_prepared_execution, "_prepare_static_item", stop)
    with Run(plan) as run, pytest.raises(ScanDone):
        _prepared_execution.prepare_static(plan, run=run)
    assert calls == {"admitted": len(plan.experiments), "priced": len(plan.experiments)} and len(plan.experiments) > 2
    calls.update(admitted=0, priced=0)
    with Run(plan, limits=ExecutionLimits(max_total_circuits=1)) as run:
        with pytest.raises(ValueError, match=rf"max_total_circuits=1 cannot admit .*\(0 counted, {len(plan.experiments)} requested"):
            _prepared_execution.prepare_static(plan, run=run)
        assert run.trace.preparations == 0
    assert calls == {"admitted": len(plan.experiments), "priced": 1}
    # The cap refusal also comes before a metadata refusal that the priced
    # first setting alone would cause.
    with Run(plan, limits=ExecutionLimits(max_total_circuits=1, max_completion_metadata_bytes=1)) as run:
        with pytest.raises(ValueError, match=rf"max_total_circuits=1 cannot admit .*\(0 counted, {len(plan.experiments)} requested"):
            _prepared_execution.prepare_static(plan, run=run)
