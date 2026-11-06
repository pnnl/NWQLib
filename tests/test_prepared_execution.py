"""Actual one-qubit preparation, native caches, noise and cumulative failures."""

import importlib.metadata
from types import SimpleNamespace

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import PreparationNotRebuilt, Run, prepare_experiment, submit_experiment
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.backends import AerBackend
from nwqlib.backends import qiskit_aer as aer
from nwqlib.blocks.lowering import INSTRUCTION_BYTES
from nwqlib.blocks.selection import NATIVE_POSITION_BYTES
from nwqlib._choice_archive import ArchiveFiles
from nwqlib.core.planning import RandomStreams
from nwqlib.execution import ExecutionLimits
from nwqlib.problems import Eigenproblem, Samples
from test_expectation_current import problem
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def base_plan(*, shots=None):
    return nwqlib.plan(problem(state=[1.0, 1.0]), method=ExpectationMethod(), shots=shots, seed=7)


def hadamard_expectation(width):
    """A Pauli expectation on ``width`` qubits, each in |+>, prepared from the circuit itself."""
    from qiskit import QuantumCircuit
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.inputs import state_input

    state = QuantumCircuit(width)
    state.h(range(width))
    observable = ingest_pauli((("Z" * width, 1.0),), num_qubits=width)
    return nwqlib.plan(nwqlib.Expectation(state=state_input(state), observable=observable),
                       method=ExpectationMethod(), seed=7)


@pytest.mark.parametrize("width, memory_mb", [(16, 1), (17, 2)])
def test_a_statevector_simulation_within_simulator_memory_is_prepared(width, memory_mb):
    prepared = nwqlib.prepare(hadamard_expectation(width), backend=AerBackend(),
                              limits=ExecutionLimits(simulator_memory_mb=memory_mb), progress=False)
    with prepared.run as run:
        assert run.trace.preparations == 1 and run.trace.submissions == ()
        assert [circuit.num_qubits for circuit in prepared.circuits] == [width]


def test_a_statevector_simulation_beyond_simulator_memory_is_refused_before_lowering(monkeypatch):
    from nwqlib.blocks import lowering

    selected = hadamard_expectation(17)
    with Run(selected, backend=AerBackend(), limits=ExecutionLimits(simulator_memory_mb=1)) as run:
        with monkeypatch.context() as patch:
            patch.setattr(lowering, "_lower_qiskit", lambda *a, **k: pytest.fail("the memory check followed lowering"))
            with pytest.raises(ValueError) as refused:
                selected.method.prepare(selected, run=run)
        message = str(refused.value)
        assert "17 qubits" in message and f"{16 << 17} bytes" in message
        assert f"simulator_memory_mb=1 ({1 << 20} bytes)" in message and "run.extend_limits(simulator_memory_mb=" in message
        assert run.trace.preparations == 0 and run.trace.submissions == ()


def test_aer_counts_the_saved_amplitudes_and_probabilities_of_a_readout():
    def admit(width, kind, **fields):
        return aer._admit_aer_saved_buffers(SimpleNamespace(kind=kind, **fields), width=width, memory_mb=1,
                                            state_threads=1)

    assert admit(16, "pauli_expectation") == 0
    assert admit(15, "amplitudes") == 16 << 15
    assert admit(15, "probabilities", qubits=tuple(range(15))) == 8 << 15
    with pytest.raises(ValueError, match="saved amplitudes"):
        admit(16, "amplitudes")
    with pytest.raises(ValueError, match="saved probabilities"):
        admit(16, "probabilities", qubits=tuple(range(15)))


def test_counts_on_aer_leave_their_memory_to_aer():
    assert AerBackend().admit_simulation_memory(SimpleNamespace(kind="counts"), width=40, memory_mb=1, run=None) == 0


def test_simulator_memory_precedes_readout_size_and_explains_binary_applicability(tmp_path, monkeypatch):
    import sympy as sp
    import nwqlib._prepared_execution as owner
    from nwqlib.algorithms.qhd import QHD, UniformState
    from nwqlib.problems import Optimization

    x, y = sp.symbols('x y', real=True)
    problem = Optimization(objective=x*x + y*y, variables=(x, y), bounds=((-1., 1.), (-1., 1.)))
    method = QHD(boundary='periodic', num_grid_points=16, num_steps=1,
                 total_time=.01, initial_state=UniformState())
    chosen = nwqlib.plan(problem, method=method, seed=7)
    limits = ExecutionLimits(simulator_memory_mb=1024)
    with Run(chosen, limits=limits, directory=tmp_path / 'refused') as run:
        with monkeypatch.context() as patch:
            patch.setattr(owner, 'readout_shape', lambda **kw: pytest.fail('readout size preceded the memory check'))
            with pytest.raises(ValueError, match='simulator_memory_mb=1024') as refused:
                method.prepare(chosen, run=run)
        message = str(refused.value)
        assert '32 qubits' in message and f'{16 << 32} bytes' in message and 'encoding="binary"' in message
        assert '8 qubits' in message
        assert run.trace.preparations == 0
    binary = nwqlib.plan(problem, method=method.revise(encoding='binary'), seed=7)
    prepared = nwqlib.prepare(binary, limits=limits, directory=tmp_path / 'binary', progress=False)
    with prepared.run:
        assert prepared.circuits and all(c.num_qubits == 8 for c in prepared.circuits)


def test_native_cache_delta_and_idle_release_follow_actual_preparations(tmp_path, monkeypatch):
    import nwqlib._run_archive as archive
    from qiskit import qpy
    import gc
    import weakref

    count = 4  # preparations; a per-preparation visit count differs from a cumulative one

    visits, encodings = [], []
    entry, dump = archive._cache_entry, qpy.dump

    def visited(token, value, files, name):
        visits.append(token)
        return entry(token, value, files, name)

    def encoded(*args, **kwargs):
        encodings.append(1)
        return dump(*args, **kwargs)

    monkeypatch.setattr(archive, "_cache_entry", visited)
    monkeypatch.setattr(qpy, "dump", encoded)
    with Run(selected(), directory=tmp_path / "delta") as run:
        handles = [prepare_experiment(run.plan.resolve("counts"), run=run) for _ in range(count)]
        assert len(visits) == count and all(token[0] == "handle" for token in visits)
        assert len(encodings) == count  # Empty one-qubit measurement circuits, no definitions.
        before = len(visits), len(encodings)
        with monkeypatch.context() as patch:
            patch.setattr(run._state["handles"], "items", lambda: pytest.fail("checkpoint visited native history"))
            for cache in run._state["definitions"]:
                patch.setattr(cache, "items", lambda: pytest.fail("checkpoint visited definition history"))
            run.checkpoint({"unchanged": True})
        assert (len(visits), len(encodings)) == before
        # One measured bit per handle. Only what release_native releases leaves the
        # count of held circuits, and restoring a released circuit brings it back.
        one = NATIVE_POSITION_BYTES
        assert run._state["native_held"] == count * one
        refused = run.release_native()
        assert not refused["released"] and len(refused["kept"]) == count
        assert run._state["native_held"] == count * one
        # Real tiny Aer acquisitions make the first two handles fully idle.
        for handle in handles[:2]:
            run.collect(submit_experiment(handle, run=run))
        stored = tuple(handles)
        circuits = tuple(handle.inspect_circuit() for handle in stored)
        native_refs = tuple(weakref.ref(handle._native) for handle in handles[:2])
        release = run.release_native()
        assert release["released"] == tuple(handle.record.content_id for handle in handles[:2])
        assert tuple(run._state["handles"].values()) == stored
        assert all(handle._native is None for handle in handles[:2])
        assert run._state["native_held"] == (count - 2) * one
        gc.collect()
        assert all(reference() is None for reference in native_refs)
        with monkeypatch.context() as patch:
            patch.setattr(aer, "_prepare_aer_execution", lambda *a, **k: pytest.fail("released handle rebuilt"))
            patch.setattr(aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("inspection acquired"))
            inspected = tuple(handle.inspect_circuit() for handle in stored)
        assert inspected == circuits
        assert run._state["native_held"] == count * one
        inspected[0].x(0)
        assert stored[0].inspect_circuit() == circuits[0]
        assert run.trace.preparations == count and sum(event.shots for event in run.trace.events) == 14
        assert (len(visits), len(encodings)) == before

    # A released circuit is new memory when it comes back. Two circuits fit 2*200 + 256, and two more once
    # the first two are released. One old circuit then fits at 600, and a second would hold 800, which is
    # refused before its file is read, with nothing charged.
    class Limited(CountsMethod):
        max_bytes: int | None = None

    limit = 2 * NATIVE_POSITION_BYTES + INSTRUCTION_BYTES
    limited = Limited(max_bytes=limit).plan(Eigenproblem(A=[[1, 0], [0, -1]]), output=Samples(), execution="quantum",
                                            shots=7, rng=RandomStreams(31))
    with Run(limited, directory=tmp_path / "limited") as run:
        old = [prepare_experiment(run.plan.resolve("counts"), run=run) for _ in range(2)]
        for handle in old:
            run.collect(submit_experiment(handle, run=run))
        assert len(run.release_native()["released"]) == 2 and run._state["native_held"] == 0
        for _ in range(2):
            prepare_experiment(run.plan.resolve("counts"), run=run)
        old[0].inspect_circuit()
        assert run._state["native_held"] == 3 * NATIVE_POSITION_BYTES
        with monkeypatch.context() as patch:
            patch.setattr(ArchiveFiles, "read_circuit", lambda *a: pytest.fail("decoded before the comparison"))
            with pytest.raises(ValueError, match=rf"restoring this prepared circuit needs about {4 * NATIVE_POSITION_BYTES} "
                                                 rf"bytes, exceeding max_bytes={limit}.*Run.release_native"):
                old[1].inspect_circuit()
        assert old[1]._native is None and run._state["native_held"] == 3 * NATIVE_POSITION_BYTES
        assert run.trace.preparations == 4

    # Saving must not decode released circuits to write them, so it cannot exceed the limit or change the Run.
    # Where the cache files cannot be copied as they are, the rows are rewritten and the files come along.
    with Run(selected(), directory=tmp_path / "source") as run:
        released = [prepare_experiment(run.plan.resolve("counts"), run=run) for _ in range(2)]
        for handle in released:
            run.collect(submit_experiment(handle, run=run))
        circuit = released[0].inspect_circuit()
        assert len(run.release_native()["released"]) == 2
        with monkeypatch.context() as patch:
            patch.setattr(archive, "_copy_cache_files", lambda *a: False)
            patch.setattr(ArchiveFiles, "read_circuit", lambda *a: pytest.fail("save decoded a released circuit"))
            run.save(tmp_path / "copy")
        assert all(handle._native is None for handle in released) and run._state["native_held"] == 0
    with archive.load(tmp_path / "copy", backend=run.backend, method=CountsMethod) as copy:
        reopened = copy._state["handles"][released[0].record.content_id]
        assert reopened.inspect_circuit() == circuit and copy._state["native_held"] == NATIVE_POSITION_BYTES


def test_progress_uses_published_work_and_never_repeats_acquisition(tmp_path):
    from nwqlib.core.planning import RandomStreams

    InjectedBackend.supports_synchronous = True
    chosen = selected()
    expected_rng = RandomStreams.restore(chosen.randomness)
    expected_rng.next_seed()  # One actual selected acquisition, including its preparation.
    notices = []

    def observe(stage, done, total):
        notices.append((stage, done, total, tuple(name for name, _ in InjectedBackend.calls)))
        if stage == "collect":
            raise RuntimeError("observer unavailable")

    prepared = nwqlib.prepare(chosen, backend=InjectedBackend(), directory=tmp_path / "progress",
                              progress=observe)
    with nwqlib.submit(prepared) as run:
        result = run.wait()
        assert result.counts == 7 and run.rng.snapshot() == expected_rng.snapshot()
        assert [name for name, _ in InjectedBackend.calls].count("prepare") == 1
        assert [name for name, _ in InjectedBackend.calls].count("launch") == 1
        assert notices[0][:3] == ("input", 1, 1)
        before, after = [row for row in notices if row[0] == "prepare"]
        assert before[1:] == (None, None, ())
        assert after[1:] == (1, None, ("prepare",))
        (collected,) = [row for row in notices if row[0] == "collect"]
        assert collected[1:3] == (len(result.data.observations.chunks), None)
        assert collected[3].count("launch") == 1
        assert len(run.warnings) == 1 and "observer unavailable" in run.warnings[0]
        assert run._state["progress"] is False


def test_prepared_resource_receiver_inspects_only_the_selected_live_native(monkeypatch):
    from qiskit import QuantumCircuit
    from nwqlib._prepared_execution import Prepared

    run = Run(base_plan())
    first = prepare_experiment(run.plan.resolve("expectation"), run=run)
    second = prepare_experiment(run.plan.resolve("expectation"), run=run)
    prepared = Prepared(run)
    monkeypatch.setattr(QuantumCircuit, "copy", lambda *a, **k: pytest.fail("raw inventory copied native data"))
    import copy
    monkeypatch.setattr(copy, "deepcopy", lambda *a, **k: pytest.fail("raw inventory deep-copied native data"))
    import qiskit
    from qiskit_aer import AerSimulator
    monkeypatch.setattr(qiskit, "transpile", lambda *a, **k: pytest.fail("raw inventory compiled"))
    monkeypatch.setattr(AerSimulator, "run", lambda *a, **k: pytest.fail("raw inventory simulated"))
    inventory = prepared.inspect_resources(index=1)
    # State (1, 1)/sqrt(2) opens the circuit and is set directly; the exact
    # expectation is one Aer save. Both are Aer directives, of depth zero.
    assert inventory["operations"] == {"set_statevector": 1, "save_expval": 1}
    assert inventory["total_operations"] == 2 and inventory["depth"] == 0
    assert inventory["circuit"] == (
        f"prepared circuit {second.record.content_id}; not executed by inspection")
    assert first.record.content_id != second.record.content_id
    assert run.trace.jobs == 0
    with pytest.raises(ValueError, match="no prepared quantum"):
        prepared.inspect_resources(index=2)
    run.close()
    with pytest.raises(ValueError, match="closed"):
        prepared.inspect_resources()


def same_native(actual, original):
    if isinstance(original, np.ndarray):
        assert actual.dtype == original.dtype and actual.shape == original.shape
        np.testing.assert_array_equal(actual, original)
    elif isinstance(original, dict):
        assert actual.keys() == original.keys()
        for key in original:
            same_native(actual[key], original[key])
    elif isinstance(original, (tuple, list)):
        assert len(actual) == len(original)
        for a, b in zip(actual, original, strict=True):
            same_native(a, b)
    else:
        assert actual == original


@pytest.mark.parametrize("preparations", [2])
def test_noise_input_and_cached_native_preparations_reopen_without_serializing_again(
    tmp_path, monkeypatch, preparations
):
    """Count noise-model serialization and poison repeated QPY writes after reopening the
    original preparation cache.
    """
    from qiskit import qpy
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error

    model = NoiseModel()
    model.add_all_qubit_readout_error(ReadoutError([[0.9, 0.1], [0.2, 0.8]]))
    model.add_all_qubit_quantum_error(amplitude_damping_error(0.25), ["ry"])
    expected = model.to_dict()
    encoded, decoded = [], []
    import nwqlib._run_archive as archive
    encode, decode = NoiseModel.to_dict, archive._load_noise_model

    def write(self, serializable=False):
        encoded.append(self)
        return encode(self, serializable=serializable)

    def read(value):
        decoded.append(value)
        return decode(value)

    monkeypatch.setattr(NoiseModel, "to_dict", write)
    monkeypatch.setattr(archive, "_load_noise_model", read)
    monkeypatch.setattr(NoiseModel, "from_dict", lambda *a: pytest.fail("deprecated noise loader"))
    backend = AerBackend.from_noise_model(model)
    metadata = AerBackend.model_validate_json(backend.model_dump_json())
    assert not encoded and not decoded and metadata.noise_model_id == backend.noise_model_id
    assert (
        backend.model_copy()._bound_noise_model() is model and backend.revise()._bound_noise_model() is model
    )
    path = tmp_path / "noise"
    with Run(base_plan(shots=8), backend=backend, directory=path) as run:
        for _ in range(preparations):
            handle = prepare_experiment(run.plan.resolve("group_0"), run=run)
            assert handle.record.backend_configuration_id == backend.content_id
            assert "explicit noise model" in handle.record.target.domain
        run.checkpoint({"prepared": preparations})
        assert run.trace.jobs == 0 and len(encoded) == 1 and not decoded
    with nwqlib.load_run(path, backend=metadata) as run:
        assert len(decoded) == 1
        # The archive keeps the basis gates beside to_dict (_run_archive._noise_model_data).
        same_native(decoded[0], {**expected, "basis_gates": sorted(model.basis_gates)})
        same_native(encode(run.backend._bound_noise_model()), expected)
        with monkeypatch.context() as patch:
            patch.setattr(
                qpy, "dump", lambda *a, **k: pytest.fail("unchanged loaded native cache serialized again")
            )
            run.checkpoint({"loaded": True})
        assert len(encoded) == len(decoded) == 1 and run.trace.jobs == 0
        assert len(tuple(path.glob("*noise-model.json"))) == 1
    # load_run bound the rebuilt model to the reopened Run's own copy of the
    # backend, so the configuration passed to it is still unbound.
    with Run(base_plan(shots=8), backend=metadata) as run:
        with pytest.raises(ValueError, match="noise model is unbound"):
            prepare_experiment(run.plan.resolve("group_0"), run=run)
        assert not run.trace.jobs
    with Run(base_plan(), backend=backend) as run:
        with pytest.raises(ValueError, match="exact readouts require a noiseless"):
            prepare_experiment(run.plan.resolve("expectation"), run=run)
        assert not run.trace.preparations
    with Run(base_plan()) as run:
        handle = prepare_experiment(run.plan.resolve("expectation"), run=run)
        assert "explicit noise model" not in handle.record.target.domain


def test_prepared_reuse_keeps_same_native_object_and_failed_actual_exposure(monkeypatch, tmp_path):
    InjectedBackend.supports_synchronous = True
    run = Run(
        selected(),
        backend=InjectedBackend(),
        limits=ExecutionLimits(max_total_shots=21),
        directory=tmp_path / "reuse",
    )
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    native = handle._native
    first = submit_experiment(handle, run=run)
    second = submit_experiment(handle, run=run)
    assert first.histogram().index_list() == second.histogram().index_list()
    assert first.histogram().weights.tolist() == second.histogram().weights.tolist()
    assert first.acquisition_key != second.acquisition_key
    assert handle._native is native and run.trace.preparations == 1
    InjectedBackend.fail = True
    with pytest.raises(RuntimeError, match="native read failed"):
        submit_experiment(handle, run=run)
    assert run.exposure["completed"]["shots"] == 14 and run.exposure["uncertain"]["shots"] == 7
    assert run.trace.submissions[-1].status == "uncertain"
    with pytest.raises(ValueError, match="max_total_shots"):
        submit_experiment(handle, run=run)
    assert sum(name == "launch" for name, _ in InjectedBackend.calls) == 3
    assert sum(name == "prepare" for name, _ in InjectedBackend.calls) == 1
    run.close()


@pytest.mark.parametrize("channel", ["readout", "sx_damping"])
def test_bound_noise_applies_after_translation_to_effective_basis(channel):
    from qiskit_aer.noise import NoiseModel, ReadoutError, amplitude_damping_error

    model = NoiseModel()
    if channel == "readout":
        model.add_all_qubit_readout_error(ReadoutError([[0.0, 1.0], [0.0, 1.0]]))
    else:
        model.add_all_qubit_quantum_error(amplitude_damping_error(1.0), ["sx"])
    plan = nwqlib.plan(problem((("Z", 1.0),), state=[1.0, 1.0]), method=ExpectationMethod(), shots=16, seed=7)
    result = nwqlib.solve(plan, backend=AerBackend.from_noise_model(model))
    # Readout maps both bits to one. Damping after the final translated SX
    # resets to zero; attaching noise only to the original preparation's name
    # would miss that channel entirely and leave the mean near zero.
    assert result.value == (-1.0 if channel == "readout" else 1.0)
    assert result.data.observations.chunks[0].returned_shots == 16
    (receipt,) = result.data.receipts
    assert "noise-target basis translation" in receipt.compiler.name
    assert "u" not in receipt.native_basis
    exact = nwqlib.solve(problem((("Z", 1.0),), state=[1.0, 1.0]), method=ExpectationMethod())
    assert exact.value == pytest.approx(0.0, abs=1e-13, rel=0)


def test_a_direct_preparation_beyond_the_physical_memory_is_refused_before_any_preparation(monkeypatch):
    from nwqlib import _limits
    from nwqlib.blocks.selection import direct_preparation, transform_block
    from nwqlib.subroutines.state_preparation.direct import direct_tree_bytes
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems import inputs
    from nwqlib.resources.records import ResourceContext

    # A four-amplitude direct PREP of magnitude/phase form.
    plan = nwqlib.plan(Expectation(state=[0.6, 0.0, 0.0, 0.8], observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    # Planning and the circuit-free estimate build no tree. Each component
    # costs 2*(2**2-2) = 4 CX slots.
    report = nwqlib.estimate(plan, context=ResourceContext(basis="cx"))
    components = report.quantity("preparation_components").fact.value
    cx = report.quantity("cx")
    assert components.denominator == cx.fact.value.denominator == 1 and components.numerator >= 1
    assert cx.interpretation == "upper_bound" and cx.fact.value.numerator == 4 * components.numerator
    # Its tree is counted as 4*2**2 + 2*2 = 20 instructions of 256 bytes.
    # It opens the circuit, so noiseless Aer sets the state instead, 48*2**2
    # = 192 bytes, while a controlled form always builds the tree.
    prep, = (block for block in plan.blocks if direct_preparation(block) is not None)
    assert direct_preparation(prep) == (2, False) and direct_tree_bytes(2) == 5120
    assert direct_preparation(transform_block("controlled", prep, control=True)) == (2, True)

    calls = []
    original = inputs.prepare_qiskit

    def recorded(state, **kwargs):
        calls.append(state)
        return original(state, **kwargs)

    monkeypatch.setattr(inputs, "prepare_qiskit", recorded)
    run = Run(plan, backend=AerBackend())
    with monkeypatch.context() as memory:
        memory.setattr(_limits, "physical_memory_bytes", lambda: 96)
        with pytest.raises(ValueError, match="direct state preparations of this circuit needs about 192 bytes"
                                             r".*Use a compact state preparation"):
            plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 0 and calls == []
    # Nothing was counted, so the same Run prepares the Plan on this computer.
    plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 1 and len(calls) == 1
    run.close()
    # At 192 bytes the direct preparation fits by itself. The check of the whole
    # circuit adds its one instruction, 256 bytes, and refuses before the charge.
    run = Run(plan, backend=AerBackend())
    with monkeypatch.context() as memory:
        memory.setattr(_limits, "physical_memory_bytes", lambda: 192)
        with pytest.raises(ValueError, match="preparing this circuit needs about 448 bytes"):
            plan.method.prepare(plan, run=run)
    assert run.trace.preparations == 0
    run.close()


def test_a_static_plan_refuses_a_direct_preparation_that_does_not_fit_before_any_preparation(monkeypatch):
    """A paid backend never sets a state directly, so the 8-qubit PREP of experiment ``wide`` needs its 266,240-byte tree.

    ``prepare_static`` checks it for every resolved point before the first preparation, so experiment
    ``small``, which calls no PREP, is not prepared and paid for first. A PREP that a Plan binds but a
    circuit does not call counts nothing for that circuit.
    """
    from nwqlib import _limits
    from nwqlib._prepared_execution import prepare_static
    from nwqlib.blocks.selection import select_preparation
    from nwqlib.core.planning import Experiment, ReadoutDetails
    from nwqlib.ir import BlockCall, Definition, MeasurementBatch, PortMap, Release, Sequence, Setting
    from nwqlib.problems import ingest_vector
    from test_ir import metadata
    from test_selected_runtime import StubBackend, ToyMethod, ToyProblem

    base = nwqlib.plan(ToyProblem(width=8), method=ToyMethod(), shots=3, seed=7)
    amplitudes = np.arange(1., 257.)
    prep = select_preparation("direct", ingest_vector(amplitudes / np.linalg.norm(amplitudes)))
    signature = prep.record.signature
    setting = Setting(label="Z", metadata=metadata("axis_counts"))
    program = base.construction.program
    definitions = (*program.definitions, Definition(id="release", node=Release(wire="q")),
        Definition(id="plain", node=Sequence(children=("allocate", "measure", "release"))),
        Definition(id="prepare", node=BlockCall(signature=signature.name,
                                                ports=(PortMap(port=signature.quantum[0].name, wire="q"),))),
        Definition(id="prepared", node=Sequence(children=("allocate", "prepare", "measure", "release"))),
        *(Definition(id=name, node=MeasurementBatch(body=body, settings=(setting,), repetitions=3,
                                                    observation_kind="counts"))
          for name, body in (("small", "plain"), ("wide", "prepared"))),
        Definition(id="root", node=Sequence(children=("small", "wide"))))
    program = program.revise(root="root", definitions=definitions, signatures=(signature,))
    selected = base.revise(construction=base.construction.revise(program=program, selections=(prep.record,)),
                           experiments=tuple(Experiment(name=name, batch=name, setting_index=0, readout=ReadoutDetails())
                                             for name in ("small", "wide")))._bind(blocks=(prep,))
    StubBackend.calls.clear()
    with Run(selected, backend=StubBackend()) as run:
        with monkeypatch.context() as memory:
            memory.setattr(_limits, "physical_memory_bytes", lambda: 133_120)
            with pytest.raises(ValueError, match="Experiment 'wide': building the direct state preparations of this "
                                                 "circuit needs about 266240 bytes"):
                prepare_static(selected, run=run)
        assert run.trace.preparations == 0 and StubBackend.calls == []


def routed_preparation_plan(*, calls, counts, prep_width, transform=None, position=None):
    """A ToyMethod Plan on 8 qubits whose circuit opens with ``calls`` calls of one direct PREP on ``prep_width`` of them.

    The readout is counts, or an exact Pauli expectation at ``position``. With ``transform``, the PREP is
    its adjoint.
    """
    from nwqlib.blocks.records import SelectedConstruction
    from nwqlib.blocks.selection import select_preparation, transform_block
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.ir import Allocate, BlockCall, Definition, PortMap, Register, Repeat, Sequence
    from nwqlib.problems import ingest_vector
    from test_selected_runtime import ToyMethod, ToyProblem

    class Routed(ToyMethod):
        def plan(self, problem, *, output, execution, shots, rng):
            base = super().plan(problem, output=output, execution=execution, shots=shots, rng=rng)
            amplitudes = np.arange(1., 2**prep_width + 1)
            prep = select_preparation("direct", ingest_vector(amplitudes / np.linalg.norm(amplitudes)))
            if transform is not None:
                prep = transform_block(transform, prep, adjoint=True)
            signature = prep.record.signature
            allocate, measure = base.construction.program.definitions[1:]
            definitions = [allocate, Definition(id="allocate-idle", node=Allocate(wire="idle")),
                Definition(id="prepare", node=BlockCall(signature=signature.name,
                                                        ports=(PortMap(port=signature.quantum[0].name, wire="q"),))),
                Definition(id="calls", node=Repeat(body="prepare", count=calls))]
            body = ["allocate", "allocate-idle", "calls"] + ["measure"] * counts
            program = base.construction.program.revise(
                registers=(Register(name="q", width=prep_width), Register(name="idle", width=8 - prep_width)),
                definitions=(Definition(id="body", node=Sequence(children=tuple(body))), *definitions,
                             *(measure,) * counts), signatures=(signature,))
            experiment = base.experiments[0]
            if not counts:
                experiment = experiment.revise(observation=ObservationSpec(kind="pauli_expectation", labels=("Z" * 8,),
                                                                           position=position))
            return base.revise(construction=SelectedConstruction(program=program, selections=(prep.record,)),
                               experiments=(experiment,))._bind(blocks=(prep,))

    return nwqlib.plan(ToyProblem(width=8), method=Routed(), shots=1, seed=7)


@pytest.mark.parametrize("calls, counts, prep_width, transform, need", [
    (2, False, 8, None, 266_240 + 12_288),  # one call is set as the state, the second builds its tree
    (1, True, 8, None, 266_240),            # a counts readout is never set as the state
    (1, False, 6, None, 12_288),            # the set state spans all 8 qubits of the circuit
    (1, False, 8, "adjoint", 266_240),      # a transformed PREP builds its tree
])
def test_direct_preparations_count_each_call_and_at_most_one_set_state(monkeypatch, calls, counts, prep_width,
                                                                        transform, need):
    """Noiseless Aer sets an untransformed PREP that opens the circuit as the initial state, 48*2**8 bytes here.

    Every other call builds a tree of 256*(4*2**8 + 2*8) = 266,240 bytes on 8 qubits. Half the need in
    physical memory refuses the point before its charge, and no tree is requested.
    """
    from nwqlib import _limits
    from nwqlib.subroutines.state_preparation.direct import DirectPreparationGate

    selected = routed_preparation_plan(calls=calls, counts=counts, prep_width=prep_width, transform=transform)
    requested = []
    monkeypatch.setattr(DirectPreparationGate, "_define", lambda gate: requested.append(gate))
    with Run(selected, backend=AerBackend()) as run:
        monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: need // 2)
        with pytest.raises(ValueError, match=f"direct state preparations of this circuit needs about {need} bytes"):
            prepare_experiment(selected.resolve(selected.experiments[0].name), run=run)
        assert run.trace.preparations == 0 and requested == []


@pytest.mark.parametrize("position, need", [(0, None), (1, None), (2, 12_288 + 266_240)])
def test_an_exact_pauli_readout_counts_only_the_direct_preparations_that_aer_runs(monkeypatch, position, need):
    """Aer runs only the instructions before an exact Pauli readout, so of three calls of one 8-qubit PREP it runs ``position``.

    At position 0 it runs none and at 1 it sets the first as the initial state, so both prepare without a
    tree in 100,000 bytes of physical memory. At 2 the second call needs its 266,240-byte tree, and the
    point is refused before its charge.
    """
    from nwqlib import _limits
    from nwqlib.subroutines.state_preparation.direct import DirectPreparationGate

    selected = routed_preparation_plan(calls=3, counts=False, prep_width=8, position=position)
    requested = []
    monkeypatch.setattr(DirectPreparationGate, "_define", lambda gate: requested.append(gate))
    with Run(selected, backend=AerBackend()) as run:
        monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 100_000)
        point = selected.resolve(selected.experiments[0].name)
        if need is None:
            prepare_experiment(point, run=run)
        else:
            with pytest.raises(ValueError, match=f"direct state preparations of this circuit needs about {need} bytes"):
                prepare_experiment(point, run=run)
        assert run.trace.preparations == int(need is None) and run.trace.jobs == 0 and requested == []


def three_preparations_plan(observation):
    """An 8-qubit Expectation Plan whose body applies its direct PREP three times and reads ``observation``."""
    from nwqlib.core.planning import Experiment
    from nwqlib.ir import Definition, Sequence
    from nwqlib.operators import ingest_pauli

    amplitudes = np.arange(1., 257.)
    base = nwqlib.plan(nwqlib.Expectation(state=list(amplitudes / np.linalg.norm(amplitudes)),
                                          observable=ingest_pauli((("Z" * 8, 1.),), num_qubits=8)),
                       method=ExpectationMethod(), seed=7)
    program = base.construction.program
    program = program.revise(definitions=tuple(item for item in program.definitions if item.id != "root") + (
        Definition(id="root", node=Sequence(children=("allocate", "prepare", "prepare", "prepare"))),))
    return base.revise(construction=base.construction.revise(program=program), experiments=(
        Experiment(name="trajectory", setting="points", observation=observation),))._bind(blocks=base.blocks)


@pytest.mark.parametrize("positions, need", [((0, 1), None), ((0, 2), 12_288 + 266_240)])
def test_a_trajectory_counts_only_the_direct_preparations_before_its_last_point(monkeypatch, positions, need):
    """Aer runs a trajectory's instructions before its last point, so of three 8-qubit PREPs it runs that many.

    With points at 0 and 1 it sets the first as the initial state and builds no tree in 100,000 bytes of
    physical memory. A last point at 2 adds the second call's 266,240-byte tree, refused before the charge.
    """
    from nwqlib import _limits
    from nwqlib.core.planning import ObservationPoint, ObservationSpec
    from nwqlib.subroutines.state_preparation.direct import DirectPreparationGate

    selected = three_preparations_plan(ObservationSpec(kind="trajectory", positions=tuple(
        ObservationPoint(id=f"at{position}", position=position, kind="pauli_expectation", labels=("Z" * 8,))
        for position in positions)))
    requested = []
    monkeypatch.setattr(DirectPreparationGate, "_define", lambda gate: requested.append(gate))
    with Run(selected, backend=AerBackend()) as run:
        monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 100_000)
        point = selected.resolve("trajectory")
        if need is None:
            prepare_experiment(point, run=run)
        else:
            with pytest.raises(ValueError, match=f"direct state preparations of this circuit needs about {need} bytes"):
                prepare_experiment(point, run=run)
        assert run.trace.preparations == int(need is None) and run.trace.jobs == 0 and requested == []


def test_a_trajectory_applies_a_view_tail_per_group_and_its_inverse_between_groups():
    """A view read at two boundaries applies its tail twice and its inverse once, between the two groups."""
    from nwqlib.core.planning import ObservationPoint, ObservationSpec, ReadoutView

    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0, 1))
    other = ReadoutView(tail="last", inverse="last_inverse", wires=(0, 1))
    observation = ObservationSpec(kind="trajectory", positions=(
        ObservationPoint(id="a", position=1, kind="pauli_expectation", labels=("ZZ",), view=view),
        ObservationPoint(id="b", position=1, kind="pauli_expectation", labels=("XX",), view=view),
        ObservationPoint(id="c", position=2, kind="pauli_expectation", labels=("ZZ",), view=view),
        ObservationPoint(id="d", position=3, kind="pauli_expectation", labels=("ZZ",), view=other)))
    saves, _ = aer._trajectory_saves(observation, (1, 1, 2, 3))
    assert aer._view_applications(saves) == {"tail": 2, "tail_inverse": 2, "last": 1}
    assert aer._view_applications(saves[:3]) == {"tail": 2, "tail_inverse": 1}


def test_a_reopened_run_restores_its_direct_preparation_unbuilt_and_still_sets_it_as_the_aer_state(tmp_path, monkeypatch):
    """A saved Run keeps its opening 6-qubit PREP as a state, and a new point after reopening sets it directly.

    Neither saving the first point, reopening, nor preparing the second point builds the 68,608-byte tree.
    """
    from nwqlib._run_archive import load as load_run
    from nwqlib.core.planning import Experiment, ObservationSpec
    from nwqlib.operators import ingest_pauli
    from nwqlib.subroutines.state_preparation.direct import DirectPreparationGate

    amplitudes = np.arange(1., 65.)
    base = nwqlib.plan(nwqlib.Expectation(state=list(amplitudes / np.linalg.norm(amplitudes)),
                                          observable=ingest_pauli((("Z" * 6, 1.),), num_qubits=6)),
                       method=ExpectationMethod(), seed=7)
    selected = base.revise(experiments=tuple(
        Experiment(name=name, setting=name, observation=ObservationSpec(kind="pauli_expectation", labels=(label,)))
        for name, label in (("first", "Z" * 6), ("second", "X" * 6))))._bind(blocks=base.blocks)
    requested = []
    monkeypatch.setattr(DirectPreparationGate, "_define", lambda gate: requested.append(gate))
    folder = tmp_path / "run"
    with Run(selected, backend=AerBackend(), directory=folder) as run:
        first = prepare_experiment(selected.resolve("first"), run=run)
        assert [item.operation.name for item in first._native.circuit.data].count("set_statevector") == 1
    with load_run(folder, backend=AerBackend()) as run:
        cached = [gate.definition.data[0].operation for gate in run._state["definitions"][0].values()]
        assert cached and all(type(gate) is DirectPreparationGate and gate._definition is None for gate in cached)
        second = prepare_experiment(run.plan.resolve("second"), run=run)
        assert [item.operation.name for item in second._native.circuit.data].count("set_statevector") == 1
        assert run.trace.preparations == 2 and run.trace.jobs == 0
    assert requested == []


def test_reading_a_direct_tree_beyond_the_physical_memory_is_refused_before_it_is_built(monkeypatch):
    """The tree is built only when its definition is first read, the one step that every path to it passes."""
    from nwqlib import _limits
    from nwqlib.subroutines.state_preparation import direct

    built = []
    monkeypatch.setattr(direct, "_build_normalized_state_preparation", lambda *args, **kwargs: built.append(args))
    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: direct.direct_tree_bytes(8) // 2)
    with pytest.raises(ValueError, match="direct state preparation tree on 8 qubits needs about 266240 bytes"
                                         r".*first instruction of the circuit"):
        direct.DirectPreparationGate(np.full(256, 1 / 16)).definition
    assert built == []


def test_an_opening_direct_preparation_is_set_as_the_aer_state_inside_the_error_bounds(monkeypatch):
    """A noiseless exact readout whose circuit opens with a direct PREP sets that state and builds no tree.

    The record bounds |‖v‖² − 1| of the set vector, adds it to state_error and probability_window, and
    does not list set_statevector as an exclusion. <ZZ + 0.5 XX> of 0.6|00> + 0.8|11> is 1 + 0.5*0.96.
    """
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.subroutines.state_preparation import direct

    built = []
    original = direct._build_normalized_state_preparation
    monkeypatch.setattr(direct, "_build_normalized_state_preparation",
                        lambda *args, **kwargs: built.append(1) or original(*args, **kwargs))
    plan = nwqlib.plan(Expectation(state=[0.6, 0.0, 0.0, 0.8],
                                   observable=ingest_pauli((("ZZ", 1.0), ("XX", 0.5)), num_qubits=2)),
                       method=ExpectationMethod(), seed=7)
    with Run(plan) as run:
        result = plan.method.execute(plan, run=run)
        record, = run._state["prepared_artifacts"].values()
        handle = run._state["handles"][record.content_id]
        assert handle._native.circuit.data[0].name == "set_statevector"
        assert built == []
        # A compiled inventory counts the preparation it replaced: the
        # two-level magnitude tree of 0.6|00> + 0.8|11> has two CX.
        compiled = handle.inspect_resources(transpile_options={"basis_gates": ["u", "cx"], "optimization_level": 0})
        assert compiled["operations"]["cx"] == 2
    assert result.value == pytest.approx(1.48, abs=1e-14)
    assert "set_statevector" not in record.probability_window_exclusions and record.initial_state_error > 0
    from nwqlib._validation import (AER_STATEVECTOR_ROUNDOFF_PER_OPERATION, add_up, exact_probability_window,
                                    native_state_error)
    delta, reason = record.state_error()
    assert reason is None and delta == add_up(
        native_state_error(record.native_operations, AER_STATEVECTOR_ROUNDOFF_PER_OPERATION), record.initial_state_error)
    assert record.probability_window == add_up(exact_probability_window(record.native_operations, 2),
                                               record.initial_state_error)


def test_the_installed_aer_release_is_one_that_the_roundoff_derivation_covers():
    """A new Aer release fails here with the comparison it needs, beside the tests that read its error bounds."""
    installed = importlib.metadata.version("qiskit-aer")
    assert installed in aer._ROUNDOFF_CHECKED_AER_VERSIONS, (
        f"Compare the fusion pass, kernels and gate matrices of qiskit-aer {installed} with the derivation in "
        "_validation.py before adding it to _ROUNDOFF_CHECKED_AER_VERSIONS (docs/MAINTENANCE.md). Until then its "
        "exact readouts have no state error, and the bounds built on it are unknown.")


def test_an_aer_release_outside_the_checked_ones_leaves_an_exact_readout_without_a_state_error(monkeypatch):
    """The state error of an exact readout rests on the sources of Aer, so another release reports why it has none.

    The readout and its numeric window do not depend on the label.
    """
    real = importlib.metadata.version

    def readout(release):
        with monkeypatch.context() as patch:
            patch.setattr(importlib.metadata, "version", lambda name: release if name == "qiskit-aer" else real(name))
            plan = hadamard_expectation(2)
            with Run(plan) as run:
                result = plan.method.execute(plan, run=run)
                record, = run._state["prepared_artifacts"].values()
        return result.value, record

    value, checked = readout("0.18.2")
    assert "unchecked qiskit-aer version" not in checked.probability_window_exclusions
    delta, reason = checked.state_error()
    assert reason is None and delta > 0
    # 0.18.1 lies between two checked releases, so a range instead of the exact set would admit it.
    for release in ("0.18.1", "0.18.3"):
        unchecked_value, unchecked = readout(release)
        assert "unchecked qiskit-aer version" in unchecked.probability_window_exclusions
        assert unchecked.state_error() == (None, "outside the roundoff derivation: unchecked qiskit-aer version")
        assert unchecked_value == value and unchecked.probability_window == checked.probability_window


def test_a_compiled_inventory_builds_a_directly_set_preparation_from_its_own_vector(tmp_path, monkeypatch):
    """A compiled inventory of a directly set PREP counts the PREP's own tree, live and after reopening.

    The set vector carries the executed circuit's phase factor, and the complex amplitudes of
    exp(0.3i)(1, 2, 3, 5)/sqrt(39) would give a tree with phase rotations. Empty options compile
    nothing, and a byte refusal comes before the PREP is built for the compiled copy, and after
    reopening before its saved vector is read. Reopened, the uncompiled inventory equals the live
    one, depth included: Aer's set_statevector is a directive, which Qiskit's depth skips.
    """
    from qiskit import QuantumCircuit, transpile
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.backends.inspection import inspect_circuit_resources
    from nwqlib.core.planning import ObservationSpec, RuntimeOptions
    from nwqlib.subroutines.state_preparation import direct

    options = {"basis_gates": ["u", "cx"], "optimization_level": 0}
    state = np.array([1., 2., 3., 5.], dtype=complex) / np.sqrt(39)
    actual = QuantumCircuit(2, global_phase=0.3)
    actual.append(direct.DirectPreparationGate(state), [0, 1])
    backend = AerBackend()
    with Run(base_plan(), backend=backend) as run:
        prepared = backend.prepare(actual, observation=ObservationSpec(kind="pauli_expectation", labels=("ZZ",)),
                                   runtime=RuntimeOptions(seed=7), position=1, source_definitions=(), run=run,
                                   snapshot="phase")
    native = prepared.native
    assert native.circuit.data[0].name == "set_statevector" and native.initial_state is state
    compiled = inspect_circuit_resources(native.circuit, native_basis=prepared.native_basis, label="phase",
                                         transpile_options=options,
                                         initial_state=(native.initial_state, native.metadata["initial_state_qubits"]))
    assert compiled["operations"] == dict(transpile(actual, **options).count_ops()) == {"u": 3, "cx": 2}

    plan = base_plan()
    with Run(plan, directory=tmp_path / "run") as run:
        handle = prepare_experiment(plan.resolve("expectation"), run=run)
        raw = handle.inspect_resources()
        assert raw["operations"] == {"set_statevector": 1, "save_expval": 1}
        assert handle.inspect_resources(transpile_options={}) == raw
        live = handle.inspect_resources(transpile_options=options)
        saved = handle._native.initial_state.tobytes()
        identity = handle.record.content_id
        with monkeypatch.context() as patch:
            patch.setattr(direct, "DirectPreparationGate", lambda *args: pytest.fail("PREP built before refusal"))
            with pytest.raises(ValueError, match="native inspection collections"):
                handle.inspect_resources(transpile_options=options, max_bytes=1)
    assert live["operations"] == {"u": 1}
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend()) as run:
        handle = run._state["handles"][identity]
        assert handle._native is None
        reads, read = [], ArchiveFiles.read_array
        monkeypatch.setattr(ArchiveFiles, "read_array", lambda files, name: reads.append(name) or read(files, name))
        assert handle.inspect_resources(transpile_options={}) == raw and raw["depth"] == 0
        with pytest.raises(ValueError, match="native inspection collections"):
            handle.inspect_resources(transpile_options=options, max_bytes=1)
        assert reads == []
        assert handle.inspect_resources(transpile_options=options) == live
        assert len(reads) == 1 and handle._native.initial_state().tobytes() == saved


# Two real four-qubit Pauli Hamiltonians with different supports and signs.
def settings_values(result):
    """Experiment names and returned outcomes, without Run, attempt or job identities."""
    return [(chunk.experiment, chunk.histogram().index_list(), chunk.histogram().weights.tolist())
            for chunk in result.data.observations.chunks]


def test_prepare_all_settings_exposes_every_circuit_and_submit_reuses_them(tmp_path, monkeypatch):
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.qpe import QCELS
    from nwqlib.blocks import lowering
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import ingest_occupation
    from nwqlib.resources.records import ResourceContext

    target = ingest_pauli((("ZIII", .4), ("XYZI", -.3), ("IYYX", .2), ("ZZZZ", .05)), num_qubits=4)
    method = QCELS(initial_state=ingest_occupation("0101", num_qubits=4), num_times=3, tau=.2, grid_size=64)
    selected = nwqlib.plan(Eigenproblem(A=target), method=method, shots=16, seed=7)
    names = tuple(experiment.name for experiment in selected.experiments)
    assert len(names) > 1
    # The saved Run names the local backend explicitly.
    directory = tmp_path / "all"
    prepared = nwqlib.prepare(selected, backend=AerBackend(), settings="all", directory=directory)
    run = prepared.run
    assert prepared.setting_names == names and len(prepared.circuits) == len(names)
    # Prepared.circuit(i) copies circuit i of the same index order, and changing
    # the copy leaves the prepared circuit unchanged.
    assert all(prepared.circuit(index) == circuit for index, circuit in enumerate(prepared.circuits))
    prepared.circuit(0).clear()
    assert prepared.circuit(0) == prepared.circuits[0] and len(prepared.circuit(0).data)
    # Nothing is submitted by preparation.
    assert run.trace.jobs == 0 and run.trace.events == ()
    # The fold counts each controlled Suzuki step by its exact CX law
    # (qpe/powers.py::suzuki_step_cx) times its repetitions and the shots of
    # its setting. The occupation preparation, H, the identity phase and the
    # feedback rotation add no CX, so transpiling every prepared circuit
    # without optimization and weighting it by its setting's shots must give
    # the same total.
    options = dict(basis_gates=["cx", "u"], optimization_level=0)
    compiled = [prepared.inspect_resources(index=index, transpile_options=options)["operations"].get("cx", 0)
                * selected.resolve(name).resolved_observation(selected)[1].shots
                for index, name in enumerate(prepared.setting_names)]
    folded = nwqlib.estimate(selected, context=ResourceContext(basis="cx")).quantity("cx")
    assert folded.interpretation == "exact" and folded.fact.value.denominator == 1
    assert sum(compiled) == folded.fact.value.numerator > 0
    prepared_trace = run.trace
    assert prepared_trace.preparations == len(names)

    def forbidden(*args, **kwargs):
        raise AssertionError("submission lowered a setting that was already prepared")

    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    # Stop before submission, then continue the saved Run.
    run.close()
    run = nwqlib.load_run(directory, backend=AerBackend())
    result = run.wait()
    with run:
        trace = run.trace
        assert trace.preparations == len(names) and trace.jobs == len(names)
        assert trace.local_prepared_ids == prepared_trace.local_prepared_ids
        assert trace.construction_work_reserved == prepared_trace.construction_work_reserved
        assert trace.synthesis_work == prepared_trace.synthesis_work
    monkeypatch.undo()
    # The default path creates each setting's workflow row, and draws its
    # seed, in the same Plan order, so it acquires the same histograms.
    default = nwqlib.solve(selected)
    assert ([receipt.runtime.seed for receipt in result.data.receipts]
            == [receipt.runtime.seed for receipt in default.data.receipts])
    assert settings_values(result) == settings_values(default)


@pytest.mark.parametrize("family", ["rwpe", "lanczos", "adapt"])
def test_prepare_all_settings_refuses_outcome_dependent_controllers(family, tmp_path):
    from math import pi
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.algorithms.lanczos import Lanczos, SensitivitySampling
    from nwqlib.algorithms.qpe import RWPE
    from nwqlib.operators import ingest_pauli

    problem, method = {
        "rwpe": (Eigenproblem(A=[[.4, 0.], [0., -.7]]), RWPE(initial_state=[1., 0.], max_steps=3, tau=.2)),
        "lanczos": (Eigenproblem(A=[[1., 1.], [1., 1.]]), Lanczos(
            initial_state=[1., 0.], krylov_dimension=2,
            sampling=SensitivitySampling(total_shots=100, pilot_fraction=.2))),
        "adapt": (Eigenproblem(A=ingest_pauli((("Z", 1.),), num_qubits=1)), ADAPT(
            initial_state=(1, 1), pool=(ingest_pauli((("Y", 1j),), num_qubits=1),), theta=pi / 8,
            max_iterations=2)),
    }[family]
    selected = nwqlib.plan(problem, method=method, shots=1 if family == "rwpe" else None, seed=7)
    # The refusal comes before the Run (Method.prepare_all_refusal), so the
    # folder stays free for the default preparation.
    with pytest.raises(ValueError, match="cannot prepare every setting in advance") as refused:
        nwqlib.prepare(selected, settings="all", directory=tmp_path / "run")
    assert "settings='first' prepares" in str(refused.value)
    assert not (tmp_path / "run").exists()
    nwqlib.prepare(selected, directory=tmp_path / "run").run.close()


def three_settings():
    """A Plan with three settings: ZZ, XX and YY need three measurement bases."""
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation

    selected = nwqlib.plan(
        Expectation(state=[1., 0., 0., 1.],
                    observable=ingest_pauli((("ZZ", 1.), ("XX", .5), ("YY", .25)), num_qubits=2)),
        method=ExpectationMethod(), shots=8, seed=7)
    assert len(selected.experiments) == 3
    return selected


def uninterrupted(selected, folder):
    """The value and the runtime seed of each setting of a Run of ``selected`` that nothing interrupts, the oracle of a recovery."""
    prepared = nwqlib.prepare(selected, settings="all", directory=folder, progress=False)
    with prepared.run as run:
        nwqlib.submit(prepared)
        return run.wait().value, seeds_of(run)


def seeds_of(run):
    return {record.realization_id: record.runtime.seed for record in run.prepared_artifacts}


def fail_the_second_lowering(patch):
    from nwqlib.blocks import lowering

    original, calls = lowering._lower_qiskit, []

    def fails_second(*args, **kwargs):
        calls.append(None)
        if len(calls) == 2:
            raise RuntimeError("lowering stopped")
        return original(*args, **kwargs)

    patch.setattr(lowering, "_lower_qiskit", fails_second)


def stop_the_second_preparation(selected, folder, monkeypatch, **options):
    """Leave ``folder`` a closed Run whose first setting is prepared and whose second is charged without a record."""
    with monkeypatch.context() as patch:
        fail_the_second_lowering(patch)
        with pytest.raises(RuntimeError, match="lowering stopped"):
            nwqlib.prepare(selected, settings="all", directory=folder, progress=False, **options)


def test_prepare_all_settings_failure_at_a_later_setting_submits_nothing_on_resume(tmp_path, monkeypatch):
    selected = three_settings()
    value, seeds = uninterrupted(selected, tmp_path / "whole")
    stop_the_second_preparation(selected, tmp_path / "run", monkeypatch)
    # The second setting keeps its charge without a receipt, and the Run never
    # repeats a preparation attempt on its own. Resuming refuses before it spends
    # shots on the first setting.
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend()) as run:
        assert run.trace.preparations == 2 and len(run.prepared_artifacts) == 1
        with pytest.raises(PreparationNotRebuilt, match="'group_1' failed or was interrupted and no replacement"):
            run.resume()
        assert run.trace.jobs == 0 and run.trace.preparations == 2
        # P_after = P_before + k_rebuild + k_new: 2 + 1 for the rebuilt setting + 1 for the third setting,
        # never prepared before. The seed, and so the Aer sample, is the uninterrupted Run's, so equality
        # is exact on this deterministic simulator.
        run.resume(rebuild_preparations=True)
        assert run.result.value == value and seeds_of(run) == seeds
        assert run.trace.preparations == 4 and run.trace.jobs == 3 and len(run.prepared_artifacts) == 3


@pytest.mark.parametrize("stop, reopened, final", [("before the new charge", 2, 4), ("after the new charge", 3, 5),
                                                   ("before the record", 3, 5), ("after the record", 3, 4)])
def test_an_interrupted_rebuild_reopens_at_a_committed_prefix(tmp_path, monkeypatch, stop, reopened, final):
    """Each persistence boundary of a rebuild, stopped and reopened, ends at the uninterrupted value.

    Counts follow from P_after = P_before + k_rebuild + k_new over committed charges: two before the rebuild, one
    for each rebuild whose charge committed, and one first preparation of the third setting. A stop before a commit
    fails that commit, and a stop after one interrupts the progress report that follows it, so what reopens is what
    the run log holds. After the record no new request is needed, and a plain resume submits the preparation restored
    from its saved circuit and collects its result.
    """
    from nwqlib._run_journal import LocalJournal

    selected = three_settings()
    value, seeds = uninterrupted(selected, tmp_path / "whole")
    stop_the_second_preparation(selected, tmp_path / "run", monkeypatch)
    commit = LocalJournal.commit

    def failing(self, records=(), **kwargs):
        if ("preparation" if stop == "before the new charge" else "prepared") in {kind for kind, _, _ in records}:
            raise KeyboardInterrupt
        commit(self, records, **kwargs)

    def observe(stage, done, total):
        if stage == "prepare" and stop == ("after the new charge" if done is None else "after the record"):
            raise KeyboardInterrupt

    with monkeypatch.context() as patch:
        if stop.startswith("before"):
            patch.setattr(LocalJournal, "commit", failing)
        with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=observe) as run:
            with pytest.raises(KeyboardInterrupt):
                run.resume(rebuild_preparations=True)
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        assert run.trace.preparations == reopened and run.trace.jobs == 0
        assert len(run.prepared_artifacts) == 1 + (stop == "after the record")
        # The row's pointer moved with the charge that committed last, in the same transaction.
        assert run._state["workflow_items"]["group_1"].preparation_id == list(run._state["preparation_charges"])[-1]
        run.resume(rebuild_preparations=stop != "after the record")
        assert run.result.value == value and seeds_of(run) == seeds
        assert run.trace.preparations == final and run.trace.jobs == 3
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        assert run.result.value == value and run.trace.preparations == final and run.trace.jobs == 3


def test_a_rebuild_is_refused_before_any_charge_when_its_preparations_exceed_the_budget(tmp_path, monkeypatch):
    """Three settings and one spent charge need four preparations, so a budget of three refuses before the rebuild.

    Otherwise the Run would measure two settings and refuse the last one. Raising the budget leads past it. Four
    is enough for four preparations and three measurement attempts, because the budget caps each of them alone.
    """
    selected = three_settings()
    value, _ = uninterrupted(selected, tmp_path / "whole")
    stop_the_second_preparation(selected, tmp_path / "run", monkeypatch, limits=ExecutionLimits(max_total_circuits=3))
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        with pytest.raises(ValueError, match="max_total_circuits=3 cannot admit"):
            run.resume(rebuild_preparations=True)
        assert run.trace.preparations == 2 and run.trace.jobs == 0
        run.extend_limits(max_total_circuits=4)
        assert run.resume(rebuild_preparations=True).result.value == value
        assert run.trace.preparations == 4 and run.trace.jobs == 3


def test_a_rebuild_request_with_nothing_to_rebuild_changes_nothing(tmp_path, monkeypatch):
    """An ordinary Run, a completed Run and a cancelled Run each continue as a plain resume does, with no new preparation."""
    selected = three_settings()
    value, _ = uninterrupted(selected, tmp_path / "whole")
    prepared = nwqlib.prepare(selected, directory=tmp_path / "ordinary", progress=False)
    with prepared.run as run:
        assert run.resume(rebuild_preparations=True).result.value == value
        assert run.trace.preparations == 3 and run.trace.jobs == 3
        assert run.resume(rebuild_preparations=True).result.value == value and run.trace.preparations == 3
    stop_the_second_preparation(selected, tmp_path / "cancelled", monkeypatch)
    with nwqlib.load_run(tmp_path / "cancelled", backend=AerBackend(), progress=False) as run:
        run.cancel()
        assert run.resume(rebuild_preparations=True).result is None
        assert run.trace.preparations == 2 and run.trace.jobs == 0


def test_an_adaptive_method_keeps_its_own_refusal_with_the_rebuild_request(tmp_path, monkeypatch):
    """RWPE chooses each next setting from data, so it has no fixed rows, and the request adds no preparation to it."""
    from nwqlib import Eigenproblem
    from nwqlib.algorithms.qpe import RWPE

    selected = nwqlib.plan(Eigenproblem(A=[[.4, 0.], [0., -.7]]), method=RWPE(initial_state=[1., 0.], max_steps=3, tau=.2),
                           shots=1, seed=7)
    with monkeypatch.context() as patch:
        fail_the_second_lowering(patch)
        prepared = nwqlib.prepare(selected, directory=tmp_path / "run", progress=False)
        with prepared.run as run, pytest.raises(RuntimeError, match="lowering stopped"):
            nwqlib.submit(prepared)
    for rebuild in (False, True):
        with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
            before = run.trace.preparations
            with pytest.raises(RuntimeError, match="no replacement was built"):
                run.resume(rebuild_preparations=rebuild)
            assert run.trace.preparations == before
