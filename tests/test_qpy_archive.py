"""Executable UCGate persistence and the tiny upstream-change canary."""

import io

import numpy as np
import pytest
from qiskit import QuantumCircuit, qpy
from qiskit.circuit import Instruction
from qiskit.circuit.library import UCGate
from qiskit.exceptions import QiskitError
from qiskit.quantum_info import Operator

from nwqlib._choice_archive import ArchiveFiles


def test_native_ucgate_qpy_limitation_canary(monkeypatch):
    """An upstream behavior change requests review, with no simulation/synthesis."""
    gate = UCGate([np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex)],
                  mux_simp=False)
    circuit = QuantumCircuit(2)
    circuit.append(gate, [0, 1])
    monkeypatch.setattr(UCGate, "_define", lambda *a: pytest.fail("canary requested synthesis"))
    stream = io.BytesIO()
    qpy.dump(circuit, stream)
    stream.seek(0)
    try:
        qpy.load(stream)
    except QiskitError as error:
        assert "single-qubit unitaries are not provided in a list" in str(error), (
            "Native UCGate QPY failure changed; reassess the archive compatibility codec")
    else:
        pytest.fail("Native UCGate QPY now loads: reassess the compatibility codec, including "
                    "simplified controls and up_to_diagonal")


@pytest.mark.parametrize("layout", ["full", "first", "second", "constant"])
@pytest.mark.parametrize("up_to_diagonal", [False, True])
def test_compact_ucgate_roundtrip_preserves_nominal_action(tmp_path, monkeypatch, layout, up_to_diagonal):
    u = np.diag(np.exp(1j*np.array([.23, -.41])))
    v = np.exp(.17j)*np.array([[np.cos(.31), -np.sin(.31)], [np.sin(.31), np.cos(.31)]])
    table = {"full": [u, v, v, u], "first": [u, u, v, v],
             "second": [u, v, u, v], "constant": [u, u, u, u]}[layout]
    gate = UCGate(table, up_to_diagonal=up_to_diagonal, mux_simp=layout != "full")
    circuit = QuantumCircuit(3, global_phase=.19)
    circuit.append(gate, [0, 1, 2], copy=False)
    original_ops = tuple(item.operation for item in circuit.data)
    path = tmp_path / "native"
    path.mkdir()
    with monkeypatch.context() as patch:
        patch.setattr(UCGate, "_define", lambda *a: pytest.fail("archive requested synthesis"))
        files = ArchiveFiles(path, 100_000)
        name = files.write_circuit("circuit.qpy", circuit)
        restored = ArchiveFiles(path, 100_000).read_circuit(name)
    back = restored.data[0].operation
    assert type(back) is UCGate
    assert back.num_qubits == 3 and back.up_to_diagonal is up_to_diagonal
    assert back.simp_contr == gate.simp_contr
    assert len(back.params) == len(gate.params)
    assert all(a is b.operation for a, b in zip(original_ops, circuit.data, strict=True))
    for actual, expected in zip(back.params, gate.params, strict=True):
        np.testing.assert_array_equal(actual, expected)
    np.testing.assert_allclose(Operator(restored).data, Operator(circuit).data, rtol=0, atol=2e-14)
    if not up_to_diagonal:
        expected = np.zeros((8, 8), dtype=complex)
        for index, matrix in enumerate(table):
            expected[2*index:2*index+2, 2*index:2*index+2] = np.exp(.19j)*matrix
        np.testing.assert_allclose(Operator(restored).data, expected, rtol=0, atol=2e-14)


def test_nested_shared_gate_and_marker_collision_preserve_user_data(tmp_path, monkeypatch):
    uc = UCGate([np.eye(2, dtype=complex), np.array([[0, 1], [1, 0]], dtype=complex)], mux_simp=False)
    definition = QuantumCircuit(2, global_phase=.2)
    definition.append(uc, [0, 1], copy=False)
    shared = definition.to_gate(label="shared")
    circuit = QuantumCircuit(2)
    circuit.append(shared, [0, 1], copy=False)
    circuit.append(shared, [0, 1], copy=False)
    ordinary = Instruction("nwqlib_stored_ucgate_v1", 2, 0, ["ordinary user data"])
    circuit.append(ordinary, [0, 1], copy=False)
    circuit.metadata = dict(version=1, marker="nwqlib_stored_ucgate_v1", original_metadata={"user": 7})
    with monkeypatch.context() as patch:
        patch.setattr(UCGate, "_define", lambda *a: pytest.fail("archive requested synthesis"))
        files = ArchiveFiles(tmp_path, 100_000)
        name = files.write_circuit("shared.qpy", circuit)
        restored = ArchiveFiles(tmp_path, 100_000).read_circuit(name)
    assert restored.metadata == circuit.metadata
    assert restored.data[-1].operation.name == ordinary.name
    assert restored.data[-1].operation.params == ordinary.params
    assert type(restored.data[0].operation.definition.data[0].operation) is UCGate
    assert type(restored.data[1].operation.definition.data[0].operation) is UCGate
    assert circuit.data[0].operation is circuit.data[1].operation is shared


def test_codec_preserves_instruction_action(tmp_path):
    gate = UCGate([np.diag([1, 1j])], mux_simp=False)
    files = ArchiveFiles(tmp_path, 10_000)
    name = files.write_instruction("instruction.qpy", gate)
    restored = ArchiveFiles(tmp_path, 10_000).read_instruction(name)
    assert type(restored) is UCGate and restored.num_qubits == 1
    np.testing.assert_allclose(Operator(restored).data, np.diag([1, 1j]), rtol=0, atol=2e-15)


def test_codec_write_preserves_existing_byte_cap(tmp_path):
    gate = UCGate([np.eye(2, dtype=complex)], mux_simp=False)
    with pytest.raises(ValueError, match="byte"):
        ArchiveFiles(tmp_path, 20).write_instruction("limited.qpy", gate)


def test_native_clifford_leaf_keeps_raw_qpy_and_ucgate_neighbor(tmp_path):
    from qiskit.quantum_info import Clifford
    from nwqlib._qpy_archive import PREFIX
    h = QuantumCircuit(1)
    h.h(0)
    clifford = Clifford(h)
    circuit = QuantumCircuit(1)
    circuit.append(clifford, [0])
    for encoded in (False, True):
        if encoded:
            circuit.append(UCGate([np.eye(2, dtype=complex)], mux_simp=False), [0])
        name = ArchiveFiles(tmp_path, 10_000).write_circuit(f"clifford-{encoded}.qpy", circuit)
        assert (tmp_path / name).read_bytes().startswith(PREFIX) is encoded
        restored = ArchiveFiles(tmp_path, 10_000).read_circuit(name)
        np.testing.assert_array_equal(restored.data[0].operation.tableau, clifford.tableau)
        if encoded:
            assert type(restored.data[1].operation) is UCGate


def test_controlled_and_inverse_multiplexors_keep_phase_and_open_control(tmp_path, monkeypatch):
    u = np.diag(np.exp(1j*np.array([.23, -.41])))
    v = np.exp(.17j)*np.array([[np.cos(.31), -np.sin(.31)], [np.sin(.31), np.cos(.31)]])
    gate = UCGate([u, v], mux_simp=False)
    circuit = QuantumCircuit(3, global_phase=.19)
    circuit.append(gate.control(1, ctrl_state=0, annotated=False), [2, 0, 1])
    circuit.append(gate.inverse(), [0, 1])
    expected = Operator(circuit).data
    with monkeypatch.context() as patch:
        patch.setattr(UCGate, "_define", lambda *a: pytest.fail("archive requested synthesis"))
        name = ArchiveFiles(tmp_path, 1_000_000).write_circuit("controlled.qpy", circuit)
        restored = ArchiveFiles(tmp_path, 1_000_000).read_circuit(name)
    np.testing.assert_allclose(Operator(restored).data, expected, rtol=0, atol=2e-14)


@pytest.mark.parametrize("durable", [False, True])
def test_public_lanczos_archive_reopens_in_fresh_process(tmp_path, monkeypatch, durable):
    import json
    import subprocess
    import sys
    import nwqlib as nw
    from nwqlib.algorithms import Lanczos
    from nwqlib._prepared_execution import Run

    plan = nw.plan(nw.Eigenproblem(A=[[1.5, -1.], [-1., .5]]),
        method=Lanczos(initial_state=[1., 0.], krylov_dimension=2), shots=32, seed=7)
    with Run(plan) as baseline:
        expected = baseline.wait().eigenvalue
        jobs = baseline.trace.jobs
        baseline.save(tmp_path / "completed")
    from nwqlib.backends import AerBackend
    with monkeypatch.context() as patch:
        import nwqlib.backends.qiskit_aer as aer
        patch.setattr(aer, "_submit_aer_execution", lambda *a: pytest.fail("completed Run submitted again"))
        with nw.load_run(tmp_path / "completed", backend=AerBackend()) as completed:
            assert completed.wait().eigenvalue == expected
            assert completed.trace.jobs == jobs
    path = tmp_path / "lanczos"
    collect = Run.collect

    def interrupt(run, chunk):
        fresh = collect(run, chunk)
        if fresh:
            raise RuntimeError("stop after committed observation")
        return fresh

    with Run(plan, directory=path if durable else None) as run:
        with monkeypatch.context() as patch:
            patch.setattr(Run, "collect", interrupt)
            with pytest.raises(RuntimeError, match="committed observation"):
                run.resume()
        prefix = [chunk.content_id for chunk in run.observations.chunks]
        assert len(prefix) == 1
        if not durable:
            run.save(path)
    child = subprocess.run([sys.executable, "-c", '''
import json, sys
import nwqlib as nw
from nwqlib.backends import AerBackend
with nw.load_run(sys.argv[1], backend=AerBackend()) as run:
    result = run.wait()
    print(json.dumps(dict(value=result.eigenvalue, jobs=run.trace.jobs,
        prefix=[run.observations.chunks[0].content_id])))
''', str(path)], text=True, capture_output=True, timeout=30)
    assert child.returncode == 0, child.stdout + child.stderr
    result = json.loads(child.stdout)
    assert result["value"] == pytest.approx(expected, rel=0, abs=2e-13)
    assert result["jobs"] == jobs and result["prefix"] == prefix
