"""Selected-source transport and stage semantics; no upstream precision claims."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit

from nwqlib import Expectation, plan, prepare, submit
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.backends.nwqec import LogicalCompilation, _static_body, compile_logical
from nwqlib.blocks import SelectedConstruction, lower_qiskit, select_preparation
from nwqlib.ir import Allocate, BlockCall, Definition, PortMap, Program, Register, Repeat, Sequence
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_vector


def expectation_plan(*, shots=None):
    """<psi|(I+Z)|psi>/<psi|psi> for the physical vector (2,2), with exact or counts readout."""
    problem = Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
    )
    return plan(problem, method=ExpectationMethod(), shots=shots, seed=7)


def lowered_preparation(state, repetitions=1):
    """Use the existing selected PREP/Program owner for the test circuit."""
    block = select_preparation("prep", state)
    program = Program(
        root="root",
        registers=(Register(name="q", width=1),),
        signatures=(block.record.signature,),
        definitions=(
            Definition(id="allocate", node=Allocate(wire="q")),
            Definition(
                id="h", node=BlockCall(signature="prep", ports=(PortMap(port="system", wire="q"),))
            ),
            Definition(id="twice", node=Repeat(body="h", count=repetitions)),
            Definition(id="root", node=Sequence(children=("allocate", "twice"))),
        ),
    )
    construction = SelectedConstruction(program=program, selections=(block.record,))
    return lower_qiskit(
        construction, blocks=(block,), max_operations=20, max_qubits=1, max_clbits=0
    )


def lowered_h_twice():
    """Public lowering of H;H, with an independently known identity action."""
    return lowered_preparation(ingest_vector(np.array([1.0, 1.0])), repetitions=2)


def lowered_rotation(angle=0.1234):
    """Selected exp(-i*.1234*Z/2)H body; no synthetic compilation identity."""
    from nwqlib.problems import bind_preparation_circuit
    from nwqlib.core import Basis, InputRef, Source

    circuit = QuantumCircuit(1)
    circuit.h(0)
    circuit.rz(angle, 0)
    state = bind_preparation_circuit(
        circuit,
        reference=InputRef(
            identity=f"declared:rz-preparation:{angle}",
            representation="circuit",
            source=Source(
                name="test RZ body",
                version="1",
                domain="one-qubit selected rotation",
                reference=f"H;RZ({angle})",
            ),
        ),
        basis=Basis(identity="computational", dimension=2, ordering="little-endian"),
    )
    return lowered_preparation(state)


def require_native():
    return pytest.importorskip(
        "nwqec", reason="native compiler extra is qualified on Python3.12 macOS/Linux"
    )


def test_actual_lowering_before_and_after_fusion_and_stored_readback(tmp_path, monkeypatch):
    require_native()
    native = lowered_h_twice()
    original = tuple(native.circuit.data)
    counts = compile_logical(native, count_only=True)
    full = compile_logical(native)
    assert dict(counts.raw_counts) == {"h": 2}
    assert dict(full.raw_counts) == {}
    assert counts.stage == "synthesis_before_fusion" and counts.output_qasm is None
    assert full.stage == "final_artifact" and full.output_qasm is not None
    assert (
        next(q for q in counts.quantities if q.metric == "logical_depth").fact.availability
        == "unknown"
    )
    assert full.source_id == native.construction_id and full.width == 1
    assert tuple(native.circuit.data) == original
    assert Path(full.compiler_module_file).is_file()
    from nwqlib.backends import _auxiliary_process

    monkeypatch.setattr(
        _auxiliary_process,
        "run_auxiliary",
        lambda *a, **k: pytest.fail("stored compilation was recompiled"),
    )
    full.save(tmp_path / "compiled.json")
    restored = LogicalCompilation.load(tmp_path / "compiled.json")
    assert restored.content_id == full.content_id
    assert "not verified achieved precision" in str(restored)


@pytest.mark.parametrize("shots", [None, 8])
def test_prepared_actual_readout_is_metadata_not_compiler_measurement(shots):
    require_native()
    prepared = prepare(expectation_plan(shots=shots))
    try:
        original = prepared.circuits[0]
        compiled = compile_logical(prepared)
        assert compiled.original_readout.kind == ("counts" if shots else "pauli_expectation")
        assert prepared.circuits[0] == original
        assert compiled.source_kind == "prepared"
        assert bool(compiled.terminal_measurements) == bool(shots)
        assert bool(compiled.simulator_saves) == (shots is None)
        if shots is None:
            assert json.loads(compiled.simulator_saves[0][3])["params"]
        assert "measure" not in dict(compiled.raw_counts)
        assert prepared.run.trace.events == ()
    finally:
        prepared.run.close()


def two_basis_plan():
    """ZZ and XX act on the same qubits with anticommuting factors, so the Plan measures them in two settings."""
    problem = Expectation(
        state=[1.0, 0.0, 0.0, 1.0],
        observable=ingest_pauli((("ZZ", 1.0), ("XX", 0.5)), num_qubits=2),
    )
    selected = plan(problem, method=ExpectationMethod(), shots=8, seed=7)
    assert len(selected.experiments) == 2
    return selected


def test_every_prepared_setting_compiles_by_index_without_submission():
    require_native()
    selected = two_basis_plan()
    prepared = prepare(selected, settings="all")
    try:
        receipts = prepared.run.prepared_artifacts
        assert prepared.setting_names == tuple(e.name for e in selected.experiments)
        assert tuple(r.realization.experiment for r in receipts) == prepared.setting_names
        for index, receipt in enumerate(receipts):
            compiled = compile_logical(prepared, index=index, count_only=True)
            assert compiled.source_id == receipt.content_id
            assert compiled.original_readout == receipt.observation
        assert prepared.run.trace.events == ()
    finally:
        prepared.run.close()


def test_released_prepared_native_compiles_from_its_saved_snapshot(tmp_path):
    require_native()
    prepared = prepare(two_basis_plan(), settings="all", directory=tmp_path / "run")
    with prepared.run as run:
        before = compile_logical(prepared, index=1, count_only=True)
        submit(prepared).wait()
        # Both settings are collected, so both saved QPY handles are released.
        assert set(run.release_native()["released"]) == {r.content_id for r in run.prepared_artifacts}
        after = compile_logical(prepared, index=1, count_only=True)
    # The reloaded native circuit is the one that was prepared and executed.
    assert (after.source_id, after.input_qasm_digest) == (before.source_id, before.input_qasm_digest)


def test_real_pbc_frame_is_not_original_readout():
    require_native()
    compiled = compile_logical(lowered_h_twice(), target="pbc")
    assert compiled.terminal_measurements == ()
    assert dict(compiled.raw_counts) == {"m_pauli": 1}
    assert "m_pauli" in compiled.output_qasm
    assert not any(q.metric == "t" for q in compiled.quantities)
    assert (
        next(q for q in compiled.quantities if q.metric == "logical_depth").fact.availability
        == "unknown"
    )


def test_released_rotation_compilation_keeps_requested_precision():
    require_native()
    compiled = compile_logical(lowered_rotation(), epsilon=1e-3, rz_err="total")
    assert dict(compiled.raw_counts).get("t", 0) > 0
    assert set(dict(compiled.raw_counts)) <= {
        "h",
        "x",
        "y",
        "z",
        "s",
        "sdg",
        "t",
        "tdg",
        "cx",
        "id",
    }
    assert "not verified achieved precision" in str(compiled)


@pytest.mark.parametrize("epsilon,rz_err,sent", [
    (3e-4, "total", 3e-4),
    # Documented values for an omitted epsilon.
    (None, "per-gate", 1e-10),
    (None, "relative", 1e-2),
])
def test_public_epsilon_and_error_policy_reach_the_compiler_request(monkeypatch, epsilon, rz_err, sent):
    from nwqlib.backends import _auxiliary_process

    requests = []

    def child(operation, request, **kwargs):
        requests.append((operation, request))
        return dict(width=1, counts={"h": 1, "t": 1}, depth=2, version="0.1.2", module_file="injected",
                    qasm="OPENQASM 2.0; qreg q[1]; h q[0]; t q[0];"), 0.0

    monkeypatch.setattr(_auxiliary_process, "run_auxiliary", child)
    compiled = compile_logical(lowered_rotation(), epsilon=epsilon, rz_err=rz_err)
    ((operation, request),) = requests
    assert operation == "nwqec" and (request["epsilon"], request["rz_err"]) == (sent, rz_err)
    # The stored effective value is the one actually sent to the compiler.
    assert (compiled.requested_epsilon, compiled.effective_epsilon) == (epsilon, sent)


def test_static_admission_preserves_source_phase_mapping_and_save_settings():
    from qiskit_aer.library import SaveStatevector
    from qiskit.quantum_info import Operator

    circuit = QuantumCircuit(2, 2)
    circuit.global_phase = 0.23
    circuit.h(0)
    circuit.ry(.9, 0)
    circuit.ry(-.4, 1)
    circuit.rz(1.3, 0)
    expected_body = Operator(circuit).data
    circuit.append(SaveStatevector(2, label="kept", pershot=True), [0, 1])
    circuit.measure(0, 1)
    original = circuit.copy()
    qasm, measurements, saves, phase = _static_body(circuit, max_operations=20, max_bytes=8192)
    assert measurements == ((0, 1),) and phase == 0.23
    assert saves[0][:3] == ("save_statevector", "kept", (0, 1))
    assert json.loads(saves[0][3])["subtype"] == "list"
    assert b"measure" not in qasm and b"save" not in qasm
    assert circuit == original
    rebuilt = QuantumCircuit.from_qasm_str(qasm.decode())
    np.testing.assert_allclose(np.exp(1j*phase)*Operator(rebuilt).data,
                               expected_body, rtol=0, atol=2e-14)
    circuit.x(1)
    with pytest.raises(ValueError, match="intermediate"):
        _static_body(circuit, max_operations=20, max_bytes=8192)
    reset = QuantumCircuit(1)
    reset.reset(0)
    with pytest.raises(ValueError, match="nonunitary"):
        _static_body(reset, max_operations=20, max_bytes=8192)
    # A Gate wrapper cannot hide a nonunitary action in its stored body.
    from qiskit.circuit import Gate

    wrapped = Gate("apparently_unitary", 1, [])
    wrapped.definition = reset
    nested = QuantumCircuit(1)
    nested.append(wrapped, [0])
    with pytest.raises(ValueError, match="unsupported native instruction"):
        _static_body(nested, max_operations=20, max_bytes=8192)


def test_invalid_options_and_input_limits_refuse_before_child(monkeypatch):
    from nwqlib.backends import _auxiliary_process

    monkeypatch.setattr(
        _auxiliary_process,
        "run_auxiliary",
        lambda *a, **k: pytest.fail("invalid input reached compiler"),
    )
    native = lowered_h_twice()
    for options in (
        {"count_only": True, "target": "pbc"},
        {"epsilon": 0},
        {"max_operations": 1},
        {"max_bytes": 1},
        {"index": 1},
    ):
        with pytest.raises(ValueError):
            compile_logical(native, **options)


@pytest.mark.parametrize("noisy", [False, True])
def test_auxiliary_timeout_or_capture_limit_reaps_child(noisy, monkeypatch):
    from nwqlib.backends import _auxiliary_process

    children = []
    real_popen = subprocess.Popen

    def fixture_process(command, **kwargs):
        code = "import os; os.write(1, b'x'*4096)" if noisy else "import time; time.sleep(10)"
        child = real_popen([sys.executable, "-c", code], **kwargs)
        children.append(child)
        return child

    monkeypatch.setattr(_auxiliary_process.subprocess, "Popen", fixture_process)
    with pytest.raises(ValueError if noisy else TimeoutError):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=1024, timeout_seconds=0.5)
    assert len(children) == 1 and children[0].poll() is not None


def test_child_forwards_exact_external_options_and_admits_output_before_qasm(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from nwqlib.backends._auxiliary_process import _compile_nwqec

    calls = []
    artifact = SimpleNamespace(
        num_qubits=lambda: 1,
        count_ops=lambda: {"t": 2},
        depth=lambda: 2,
        to_qasm=lambda: "OPENQASM 2.0; qreg q[1]; t q[0]; t q[0];",
    )

    def transform(name):
        def invoke(circuit, **options):
            assert circuit is artifact
            calls.append((name, options))
            return {"t": 2} if name == "counts" else artifact

        return invoke

    monkeypatch.setitem(
        sys.modules,
        "nwqec",
        SimpleNamespace(
            __version__="0.1.2",
            __file__="test-native-contract",
            load_qasm=lambda path: artifact,
            get_clifford_t_counts=transform("counts"),
            to_clifford_t=transform("ct"),
            to_pbc=transform("pbc"),
            to_clifford_reduction=transform("reduction"),
        ),
    )
    common = dict(rz_err="relative", epsilon=0.003)
    cases = [
        ("clifford_t", False, "ct", dict(keep_ccx=False)),
        ("clifford_t", True, "counts", dict(keep_ccx=False)),
        ("pbc", False, "pbc", dict(keep_cx=False, optimize_t_count=False)),
        ("pbc_tfuse", False, "pbc", dict(keep_cx=False, optimize_t_count=True)),
        ("clifford_reduction", False, "reduction", {}),
    ]
    for target, count_only, name, options in cases:
        request = dict(
            target=target,
            count_only=count_only,
            max_operations=2,
            **common,
        )
        _compile_nwqec(request, tmp_path)
        assert calls[-1] == (name, {**common, **options})
    artifact.to_qasm = lambda: pytest.fail("oversized compiled output was serialized")
    request.update(target="clifford_t", count_only=False, max_operations=1)
    with pytest.raises(ValueError, match="output exceeds"):
        _compile_nwqec(request, tmp_path)


def test_auxiliary_diagnostics_use_the_selected_byte_allowance(monkeypatch):
    """A successful child's 128 KiB log is valid under the caller's 1 MB budget."""
    from nwqlib.backends import _auxiliary_process

    real_popen = subprocess.Popen

    def fixture_process(command, **kwargs):
        directory = command[-2]
        code = (
            "import os, pathlib; os.write(1, b'x' * 131072); "
            f"pathlib.Path({directory!r}, 'result.json').write_text('{{}}')"
        )
        return real_popen([sys.executable, "-c", code], **kwargs)

    monkeypatch.setattr(_auxiliary_process.subprocess, "Popen", fixture_process)
    result, _ = _auxiliary_process.run_auxiliary(
        "nwqec", {}, max_bytes=1_000_000, timeout_seconds=5
    )
    assert result["diagnostics"] == "x" * 131072
