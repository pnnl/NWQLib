"""Selected-source transport and stage semantics; no upstream precision claims."""

import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
from qiskit import QuantumCircuit

from nwqlib.backends.nwqec import LogicalCompilation, _static_body, compile_logical
from nwqlib.blocks import SelectedConstruction, lower_qiskit, select_preparation
from nwqlib.ir import Allocate, BlockCall, Definition, PortMap, Program, Register, Repeat, Sequence
from nwqlib.problems.inputs import ingest_vector


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
    return lower_qiskit(construction, blocks=(block,))


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
    # The default byte limit is the physical memory, read when the compilation runs, and the
    # record keeps None, so a reopened record names no machine-specific value.
    assert counts.max_bytes is None and full.max_bytes is None and restored.max_bytes is None
    assert restored.max_operations is None and restored.timeout_seconds is None


def test_clifford_reduction_gives_no_t_count_because_its_output_writes_t_as_a_rotation():
    """RZ(pi/4) is T up to phase, so clifford_t counts one T. clifford_reduction writes the same gate
    as rx(pi/4), which counting t and tdg would read as zero T."""
    require_native()
    native = lowered_rotation(np.pi / 4)
    known = compile_logical(native, target="clifford_t")
    assert next(q for q in known.quantities if q.metric == "t").fact.value.numerator == 1
    reduced = compile_logical(native, target="clifford_reduction")
    assert dict(reduced.raw_counts).get("rx(pi/4)") == 1 and "t" not in dict(reduced.raw_counts)
    t = next(q for q in reduced.quantities if q.metric == "t")
    assert t.fact.availability == "unknown" and t.interpretation == "unavailable"
    assert "rx(pi/4)" in t.fact.reason and "clifford_t" in t.fact.reason
    cx = next(q for q in reduced.quantities if q.metric == "cx")
    assert cx.fact.value.numerator == 0 and cx.interpretation == "exact"


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


@pytest.mark.parametrize("epsilon,rz_err,sent", [
    (3e-4, "total", 3e-4),
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
    # The fixture child stands in for NWQEC, so the parent's package check passes without it.
    monkeypatch.setattr(_auxiliary_process, "find_spec", lambda package: object())
    with pytest.raises(ValueError if noisy else TimeoutError):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=1024, timeout_seconds=0.5)
    assert len(children) == 1 and children[0].poll() is not None


# The real Popen, before any test replaces it, so that one test can script several children.
_REAL_POPEN = subprocess.Popen


def _scripted_child(monkeypatch, *, stdout=b"", stderr=b"", result=b"{}", code=0):
    """Replace the auxiliary child with a real Python child that writes these bytes, and return the commands.

    The child writes ``result`` to the call's ``result.json``, then ``stdout``
    and ``stderr`` to its two pipes, and exits with ``code``.
    """
    from nwqlib.backends import _auxiliary_process

    commands = []
    program = ("import os, sys; from pathlib import Path; "
               "Path(sys.argv[1], 'result.json').write_bytes(bytes.fromhex(sys.argv[2])); "
               "os.write(1, bytes.fromhex(sys.argv[3])); os.write(2, bytes.fromhex(sys.argv[4])); "
               "sys.exit(int(sys.argv[5]))")

    def fixture_process(command, **kwargs):
        commands.append(command)
        return _REAL_POPEN([sys.executable, "-c", program, command[-2], result.hex(), stdout.hex(),
                           stderr.hex(), str(code)], **kwargs)

    monkeypatch.setattr(_auxiliary_process.subprocess, "Popen", fixture_process)
    monkeypatch.setattr(_auxiliary_process, "find_spec", lambda package: object())
    return commands


# With max_bytes=8192 and the 3-byte request "{}\n", the capture may hold (1 + k) n <= 8189
# bytes, k = 1 for ASCII, 2 without a byte of 0xF0 or more, and 4 otherwise.
_CAPTURE_CASES = [
    ("ascii", b"a" * 4094, 1),
    ("bmp", b"a" * 2726 + "€".encode(), 2),
    ("supplementary", b"a" * 1633 + "😀".encode(), 4),
    ("invalid", b"\xff" + b"a" * 1636, 4),
]


@pytest.mark.parametrize("declared", [8192, None])
@pytest.mark.parametrize("name, admitted, width", _CAPTURE_CASES, ids=[case[0] for case in _CAPTURE_CASES])
def test_auxiliary_capture_counts_its_decoded_text_by_byte_class(monkeypatch, declared, name, admitted, width):
    """The largest admitted output decodes completely, and one more byte is refused before it is decoded.

    An explicit cap and the physical memory (mocked at 8192 bytes) give the
    same boundary, and the refusal names the one that applied. Invalid UTF-8
    decodes with U+FFFD. The child receives the resolved limit.
    """
    from nwqlib import _limits
    from nwqlib.backends import _auxiliary_process

    monkeypatch.setattr(_limits, "physical_memory_bytes", lambda: 8192)
    assert (1 + width) * len(admitted) <= 8189 < (1 + width) * (len(admitted) + 1)
    commands = _scripted_child(monkeypatch, stdout=admitted)
    result, _ = _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=declared, timeout_seconds=30)
    assert result["diagnostics"] == admitted.decode("utf-8", errors="replace")
    assert ("�" in result["diagnostics"]) == (name == "invalid") and commands[0][-1] == "8192"
    refused = b"a" + admitted
    source = "max_bytes=8192" if declared is not None else r"the physical memory \(8192 bytes\)"
    _scripted_child(monkeypatch, stdout=refused)
    with pytest.raises(ValueError, match=rf"diagnostics of {len(refused)} bytes need "
                                         rf"{(1 + width) * len(refused)} bytes .* beyond the 8189 bytes that {source}"):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=declared, timeout_seconds=30)


def test_auxiliary_capture_counts_both_pipes_together(monkeypatch):
    """Output split between stdout and stderr counts as one capture against the same limit."""
    from nwqlib.backends import _auxiliary_process

    _scripted_child(monkeypatch, stdout=b"a" * 2047, stderr=b"b" * 2047)
    result, _ = _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)
    diagnostics = result["diagnostics"]
    assert len(diagnostics) == 4094 and diagnostics.count("a") == diagnostics.count("b") == 2047
    _scripted_child(monkeypatch, stdout=b"a" * 2048, stderr=b"b" * 2048)
    with pytest.raises(ValueError, match="diagnostics of 4096 bytes need 8192 bytes"):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)


def test_auxiliary_result_needs_four_copies_in_what_the_diagnostics_leave(monkeypatch):
    """After 4094 ASCII bytes of diagnostics, 4095 bytes remain, so a result file of 1023 bytes is read and 1024 is not."""
    from nwqlib.backends import _auxiliary_process

    def result(size):
        return b'{"x":"' + b"y" * (size - 8) + b'"}'

    _scripted_child(monkeypatch, stdout=b"a" * 4094, result=result(1023))
    read, _ = _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)
    assert read["x"] == "y" * 1015 and read["diagnostics"] == "a" * 4094
    _scripted_child(monkeypatch, stdout=b"a" * 4094, result=result(1024))
    with pytest.raises(ValueError, match="result of 1024 bytes needs 4096 bytes for its copies, beyond the 4095 bytes"):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)


def test_auxiliary_child_failure_keeps_its_complete_diagnostics(monkeypatch):
    """A nonzero exit raises RuntimeError with the complete decoded diagnostics after the exit status.

    Diagnostics beyond the limit are refused while they are captured, before
    any decoding, under the same count.
    """
    from nwqlib.backends import _auxiliary_process

    text = "warning: é\n" + "x" * 1000
    _scripted_child(monkeypatch, stderr=text.encode(), code=3)
    with pytest.raises(RuntimeError) as failed:
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)
    assert str(failed.value) == f"nwqec child failed (3): {text}"
    _scripted_child(monkeypatch, stderr=b"x" * 4095, code=3)
    with pytest.raises(ValueError, match="diagnostics of 4095 bytes need 8190 bytes"):
        _auxiliary_process.run_auxiliary("nwqec", {}, max_bytes=8192, timeout_seconds=30)


def test_child_forwards_exact_external_options_and_checks_four_copies_of_the_output_text(monkeypatch, tmp_path):
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
        request = dict(target=target, count_only=count_only, **common)
        _compile_nwqec(request, tmp_path, 4096)
        assert calls[-1] == (name, {**common, **options})
    text = artifact.to_qasm()
    request = dict(target="clifford_t", count_only=False, **common)
    assert _compile_nwqec(request, tmp_path, 4 * len(text))["qasm"] == text
    with pytest.raises(ValueError, match="copies"):
        _compile_nwqec(request, tmp_path, 4 * len(text) - 1)
    # A count-only result is a short count table and has no output check.
    request.update(count_only=True)
    assert _compile_nwqec(request, tmp_path, 1)["counts"] == {"t": 2}
