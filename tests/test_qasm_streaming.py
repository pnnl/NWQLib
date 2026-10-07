"""Small syntax, source-binding and lifecycle checks; no numerical execution."""

from io import BytesIO

import numpy as np
import openqasm3
from openqasm3 import ast
import pytest

import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.blocks import SelectedConstruction, select_preparation, select_zero_reflection, transform_block
from nwqlib.io import (
    QasmMaterializationBudget, QasmWriteBudget, QasmWriteError, QasmWriteReceipt,
    materialize_qasm3_file, write_qasm3, write_qasm3_file,
)
from nwqlib.ir import (
    Allocate, BlockCall, ClassicalValue, Definition, Measure, MeasurementBatch, PortMap, Program,
    Register, Release, Repeat, Reset, Sequence, Setting,
)
from nwqlib.operators import ingest_pauli
from nwqlib.problems.inputs import ingest_occupation, ingest_vector

WRITE = QasmWriteBudget(max_metadata_bytes=1000000, max_walk_steps=100000,
                        max_qubits=3, max_clbits=3, max_bytes=4096, max_instructions=100, max_chunk_bytes=13)
MATERIALIZE = QasmMaterializationBudget(max_bytes=4096, max_qubits=3, max_clbits=3,
                                      max_dynamic_visits=100, max_operations=100)


def expectation_plan():
    """Exact <psi|(I+Z)|psi>/<psi|psi> for the physical vector (2,2), prepared as |+> by one H."""
    problem = nwqlib.Expectation(
        state=[2.0, 2.0], observable=ingest_pauli((("I", 1.0), ("Z", 1.0)), num_qubits=1)
    )
    return nwqlib.plan(problem, method=ExpectationMethod(), seed=7)


def one(block, *, repeat=1, reverse_registers=False):
    ports = block.record.signature.quantum
    registers = tuple(Register(name=p.name, width=p.width) for p in ports)
    if reverse_registers:
        registers = registers[::-1]
    return SelectedConstruction(program=Program(
        root="root", registers=registers, signatures=(block.record.signature,),
        definitions=tuple(Definition(id=f"alloc{i}", node=Allocate(wire=p.name)) for i, p in enumerate(ports)) + (
            Definition(id="call", node=BlockCall(signature=block.record.signature.name,
                       ports=tuple(PortMap(port=p.name, wire=p.name) for p in ports))),
            Definition(id="repeat", node=Repeat(body="call", count=repeat)),
            Definition(id="root", node=Sequence(children=tuple(f"alloc{i}" for i in range(len(ports))) + ("repeat",))),
        ),
    ), selections=(block.record,))


def emit(construction, budget=WRITE):
    sink = BytesIO()
    receipt = write_qasm3(construction, sink, budget=budget)
    text = sink.getvalue().decode("ascii")
    return text, receipt, openqasm3.parse(text)


def gate_body(parsed):
    return next(node.body for node in parsed.statements if isinstance(node, ast.QuantumGateDefinition))


def literal(node):
    if isinstance(node, ast.UnaryExpression):
        assert node.op.name == "-"
        return -literal(node.expression)
    return node.value


def test_actual_expectation_selection_and_json_file_consumer(tmp_path):
    from nwqlib import Expectation, plan as select_plan
    plan = expectation_plan()
    states = ((plan.problem.state, "native", ["h"]),
              (ingest_occupation("1", num_qubits=1), "native", ["x"]),
              (ingest_occupation("1", num_qubits=1), "hzh", ["h", "z", "h"]))
    for state, choice, expected in states:
        selected = select_plan(Expectation(state=state, observable=plan.problem.observable),
            method=ExpectationMethod(preparation_choice=choice), seed=7)
        construction = SelectedConstruction.model_validate_json(selected.construction.model_dump_json())
        text, receipt, parsed = emit(construction)
        # Independent intended preparations: |+>, |1>, and HZH = X.
        assert [node.name.name for node in gate_body(parsed)] == expected
        assert "measure" not in text and "gphase" not in text
        assert selected.output.kind == "normalized_expectation"
        assert receipt.expanded_operations == len(expected)
        path = tmp_path / f"{choice}-{expected[0]}.qasm"
        completed = write_qasm3_file(construction, path, budget=WRITE)
        loaded = materialize_qasm3_file(construction, path, completed, writer_budget=WRITE,
                                       budget=MATERIALIZE.revise(max_bytes=2**100))
        assert loaded.output_nodes == 1
        assert loaded.circuit.num_qubits == 1
        assert [item.operation.name for item in loaded.circuit.data[0].operation.definition.data] == expected
        assert QasmWriteReceipt.model_validate_json(completed.model_dump_json()) == completed
    # The original physical request keeps its norm and external I+Z observable.
    assert plan.reconstruction.physical_scale.as_float() ** 2 == pytest.approx(8., rel=0., abs=2e-15)


def test_phase_adjoint_recipe_and_signature_order():
    block = select_preparation("unsafe name; do not emit", ingest_vector(np.array([0, 1j, 0, 0])))
    cases = ((block, False, False),
             (transform_block("ctrl", block, control=True), True, False),
             (transform_block("inv", block, adjoint=True), False, True),
             (transform_block("both", block, control=True, adjoint=True), True, True))
    for selected, controlled, adjoint in cases:
        text, receipt, parsed = emit(one(selected, reverse_registers=True))
        assert "unsafe name" not in text
        body = gate_body(parsed)
        phase = body[0 if adjoint else 1]
        flip = body[1 if adjoint else 0]
        assert isinstance(phase, ast.QuantumPhase)
        assert literal(phase.argument) == (-1 if adjoint else 1) * float(np.pi / 2)
        assert flip.name.name == "x"
        assert flip.qubits[0].name == ("a0" if controlled else "a0")
        if controlled:
            assert [literal(mod.argument) for mod in flip.modifiers] == [1]
            assert [q.name for q in flip.qubits] == ["a0", "a1"]
            assert [q.name for q in phase.qubits] == ["a0"]
            assert [literal(mod.argument) for mod in phase.modifiers] == [1]
            # Declarations reversed, but formal control still precedes target.
            assert "g0 q1[0], q0[0], q0[1];" in text
        else:
            assert not phase.modifiers and not phase.qubits
        assert receipt.expanded_operations == 2
        # Stored binary64 is serialized exactly, including sign under adjoint.
        stored = block.record.decomposition[-1].angle
        assert literal(phase.argument) == (-stored if adjoint else stored)


def test_multisite_xy_parity_forward_inverse_control_parser_and_native_recipe(tmp_path):
    from nwqlib.blocks import lower_qiskit
    from nwqlib.blocks.selection import select_pauli_parity
    from nwqlib.core import Basis

    block=select_pauli_parity("xy", "XY",basis=Basis(identity="two-sites",dimension=4,ordering="LSB"))
    # For |+> on q1 and |+i> on q0, both X and Y eigenvalues are +1.
    # Z expectation on an unrotated |+i> is zero, so pivot-only rotation fails.
    assert [(op.gate,op.qubits) for op in block.record.decomposition]==[
        ("sdg",(0,)),("h",(0,)),("h",(1,)),("cx",(0,1))]
    for controlled,adjoint in ((False,False),(False,True),(True,False),(True,True)):
        selected=(transform_block(f"parity_{controlled}_{adjoint}",block,control=controlled,adjoint=adjoint)
                  if controlled or adjoint else block)
        construction=one(selected)
        text,receipt,parsed=emit(construction)
        expected=["cx","h","h","s"] if adjoint else ["sdg","h","h","cx"]
        assert [node.name.name for node in gate_body(parsed)]==expected
        assert receipt.expanded_operations==4
        if controlled:
            assert all([literal(mod.argument) for mod in node.modifiers]==[1] for node in gate_body(parsed))
        path=tmp_path/f"parity-{controlled}-{adjoint}.qasm"
        written=write_qasm3_file(construction,path,budget=WRITE)
        loaded=materialize_qasm3_file(construction,path,written,writer_budget=WRITE,budget=MATERIALIZE)
        assert loaded.circuit.num_qubits==2+controlled
        if not controlled:
            native=lower_qiskit(construction,blocks=(selected,),max_operations=1000,max_qubits=3,max_clbits=0).circuit
            assert [item.operation.name for item in native.data[0].operation.definition.data]==expected
        # Whole-composite native control may decompose to another gate basis.
        # Its action remains source-qualified through the existing control owner;
        # this offline recipe/import witness does not numerically validate it.


def test_reflection_ast_has_positive_zero_sign_and_reversed_inverse():
    block = select_zero_reflection("reflection", 2)
    for selected, inverse in ((block, False), (transform_block("inverse", block, adjoint=True), True)):
        text, receipt, parsed = emit(one(selected))
        body = gate_body(parsed)
        phase = body[0] if inverse else body[-1]
        assert literal(phase.argument) == (-float(np.pi) if inverse else float(np.pi))
        gates = body[1:] if inverse else body[:-1]
        assert [node.name.name for node in gates] == ["x", "x", "z", "x", "x"]
        assert [node.qubits[0].name for node in gates] == (["a1", "a0", "a0", "a1", "a0"] if inverse
                                                          else ["a0", "a1", "a0", "a0", "a1"])
        assert literal(gates[2].modifiers[0].argument) == 1
        assert [q.name for q in gates[2].qubits] == ["a0", "a1"]
        # X on both axes conjugates CZ's -1 at |11> to |00>;
        # the stored pi phase yields the selected (+1,-1,-1,-1) convention.
        assert receipt.expanded_operations == 6
        assert text.count("gphase") == 1


def test_repeat_text_is_lazy_nested_and_dynamic_admission_precedes_sdk(tmp_path, monkeypatch):
    """A trillion repetitions remain one textual loop, but dynamic materialization must reject
    before SDK allocation.
    """
    block = select_preparation("hzh", ingest_occupation("1", num_qubits=1), choice="hzh")
    texts = []
    for count in (0, 1, 2, 10**12, 2**63 - 1):
        text, receipt, parsed = emit(one(block, repeat=count))
        texts.append(text)
        assert receipt.expanded_operations == 3 * count
        assert receipt.emitted_instructions == (4 if count else 0)
        loops = [node for node in parsed.statements if isinstance(node, ast.ForInLoop)]
        assert len(loops) == bool(count)
        if count:
            assert loops[0].set_declaration.end.value == count - 1
            assert len(loops[0].block) == 1
    assert len(texts[3]) - len(texts[2]) == 11  # inclusive endpoint digits: 1 versus 999999999999
    huge = one(block, repeat=10**12)
    path = tmp_path / "huge.qasm"
    receipt = write_qasm3_file(huge, path, budget=WRITE)
    import qiskit.qasm3
    import qiskit.transpiler.passes
    def forbidden(*args, **kwargs):
        raise AssertionError("consumer allocation happened before admission")
    with monkeypatch.context() as patch:
        patch.setattr(qiskit.qasm3, "loads", forbidden)
        patch.setattr(qiskit.transpiler.passes, "UnrollForLoops", forbidden)
        with pytest.raises(ValueError, match="dynamic materialization"):
            materialize_qasm3_file(huge, path, receipt, writer_budget=WRITE, budget=MATERIALIZE)
    nested = one(block, repeat=2)
    definitions = tuple(d.revise(node=Repeat(body="inner", count=2)) if d.id == "repeat" else d
                        for d in nested.program.definitions) + (Definition(id="inner", node=Repeat(body="call", count=3)),)
    nested = nested.revise(program=nested.program.revise(definitions=definitions))
    receipt = write_qasm3_file(nested, path, budget=WRITE)
    assert receipt.expanded_operations == 18
    imported = materialize_qasm3_file(nested, path, receipt, writer_budget=WRITE, budget=MATERIALIZE)
    assert imported.output_nodes == 6  # six logical HZH calls, eighteen primitive applications
    assert all(item.operation.name == "g0" for item in imported.circuit.data)


def test_identity_loop_has_dynamic_cost_and_zero_width_emits_no_call(tmp_path, monkeypatch):
    identity = select_preparation("identity", ingest_occupation("0", num_qubits=1))
    huge = one(identity, repeat=10**12)
    path = tmp_path / "empty.qasm"
    receipt = write_qasm3_file(huge, path, budget=WRITE)
    assert receipt.expanded_operations == 0 and receipt.dynamic_visits > 10**12
    import qiskit.qasm3
    monkeypatch.setattr(qiskit.qasm3, "loads", lambda *_: pytest.fail("import must not happen"))
    with pytest.raises(ValueError, match="dynamic materialization"):
        materialize_qasm3_file(huge, path, receipt, writer_budget=WRITE, budget=MATERIALIZE)
    text, receipt, _ = emit(one(select_zero_reflection("empty", 0)))
    assert "qubit[0]" not in text and "gate " not in text and "g0 " not in text
    assert receipt.expanded_operations == receipt.emitted_instructions == 0


def test_byte_boundaries_partial_sink_failure_and_cancellation(tmp_path, monkeypatch):
    """Exact byte caps and partial writes must report the accepted prefix while failed file
    replacement preserves old content.
    """
    construction = one(select_zero_reflection("reflect", 2), repeat=2)
    text, receipt, _ = emit(construction)
    assert emit(construction, WRITE.revise(max_bytes=len(text)))[1].bytes_written == len(text)
    sink = BytesIO()
    with pytest.raises(QasmWriteError) as failure:
        write_qasm3(construction, sink, budget=WRITE.revise(max_bytes=len(text) - 1))
    assert sink.getvalue() == text.encode()[:-1]
    assert failure.value.prefix.bytes_written == len(text) - 1
    class Partial:
        def __init__(self):
            self.data = bytearray()
            self.offered = []
        def write(self, chunk):
            self.offered.append(len(chunk))
            if len(self.data) >= 17:
                raise OSError("sink gone")
            size = min(len(chunk), 3, 17 - len(self.data))
            self.data.extend(chunk[:size])
            return size
    partial = Partial()
    with pytest.raises(QasmWriteError) as failure:
        write_qasm3(construction, partial, budget=WRITE)
    assert isinstance(failure.value.__cause__, OSError)
    assert bytes(partial.data) == text.encode()[:17]
    assert failure.value.prefix.bytes_written == 17
    assert max(partial.offered) <= WRITE.max_chunk_bytes
    partial = Partial()
    with pytest.raises(QasmWriteError) as failure:
        write_qasm3(construction, partial, budget=WRITE, cancel=lambda: len(partial.data) >= 7)
    assert isinstance(failure.value.__cause__, InterruptedError)
    assert bytes(partial.data) == text.encode()[:len(partial.data)]
    path = tmp_path / "existing.qasm"
    path.write_bytes(b"previous")
    for kwargs in ({"budget": WRITE.revise(max_bytes=10)}, {"budget": WRITE, "cancel": lambda: True}):
        with pytest.raises(QasmWriteError):
            write_qasm3_file(construction, path, **kwargs)
        assert path.read_bytes() == b"previous"
        assert list(tmp_path.iterdir()) == [path]
    import os
    with monkeypatch.context() as patch:
        def refused(*args):
            raise OSError("replacement refused")
        patch.setattr(os, "replace", refused)
        with pytest.raises(QasmWriteError) as failure:
            write_qasm3_file(construction, path, budget=WRITE)
        assert isinstance(failure.value.__cause__, OSError)
        assert path.read_bytes() == b"previous" and list(tmp_path.iterdir()) == [path]
    write_qasm3_file(construction, path, budget=WRITE)
    assert path.read_bytes() == text.encode()


@pytest.mark.parametrize("returned", [0, None, -1, True, 10000])
def test_invalid_binary_sink_contract(returned):
    class Sink:
        def write(self, chunk):
            return returned
    with pytest.raises(QasmWriteError) as failure:
        write_qasm3(one(select_zero_reflection("identity", 0)), Sink(), budget=WRITE)
    assert failure.value.prefix.bytes_written == 0


def test_measure_reset_position_and_bit_register_order():
    block = select_preparation("x", ingest_occupation("10", num_qubits=2))
    construction = one(block)
    program = construction.program
    definitions = tuple(d for d in program.definitions if d.id != "root") + (
        Definition(id="measure", node=Measure(wire="system", result="read out;")),
        Definition(id="reset", node=Reset(wire="system")),
        Definition(id="release", node=Release(wire="system")),
        Definition(id="root", node=Sequence(children=("alloc0", "call", "measure", "reset", "call", "release"))),
    )
    construction = construction.revise(program=program.revise(
        definitions=definitions, classical=(ClassicalValue(name="read out;", dtype="bits", width=2),)))
    text, receipt, parsed = emit(construction)
    operations = [type(node).__name__ for node in parsed.statements if isinstance(node, (
        ast.QuantumGate, ast.QuantumMeasurementStatement, ast.QuantumReset))]
    assert operations == ["QuantumGate", "QuantumMeasurementStatement", "QuantumMeasurementStatement",
                          "QuantumReset", "QuantumReset", "QuantumGate"]
    assert "c0[0] = measure q0[0];\nc0[1] = measure q0[1];" in text
    assert "reset q0[0];\nreset q0[1];" in text
    assert receipt.classical_layout == (("read out;", "c0", 2),)
    assert receipt.expanded_operations == 6
    # Every recipe/call/measure/reset statement completes at its semicolon.
    # Declarations consume bytes but are not application instructions.
    offset = completed = 0
    for line in text.splitlines(keepends=True):
        if ";" in line and not line.startswith(("OPENQASM", "include", "qubit[", "bit[")):
            position = offset + line.index(";")
            for cap, expected in ((position, completed), (position + 1, completed + 1)):
                sink = BytesIO()
                with pytest.raises(QasmWriteError) as failure:
                    write_qasm3(construction, sink, budget=WRITE.revise(max_bytes=cap))
                assert sink.getvalue() == text.encode()[:cap]
                assert failure.value.prefix.emitted_instructions == expected
            completed += 1
        offset += len(line)
    assert receipt.emitted_instructions == completed


def test_admission_rejects_opaque_width_instruction_integer_and_metadata_before_output():
    block = select_preparation("x", ingest_occupation("1", num_qubits=1))
    cases = ((one(block, repeat=2**63), WRITE, "int"),
             (one(block), WRITE.revise(max_qubits=0), "width"),
             (one(block), WRITE.revise(max_instructions=1), "instructions"),
             (one(block), WRITE.revise(max_walk_steps=1), "max_walk_steps"),
             (one(block), WRITE.revise(max_metadata_bytes=1), "max_metadata_bytes"),
             (one(select_preparation("opaque", ingest_vector(np.array([1., 2.])))), WRITE, "opaque"))
    for construction, budget, message in cases:
        sink = BytesIO()
        with pytest.raises(ValueError, match=message):
            write_qasm3(construction, sink, budget=budget)
        assert sink.getvalue() == b""


def test_materializer_checks_file_bytes_before_reading(tmp_path, monkeypatch):
    construction = one(select_zero_reflection("reflection", 2))
    path = tmp_path / "source.qasm"
    receipt = write_qasm3_file(construction, path, budget=WRITE)
    import qiskit.qasm3
    monkeypatch.setattr(qiskit.qasm3, "loads", lambda *_: pytest.fail("unadmitted import"))

    from contextlib import nullcontext
    from pathlib import Path
    from types import SimpleNamespace

    def forbidden_read(*args):
        pytest.fail("QASM text read before byte admission")

    with path.open("rb") as actual:
        unopened_text = SimpleNamespace(fileno=actual.fileno, read=forbidden_read)
        with monkeypatch.context() as guard:
            guard.setattr(Path, "open", lambda *a, **k: nullcontext(unopened_text))
            with pytest.raises(ValueError, match="consumer budget"):
                materialize_qasm3_file(construction, path, receipt, writer_budget=WRITE,
                                      budget=MATERIALIZE.revise(max_bytes=1))


def test_materializer_control_cap_rejects_before_open_import_or_unroll(tmp_path, monkeypatch):
    # Four wires are metadata/text only: never import this higher-control fixture.
    construction = one(transform_block("controlled", select_zero_reflection("r", 3), control=True))
    writer = WRITE.revise(max_qubits=4)
    path = tmp_path / "control.qasm"
    receipt = write_qasm3_file(construction, path, budget=writer)
    assert b"ctrl(3) @ z" in path.read_bytes()
    import qiskit.qasm3
    import qiskit.transpiler.passes
    from pathlib import Path
    def forbidden(*args, **kwargs):
        pytest.fail("higher-control consumer must reject before allocation or file access")
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", forbidden)
        patch.setattr(qiskit.qasm3, "loads", forbidden)
        patch.setattr(qiskit.transpiler.passes, "UnrollForLoops", forbidden)
        with pytest.raises(ValueError, match="more than two controls"):
            materialize_qasm3_file(construction, path, receipt, writer_budget=writer,
                                  budget=MATERIALIZE.revise(max_qubits=4))
    # The nearest legal case remains accepted by the actual installed consumer.
    legal = one(transform_block("controlled", select_zero_reflection("r", 2), control=True))
    receipt = write_qasm3_file(legal, path, budget=WRITE)
    result = materialize_qasm3_file(legal, path, receipt, writer_budget=WRITE, budget=MATERIALIZE)
    assert result.circuit.num_qubits == 3 and result.output_nodes == 1
    assert "ccz" in [item.operation.name for item in result.circuit.data[0].operation.definition.data]


def test_batch_context_is_external_and_reallocation_is_rejected():
    from nwqlib.core import InputRef, Source
    from nwqlib.ir import MetadataRef
    source = Source(name="readout", version="1", domain="metadata", reference="test:readout")
    metadata = MetadataRef(format=source, data=InputRef(identity="test:readout", representation="readout", source=source))
    construction = one(select_preparation("x", ingest_occupation("1", num_qubits=1)))
    program = construction.program
    batch = Definition(id="batch", node=MeasurementBatch(body="experiment", repetitions=99,
                       observation_kind="counts", settings=(Setting(label="Z", metadata=metadata),)))
    construction = construction.revise(program=program.revise(root="batch", definitions=program.definitions + (
        Definition(id="release", node=Release(wire="system")),
        Definition(id="experiment", node=Sequence(children=("root", "release"))), batch)))
    text, receipt, _ = emit(construction)
    assert (receipt.batch_repetitions, receipt.observation_kind, receipt.setting_label) == (99, "counts", "Z")
    assert receipt.expanded_operations == 1 and "99" not in text and "measure" not in text
    reallocated = program.revise(definitions=tuple(d for d in program.definitions if d.id != "root") + (
        Definition(id="release", node=Release(wire="system")),
        Definition(id="root", node=Sequence(children=("alloc0", "call", "release", "alloc0", "call"))),))
    sink = BytesIO()
    with pytest.raises(ValueError, match="reallocation"):
        write_qasm3(construction.revise(program=reallocated), sink, budget=WRITE)
    assert sink.getvalue() == b""


@pytest.mark.parametrize("stage", ["write_close", "write_close_unlink", "flush", "close", "replace_unlink", "cancel",
                                  "keyboard", "system_exit", "keyboard_unlink"])
def test_file_finalization_preserves_primary(tmp_path, monkeypatch, stage):
    """Inject failures during write, flush, close and cleanup so secondary errors cannot hide the
    primary failure.
    """
    from pathlib import Path
    import nwqlib.io.streaming as streaming

    plan = expectation_plan()
    text, _, _ = emit(plan.construction)
    path = tmp_path / "existing.qasm"
    path.write_bytes(b"previous")
    original_temporary = streaming.tempfile.NamedTemporaryFile
    original_write = streaming._write
    primary = []
    events = []
    close_error = OSError("close refused")
    cleanup_error = PermissionError("unlink refused")
    operation_error = OSError("operation refused")
    raw_control = stage.startswith("keyboard") or stage == "system_exit"
    control = KeyboardInterrupt("injected unwind") if stage.startswith("keyboard") else SystemExit("injected unwind")

    class Sink:
        def __init__(self, **kwargs):
            self.inner = original_temporary(**kwargs)
            self.name = self.inner.name
        def __enter__(self):
            return self
        def __exit__(self, *args):
            self.close()
        def write(self, chunk):
            if raw_control:
                raise control
            return self.inner.write(chunk)
        def flush(self):
            if stage == "flush":
                raise operation_error
            self.inner.flush()
        def close(self):
            self.inner.close()
            events.append("close")
            if stage in {"write_close", "write_close_unlink", "close"}:
                raise close_error

    def writing(*args):
        try:
            return original_write(*args)
        except QasmWriteError as error:
            primary.append(error)
            raise

    def replace(*args):
        events.append("replace")
        raise operation_error

    def unlink(*args, **kwargs):
        raise cleanup_error

    with monkeypatch.context() as patch:
        patch.setattr(streaming.tempfile, "NamedTemporaryFile", Sink)
        patch.setattr(streaming, "_write", writing)
        patch.setattr(streaming.os, "replace", replace)
        if stage.endswith("unlink"):
            patch.setattr(Path, "unlink", unlink)
        expected_type = type(control) if raw_control else QasmWriteError
        with pytest.raises(expected_type) as failure:
            write_qasm3_file(plan.construction, path,
                            budget=WRITE.revise(max_bytes=10) if stage.startswith("write") else WRITE,
                            cancel=(lambda: "close" in events) if stage == "cancel" else None)
        error = failure.value
        assert path.read_bytes() == b"previous"
        assert events == (["close", "replace"] if stage == "replace_unlink" else ["close"])
        if raw_control:
            assert error is control
            if stage.endswith("unlink"):
                remaining = next(p for p in tmp_path.iterdir() if p != path)
                assert any(repr(cleanup_error) in note for note in error.__notes__)
                assert any(str(remaining) in note for note in error.__notes__)
        else:
            accepted = 10 if stage.startswith("write") else len(text)
            assert error.prefix.bytes_written == accepted
            if stage.startswith("write"):
                assert error is primary[0]
                assert isinstance(error.__cause__, ValueError)
                assert "byte cap" in str(error.__cause__)
            elif stage == "cancel":
                assert isinstance(error.__cause__, InterruptedError)
            else:
                assert error.__cause__ is (close_error if stage == "close" else operation_error)
            secondary = ((close_error,) if stage.startswith("write") else ())
            secondary += (cleanup_error,) if stage.endswith("unlink") else ()
            assert error.secondary == secondary
            if stage.endswith("unlink"):
                assert error.temporary.parent == tmp_path and error.temporary != path
                assert error.temporary.read_bytes() == text.encode()[:accepted]
            else:
                assert error.temporary is None
    if stage.endswith("unlink"):
        (remaining if raw_control else error.temporary).unlink()
    assert list(tmp_path.iterdir()) == [path]


def test_valid_long_destination_replacement(tmp_path):
    plan = expectation_plan()
    text, receipt, _ = emit(plan.construction)
    path = tmp_path / ("a" * 250)
    try:
        path.write_bytes(b"previous")
    except OSError as error:
        pytest.skip(f"filesystem cannot create the long destination witness: {error}")
    write_qasm3_file(plan.construction, path, budget=WRITE)
    assert path.read_bytes() == text.encode()
    assert list(tmp_path.iterdir()) == [path]
