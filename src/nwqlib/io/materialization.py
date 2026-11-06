"""Import an unchanged file of the OpenQASM 3 writer into Qiskit after checking it against its write record."""

from dataclasses import dataclass
import os
from pathlib import Path

from nwqlib._limits import check_memory
from .streaming import QasmWriteReceipt

# Coarse memory of Qiskit's OpenQASM 3 import and UnrollForLoops: bytes per
# top-level instruction after unrolling, per declared qubit or bit, per byte of
# text, and a fixed part.
# ENGINEERING_CONSTANTS.md, "OpenQASM import estimate", records the measurement.
_INSTRUCTION_BYTES = 4096
_WIRE_BYTES = 2048
_TEXT_BYTES = 512
_FIXED_BYTES = 16 * 2**20


@dataclass(frozen=True)
class QasmMaterialization:
    """A Qiskit circuit imported from an OpenQASM 3 file, with its loops unrolled.

    [`materialize_qasm3_file`][nwqlib.io.materialization.materialize_qasm3_file]
    returns it. The fields below are read-only. Gates are not decomposed and
    nothing is simulated.

    Attributes:
        circuit: The imported Qiskit circuit without loops. Gate definitions from
            the file stay as logical gates.
        source: The [`QasmWriteReceipt`][nwqlib.io.streaming.QasmWriteReceipt]
            passed as `receipt`, whose `construction_id` names the construction.
        output_nodes: Top-level instructions of `circuit` after unrolling, at
            most `source.unrolled_instructions`. It can differ from
            `source.expanded_operations`.
    """

    circuit: object
    source: QasmWriteReceipt
    output_nodes: int


def materialize_qasm3_file(construction, path, receipt: QasmWriteReceipt, *,
                           max_bytes: int | None = None) -> QasmMaterialization:
    """Import an OpenQASM 3 file written by `write_qasm3_file` into Qiskit, after checking it against its write record.

    Use the unchanged file, construction and receipt from the same writer
    call. Before the file is opened, the receipt's `construction_id` is
    compared with `construction.content_id`, and the memory of the import is
    estimated from the receipt's counts and checked against `max_bytes`. The
    file's actual size must equal the receipt's `bytes_written`. The text
    then goes to Qiskit's OpenQASM 3 importer and the `UnrollForLoops` pass.
    Gate definitions stay logical. Z with more than two controls in total is
    rejected before import, because the installed importer can synthesize such
    gates eagerly. It needs the `qasm` extra
    (`pip install "nwqlib[qasm]"`). The checked combination is OpenQASM parser
    1.0.1, qiskit-qasm3-import 0.6.0 and Qiskit 2.6.0.

    Args:
        construction (SelectedConstruction): The construction the file was
            written from.
        path (str | Path): The file.
        receipt (QasmWriteReceipt): The record that `write_qasm3_file` returned
            for `construction`.
        max_bytes (int | None): Limit on the estimated memory of the import,
            4096 bytes per top-level instruction after unrolling
            (`receipt.unrolled_instructions`), 2048 bytes per qubit and bit of
            the receipt's layouts, 512 bytes per byte of text and 16 MiB.
            Default `None`, the physical memory.

    Returns:
        materialization (QasmMaterialization): The imported circuit and its writer's `QasmWriteReceipt`.

    Raises:
        ValueError: If `receipt.construction_id` is not `construction.content_id`,
            a Z gate has more than two controls, the estimated memory exceeds
            `max_bytes`, or the file's size differs from `receipt.bytes_written`.
    """
    if receipt.construction_id != construction.content_id:
        raise ValueError(f"receipt.construction_id {receipt.construction_id} is not "
                         f"construction.content_id {construction.content_id}. Pass as receipt "
                         "the QasmWriteReceipt that the writer returned for this construction")
    written = set(receipt.selected_definitions)
    for record in construction.selections:
        if record.content_id in written and any(
                operation.gate == "mc_z" and len(operation.qubits) - 1 + int(record.controlled) > 2
                for operation in record.decomposition):
            raise ValueError("consumer capability: Z with more than two controls requires unadmitted synthesis")
    wires = sum(width for *_, width in receipt.quantum_layout + receipt.classical_layout)
    check_memory(_INSTRUCTION_BYTES * receipt.unrolled_instructions + _WIRE_BYTES * wires
                 + _TEXT_BYTES * receipt.bytes_written + _FIXED_BYTES, max_bytes, "OpenQASM 3 import with unrolled loops")
    with Path(path).open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size != receipt.bytes_written:
            raise ValueError(f"the file has {size} bytes, but its write record has {receipt.bytes_written}. "
                             "Import the unchanged file of the writer call that returned the record")
        data = source.read(size)
    text = data.decode("ascii")
    from nwqlib._optional import optional_import
    optional_import("qiskit", extra="qasm")
    optional_import("qiskit_qasm3_import", extra="qasm")
    from qiskit.qasm3 import loads
    from qiskit.transpiler import PassManager
    from qiskit.transpiler.passes import UnrollForLoops

    circuit = loads(text)
    circuit = PassManager(UnrollForLoops()).run(circuit)
    if any(instruction.operation.name in {"for_loop", "while_loop"} for instruction in circuit.data):
        raise ValueError("consumer did not produce a loop-free artifact")
    return QasmMaterialization(circuit, receipt, len(circuit.data))
