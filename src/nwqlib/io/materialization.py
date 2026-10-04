"""Import unchanged QASM writer output after checking its construction's work limits."""

from dataclasses import dataclass
import os
from pathlib import Path

from nwqlib.core.records import Record
from nwqlib.operators.access import Count
from .streaming import QasmWriteBudget, QasmWriteReceipt, _Prepared


class QasmMaterializationBudget(Record):
    """Limits for importing an OpenQASM 3 file back into Qiskit with `materialize_qasm3_file`.

    Build it with keyword arguments, for example
    `QasmMaterializationBudget(max_bytes=4096, max_qubits=1, max_clbits=0, max_dynamic_visits=100, max_operations=100)`,
    and pass it as `budget=`. The first five fields are required. The counts are
    derived from the matching construction and checked before the file is
    opened. Use the unchanged file from that writer call. No limit here
    estimates the importer's memory.

    Attributes:
        max_bytes: Required. Limit on the bytes of the file text.
        max_qubits: Required. Limit on the total declared qubits.
        max_clbits: Required. Limit on the total declared classical bits.
        max_dynamic_visits: Required. Limit on Program node visits after
            expanding loops, empty and identity loops included.
        max_operations: Required. Limit on primitive gate, measurement and reset
            applications after expanding loops. Gates that Qiskit synthesizes for
            controls are not counted.
        required_max_native_bytes: Default `None`, the only accepted value. Any
            other value is rejected before import, because this importer cannot
            bound its memory.
        required_max_peak_rss: Default `None`, the only accepted value, for the
            same reason.
    """

    max_bytes: Count
    max_qubits: Count
    max_clbits: Count
    max_dynamic_visits: Count
    max_operations: Count
    required_max_native_bytes: Count | None = None
    required_max_peak_rss: Count | None = None


@dataclass(frozen=True)
class QasmMaterialization:
    """A Qiskit circuit imported from an OpenQASM 3 file, with its loops unrolled.

    [`materialize_qasm3_file`][nwqlib.io.materialization.materialize_qasm3_file]
    returns it. The fields below are read-only. Gates are not decomposed and
    nothing is simulated.

    Attributes:
        circuit: The imported Qiskit circuit without loops. Gate definitions from
            the file stay as logical gates.
        source: The writer’s [`QasmWriteReceipt`][nwqlib.io.streaming.QasmWriteReceipt].
        output_nodes: Top-level instructions of `circuit` after unrolling. It can
            differ from `source.expanded_operations`.
        native_bytes: Always `None`, because the allocation size is not known.
        peak_rss: Always `None`, because peak process memory is not known.
    """

    circuit: object
    source: QasmWriteReceipt
    output_nodes: int
    native_bytes: None = None
    peak_rss: None = None


def materialize_qasm3_file(construction, path, receipt: QasmWriteReceipt, *,
                           writer_budget: QasmWriteBudget,
                           budget: QasmMaterializationBudget) -> QasmMaterialization:
    """Import an OpenQASM 3 file written by `write_qasm3_file` into Qiskit within explicit limits.

    Use the unchanged file, construction and receipt from the same writer
    call. The sizes are derived again from the construction and checked against
    `budget` before the file is opened. Its actual byte size is checked before
    reading. The text then goes to Qiskit’s OpenQASM 3
    importer and the `UnrollForLoops` pass. Gate definitions stay logical. Z
    with more than two controls in total is rejected before import, because
    the installed importer can synthesize such gates eagerly. It needs the
    `qasm` extra
    (`pip install "nwqlib[qasm]"`). The checked combination is OpenQASM parser
    1.0.1, qiskit-qasm3-import 0.6.0 and Qiskit 2.5.2.

    Args:
        construction (SelectedConstruction): The construction the file was
            written from.
        path (str | Path): The file.
        receipt (QasmWriteReceipt): The record that `write_qasm3_file` returned.
        writer_budget (QasmWriteBudget): The budget the file was written with.
        budget (QasmMaterializationBudget): The import limits.

    Returns:
        materialization (QasmMaterialization): The imported circuit and its writer’s `QasmWriteReceipt`.

    Raises:
        ValueError: If the construction exceeds `budget`, `budget` asks for a
            memory bound, a Z gate has more than two controls, or the actual
            file exceeds the byte limit.
    """
    prepared = _Prepared(construction, writer_budget)
    if budget.required_max_native_bytes is not None or budget.required_max_peak_rss is not None:
        raise ValueError("consumer cannot establish a hard native-byte or peak-RSS bound")
    if (sum(prepared.widths.values()) > budget.max_qubits
            or sum(prepared.cwidths.values()) > budget.max_clbits
            or prepared.visits > budget.max_dynamic_visits
            or prepared.operations > budget.max_operations):
        raise ValueError("QASM dynamic materialization exceeds consumer budget")
    for record in prepared.selected.values():
        if any(operation.gate == "mc_z" and len(operation.qubits) - 1 + int(record.controlled) > 2
               for operation in record.decomposition):
            raise ValueError("consumer capability: Z with more than two controls requires unadmitted synthesis")
    with Path(path).open("rb") as source:
        size = os.fstat(source.fileno()).st_size
        if size > budget.max_bytes:
            raise ValueError("QASM stored parsing text exceeds consumer budget")
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
