"""Explicit source-bound QASM import and loop unrolling, never execution."""

from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path

from nwqlib.core.records import Record
from nwqlib.operators.access import Count
from .streaming import QasmWriteBudget, QasmWriteReceipt, _Prepared, _write


class QasmMaterializationBudget(Record):
    """Finite parser text and source-expansion capacity; no native RSS estimate.

    max_bytes bounds stored ASCII source bytes. max_dynamic_visits also caps
    identity/empty loops. max_operations counts expanded primitive applications,
    not synthesized controlled gates. Required hard RSS/native-byte limits are
    unavailable and reject before import.
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
    """Actual loop-free logical import, without gate decomposition or simulation.

    Attributes:
        circuit: Imported/unrolled Qiskit circuit; user gates remain logical.
        source: Verified completed artifact receipt.
        output_nodes: Actual top-level circuit instruction count after unrolling.
        native_bytes: Unknown allocation size.
        peak_rss: Unknown peak process memory.
    """

    circuit: object
    source: QasmWriteReceipt
    output_nodes: int
    native_bytes: None = None
    peak_rss: None = None


def materialize_qasm3_file(construction, path, receipt: QasmWriteReceipt, *,
                           writer_budget: QasmWriteBudget,
                           budget: QasmMaterializationBudget) -> QasmMaterialization:
    """Admit source work, verify exact bytes, then import and UnrollForLoops.

    Re-emission only hashes bounded text; it does not reconstruct native gates.
    Caller-supplied receipts/counts never supply trusted capacity. The same bytes
    read once and verified are passed to loads, preventing path substitution.
    User definitions remain logical; controls of Z with at most two controls
    use the importer's standard gates. Higher arity is rejected before import
    because the installed importer can eagerly synthesize generic controls. The installed
    qiskit-qasm3-import dependency must support this documented language subset.
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
    if receipt.bytes_written > budget.max_bytes:
        raise ValueError("QASM stored parsing text exceeds consumer budget")

    class DigestSink:
        """Accept every chunk and keep nothing, so ``_write`` recomputes the receipt without storing text."""

        def write(self, chunk):
            return len(chunk)

    prepared.budget = writer_budget.revise(max_bytes=min(writer_budget.max_bytes, budget.max_bytes))
    expected = _write(prepared, DigestSink(), None, "file")
    if expected != receipt:
        raise ValueError("receipt does not match source-derived completed file artifact")
    with Path(path).open("rb") as source:
        data = source.read(expected.bytes_written + 1)
    if len(data) != expected.bytes_written or "sha256:" + sha256(data).hexdigest() != expected.digest:
        raise ValueError("QASM file bytes differ from source-bound artifact")
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
