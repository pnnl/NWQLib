"""Export of a supplied Qiskit circuit to OpenQASM, with limits on its size."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from nwqlib._limits import exceeds, memory_limit, optional_limit

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class _TextSink:
    """A text stream that counts UTF-8 bytes and refuses a write that would pass ``maximum``, None for no limit.

    ``source`` names the limit in the refusal, ``max_text_bytes`` or the physical memory.
    """

    def __init__(self, stream, maximum, source):
        self.stream, self.maximum, self.source, self.bytes = stream, maximum, source, 0

    def write(self, text):
        """Write ``text`` if its UTF-8 size still fits, before it reaches the stream."""
        size = len(text.encode("utf-8"))
        if exceeds(self.bytes + size, self.maximum):
            raise ValueError(f"QASM output exceeds {self.source}")
        written = self.stream.write(text)
        self.bytes += size
        return written


def export_qasm(
    circuit: QuantumCircuit,
    *,
    format: str = "qasm3",
    path: str | Path | None = None,
    max_text_bytes: int | None = None,
    max_operations: int | None = None,
) -> str | Path:
    """Return the OpenQASM 3 or 2 text of a Qiskit circuit, or write it to a file.

    Use it for any built Qiskit circuit, for example one from
    `prepared.circuit(0)` or [`lower_qiskit`][nwqlib.blocks.lowering.lower_qiskit].
    Without `path` it returns the text. With `path` it writes a temporary file in
    the same directory and replaces the destination only after the export
    succeeds, so a failed export leaves an earlier file unchanged. It does not
    transpile or simulate. Qiskit's OpenQASM 3 exporter streams the text but still
    builds a complete syntax tree, and its OpenQASM 2 exporter builds the whole
    string first, so for OpenQASM 2 `max_text_bytes` limits only the returned or
    written text. Neither limit bounds expanded gate definitions or Qiskit's
    workspace.

    Args:
        circuit (qiskit.QuantumCircuit): The circuit to export.
        format (str): `"qasm3"` or `"qasm2"`.
        path (str | Path | None): A file path whose parent directory exists, or
            `None` to return the text.
        max_text_bytes (int | None): Positive limit on the UTF-8 bytes of the
            exported text. Default `None`: the physical memory of this
            computer for returned text, and no limit for a file.
        max_operations (int | None): Default `None`, no limit. Positive limit
            on the top-level instructions of `circuit`, checked before export.

    Returns:
        output (str | Path): The OpenQASM text, or the destination path when `path` is
            given.

    Raises:
        ValueError: If `format` is not `"qasm2"` or `"qasm3"`, a limit is neither
            a positive integer nor `None`, the circuit has more than
            `max_operations` instructions, or the text exceeds
            `max_text_bytes` (or, for returned text without that limit, the
            physical memory).
        TypeError: If `circuit` is not a Qiskit `QuantumCircuit`.

    Examples:
        >>> from qiskit import QuantumCircuit
        >>> from nwqlib.io import export_qasm
        >>> bell = QuantumCircuit(2)
        >>> _ = bell.h(0)
        >>> _ = bell.cx(0, 1)
        >>> print(export_qasm(bell))
        OPENQASM 3.0;
        include "stdgates.inc";
        qubit[2] q;
        h q[0];
        cx q[0], q[1];
    """
    if format not in {"qasm2", "qasm3"}:
        raise ValueError('format must be either "qasm2" or "qasm3"')
    maximum = optional_limit(max_text_bytes, "max_text_bytes")
    source = f"max_text_bytes={maximum}"
    if maximum is None and path is None:
        # Without max_text_bytes, the returned string, which stays in memory, is
        # limited to the physical memory. A written file has no output limit.
        # Neither limit covers Qiskit's own workspace (the QASM 2 note below).
        maximum = memory_limit(None)
        source = f"the physical memory ({maximum} bytes)"
    operation_limit = optional_limit(max_operations, "max_operations")
    from qiskit import QuantumCircuit, qasm2, qasm3

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("QASM export requires a Qiskit circuit")
    if exceeds(len(circuit.data), operation_limit):
        raise ValueError("QASM native input exceeds max_operations")

    def emit(stream):
        """Export the circuit through the byte-counting sink in the selected format."""
        sink = _TextSink(stream, maximum, source)
        if format == "qasm3":
            qasm3.dump(circuit, sink)
        else:
            # qasm2.dump builds the whole text with dumps as well, so the QASM 2 path
            # does not stream. The whole text exists in memory before the sink
            # checks the output limit.
            sink.write(qasm2.dumps(circuit))

    if path is None:
        stream = StringIO()
        emit(stream)
        return stream.getvalue()
    destination = Path(path)
    temporary = None
    try:
        with NamedTemporaryFile(mode="w", encoding="utf-8", newline="", dir=destination.parent,
                                prefix=f".{destination.name}.", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            emit(stream)
        temporary.replace(destination)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return destination
