"""Explicit bounded QASM export of a supplied native circuit."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import TYPE_CHECKING

from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib._validation import integer

if TYPE_CHECKING:
    from qiskit import QuantumCircuit


class _TextSink:
    """A text stream that counts UTF-8 bytes and refuses a write that would pass ``maximum``."""

    def __init__(self, stream, maximum):
        self.stream, self.maximum, self.bytes = stream, maximum, 0

    def write(self, text):
        """Write ``text`` if its UTF-8 size still fits, before it reaches the stream."""
        size = len(text.encode("utf-8"))
        if size > self.maximum - self.bytes:
            raise ValueError("QASM output exceeds max_text_bytes")
        written = self.stream.write(text)
        self.bytes += size
        return written


def export_qasm(
    circuit: QuantumCircuit,
    *,
    format: str = "qasm3",
    path: str | Path | None = None,
    max_text_bytes: int = DEFAULT_MAX_BYTES,
    max_operations: int = 100_000,
) -> str | Path:
    """Export text, or atomically write a file and return its Path.

    ``max_operations`` bounds the supplied top-level native instruction count
    before export, as in native resource inspection. It does not bound expanded
    gate definitions or compiler workspace. ``max_text_bytes`` bounds emitted
    UTF-8 text. QASM3 streams text but still builds its full AST; QASM2's exporter
    first builds its full string, so its text cap is a publication bound only.
    Parent directories must already exist. A failed export preserves any prior
    destination file. No transpilation or simulation is performed. The
    ``max_operations`` default of 100,000 is registered in
    docs/ENGINEERING_CONSTANTS.md ("Direct native output allowances").
    """
    if format not in {"qasm2", "qasm3"}:
        raise ValueError('format must be either "qasm2" or "qasm3"')
    maximum = integer(max_text_bytes, "max_text_bytes", 1)
    operation_limit = integer(max_operations, "max_operations", 1)
    from qiskit import QuantumCircuit, qasm2, qasm3

    if not isinstance(circuit, QuantumCircuit):
        raise TypeError("QASM export requires a Qiskit circuit")
    if len(circuit.data) > operation_limit:
        raise ValueError("QASM native input exceeds max_operations")

    def emit(stream):
        """Export the circuit through the byte-counting sink in the selected format."""
        sink = _TextSink(stream, maximum)
        if format == "qasm3":
            qasm3.dump(circuit, sink)
        else:
            # Qiskit qasm2.dump also calls dumps; no streaming claim here.
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
