"""Compact UCGate storage around QPY's builtin-constructor limitation.

With Qiskit 2.5.2, QPY writes a ``UCGate`` but cannot load it. Its loader passes
the gate's 2x2 matrices to the constructor as separate positional arguments,
while ``UCGate`` expects one list. The defect is in constructor reconstruction,
not in qubit order. This module stores each UCGate as a plain
instruction whose first parameter is a JSON header (nominal width, name, label,
``up_to_diagonal``, whether the table is already simplified, and the one-based
active controls) followed by the stored matrices. Decoding rebuilds the gate
directly, without simplifying the table again or synthesizing its matrix.
``docs/development/execution.md`` (Circuit entries and Qiskit UCGate
compatibility) is the public contract, including when this storage may change.

The container prefix selects this codec, and ordinary raw QPY is never
interpreted as this storage format. Within a container the marker is chosen
outside all original operation names, so an ordinary custom instruction cannot
collide.

Revisit condition: ``test_native_ucgate_qpy_limitation_canary`` in
``tests/test_qpy_archive.py`` loads one two-qubit UCGate through raw QPY. It
fails, asking for a compatibility review, when the installed Qiskit loads the
gate natively or fails differently. Check simplified controls, the
``up_to_diagonal`` flag and executable round trips when choosing the codec
for that SDK.
"""

from copy import copy
import json


PREFIX = b"NWQLIB-QPY-UC1\n"


class QpyStream:
    """Expose QPY-relative offsets for its indexed writer and seeking loader.

    In a container file the QPY data starts after ``PREFIX``, not at byte 0.
    QPY records positions from ``tell`` and seeks back to them. ``tell``
    therefore subtracts the position where the QPY data starts, and an absolute
    ``seek`` adds it back, so the positions QPY sees are the same as in a
    standalone QPY file. Relative seeks pass through unchanged.
    """

    def __init__(self, stream):
        self.stream, self.start = stream, stream.tell()

    def read(self, size=-1):
        return self.stream.read(size)

    def write(self, data):
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def seekable(self):
        return self.stream.seekable()

    def tell(self):
        return self.stream.tell() - self.start

    def seek(self, offset, whence=0):
        self.stream.seek(offset + self.start if whence == 0 else offset, whence)
        return self.tell()


def _graph(circuit, convert, *, writing):
    """Visit the operation graph QPY serializes, preserving shared definitions."""
    from qiskit import circuit as circuit_module
    from qiskit.circuit import AnnotatedOperation, ControlFlowOp, ControlledGate, library
    from qiskit.circuit import controlflow
    from qiskit.circuit.library import PauliEvolutionGate, PauliProductMeasurement
    from qiskit.quantum_info import Clifford

    memo, active, names = {}, set(), set()

    def visit(value, build):
        """Build each distinct object once, by identity, and reject a cyclic definition.

        An instruction or circuit shared by several parents therefore stays one
        shared object in the result, as QPY expects. ``memo`` keeps the original
        object alive, so its ``id`` cannot be reused during the traversal.
        """
        identity = id(value)
        if identity in memo:
            return memo[identity][1]
        if identity in active:
            raise ValueError("QPY storage requires acyclic instruction definitions")
        active.add(identity)
        try:
            result = build(value)
            memo[identity] = value, result
            return result
        finally:
            active.remove(identity)

    def operation(original):
        return visit(original, instruction)

    def instruction(original):
        """Convert one operation and every definition or block QPY will serialize with it.

        Control-flow blocks, annotated bases, controlled base gates and custom
        definitions are visited because QPY stores them. Library gates are left
        alone, so their definitions are never forced. An unchanged subgraph returns
        the original object, and a changed one gets a shallow copy when writing, so
        the caller's circuit is never modified.
        """
        names.add(original.name)
        converted = convert(original)
        if converted is not original:
            definition = getattr(original, "_definition", None)
            if definition is not None:
                converted._definition = block(definition)
            return converted
        if isinstance(original, ControlFlowOp):
            blocks = tuple(block(child) for child in original.blocks)
            return (original if all(a is b for a, b in zip(blocks, original.blocks, strict=True))
                    else original.replace_blocks(blocks))
        if isinstance(original, AnnotatedOperation):
            base = operation(original.base_op)
            if base is original.base_op:
                return original
            result = copy(original) if writing else original
            result.base_op = base
            return result
        if isinstance(original, (PauliEvolutionGate, Clifford, PauliProductMeasurement)):
            # QPY serializes operator/settings, tableau or Pauli data, not an
            # instruction definition. These payloads cannot contain UCGate.
            return original
        kind = getattr(original, "base_class", type(original)).__name__
        custom = kind in {"Gate", "Instruction", "ControlledGate", "MCMTGate"} or not any(
            hasattr(module, kind) for module in (library, circuit_module, controlflow))
        if not custom:
            return original  # Do not force definitions of builtin library gates.
        if writing:
            # QPY itself requests this definition for a custom operation. For
            # open controls it stores _definition, not the X-conjugated getter.
            original.definition
        definition = getattr(original, "_definition", None)
        transformed = None if definition is None else block(definition)
        base = operation(original.base_gate) if isinstance(original, ControlledGate) else None
        if transformed is definition and (base is None or base is original.base_gate):
            return original
        result = copy(original) if writing else original
        result._definition = transformed
        if base is not None:
            result.base_gate = base
        return result

    def block(original):
        return visit(original, container)

    def container(original):
        """Convert every operation of one circuit or block.

        The original circuit comes back when nothing changed. Otherwise a writer
        gets an empty copy with the converted operations, and a reader converts
        its freshly loaded circuit in place.
        """
        items = tuple(original.data)
        operations = tuple(operation(item.operation) for item in items)
        if all(op is item.operation for op, item in zip(operations, items, strict=True)):
            return original
        result = original.copy_empty_like() if writing else original
        result.data = [item.replace(operation=op) for item, op in zip(items, operations, strict=True)]
        return result

    return block(circuit), names


def encode_circuit(circuit):
    """Return a shallow storage view, or None when raw QPY already suffices."""
    from qiskit.circuit import Instruction
    from qiskit.circuit.library import UCGate

    encoded = []
    marker = "nwqlib_stored_ucgate_v1"

    def convert(gate):
        """Replace a UCGate by a storage instruction and leave every other operation.

        The storage instruction has the gate's width, no clbits and parameters
        ``[header, *matrices]``. The header keeps what the constructor would
        otherwise recompute or lose: ``simp_contr`` (whether
        the table is already simplified, and its one-based active controls),
        ``up_to_diagonal``, name and label. The storage instruction carries
        the nominal width.
        """
        if type(gate) is not UCGate:
            return gate
        simplified, controls = gate.simp_contr
        header = json.dumps(dict(name=gate.name,
            label=gate.label, up_to_diagonal=gate.up_to_diagonal,
            simplified=simplified, controls=sorted(controls)), separators=(",", ":"))
        stored = Instruction(marker, gate.num_qubits, 0, [header, *gate.params])
        encoded.append(stored)
        return stored

    result, names = _graph(circuit, convert, writing=True)
    if not encoded:
        return None
    if marker in names:
        # CircuitData captures instruction names on insertion. Choose a free
        # name before rebuilding the view, rather than mutating inserted ops.
        while marker in names:
            marker += "_"
        del result
        encoded.clear()
        result, _ = _graph(circuit, convert, writing=True)
    result.metadata = dict(marker=marker, original_metadata=circuit.metadata)
    return result


def decode_circuit(circuit):
    """Restore an explicitly identified codec container without synthesis."""
    from qiskit.circuit import Gate
    from qiskit.circuit.library import UCGate

    metadata = circuit.metadata
    marker = metadata["marker"]

    def convert(stored):
        """Rebuild one UCGate with its saved table and active controls."""
        if stored.name != marker:
            return stored
        header = json.loads(stored.params[0])
        params, controls = stored.params[1:], header["controls"]
        # The public constructor would simplify the already simplified table
        # again and infer the wrong nominal width. Base initialization invokes
        # UCGate's ndarray parameter validator without matrix synthesis.
        gate = UCGate.__new__(UCGate)
        Gate.__init__(gate, header["name"], stored.num_qubits, params, label=header["label"])
        gate.up_to_diagonal = header["up_to_diagonal"]
        gate.simp_contr = header["simplified"], set(controls)
        return gate

    result, _ = _graph(circuit, convert, writing=False)
    result.metadata = metadata["original_metadata"]
    return result
