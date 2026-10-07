"""SDK-free, bounded OpenQASM 3.0 emission of selected primitive recipes."""

from dataclasses import dataclass
import os
from pathlib import Path
import tempfile
from typing import Literal

from nwqlib.core.records import PositiveInt, Record
from nwqlib.ir import Allocate, BlockCall, CoherentRegion, Measure, MeasurementBatch, Release, Repeat, Reset, Sequence
from nwqlib.ir.expressions import walk_kept
from nwqlib.ir.validation import _Admission
from nwqlib.operators.access import Count


class QasmWriteBudget(Record):
    """Limits for writing a Plan's circuit (`plan.construction`) as OpenQASM 3 text with `write_qasm3` or `write_qasm3_file`.

    Build it with keyword arguments, for example
    `QasmWriteBudget(max_metadata_bytes=1_000_000, max_walk_steps=100_000, max_qubits=1, max_clbits=0, max_bytes=4096, max_instructions=100, max_chunk_bytes=256)`,
    and pass it as `budget=`. Every field is required. The graph and metadata
    checks run before any text is written. The limits do not grow with the count
    of a `Repeat`, which the text writes as one loop.

    Attributes:
        max_metadata_bytes: Required. Limit in bytes on a conservative JSON size
            of the [Program](../glossary.md#program) graph and of the final
            `QasmWriteReceipt`, each checked before it is used.
        max_walk_steps: Required. Limit on the steps of each walk over the graph
            and of the counting and writing passes. A `Repeat` count is not
            multiplied in.
        max_qubits: Required. Limit on the total declared qubits.
        max_clbits: Required. Limit on the total declared classical bits.
        max_bytes: Required. Limit on the bytes of text written. It is checked
            while the text is written, so exceeding it raises `QasmWriteError`
            after part of the text has been delivered.
        max_instructions: Required. Limit on the gate, measurement and reset
            statements in the text, gate definitions included.
        max_chunk_bytes: Required. Positive. Limit in bytes on each buffer offered
            to the sink. It does not bound how long a blocking sink takes.
    """

    max_metadata_bytes: Count
    max_walk_steps: Count
    max_qubits: Count
    max_clbits: Count
    max_bytes: Count
    max_instructions: Count
    max_chunk_bytes: PositiveInt


class QasmWriteReceipt(Record):
    """The record of a completed OpenQASM 3 file or stream: what was written, its layouts and counts.

    `write_qasm3` and `write_qasm3_file` return it, and
    [`materialize_qasm3_file`][nwqlib.io.materialization.materialize_qasm3_file]
    imports its written file. The fields below are read-only. The text describes
    one circuit template. Writing it runs nothing and collects no observation.
    The counts are syntactic. They count no shots, gate synthesis or SDK
    memory, and they are not hardware estimates.

    Attributes:
        construction_id: Content hash (`content_id`) of the written
            construction. `materialize_qasm3_file` refuses a receipt whose
            `construction_id` is not the `content_id` of the construction
            passed with it.
        subset: `"nwqlib.direct-qasm3.v1"`, the supported subset of OpenQASM 3.0
            with `stdgates.inc`.
        angle_convention: `"binary64-repr"`, meaning that each angle is the
            shortest decimal that rounds back to the same binary64 value.
        selected_definitions: Content hashes of the block definitions that the
            written circuit uses.
        quantum_layout: `(original name, generated name, width)` of each quantum
            register, in declaration order. Qubit index 0 is the least
            significant bit.
        classical_layout: The same for each classical register.
        batch_repetitions: Repetitions of the measurement batch, kept here and
            not written as a loop, or `None`.
        observation_kind: Observation kind of the batch, or `None`.
        setting_label: Label of the one written setting, or `None`.
        bytes_written: Bytes of text written, punctuation included.
        emitted_instructions: Gate, measurement and reset statements written,
            counting a definition body once for each time it appears in the text.
        expanded_operations: Primitive gates plus one per measured or reset bit,
            after expanding loops and calls.
        dynamic_visits: Visits to [Program](../glossary.md#program) nodes,
            empty calls and loop bodies included.
        completion: `"stream"` from `write_qasm3` or `"file"` from
            `write_qasm3_file`.
    """

    construction_id: str
    subset: Literal["nwqlib.direct-qasm3.v1"] = "nwqlib.direct-qasm3.v1"
    angle_convention: Literal["binary64-repr"] = "binary64-repr"
    selected_definitions: tuple[str, ...]
    quantum_layout: tuple[tuple[str, str, Count], ...]
    classical_layout: tuple[tuple[str, str, Count], ...]
    batch_repetitions: Count | None
    observation_kind: str | None
    setting_label: str | None
    bytes_written: Count
    emitted_instructions: Count
    expanded_operations: Count
    dynamic_visits: Count
    completion: Literal["stream", "file"]


@dataclass(frozen=True)
class QasmPrefix:
    """The text that a failed OpenQASM 3 write had already delivered, never a completed file.

    Read it from `QasmWriteError.prefix`. The fields below are read-only.

    Attributes:
        bytes_written: Bytes the sink accepted.
        emitted_instructions: Statements the sink accepted completely.
    """

    bytes_written: int
    emitted_instructions: int


class QasmWriteError(RuntimeError):
    """Raised when writing OpenQASM 3 text fails after the limit checks passed.

    `prefix` is the [`QasmPrefix`][nwqlib.io.streaming.QasmPrefix] of the text
    already delivered, and `__cause__` is the original failure. `secondary`
    holds the failures of closing or cleaning up, in the order they occurred.
    `temporary` names the temporary file of `write_qasm3_file` that could not be
    removed, or is `None`.
    """

    def __init__(self, message, prefix, *, secondary=(), temporary=None):
        super().__init__(message)
        self.prefix = prefix
        self.secondary = secondary
        self.temporary = temporary


def _metadata(value, budget):
    """Bound a conservative JSON envelope of ``value`` before any text is produced.

    Each kept slot costs one walk step and 8 bytes of punctuation. Strings cost
    six bytes per character (worst-case JSON escaping), integers their bit length
    plus one (a decimal digit count never exceeds the bit length) and floats 32
    bytes. Both limits are checked during the walk, so an oversized graph stops
    early.
    """
    slots = payload = 0

    def reserve(amount):
        """Charge ``amount`` kept slots, one walk step and 8 bytes each, before walk_kept visits them."""
        nonlocal slots, payload
        slots += amount
        payload += 8 * amount  # punctuation and scalar overhead, no RSS claim
        if slots > budget.max_walk_steps or payload > budget.max_metadata_bytes:
            raise ValueError("QASM metadata traversal exceeds max_walk_steps or max_metadata_bytes")

    for item in walk_kept(value, reserve):
        if isinstance(item, str):
            payload += 6 * len(item)  # worst-case JSON character escaping
        elif type(item) is int:
            payload += item.bit_length() + 1  # conservative decimal envelope
        elif type(item) is float:
            payload += 32
        elif isinstance(item, Record):
            payload += sum(6 * len(field) for field in type(item).model_fields)
        if slots > budget.max_walk_steps or payload > budget.max_metadata_bytes:
            raise ValueError("QASM metadata traversal exceeds max_walk_steps or max_metadata_bytes")


class _Prepared:
    """One admitted construction, counted completely before the first byte is written.

    Writing has two passes over the same kept graph. The counting pass here checks
    the supported subset (one root batch setting without axes, bit registers,
    Repeat, CoherentRegion, Sequence, computational Measure and Reset,
    Allocate/Release without reallocation, and the five registered primitive
    recipes without parameters, ``preparation.native``, ``preparation.hzh``,
    ``reflection.positive_zero``, ``pauli.parity`` and ``pauli.group_basis``),
    folds dynamic work through Repeat counts without
    expanding them, and counts static emission with each shared definition once.
    It also bounds the final receipt. The emission pass (``fragments``) then
    streams text whose instruction and walk counts are already known to fit. The
    byte cap is enforced while streaming, so exceeding it raises ``QasmWriteError``
    with the accepted prefix. An admission rejection here leaves no output, and
    the materializer can rebuild the same counts from the construction alone.
    """

    def __init__(self, construction, budget):
        """Admit the supported QASM subset and count its static output before any byte is exposed."""
        _metadata(construction, budget)
        self.construction = construction
        self.budget = budget
        program = construction.program
        admission = _Admission(program)
        admission.check().require_ready()
        binding = admission.binding_map(program.bindings)
        root = admission.nodes[program.root]
        self.batch = root if isinstance(root, MeasurementBatch) else None
        if self.batch:
            if len(root.settings) != 1 or root.axes:
                raise ValueError("select one root batch setting without axes")
            binding.update(admission.binding_map(root.settings[0].bindings))
        context = admission.expressions(binding)
        self.repetitions = (admission.integer(root.repetitions, context, "batch repetitions")
                            if self.batch and root.repetitions is not None else None)
        self.widths = {reg.name: admission.integer(reg.width, context, "qubit width") for reg in program.registers}
        if any(value.dtype != "bits" for value in program.classical):
            raise ValueError("QASM writer supports only classical bits")
        self.cwidths = {reg.name: admission.integer(reg.width, context, "bit width") for reg in program.classical}
        if sum(self.widths.values()) > budget.max_qubits or sum(self.cwidths.values()) > budget.max_clbits:
            raise ValueError("QASM width exceeds budget")
        self.qnames = {name: f"q{i}" for i, name in enumerate(self.widths)}
        self.cnames = {name: f"c{i}" for i, name in enumerate(self.cwidths)}
        records = {item.signature.name: item for item in construction.selections}
        self.selected = {}
        self.counts = {}
        self.nodes = admission.nodes
        memo = {}
        work = 0

        def multiply(a, b):
            """Return ``a * b`` for nonnegative counts, rejecting growth past the Program's integer limit.

            For positive a and b the product has at least
            ``bit_length(a) + bit_length(b) - 1`` bits, so a product that fails
            this test is too large before it is formed. ``admission.bounded``
            then checks the actual product.
            """
            if a and b and a.bit_length() + b.bit_length() - 1 > program.limits.max_integer_bits:
                raise ValueError("QASM expansion exceeds integer growth envelope")
            return admission.bounded(a * b)

        def count(name):
            """Count dynamic work and static emission of node ``name`` without expanding a repeated body."""
            nonlocal work
            if name in memo:
                return memo[name]
            node = self.nodes[name]
            visits, operations, statements, static_visits = 1, 0, 0, 1
            allocations = {}
            children = ()
            factor = 1
            if isinstance(node, Sequence):
                children = node.children
            elif isinstance(node, (Repeat, CoherentRegion, MeasurementBatch)):
                if isinstance(node, MeasurementBatch) and name != program.root:
                    raise ValueError("nested batches are unsupported by QASM writer")
                if isinstance(node, Repeat):
                    factor = admission.integer(node.count, context, "Repeat")
                    # The emitted loop variable is int[64], so the count must fit
                    # a signed 64-bit integer.
                    if factor > 2**63 - 1:
                        raise ValueError("Repeat exceeds signed int[64] subset")
                    self.counts[name] = factor
                children = (node.body,) if factor else ()
            elif isinstance(node, BlockCall):
                record = records[node.signature]
                if node.arguments or record.signature.parameters:
                    raise ValueError("parameterized selected definitions are unsupported")
                if ((record.implementation.name, record.implementation.version) not in {
                    ("preparation.native", "1"), ("preparation.hzh", "1"), ("reflection.positive_zero", "1"), ("pauli.parity", "1"),
                    ("pauli.group_basis", "1")
                } or record.decomposition is None or record.blocker):
                    raise ValueError("unsupported opaque/unregistered selected recipe")
                self.selected[node.signature] = record
                operations = len(record.decomposition)
                width = sum(admission.integer(port.width, context, "port width") for port in record.signature.quantum)
                statements = 1 if width else operations
                static_visits += width
            elif isinstance(node, Measure):
                operations = statements = self.widths[node.wire]
            elif isinstance(node, Reset):
                operations = statements = self.widths[node.wire]
            elif isinstance(node, Allocate):
                allocations[node.wire] = 1
            elif not isinstance(node, Release):
                raise ValueError(f"unsupported QASM node: {node.kind}")
            for child in children:
                v, o, s, t, allocated = count(child)
                visits = admission.bounded(visits + multiply(factor, v))
                operations = admission.bounded(operations + multiply(factor, o))
                statements += s
                static_visits += t
                work += len(allocated) + 1
                if work > budget.max_walk_steps:
                    raise ValueError("QASM accounting exceeds max_walk_steps")
                # Allocation counts saturate at 2. Only "more than one" matters
                # (a reallocation, rejected below), so a large Repeat factor
                # never has to be multiplied out.
                for wire, number in allocated.items():
                    allocations[wire] = min(2, allocations.get(wire, 0) + min(2, factor) * number)
            if statements > budget.max_instructions or static_visits > budget.max_walk_steps:
                raise ValueError("QASM static emission exceeds max_walk_steps or max_instructions")
            memo[name] = visits, operations, statements, static_visits, allocations
            return memo[name]

        # Count a shared definition once for emission while preserving its dynamic
        # repetition cost. Reallocation is rejected before streaming starts.
        self.visits, self.operations, statements, static_visits, allocations = count(program.root)
        if any(value > 1 for value in allocations.values()):
            raise ValueError("QASM reallocation is unsupported")
        self.gnames = {name: f"g{i}" for i, name in enumerate(self.selected)}
        self.gwidths = {name: sum(admission.integer(port.width, context, "port width") for port in rec.signature.quantum)
                        for name, rec in self.selected.items()}
        statements += sum(len(rec.decomposition) for name, rec in self.selected.items() if self.gwidths[name])
        definition_work = sum(self.gwidths[name] + len(rec.decomposition) for name, rec in self.selected.items())
        if static_visits + definition_work > budget.max_walk_steps:
            raise ValueError("QASM definition/emission traversal exceeds max_walk_steps")
        if statements > budget.max_instructions:
            raise ValueError("QASM emitted instructions exceed budget")
        self.receipt_fields = dict(
            construction_id=construction.content_id,
            selected_definitions=tuple(rec.content_id for rec in self.selected.values()),
            quantum_layout=tuple((name, self.qnames[name], width) for name, width in self.widths.items()),
            classical_layout=tuple((name, self.cnames[name], width) for name, width in self.cwidths.items()),
            batch_repetitions=self.repetitions,
            observation_kind=self.batch.observation_kind if self.batch else None,
            setting_label=self.batch.settings[0].label if self.batch else None,
            expanded_operations=self.operations, dynamic_visits=self.visits,
        )
        # Validate/bound even the final receipt before exposing artifact bytes.
        _metadata(self.receipt(budget.max_bytes, budget.max_instructions, "stream"), budget)

    def receipt(self, size, instructions, completion):
        """The receipt for ``size`` accepted bytes and ``instructions`` statements."""
        return QasmWriteReceipt(**self.receipt_fields, bytes_written=size, emitted_instructions=instructions,
                                completion=completion)

    def recipe(self, record, operand):
        """Yield one selected primitive recipe as QASM statements.

        The adjoint of ``U_1 ... U_n`` is ``U_n^dagger ... U_1^dagger``, so an
        adjoint recipe runs in reverse order, ``sdg`` becomes ``s`` and a phase
        angle changes sign. The other primitives (``x``, ``h``, ``z``, ``cx`` and
        ``mc_z``) are self-adjoint. A controlled
        recipe adds operand 0 as a control of every statement, and ``mc_z`` is
        written as ``z`` controlled by its leading qubits. The phase primitive is
        written as ``gphase``, a global phase in OpenQASM 3. Under ``ctrl`` it
        becomes a relative phase on the branch where all controls are one, which
        is why a selected phase is emitted rather than dropped.
        """
        recipe = reversed(record.decomposition) if record.adjoint else record.decomposition
        for operation in recipe:
            offset = int(record.controlled)
            targets = [operand(index + offset) for index in operation.qubits]
            controls = [operand(0)] if record.controlled else []
            gate = operation.gate
            if gate == "mc_z":
                controls += targets[:-1]
                targets = targets[-1:]
                gate = "z"
            elif gate == "sdg" and record.adjoint:
                gate = "s"
            elif gate == "phase":
                angle = -operation.angle if record.adjoint else operation.angle
                gate = f"gphase({repr(float(angle))})"
            prefix = f"ctrl({len(controls)}) @ " if controls else ""
            yield prefix + gate, False
            for i, target in enumerate(controls + targets):
                yield (" " if i == 0 else ", ") + target, False
            yield ";", True
            yield "\n", False

    def fragments(self):
        """Stream the admitted definitions and calls, marking instruction boundaries for the byte and count checks."""
        yield 'OPENQASM 3.0;\ninclude "stdgates.inc";\n', False
        for name, record in self.selected.items():
            width = self.gwidths[name]
            if width:
                yield f"gate {self.gnames[name]} ", False
                for i in range(width):
                    yield (", " if i else "") + f"a{i}", False
                yield " {\n", False
                yield from self.recipe(record, lambda index: f"a{index}")
                yield "}\n", False
        for name, width in self.widths.items():
            if width:
                yield f"qubit[{width}] {self.qnames[name]};\n", False
        for name, width in self.cwidths.items():
            if width:
                yield f"bit[{width}] {self.cnames[name]};\n", False

        # Emit Repeat as a QASM loop, not as repeated text, and preserve the
        # selected ordered body and its actual wire maps.
        def emit(name, depth=0):
            """Yield the statements of node ``name`` and its children in Program order.

            A Repeat with count n > 0 becomes one ``for int[64] i<depth> in
            [0:n-1]`` loop around its body, emitted once, so the text does not
            grow with n. The loop variable is named by nesting depth, so nested
            loops never shadow each other. A call with quantum width applies its
            gate definition ``g<k>`` to the bits of its port wires in order. A
            zero-width call has no operands, so its recipe is written inline.
            Measure and Reset are written once per bit of their register.
            """
            node = self.nodes[name]
            if isinstance(node, Sequence):
                for child in node.children:
                    yield from emit(child, depth)
            elif isinstance(node, Repeat):
                count = self.counts[name]
                if count:
                    yield f"for int[64] i{depth} in [0:{count - 1}] {{\n", False
                    yield from emit(node.body, depth + 1)
                    yield "}\n", False
            elif isinstance(node, (CoherentRegion, MeasurementBatch)):
                yield from emit(node.body, depth)
            elif isinstance(node, BlockCall):
                record = self.selected[node.signature]
                if self.gwidths[node.signature]:
                    yield self.gnames[node.signature], False
                    first = True
                    for port in node.ports:
                        for i in range(self.widths[port.wire]):
                            yield (" " if first else ", ") + f"{self.qnames[port.wire]}[{i}]", False
                            first = False
                    yield ";", True
                    yield "\n", False
                else:
                    yield from self.recipe(record, None)
            elif isinstance(node, (Measure, Reset)):
                for i in range(self.widths[node.wire]):
                    operand = f"{self.qnames[node.wire]}[{i}]"
                    yield (f"{self.cnames[node.result]}[{i}] = measure {operand};" if isinstance(node, Measure)
                           else f"reset {operand};"), True
                    yield "\n", False
        yield from emit(self.construction.program.root)


def _write(prepared, sink, cancel, completion):
    """Stream the admitted text in bounded chunks to a blocking binary sink.

    A sink may accept fewer bytes than offered. The remainder is offered again
    until the chunk is complete. Cancellation is checked before each fragment and each retry. Any error
    becomes ``QasmWriteError`` carrying the accepted prefix, so a caller can tell
    how much of the file exists without a completed receipt.
    """
    budget = prepared.budget
    size = instructions = 0
    try:
        fragments = iter(prepared.fragments())
        while True:
            if cancel is not None and cancel():
                raise InterruptedError("QASM write cancelled")
            try:
                fragment, instruction = next(fragments)
            except StopIteration:
                break
            # Fragments are ASCII. No full artifact or unrolled body is kept.
            for start in range(0, len(fragment), budget.max_chunk_bytes):
                chunk = fragment[start:start + budget.max_chunk_bytes].encode("ascii")
                offset = 0
                while offset < len(chunk):
                    if cancel is not None and cancel():
                        raise InterruptedError("QASM write cancelled")
                    remaining = budget.max_bytes - size
                    if remaining <= 0:
                        raise ValueError("QASM byte cap exceeded")
                    offered = memoryview(chunk)[offset:offset + remaining]
                    accepted = sink.write(offered)
                    if type(accepted) is not int or not 0 < accepted <= len(offered):
                        raise ValueError("binary sink must return a positive accepted-byte count")
                    size += accepted
                    offset += accepted
            instructions += int(instruction)
        return prepared.receipt(size, instructions, completion)
    except Exception as exc:
        raise QasmWriteError(str(exc), QasmPrefix(size, instructions)) from exc


def write_qasm3(construction, sink, *, budget: QasmWriteBudget, cancel=None) -> QasmWriteReceipt:
    """Write a Plan's circuit (`plan.construction`) as OpenQASM 3 text to a binary stream, without any quantum SDK.

    The construction must use the writer subset of the
    [OpenQASM guide](../qasm-streaming.md#writer-subset): exact gate recipes,
    sequences, bound repeats written as loops, allocation, computational
    measurement and reset, and at most one batch setting. Other constructions,
    such as generic state preparation or Pauli SELECT blocks, are rejected before
    any text is written. The sink's `write` must return a positive integer no
    larger than the buffer offered, and a partial write is completed before more
    text is produced. Cancellation is checked between writes, so a blocking write
    cannot be interrupted.

    Args:
        construction (SelectedConstruction): The
            [Program](../glossary.md#program) and its selected definitions,
            such as `plan.construction`.
        sink (BinaryIO): Binary stream with a `write` method.
        budget (QasmWriteBudget): The limits.
        cancel (Callable[[], bool] | None): Called between writes. A true
            result stops the write and raises `QasmWriteError`.

    Returns:
        receipt (QasmWriteReceipt): The receipt, with
            `completion="stream"`.

    Raises:
        ValueError: If the construction is outside the writer subset or
            exceeds a limit of `budget` other than `max_bytes`, before any
            text is written.
        QasmWriteError: If writing fails after it started, including when the
            text exceeds `budget.max_bytes` or `cancel` returns true. After a
            cancellation its `__cause__` is an `InterruptedError`. Its
            `prefix` describes the text already delivered.
    """
    return _write(_Prepared(construction, budget), sink, cancel, "stream")


def write_qasm3_file(construction, path, *, budget: QasmWriteBudget, cancel=None) -> QasmWriteReceipt:
    """Write a Plan's circuit (`plan.construction`) as an OpenQASM 3 file, replacing the destination only when the write is complete.

    It writes the text as [`write_qasm3`][nwqlib.io.streaming.write_qasm3] does,
    into a temporary file in the destination's directory, and replaces the
    destination only after the text and its `QasmWriteReceipt` are complete. On failure or
    cancellation it removes the temporary file and leaves an existing destination
    unchanged. No `fsync` or durability across power loss is promised.

    Args:
        construction (SelectedConstruction): The
            [Program](../glossary.md#program) and its selected definitions,
            such as `plan.construction`.
        path (str | Path): Destination file. Its directory must exist.
        budget (QasmWriteBudget): The limits.
        cancel (Callable[[], bool] | None): Called between writes. A true
            result stops the write and raises `QasmWriteError`.

    Returns:
        receipt (QasmWriteReceipt): The receipt, with
            `completion="file"`.

    Raises:
        ValueError: If the construction is outside the writer subset or
            exceeds a limit of `budget` other than `max_bytes`, before any
            file is created.
        QasmWriteError: If writing fails after it started, including when the
            text exceeds `budget.max_bytes` or `cancel` returns true. The
            destination is left unchanged. Its `prefix` describes the text
            already delivered, its `temporary` names the temporary file if it
            could not be removed, and `secondary` holds later cleanup
            failures. An exception that is not an `Exception`, such as
            `KeyboardInterrupt`, keeps its type, with cleanup failures in its
            notes.
    """
    prepared = _Prepared(construction, budget)
    path = Path(path)
    temporary = None
    receipt = None
    primary = None
    secondary = []
    try:
        sink = tempfile.NamedTemporaryFile(mode="wb", dir=path.parent, prefix=".nwqlib-qasm3-", delete=False)
        temporary = Path(sink.name)
        try:
            receipt = _write(prepared, sink, cancel, "file")
            sink.flush()
        except BaseException as exc:
            primary = exc
        try:
            sink.close()
        except BaseException as exc:
            if primary is None:
                primary = exc
            else:
                secondary.append(exc)
        if primary is not None:
            raise primary
        if cancel is not None and cancel():
            raise InterruptedError("QASM write cancelled")
        os.replace(temporary, path)
        return receipt
    except BaseException as exc:
        remaining = None
        if temporary is not None:
            try:
                temporary.unlink(missing_ok=True)
            except BaseException as cleanup:
                secondary.append(cleanup)
                remaining = temporary
        if isinstance(exc, Exception):
            error = exc
            if not isinstance(error, QasmWriteError):
                prefix = (QasmPrefix(receipt.bytes_written, receipt.emitted_instructions) if receipt
                          else QasmPrefix(0, 0))
                error = QasmWriteError(str(exc), prefix)
                error.__cause__ = exc
            error.secondary += tuple(secondary)
            error.temporary = remaining
            raise error
        for failure in secondary:
            exc.add_note(f"QASM finalization also failed: {failure!r}")
        if remaining is not None:
            exc.add_note(f"QASM temporary remains owned: {remaining}")
        raise
