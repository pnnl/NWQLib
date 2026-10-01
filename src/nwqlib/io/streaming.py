"""SDK-free, bounded OpenQASM 3.0 emission of selected primitive recipes."""

from dataclasses import dataclass
from hashlib import sha256
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
    """Finite text/metadata envelope, independent of dynamic loop multiplicity.

    max_metadata_bytes bounds a conservative JSON envelope for the selected
    graph and final receipt, checked separately before either is published.
    max_walk_steps bounds each kept-graph walk and static accounting/emission
    traversal; it does not count represented Repeat multiplicity.
    max_instructions counts textual applications, including gate definitions.
    max_chunk_bytes bounds each offered binary buffer, not a blocking sink's latency.
    """

    max_metadata_bytes: Count
    max_walk_steps: Count
    max_qubits: Count
    max_clbits: Count
    max_bytes: Count
    max_instructions: Count
    max_chunk_bytes: PositiveInt


class QasmWriteReceipt(Record):
    """Completed template artifact; no execution or observation is implied.

    quantum_layout/classical_layout map original names to generated ASCII names
    and widths in declaration order. selected_definitions binds reachable recipes.
    expanded_operations counts primitives and per-bit measure/reset applications;
    dynamic_visits includes empty calls and loop bodies. Neither counts shots,
    synthesis or native allocations. bytes_written and emitted_instructions are
    actual accepted text counts. Batch fields describe external template context.
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
    digest: str
    completion: Literal["stream", "file"]
    native_bytes: None = None
    peak_rss: None = None


@dataclass(frozen=True)
class QasmPrefix:
    """Exact bytes accepted before a failed write, never a completed artifact.

    Attributes:
        bytes_written: Accepted byte count.
        emitted_instructions: Fully accepted application statements.
        digest: SHA256 of precisely the accepted prefix.
    """

    bytes_written: int
    emitted_instructions: int
    digest: str


class QasmWriteError(RuntimeError):
    """Emission failed; prefix exposes progress and __cause__ keeps the failure.

    secondary keeps finalization failures in occurrence order. temporary names
    the still-owned file when cleanup fails; otherwise it is None.
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
    Allocate/Release without reallocation, and the four registered primitive
    recipes without parameters), folds dynamic work through Repeat counts without
    expanding them, and counts static emission with each shared definition once.
    It also bounds the final receipt. The emission pass (``fragments``) then
    streams text whose instruction and walk counts are already known to fit. The
    byte cap is enforced while streaming, so exceeding it raises ``QasmWriteError``
    with the accepted prefix. An admission rejection here leaves no output, and
    the materializer can rebuild the same counts from the construction alone.
    """

    def __init__(self, construction, budget):
        """Admit the supported QASM subset and count bounded static output before exposing any
        bytes.
        """
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
            """Fold repeated dynamic work and static emission separately without expanding the
            repeated body.
            """
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
                if node.basis != "computational":
                    raise ValueError("only computational measurement is supported")
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
        _metadata(self.receipt(budget.max_bytes, budget.max_instructions, "sha256:" + "0" * 64, "stream"), budget)

    def receipt(self, size, instructions, digest, completion):
        """The receipt for ``size`` accepted bytes and ``instructions`` statements with their digest."""
        return QasmWriteReceipt(**self.receipt_fields, bytes_written=size, emitted_instructions=instructions,
                                digest=digest, completion=completion)

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
        """Stream admitted definitions and calls, marking instruction boundaries for byte/count
        accounting.
        """
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
    until the chunk is complete, and the digest covers exactly the accepted
    bytes. Cancellation is checked before each fragment and each retry. Any error
    becomes ``QasmWriteError`` carrying the accepted prefix, so a caller can tell
    how much of the file exists without a completed receipt.
    """
    budget = prepared.budget
    digest = sha256()
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
                    digest.update(offered[:accepted])
                    size += accepted
                    offset += accepted
            instructions += int(instruction)
        return prepared.receipt(size, instructions, "sha256:" + digest.hexdigest(), completion)
    except Exception as exc:
        raise QasmWriteError(str(exc), QasmPrefix(size, instructions, "sha256:" + digest.hexdigest())) from exc


def write_qasm3(construction, sink, *, budget: QasmWriteBudget, cancel=None) -> QasmWriteReceipt:
    """Write synchronously to a binary sink; partial sinks expose accepted prefixes.

    Cancellation is cooperative between writes; a blocking write cannot be
    preempted. Admission errors precede output; emission errors carry QasmPrefix.
    """
    return _write(_Prepared(construction, budget), sink, cancel, "stream")


def write_qasm3_file(construction, path, *, budget: QasmWriteBudget, cancel=None) -> QasmWriteReceipt:
    """Replace a destination atomically after complete text and receipt validation.

    Cleanup of the same-directory temporary is attempted on failure. If removal
    fails, QasmWriteError exposes the remaining temporary and secondary errors.
    Control-flow exceptions keep their type, with cleanup failures in notes.
    No fsync/power-loss promise.
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
                prefix = (QasmPrefix(receipt.bytes_written, receipt.emitted_instructions, receipt.digest) if receipt
                          else QasmPrefix(0, 0, "sha256:" + sha256().hexdigest()))
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
