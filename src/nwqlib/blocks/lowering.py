"""Construction of the Qiskit circuit of a selected Program.

Only the supported Program nodes are built, after a check of the memory of the circuit's instructions.
"""

from dataclasses import dataclass
from typing import NamedTuple

from nwqlib._limits import ResourceLimitError, check_memory, known_sum, optional_limit
from nwqlib.ir import Allocate, BlockCall, CoherentRegion, Measure, MeasurementBatch, Release, Repeat, Reset, Sequence
from nwqlib.ir.validation import _Admission
from .records import SelectedConstruction
from .selection import NATIVE_POSITION_BYTES, SelectedBlock, direct_preparation


class _ConstructorContext:
    """The context argument that circuit construction passes to every constructor of a Qiskit gate.

    Calling it with a factory returns the method context of the Run, or of
    one standalone circuit construction or explicit sample.

    A constructor that appends instructions by copying a definition that
    already exists, beyond the size rule of its selection, reports them
    with ``report_copies`` before it copies. They join this construction's
    ``LogicalCircuit.construction_work``. ``owner`` is an opaque token of
    this construction, which a method context may keep to recognize a copy
    made by it. In a Run, the ``commit`` callback of each report runs once
    the preparation record that holds this construction is saved, and the
    ``withdraw`` callback instead once the preparation ends without a record:
    a refusal during lowering or backend preparation, a refused record, or
    a remote preparation that fails or is cancelled. A failed preparation also withdraws
    the gates it built for blocks with arguments, so a retry builds and
    reports them again, while the shared cache of static definitions keeps
    its gates. Only a constructor of a block with arguments may therefore
    report copies. A standalone construction calls ``commit`` at once.
    """

    __slots__ = ("_get", "owner", "copied_work", "_commits")

    def __init__(self, get, commits=None):
        self._get = get
        self.owner = object()
        self.copied_work = 0
        self._commits = commits

    def __call__(self, factory, **options):
        return self._get(factory, **options)

    def report_copies(self, count, commit, withdraw):
        """Add ``count`` copied instructions to this construction's work and register ``commit`` and ``withdraw``."""
        self.copied_work += count
        if self._commits is None:
            commit()
        else:
            self._commits.append((commit, withdraw))


def _local_method_context():
    """One empty context, created on first use, for a standalone circuit construction or explicit sample."""
    context = None

    def get(factory, *, marks_changes=False):
        nonlocal context
        if context is None:
            context = factory()
        return context

    return _ConstructorContext(get)


@dataclass(frozen=True)
class LogicalCircuit:
    """A Qiskit circuit built from a selected construction, with its register and measurement layouts, before compilation.

    [`lower_qiskit`][nwqlib.blocks.lowering.lower_qiskit] returns it. The fields
    below are read-only. Bit and qubit indices are little-endian, index 0 being
    the least significant.

    Attributes:
        construction_id: Content hash of the construction the circuit was built from.
        circuit: The Qiskit circuit. It is not compiled or run.
        quantum_layout: `(register, qubit indices)` pairs in declaration order.
        measurement_layout: `(classical value, bit indices)` pairs.
        dynamic_visits: Program node visits counted while building the circuit,
            not a gate count.
        construction_work: Sum of the size rules of the selected definitions,
            plus the instructions that constructors appended by copying
            existing definitions while building this circuit
            (``_ConstructorContext.report_copies``, at present the
            definitions of multi-controlled RY gates that the ADAPT
            ``"vacuum_restore"`` adjoints copy). It is not measured CPU work
            or a limit. `None` when a selected definition reached by the
            circuit has unknown construction work, such as the controlled
            query of a supplied gate whose definition was not built at
            planning.
        defined_selections: Content hashes of the gate definitions used, base
            definitions included. Several calls with different arguments can share
            one definition.
    """

    construction_id: str
    circuit: object
    quantum_layout: tuple
    measurement_layout: tuple
    dynamic_visits: int
    construction_work: int | None
    defined_selections: tuple[str, ...]


class _NodeCount(NamedTuple):
    """The counts of one Program node that ``_count_definition`` makes, Repeat multiplicity included.

    ``instructions`` are those that emission appends. ``lazy`` and ``eager``
    are the direct preparation tree bytes of its untransformed calls, built
    when a backend translates them, and of their controls and adjoints,
    built by lowering. ``first`` is the node of its first emitted
    instruction, or None. ``native`` adds the estimated bytes that its calls
    and measured bits leave in a flattened native circuit, ``unknown`` counts
    its calls without such an estimate and ``unknown_first`` names the block
    of the first of them.
    """

    visits: int
    work: int | None
    instructions: int
    lazy: int
    eager: int
    first: str | None
    native: int = 0
    unknown: int = 0
    unknown_first: str | None = None


@dataclass(frozen=True)
class _Count:
    """The counting pass of one definition of a selected Program, from which ``_lower_qiskit`` emits the circuit.

    ``nodes`` are the Program nodes by name. ``memo`` maps each reached
    node to its ``_NodeCount``, ``selected`` maps each reached
    BlockCall node to its ``(block, arguments)`` and ``repeat_counts`` each
    reached Repeat to its count.
    """

    start: str
    nodes: dict
    widths: dict
    cwidths: dict
    memo: dict
    selected: dict
    repeat_counts: dict

    @property
    def instructions(self):
        """Instructions that the circuit emits."""
        return self.memo[self.start].instructions

    @property
    def native_bytes(self):
        """Estimated bytes of the calls and measured bits with a memory estimate, in a flattened native circuit."""
        return self.memo[self.start].native

    @property
    def native_unknown(self):
        """Calls that have no memory estimate, Repeat counts included."""
        return self.memo[self.start].unknown

    @property
    def first_unknown(self):
        """Name of the block of the first call without a memory estimate, or None."""
        return self.memo[self.start].unknown_first

    @property
    def lazy_bytes(self):
        """Tree bytes of the untransformed direct preparation calls, which a backend builds when it translates them."""
        return self.memo[self.start].lazy

    @property
    def eager_bytes(self):
        """Tree bytes of the controlled and inverted direct preparation calls, which lowering builds."""
        return self.memo[self.start].eager

    def lazy_bytes_before(self, stop):
        """Lazy tree bytes among the first ``stop`` instructions that the circuit emits.

        A node that lies wholly before the boundary adds its subtotal from
        ``memo``, and only the node that holds the boundary is entered, so
        a Repeat costs one division and nothing is unrolled.
        """
        def prefix(name, stop):
            counted = self.memo[name]
            if stop <= 0 or not counted.lazy:
                return 0
            if stop >= counted.instructions:
                return counted.lazy
            node = self.nodes[name]
            if isinstance(node, Sequence):
                total = 0
                for child in node.children:
                    size = self.memo[child].instructions
                    if stop < size:
                        return total + prefix(child, stop)
                    total, stop = total + self.memo[child].lazy, stop - size
                return total
            if isinstance(node, Repeat):
                body = self.memo[node.body]
                whole, rest = divmod(stop, body.instructions)
                return whole * body.lazy + prefix(node.body, rest)
            return prefix(node.body, stop)

        return prefix(self.start, stop)

    @property
    def opening_direct(self):
        """Qubits of the untransformed direct preparation that the circuit emits as its first instruction, or None."""
        first = self.memo[self.start].first
        if first is None or first not in self.selected:
            return None
        declared = direct_preparation(self.selected[first][0])
        if declared is None:
            return None
        qubits, transformed = declared
        return None if transformed else qubits


# Bytes per appended instruction of the memory check, registered in
# ENGINEERING_CONSTANTS.md ("Selected block lowering limits").
INSTRUCTION_BYTES = 256


def lower_qiskit(construction: SelectedConstruction, *, blocks: tuple[SelectedBlock, ...],
                 max_bytes=None) -> LogicalCircuit:
    """Build the Qiskit circuit of one experiment of a selected construction, after checking its memory.

    The construction must be one static experiment, a Program that
    describes exactly one circuit, for example from
    [`Program.select_experiment`][nwqlib.ir.records.Program.select_experiment].
    It supports sequences, bound repeats, block calls, coherent regions,
    allocation, release, computational measurement and reset. Reallocation and a
    register lifetime split across jobs are not supported, and there is no
    OpenQASM fallback. A counting pass first checks every reachable binding
    and supported node and counts the repeats arithmetically. It also counts
    the instructions that the circuit will hold, one for each block call and
    one for each register qubit of a measurement or reset, and checks 256
    bytes for each against the memory limit, so an oversized or unbound
    construction is rejected before any Qiskit register or gate is built. It reads the same Program and selected definitions
    as the resource estimate, so the circuit and the estimate describe one
    construction. Each block call is built by the `SelectedBlock` whose record
    is the selection the Program names. A final measurement batch gives one
    circuit, whatever its repetitions or observation kind, and running it is the
    Run's task.

    Args:
        construction (SelectedConstruction): The Program and its selected
            definitions.
        blocks (tuple[SelectedBlock, ...]): The bound blocks of the reachable
            selections, matched to them by their exact records.
        max_bytes (int | None): Default `None`, the physical memory of this
            computer. Positive limit on the memory of the circuit's
            instructions, counted as 256 bytes each before any Qiskit gate is built,
            and of the exact syntheses of dense unitaries that controlling a
            transformed block makes, checked before they start.

    Returns:
        circuit (LogicalCircuit): The circuit with its register and measurement layouts.

    Raises:
        ValueError: If `max_bytes` is invalid or exceeded, the root is not one static
            experiment, a block is missing or does not match its selection, or a
            node is outside the supported subset.
    """
    return _lower_qiskit(construction, blocks=blocks, max_bytes=max_bytes)


def _definition_closure(selected):
    """Return ``{(content hash, arguments): block}`` of the definitions that the calls in ``selected`` reach.

    ``selected`` holds ``(block, arguments)`` pairs. The key is what lowering
    caches each built gate under: a static gate by its content hash (arguments
    are empty) and a gate with parameters per argument values. A control or
    adjoint adds its base, which lowering builds first.
    """
    closure, pending = {}, list(selected)
    while pending:
        block, arguments = pending.pop()
        key = block.record.content_id, arguments
        if key not in closure:
            closure[key] = block
            if block._base is not None:
                pending.append((block._base, arguments))
    return closure


def _count_definition(construction, *, blocks, admission=None, definition=None):
    """Check the reachable bindings of one definition of a selected Program and count it, building nothing.

    ``definition`` names the Program definition, the root by default. Each
    reached node gets a ``_NodeCount``.
    visits is the number of dynamic Program node visits, with every
    enclosing Repeat multiplying its body's count. work adds each visited
    BlockCall's selected construction_work and each Measure or Reset
    register width, multiplied the same way, and is None once a visited
    definition's work is unknown. instructions counts what emission
    appends, one per BlockCall and one per register qubit of a Measure or
    Reset, multiplied the same way. lazy adds ``direct_tree_bytes`` of each
    call of a block that declares an untransformed direct state
    preparation, whose tree the backend builds when it translates that
    occurrence, and eager the same for its controls and adjoints, whose
    trees lowering builds, both multiplied the same way. first is the node of the first
    instruction that the node emits, or None. native adds the
    ``native_bytes`` of each call and ``NATIVE_POSITION_BYTES`` for each
    register qubit of a Measure or Reset, multiplied the same way, and
    unknown counts the calls whose block has no such estimate. A direct
    preparation adds nothing to either, because its tree bytes above are the
    estimate of what the backend keeps for it. Everything is computed from
    the cached body counts, so a huge Repeat costs one multiplication and
    nothing is unrolled.
    """
    program = construction.program
    if admission is None:
        admission = _Admission(program)
        admission.admitted().require_ready()
    binding = admission.binding_map(program.bindings)
    root = admission.nodes[program.root]
    if isinstance(root, MeasurementBatch):
        if len(root.settings) != 1 or root.axes:
            raise ValueError("select one static experiment with Program.select_experiment")
        binding.update(admission.binding_map(root.settings[0].bindings))
    context = admission.expressions(binding)
    native = {block.record.content_id: block for block in blocks}
    if len(native) != len(blocks) or not native.keys() <= {item.content_id for item in construction.selections}:
        raise ValueError("native selections must match exact persisted selected identities")
    records = {record.signature.name: record for record in construction.selections}
    selected = {}
    widths = {item.name: admission.integer(item.width, context, "register width") for item in program.registers}
    cwidths = {item.name: admission.integer(item.width, context, "classical width")
               for item in program.classical if item.dtype == "bits"}
    nodes = admission.nodes
    memo = {}
    repeat_counts = {}

    def count(name):
        """Return the counts of one node, checking its reachable bindings (see ``_count_definition``)."""
        if name in memo:
            return memo[name]
        node = nodes[name]
        visits, work, instructions, lazy, eager, first = 1, 0, 0, 0, 0, None
        kept, unknown, unknown_first = 0, 0, None
        if isinstance(node, Sequence):
            for name_of_child in node.children:
                child = count(name_of_child)
                visits += child.visits
                work = known_sum(work, child.work)
                instructions += child.instructions
                lazy += child.lazy
                eager += child.eager
                first = child.first if first is None else first
                kept += child.native
                unknown += child.unknown
                unknown_first = child.unknown_first if unknown_first is None else unknown_first
        elif isinstance(node, (Repeat, CoherentRegion, MeasurementBatch)):
            if isinstance(node, MeasurementBatch) and name != program.root:
                raise ValueError("nested experiment batches are outside the logical subset")
            multiplicity = admission.integer(node.count, context, "Repeat") if isinstance(node, Repeat) else 1
            if isinstance(node, Repeat):
                repeat_counts[name] = multiplicity
            body = count(node.body) if multiplicity else _NodeCount(0, 0, 0, 0, 0, None)
            visits += multiplicity * body.visits
            work = known_sum(work, None if body.work is None else multiplicity * body.work)
            instructions += multiplicity * body.instructions
            lazy += multiplicity * body.lazy
            eager += multiplicity * body.eager
            first = body.first
            kept += multiplicity * body.native
            unknown += multiplicity * body.unknown
            unknown_first = body.unknown_first
        elif isinstance(node, BlockCall):
            record = records[node.signature]
            block = native.get(record.content_id)
            # A legal definition with unknown construction work is built, and that work stays unknown.
            if block is None or block.record != record or record.blocker:
                raise ValueError(record.blocker or "selected definition has no exact persisted native binding")
            admission.tick(2 * len(node.arguments))
            supplied = {argument.parameter: argument.value.expression for argument in node.arguments}
            arguments = tuple((parameter.name, context[supplied[parameter.name]])
                              for parameter in record.signature.parameters)
            if arguments and block._base is not None:
                raise ValueError("parameterized control/adjoint requires its compact native selected implementation")
            selected[name] = block, arguments
            work, instructions, first = record.construction_work, 1, name
            declared = direct_preparation(block)
            if declared is not None:
                from nwqlib._optional import optional_import
                optional_import("qiskit", extra="qiskit")
                from nwqlib.subroutines.state_preparation.direct import direct_tree_bytes
                qubits, transformed = declared
                if transformed:
                    eager = direct_tree_bytes(qubits)
                else:
                    lazy = direct_tree_bytes(qubits)
            elif block._native_bytes is None:
                unknown, unknown_first = 1, record.signature.name
            else:
                kept = block._native_bytes
        elif isinstance(node, (Measure, Reset)):
            work = instructions = widths[node.wire]
            first = name if instructions else None
            kept = NATIVE_POSITION_BYTES * instructions
        elif not isinstance(node, (Allocate, Release)):
            raise ValueError(f"unsupported logical node: {node.kind}; select one static experiment")
        memo[name] = _NodeCount(visits, work, instructions, lazy, eager, first, kept, unknown, unknown_first)
        return memo[name]

    start = program.root if definition is None else definition
    if start not in nodes:
        raise ValueError(f"definition {start!r} is not in the selected Program")
    count(start)
    return _Count(start, nodes, widths, cwidths, memo, selected, repeat_counts)


def _lower_qiskit(construction, *, blocks, definition_cache=None, specialization_cache=None,
                  method_context=None, record_synthesis=None, definition=None, templates=None,
                  receipt_commits=None, max_bytes=None, admission=None, counted=None):
    """Build the LogicalCircuit of a selected Program, optionally reusing a live Run's definition caches and method context.

    ``definition`` names the Program definition to lower, the Program's root by
    default. Another definition, such as the coherent tail or inverse of a
    trajectory readout view, is built onto the same registers, with the
    same bindings, selected blocks, caches and limits, so its gates act on
    the wires its ports name in the bound body.

    ``templates`` maps the content hash of a selected block to a
    parameterized circuit (`RowTemplate.circuit`) that replaces every call
    of that block, so the result is the Program's circuit with that
    block's angles left as parameters. The Run's template preparation
    passes it; other callers lower bound circuits.

    The context getter creates empty method metadata. Only the checked
    constructors of Qiskit gates may fill it, under the work and workspace
    rules of their selections. A standalone circuit construction shares one
    context during this call, then releases it.
    Constructors must not keep the getter in payloads, contexts or gates.
    A controlled transformed block checks the memory of the exact syntheses
    of its base and of Qiskit's control of them against ``max_bytes`` before
    they start. A Run passes its ``_record_synthesis`` as
    ``record_synthesis``, which receives their work.
    It also passes a list as ``receipt_commits``, which collects the
    ``(commit, withdraw)`` callbacks of copies that constructors report,
    for the Run to call when the preparation record is saved or when the
    preparation ends without one. The Run passes no ``max_bytes``, so its
    lowerings are checked against the physical memory here, after its own
    check against the Method's limit (``_admit_circuit_memory``). A Run's preparation
    also passes ``admission``, the ready ``_Admission`` of this
    construction's Program that its readout check made, so the Program is
    not checked again here, and ``counted``, the ``_count_definition`` of
    this definition that its direct preparation check made, so it is not
    counted again.
    """
    max_bytes = optional_limit(max_bytes, "max_bytes")
    start = construction.program.root if definition is None else definition
    if counted is None:
        counted = _count_definition(construction, blocks=blocks, admission=admission, definition=definition)
    elif counted.start != start:
        raise ValueError("the counting pass belongs to another definition")
    nodes, widths, cwidths, memo, selected, repeat_counts = (counted.nodes, counted.widths, counted.cwidths,
                                                             counted.memo, counted.selected, counted.repeat_counts)
    visits, work, instructions = memo[start].visits, memo[start].work, memo[start].instructions
    # Reserve only reachable native definitions, including base gates built once
    # for selected transforms. A transformed law may conservatively include base
    # work too. This is a reservation, never an observed CPU or instruction count.
    closure = _definition_closure(selected.values())
    if any(block._base is None and not callable(block._constructor) for block in closure.values()):
        raise ValueError("selected leaf has no bound native constructor")
    unique_work = known_sum(*(block.record.construction_work for block in closure.values()))
    # Detect unsupported repeated lifetimes compactly, without unrolling a Repeat.
    allocation_memo = {}

    def allocations(name):
        """Return how often each wire is allocated during the dynamic run of one node.

        Repeat counts multiply their body's allocations and a zero count
        contributes none. A total above one for any wire means reallocation.
        """
        if name in allocation_memo:
            return allocation_memo[name]
        node = nodes[name]
        result = {}
        if isinstance(node, Allocate):
            result[node.wire] = 1
        elif isinstance(node, (Sequence, Repeat, CoherentRegion, MeasurementBatch)):
            children = node.children if isinstance(node, Sequence) else (node.body,)
            factor = repeat_counts[name] if isinstance(node, Repeat) else 1
            for child in children if factor else ():
                for wire, number in allocations(child).items():
                    result[wire] = result.get(wire, 0) + factor * number
        allocation_memo[name] = result
        return result

    if any(number > 1 for number in allocations(start).values()):
        raise ValueError("reallocation requires another logical lifetime implementation")
    # The circuit's instruction list dominates the memory that emission adds,
    # since append(copy=False) shares each gate. The 256 bytes per
    # instruction are registered in ENGINEERING_CONSTANTS.md ("Selected block
    # lowering limits"). Gate definitions are counted by their constructors.
    try:
        check_memory(INSTRUCTION_BYTES * instructions, max_bytes, f"logical circuit lowering of {instructions} instructions")
    except ResourceLimitError as error:
        raise ResourceLimitError(f"{error}. Raise max_bytes or the physical memory, or reduce the repetitions") from None

    from nwqlib._optional import optional_import
    optional_import("qiskit", extra="qiskit")
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate

    qregs = {name: QuantumRegister(width, name) for name, width in widths.items() if width}
    cregs = {name: ClassicalRegister(width, name) for name, width in cwidths.items() if width}
    circuit = QuantumCircuit(*qregs.values(), *cregs.values())
    gates, controlled_gates = ({}, {}) if definition_cache is None else definition_cache
    previous, current = ({}, {}) if specialization_cache is None else specialization_cache
    defined = {}
    measured_results = set()
    if method_context is None:
        method_context = _local_method_context()
    else:
        method_context = _ConstructorContext(method_context, receipt_commits)

    def admit_control(base, name):
        """Check, before they start, the memory of the exact syntheses that controlling ``base`` makes and of Qiskit's control of them, and record their work.

        ``qiskit_compat.controlled`` synthesizes each distinct dense unitary
        of the built base once and Qiskit controls every occurrence of
        the synthesized gates, so ``_dense_synthesis.admit_dense_syntheses``
        sizes them by ``dense_synthesis_size`` of each width and
        ``gatewise_control_size`` of the counts, both from one walk
        (``qiskit_compat.dense_gatewise_charges``), and checks them against
        ``max_bytes``. Their work goes to ``record_synthesis``.
        """
        from nwqlib.subroutines._dense_synthesis import admit_dense_syntheses
        from nwqlib.subroutines.qiskit_compat import dense_gatewise_charges
        widths, counts, _ = dense_gatewise_charges(base, 1)
        _, work = admit_dense_syntheses(widths, controls=(counts,), max_work=None, max_bytes=max_bytes,
                                        operation=f"exact synthesis for the controlled transform {name!r}")
        if work and record_synthesis is not None:
            record_synthesis(work)

    def gate(block, arguments=()):
        """Return the Qiskit gate of one selected block at one argument tuple, building it once.

        A static definition is cached by the content hash of its selection in
        ``gates``, which a live Run shares across preparations. A
        parameterized one is cached per ``(content hash, arguments)`` in
        ``current``, reusing an entry of ``previous``, the gates for fixed
        argument values that the Run kept from its preceding preparation. A
        control or adjoint is derived from the built base gate, so the base's
        global phase is kept. A controlled base is built once and shared by
        the controlled and controlled-adjoint forms. Every content hash used,
        including a base reached through a cache hit, is recorded in
        ``defined``.
        """
        identity = block.record.content_id
        if templates and identity in templates:
            defined[identity] = None
            return templates[identity].to_gate(label=block.record.signature.name)
        key = (identity, arguments) if arguments else identity
        cache = current if arguments else gates
        if arguments and key not in cache and key in previous:
            cache[key] = previous[key]
        if key in cache:
            if block._base is not None:
                gate(block._base)
            defined[identity] = None
            return cache[key]
        record = block.record
        if block._base is not None:
            result = gate(block._base)
            if record.controlled:
                base_key = block._base.record.content_id
                if base_key not in controlled_gates:
                    admit_control(result, record.signature.name)
                    controlled_gates[base_key] = controlled(result, 1)
                result = controlled_gates[base_key]
            if record.adjoint:
                result = inverse_realized_gate(result)
        else:
            built = block._constructor(block, arguments, method_context)
            result = built.to_gate(label=record.signature.name)
        cache[key] = result
        defined[identity] = None
        return result

    realized = {}
    emitted_empty = set()

    def emit(name):
        """Append one node's instructions to the circuit in Program order.

        Repeat bodies are appended count times, a total the memory check has
        already bounded. A node that appends no instruction, as the counting
        pass recorded, is processed once however often the Program reaches
        it, by a Repeat or by Sequences that share it, which keeps the
        measured-result names it records. BlockCall ports map to whole
        registers in signature order, and a zero-width register contributes
        no qubits. A BlockCall node's gate is looked up once and kept in
        ``realized``, so a Repeat body appends the same gate without looking
        up its selection again.
        """
        empty = not memo[name].instructions
        if empty:
            if name in emitted_empty:
                return
            emitted_empty.add(name)
        node = nodes[name]
        if isinstance(node, Sequence):
            for child in node.children:
                emit(child)
        elif isinstance(node, Repeat):
            for _ in range(min(repeat_counts[name], 1) if empty else repeat_counts[name]):
                emit(node.body)
        elif isinstance(node, (CoherentRegion, MeasurementBatch)):
            emit(node.body)
        elif isinstance(node, BlockCall):
            if name not in realized:
                realized[name] = gate(*selected[name])
            wires = [qubit for port in node.ports for qubit in qregs.get(port.wire, ())]
            circuit.append(realized[name], wires, copy=False)
        elif isinstance(node, Measure):
            circuit.measure(qregs[node.wire], cregs[node.result])
            measured_results.add(node.result)
        elif isinstance(node, Reset):
            circuit.reset(qregs.get(node.wire, ()))

    emit(start)
    # construction_work is the per-visit sum from the counting pass plus the
    # construction law of each distinct reachable (definition, arguments) pair,
    # transformed bases included, plus the copies its constructors reported.
    return LogicalCircuit(
        construction.content_id, circuit,
        tuple((name, tuple(circuit.find_bit(q).index for q in reg)) for name, reg in qregs.items()),
        tuple((name, tuple(circuit.find_bit(c).index for c in reg)) for name, reg in cregs.items() if name in measured_results),
        visits, known_sum(work, unique_work, method_context.copied_work), tuple(defined),
    )
