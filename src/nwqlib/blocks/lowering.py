"""Bounded explicit Qiskit consumer of the selected shared Program subset."""

from dataclasses import dataclass

from nwqlib._limits import DEFAULT_MAX_DIRECT_AMPLITUDES
from nwqlib.ir import Allocate, BlockCall, CoherentRegion, Measure, MeasurementBatch, Release, Repeat, Reset, Sequence
from nwqlib.ir.validation import _Admission
from .records import SelectedConstruction
from .selection import SelectedBlock, direct_preparation_amplitudes


class _ConstructorContext:
    """The context argument that lowering passes to every native constructor.

    Calling it with a factory returns the method context of the Run, or of
    one standalone lowering or explicit sample. ``max_direct_amplitudes`` is
    the limit against which lowering admitted every declared direct state
    preparation (``selection.direct_preparation_amplitudes``), and the
    constructors that call ``prepare_qiskit`` pass it on.
    """

    __slots__ = ("_get", "max_direct_amplitudes")

    def __init__(self, get, max_direct_amplitudes):
        self._get = get
        self.max_direct_amplitudes = max_direct_amplitudes

    def __call__(self, factory, **options):
        return self._get(factory, **options)


def _local_method_context(max_direct_amplitudes=DEFAULT_MAX_DIRECT_AMPLITUDES):
    """One lazy empty context for a standalone lowering or explicit sample."""
    context = None

    def get(factory, *, marks_changes=False):
        nonlocal context
        if context is None:
            context = factory()
        return context

    return _ConstructorContext(get, max_direct_amplitudes)


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
        construction_work: Sum of the size rules of the selected definitions, not
            measured work or a limit.
        defined_selections: Content hashes of the gate definitions used, base
            definitions included. Several calls with different arguments can share
            one definition.
    """

    construction_id: str
    circuit: object
    quantum_layout: tuple
    measurement_layout: tuple
    dynamic_visits: int
    construction_work: int
    defined_selections: tuple[str, ...]


# The default limits are finite ceilings on one explicit lowering, registered
# in ENGINEERING_CONSTANTS.md ("Selected block lowering limits"). Revisit them
# for a concrete selected construction that needs more.
def lower_qiskit(construction: SelectedConstruction, *, blocks: tuple[SelectedBlock, ...],
                 max_operations=100_000, max_qubits=4096, max_clbits=4096,
                 max_direct_amplitudes=DEFAULT_MAX_DIRECT_AMPLITUDES,
                 max_synthesis_work=1_000_000_000) -> LogicalCircuit:
    """Build the Qiskit circuit of one experiment of a selected construction, after checking its size.

    The construction must be one static experiment, a Program that
    describes exactly one circuit, for example from
    [`Program.select_experiment`][nwqlib.ir.records.Program.select_experiment].
    It supports sequences, bound repeats, block calls, coherent regions,
    allocation, release, computational measurement and reset. Reallocation and a
    register lifetime split across jobs are not supported, and there is no
    OpenQASM fallback. A counting pass first checks every reachable binding,
    width, direct-preparation size and supported node and counts the repeats
    arithmetically, so an oversized or unbound construction is rejected before
    Qiskit is imported. It reads the same Program and selected definitions
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
        max_operations (int): Positive, inclusive limit on the
            Program node visits, repeat counts included.
        max_qubits (int): Limit on the total declared qubits.
        max_clbits (int): Limit on the total declared classical
            bits.
        max_direct_amplitudes (int): Default `65536`. Positive limit on the
            amplitudes of each reachable direct state preparation, whose synthesis
            work grows as `q * 2**q`.
        max_synthesis_work (int): Positive limit on the
            total work of the exact dense-unitary syntheses, and Qiskit's control
            of them, that the controlled blocks of this circuit make, in the work
            units of the dense synthesis size rule
            ([Circuit-free synthesis laws](../ENGINEERING_CONSTANTS.md#circuit-free-synthesis-laws)).
            Each synthesis is counted before it starts.

    Returns:
        circuit (LogicalCircuit): The circuit with its register and measurement layouts.

    Raises:
        ValueError: If a limit is invalid or exceeded, the root is not one static
            experiment, a block is missing or does not match its selection, or a
            node is outside the supported subset.
    """
    if type(max_synthesis_work) is not int or max_synthesis_work < 1:
        raise ValueError("lowering max_synthesis_work must be a positive integer")
    spent = 0

    def charge(work, operation):
        nonlocal spent
        if spent + work > max_synthesis_work:
            raise ValueError(f"{operation} needs {work} work units, and this lowering has already charged "
                             f"{spent} of max_synthesis_work={max_synthesis_work}")
        spent += work

    return _lower_qiskit(construction, blocks=blocks, max_operations=max_operations, max_qubits=max_qubits,
                         max_clbits=max_clbits, max_direct_amplitudes=max_direct_amplitudes,
                         synthesis_charge=charge)


def _lower_qiskit(construction, *, blocks, max_operations=100_000, max_qubits=4096, max_clbits=4096,
                  max_direct_amplitudes=DEFAULT_MAX_DIRECT_AMPLITUDES,
                  definition_cache=None, specialization_cache=None,
                  method_context=None, synthesis_charge=None, definition=None):
    """Build the LogicalCircuit of a selected Program, optionally reusing a live Run's definition caches and method context.

    ``definition`` names the Program definition to lower, the Program's root by
    default. Another definition, such as the coherent tail or inverse of a
    trajectory readout view, is lowered onto the same registers, with the
    same bindings, selected blocks, caches and limits, so its gates act on
    the wires its ports name in the bound body.

    The context getter creates empty method metadata. Only admitted native
    constructors may populate it, under their selected work/workspace laws.
    Standalone lowering shares one context during this call, then releases it.
    Constructors must not keep the getter in payloads, contexts or gates.
    A Run passes its ``ExecutionLimits.max_direct_amplitudes``, and its
    ``_charge_synthesis`` as ``synthesis_charge``, which a controlled
    transformed block calls before its base is synthesized and controlled.
    """
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("lowering max_operations must be a positive integer")
    if any(type(value) is not int or value < 0 for value in (max_qubits, max_clbits)):
        raise ValueError("lowering width limits must be nonnegative integers")
    if type(max_direct_amplitudes) is not int or max_direct_amplitudes < 1:
        raise ValueError("lowering max_direct_amplitudes must be a positive integer")
    program = construction.program
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
    if sum(widths.values()) > max_qubits or sum(cwidths.values()) > max_clbits:
        raise ValueError("circuit width exceeds the selected lowering limit")
    nodes = admission.nodes
    memo = {}
    repeat_counts = {}

    def count(name):
        """Admit the reachable bindings of one node and return its ``(visits, work)``.

        visits is the number of dynamic IR visits, with every enclosing
        Repeat multiplying its body's count. work adds each visited
        BlockCall's selected construction_work and each Measure or Reset
        register width, multiplied the same way. Both are computed from the
        memoized body counts, so a huge Repeat costs one multiplication and
        rejects against max_operations before Qiskit is imported.
        """
        if name in memo:
            return memo[name]
        node = nodes[name]
        visits, work = 1, 0
        if isinstance(node, Sequence):
            for child in node.children:
                v, w = count(child)
                visits += v
                work += w
        elif isinstance(node, (Repeat, CoherentRegion, MeasurementBatch)):
            if isinstance(node, MeasurementBatch) and name != program.root:
                raise ValueError("nested experiment batches are outside the logical subset")
            multiplicity = admission.integer(node.count, context, "Repeat") if isinstance(node, Repeat) else 1
            if isinstance(node, Repeat):
                repeat_counts[name] = multiplicity
            v, w = count(node.body) if multiplicity else (0, 0)
            visits += multiplicity * v
            work += multiplicity * w
        elif isinstance(node, BlockCall):
            record = records[node.signature]
            block = native.get(record.content_id)
            if block is None or block.record != record or record.blocker or record.construction_work is None:
                raise ValueError(record.blocker or "selected definition has no exact persisted native binding/work law")
            amplitudes = direct_preparation_amplitudes(block)
            if amplitudes > max_direct_amplitudes:
                raise ValueError(f"selected direct preparation has {amplitudes} amplitudes, "
                                 f"exceeding max_direct_amplitudes={max_direct_amplitudes}")
            admission.tick(2 * len(node.arguments))
            supplied = {argument.parameter: argument.value.expression for argument in node.arguments}
            arguments = tuple((parameter.name, context[supplied[parameter.name]])
                              for parameter in record.signature.parameters)
            if arguments and block._base is not None:
                raise ValueError("parameterized control/adjoint requires its compact native selected implementation")
            selected[name] = block, arguments
            work = record.construction_work
        elif isinstance(node, Measure):
            work = widths[node.wire]
        elif isinstance(node, Reset):
            work = widths[node.wire]
        elif not isinstance(node, (Allocate, Release)):
            raise ValueError(f"unsupported logical node: {node.kind}; select one static experiment")
        if visits > max_operations:
            raise ValueError("dynamic Program visits exceed max_operations")
        memo[name] = visits, work
        return visits, work

    start = program.root if definition is None else definition
    if start not in nodes:
        raise ValueError(f"definition {start!r} is not in the selected Program")
    visits, work = count(start)
    # Reserve only reachable native definitions, including base gates built once
    # for selected transforms. A transformed law may conservatively include base
    # work too. This is a reservation, never an observed CPU or instruction count.
    closure = {}
    pending = list(selected.values())
    while pending:
        block, arguments = pending.pop()
        key = block.record.content_id, arguments
        if key in closure:
            continue
        if block._base is None and not callable(block._constructor):
            raise ValueError("selected leaf has no bound native constructor")
        closure[key] = block
        if block._base is not None:
            pending.append((block._base, arguments))
    unique_work = sum(block.record.construction_work for block in closure.values())
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
        method_context = _local_method_context(max_direct_amplitudes)
    else:
        method_context = _ConstructorContext(method_context, max_direct_amplitudes)

    def charge_control(base, name):
        """Charge the exact syntheses that controlling ``base`` makes, and Qiskit's control of them.

        ``qiskit_compat.controlled`` synthesizes each distinct dense unitary
        of the realized base once and Qiskit controls every occurrence of
        the synthesized gates, so the work is ``dense_synthesis_size`` of
        each width that ``dense_synthesis_widths`` lists plus
        ``gatewise_control_size`` of the ``dense_control_counts`` triple.
        """
        from nwqlib.subroutines._dense_synthesis import dense_synthesis_size, gatewise_control_size
        from nwqlib.subroutines.qiskit_compat import dense_control_counts, dense_synthesis_widths
        work = sum(dense_synthesis_size(width)[0] for width in dense_synthesis_widths(base))
        counts = dense_control_counts(base, 1)
        if counts[0]:
            work += gatewise_control_size(*counts)[0]
        if work:
            synthesis_charge(work, f"controlled transform {name!r}")

    def gate(block, arguments=()):
        """Return the native gate of one selected block at one argument tuple, building it once.

        A static definition is cached by its selected identity in ``gates``,
        which a live Run shares across preparations. A parameterized one is
        cached per ``(identity, arguments)`` in ``current``, reusing an entry
        of ``previous``, the specializations the Run kept from its preceding
        preparation. A control or adjoint is derived from the realized base
        gate, so the base's global phase is kept. A controlled base is built
        once and shared by the controlled and controlled-adjoint forms. Every
        identity used, including a base reached through a cache hit, is
        recorded in ``defined``.
        """
        identity = block.record.content_id
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
                    if synthesis_charge is not None:
                        charge_control(result, record.signature.name)
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

    def emit(name):
        """Append one node's instructions to the circuit in Program order.

        Repeat bodies are appended count times, a total the counting pass has
        already bounded. BlockCall ports map to whole registers in signature
        order, and a zero-width register contributes no qubits.
        A BlockCall node's gate is looked up once and kept in ``realized``,
        so a Repeat body appends the same gate without reading its selected
        identity again.
        """
        node = nodes[name]
        if isinstance(node, Sequence):
            for child in node.children:
                emit(child)
        elif isinstance(node, Repeat):
            for _ in range(repeat_counts[name]):
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
    # transformed bases included.
    return LogicalCircuit(
        construction.content_id, circuit,
        tuple((name, tuple(circuit.find_bit(q).index for q in reg)) for name, reg in qregs.items()),
        tuple((name, tuple(circuit.find_bit(c).index for c in reg)) for name, reg in cregs.items() if name in measured_results),
        visits, work + unique_work, tuple(defined),
    )
