"""Private copies of supplied Qiskit circuits for execution, with a coarse byte estimate checked against a limit as they are copied."""

from copy import copy
from functools import cache
from math import isfinite

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import (AnnotatedOperation, Barrier, CircuitInstruction, ControlledGate, ControlModifier,
                               Gate, Instruction, InverseModifier, PowerModifier)
    from qiskit.circuit.library import (DiagonalGate, Isometry, QFTGate, StatePreparation,
        UCGate, UCRXGate, UCRYGate, UCRZGate, UniformSuperpositionGate, UnitaryGate)
    from qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal import MCGupDiag
except ModuleNotFoundError as error:
    if error.name == "qiskit":
        error.add_note("Native circuit intake requires its extra: pip install 'nwqlib[qiskit]'.")
    raise

from nwqlib.backends.resources import _standard_gate_classes
from nwqlib.operators.access import _check_bytes

# Bytes estimated for each stored instruction and each qubit of a copied
# circuit, and for each annotation modifier. Lowering counts the same 256
# bytes per instruction, about twice the 109-122 bytes measured for a Qiskit
# standard-gate instruction (ENGINEERING_CONSTANTS.md, "Supplied circuit
# snapshot estimate").
SLOT_BYTES = 256

# The way past a refusal of an operation that may have a definition. Expanding it
# is the caller's step, because this copy reads stored definitions and builds none.
_DECOMPOSE_ADVICE = (". circuit.decompose() replaces it by its definition, and it can be repeated until no refused "
                     "operation remains. That definition may approximate the operation, as for a PauliEvolutionGate "
                     "of noncommuting terms, and the decomposed circuit is then what runs")


@cache
def _stock_gates():
    """Return Qiskit's standard gates by name, as the ``StandardGate`` that ``CircuitInstruction.from_standard`` takes.

    docs/dependency_issues.md, "Standard-gate instructions", records this
    interface. A name is included only when Qiskit's name mapping gives a
    gate of that same standard gate.
    """
    from qiskit._accelerate.circuit import StandardGate
    from qiskit.circuit.library import get_standard_gate_name_mapping

    mapping = get_standard_gate_name_mapping()
    return {standard.name: standard for standard in StandardGate.all_gates()
            if getattr(mapping.get(standard.name), "_standard_gate", None) is standard}


def snapshot_circuit(circuit, *, max_bytes=None):
    """Return a private copy of a supplied circuit and a coarse estimate of its bytes.

    The copy isolates the selected action from later changes the caller makes
    to its own circuit, gates or parameter arrays. One cache preserves acyclic
    sharing, so a definition or array referenced twice is copied once and
    stays shared. The set of objects being copied detects recursion before
    the cache is reused. Input-circuit metadata and extra attributes are
    omitted. Supported Qiskit operation objects keep their Qiskit class and
    scalar execution flags. Supported immutable Qiskit singletons are shared,
    as Qiskit permits, so their global internals are outside this isolation.
    Top-level operations must be gates or gate-based annotations, because the
    code that uses the copy inverts and controls it. Barriers carry no action
    and are omitted at every stored level.

    A standard gate whose parameters are bound floats is copied from the
    gate, parameters and label that the circuit stores, without reading a
    Python gate object for it, which Qiskit may keep from the caller.

    The byte estimate has three terms, and a shared object is counted once.
    Each copied circuit, the supplied one and each stored definition, costs
    ``SLOT_BYTES`` per stored instruction and per qubit. Each copied mutable
    operation other than a standard gate costs its numerical data, 16 bytes
    per scalar parameter and the bytes of each array. Each annotation
    modifier costs ``SLOT_BYTES``, and a control modifier also the bytes of
    its count-bit control state. Each term is checked against ``max_bytes``
    before the allocation it covers, so a refusal can come after part of the
    copy is built, and that part is freed with the exception.
    ENGINEERING_CONSTANTS.md, "Supplied circuit snapshot estimate", gives the
    basis of the unit. The estimate is of this copy, not of the process's
    resident memory or of synthesis work.

    Args:
        circuit: Ordinary QuantumCircuit without classical bits.
        max_bytes: Limit on the cumulative byte estimate.

    Returns:
        ``(snapshot, data_bytes)``: the copied circuit and its byte estimate.
    """
    # CircuitInstruction.operation may materialize a temporary SDK wrapper.
    # Keep each source alive with its clone so a later wrapper cannot reuse its
    # id. This applies to every stored payload, not just top-level gates.
    memo, active = {}, set()
    data_bytes = 0

    def charge(size):
        """Add ``size`` bytes to the estimate, raising before the allocation they cover."""
        nonlocal data_bytes
        _check_bytes(data_bytes + size, max_bytes, "circuit snapshot")
        data_bytes += size
    stock = _stock_gates()
    standard_classes = frozenset(_standard_gate_classes().values())
    native_classes = set(standard_classes)
    native_classes.update((DiagonalGate, QFTGate, StatePreparation, UCGate, UCRXGate,
                           UCRYGate, UCRZGate, UniformSuperpositionGate, UnitaryGate,
                           MCGupDiag))

    def array(value):
        """Copy one finite real or complex ndarray parameter once, so a shared array stays shared."""
        if type(value) is not np.ndarray or value.dtype.kind not in "fc":
            raise ValueError("native parameters require ordinary real/complex ndarrays")
        if id(value) in memo:
            return memo[id(value)][1]
        # The operation's estimate (data_size) already holds value.nbytes for
        # the kept copy. The one-byte-per-entry np.isfinite mask is freed
        # before tobytes makes that copy, and value.nbytes >= value.size for
        # every real or complex dtype, so the estimate also bounds the moment
        # when the mask is alive.
        if not np.isfinite(value).all():
            raise ValueError("native array parameters must be finite")
        result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
        memo[id(value)] = (value, result)
        return result

    def scalar(value):
        """Check one bound scalar: a finite int, float or complex, returned as a Python number."""
        if type(value) in (int, float, complex) or isinstance(value, np.number):
            value = value.item() if isinstance(value, np.number) else value
            if not (isfinite(value.real) and isfinite(value.imag)):
                raise ValueError("native scalar parameters must be finite")
            return value
        raise ValueError("native gate parameters must be bound numerical values")

    def parameter(value):
        """Check and copy one bound numerical gate parameter: an ndarray or a scalar."""
        return array(value) if type(value) is np.ndarray else scalar(value)

    def data_size(original):
        """Bytes of the numerical data that copying a mutable operation adds.

        16 per scalar parameter and the bytes of each array among its
        parameters and an Isometry's ``iso_data`` that no earlier operation
        copied. The characters of a label-form StatePreparation fall under
        the instruction unit.
        """
        values = original.params
        arrays = {id(value): value for value in values if type(value) is np.ndarray}
        if type(original) is Isometry and type(original.iso_data) is np.ndarray:
            arrays[id(original.iso_data)] = original.iso_data
        scalars = sum(type(value) not in (np.ndarray, str) for value in values)
        return 16*scalars + sum(value.nbytes for key, value in arrays.items() if key not in memo)

    def definition(original, *, top_level=False):
        """Copy one definition circuit instruction by instruction, once per shared circuit.

        Before the copy is created, the estimate adds ``SLOT_BYTES`` for each
        stored instruction, barriers included, and each qubit. The unit also
        covers the gates, labels and scalar parameters of standard gates.
        ``top_level`` marks the supplied circuit, whose operations must be
        gates.
        """
        if type(original) is not QuantumCircuit or original.num_clbits:
            raise ValueError("native definitions require ordinary circuits without classical bits")
        identity = id(original)
        if identity in active:
            raise ValueError("recursive native gate definitions are unsupported")
        if identity in memo:
            return memo[identity][1]
        charge(SLOT_BYTES * (len(original.data) + original.num_qubits))
        result = QuantumCircuit(original.num_qubits, global_phase=scalar(original.global_phase))
        memo[identity] = (original, result)
        active.add(identity)
        try:
            for instruction in original.data:
                standard = stock.get(instruction.name) if instruction.is_standard_gate() else None
                params = instruction.params
                if standard is not None and all(type(value) is float for value in params):
                    if not all(map(isfinite, params)):
                        raise ValueError("native scalar parameters must be finite")
                    qubits = tuple(result.qubits[original.find_bit(bit).index] for bit in instruction.qubits)
                    result._append(CircuitInstruction.from_standard(standard, qubits, params, instruction.label))
                    continue
                if type(instruction.operation) is Barrier:
                    continue
                gate = clone(instruction.operation, require_gate=top_level, width=len(instruction.qubits))
                result.append(gate, [original.find_bit(bit).index for bit in instruction.qubits], copy=False)
        finally:
            active.remove(identity)
        return result

    def modifier(original):
        """Copy one control, inverse or power annotation modifier after checking its data.

        A power must be a finite real scalar. A control count must be a
        nonnegative integer and its control state must fit that count. Each
        modifier adds ``SLOT_BYTES`` to the estimate, and a control modifier
        also the bytes of its count-bit control state.
        """
        if type(original) not in (ControlModifier, InverseModifier, PowerModifier):
            raise ValueError("unsupported native annotation modifier")
        if id(original) in memo:
            return memo[id(original)][1]
        if type(original) is ControlModifier:
            count, state = original.num_ctrl_qubits, original.ctrl_state
            count = count.item() if isinstance(count, np.integer) else count
            state = state.item() if isinstance(state, np.integer) else state
            if type(count) is not int or count < 0:
                raise ValueError("native annotation controls require a nonnegative integer width")
            # The SDK constructor may form 2**count. Charge the bytes of a
            # count-bit control-state integer before that allocation, even when
            # a caller has mutated the modifier's data.
            charge(SLOT_BYTES + (count+7)//8)
            if state is not None:
                if type(state) is str:
                    if len(state) != count or not state or set(state) - {"0", "1"}:
                        raise ValueError("native annotation control state must match its width")
                elif type(state) is not int or state < 0 or state.bit_length() > count:
                    raise ValueError("native annotation control state must match its width")
            result = ControlModifier(count, state)
        else:
            charge(SLOT_BYTES)
            if type(original) is PowerModifier:
                power = original.power
                if type(power) not in (int, float) and not isinstance(power, (np.integer, np.floating)):
                    raise ValueError("native annotation power must be a finite real scalar")
                result = PowerModifier(scalar(power))
            else:
                result = InverseModifier()
        memo[id(original)] = (original, result)
        return result

    def admit_operation(original, *, require_gate, width):
        """Check that a non-annotated operation is supported and has the stored width.

        Supported operations are Qiskit gates whose base class is a standard or
        listed library gate, Isometry, the library's ``DirectPreparationGate``
        and plain operations. A plain operation has type exactly Gate,
        ControlledGate or Instruction, and its already stored definition gives
        its action. A ``DirectPreparationGate`` builds its definition, the
        direct tree, when it is read here. ``require_gate`` also rejects any
        operation that is not a Gate. Returns ``(plain, stored)``, whether the
        operation is plain and its stored definition or None.

        The refusal of an unsupported operation names it. For a gate, or a
        top-level operation with a stored definition such as an appended
        subcircuit, it also points to ``decompose``, which replaces the
        operation by Qiskit's definition of it. The test reads ``_definition``
        because ``definition`` could build one. An evolution gate's exact matrix and its approximate synthesized
        decomposition can differ, as for a ``PauliEvolutionGate`` of
        noncommuting terms. A caller therefore compares this copy with the
        actual decomposed circuit for its implemented action, and separately
        compares that circuit with the desired exact propagator. Control,
        inverse and global phase remain part of the action when relevant, and
        agreement only up to global phase need not suffice for a controlled
        consumer.
        """
        from nwqlib.subroutines.state_preparation.direct import DirectPreparationGate

        if require_gate and not isinstance(original, Gate):
            advice = "" if getattr(original, "_definition", None) is None else _DECOMPOSE_ADVICE
            raise ValueError("top-level native operations must be gates or gate-based annotations, "
                             f"and {original.name!r} is neither{advice}")
        module = type(original).__module__
        sdk = (module or "").startswith("qiskit.") or (
            module is None and type(original).__name__.startswith("_Singleton"))
        # Support is checked before the immutable fast path, so custom mutable
        # or copy properties cannot bypass admission by imitating a singleton.
        plain = type(original) in (Gate, ControlledGate, Instruction)
        direct = type(original) is DirectPreparationGate
        supported = ((sdk and isinstance(original, Gate) and original.base_class in native_classes)
            or plain or direct or type(original) is Isometry)
        if not supported or original.num_clbits:
            advice = ("" if not isinstance(original, Gate) else
                      f"{_DECOMPOSE_ADVICE}. A gate without a definition stays unchanged under decompose(), and "
                      "UnitaryGate(Operator(gate)) holds its matrix")
            raise ValueError(f"unsupported native instruction {original.name!r}{advice}")
        if width is not None and original.num_qubits != width:
            raise ValueError("native gate width differs from its stored instruction")
        stored = original.definition if direct else getattr(original, "_definition", None)
        if plain and stored is None:
            raise ValueError("plain native gates require an already stored definition")
        return plain, stored

    def copied_parameters(original):
        """Return checked copies of an operation's parameters.

        A label-form StatePreparation keeps its label characters, each one of
        ``0 1 + - r l``. Amplitudes that are all Python floats or complex
        numbers, the form Qiskit stores, are checked together, with one
        finiteness check of their complex128 array. The kept values are the
        same objects. Every other parameter, including a Python int, is
        checked by ``parameter``.
        """
        values = original.params
        if isinstance(original, StatePreparation) and original._from_label:
            if any(type(value) is not str or value not in ("0", "1", "+", "-", "r", "l")
                   for value in values):
                raise ValueError("StatePreparation labels require 0, 1, +, -, r or l")
            return list(values)
        if (isinstance(original, StatePreparation) and not original._from_int
                and set(map(type, values)) <= {float, complex}):
            if not np.isfinite(np.asarray(values, dtype=np.complex128)).all():
                raise ValueError("native scalar parameters must be finite")
            return list(values)
        return [parameter(value) for value in values]

    def copy_class_data(original, result):
        """Replace the UCGate, StatePreparation and Isometry data of ``result`` by checked copies.

        None of this data refers to another operation, so nothing here
        recurses into ``clone``.
        """
        if isinstance(original, UCGate):
            simplify, controls = original.simp_contr
            if any(type(bit) is not int for bit in controls):
                raise ValueError("UCGate control selection requires integer indices")
            result.simp_contr = (simplify, set(controls))
        if isinstance(original, StatePreparation):
            # Its inverse() rebuilds the gate from this argument rather than
            # from _params, so bind that call to the admitted current amplitudes,
            # sharing the admitted list itself.
            result._params_arg = ("".join(result._params) if original._from_label else
                int(original._params_arg) if original._from_int else result._params)
        if type(original) is Isometry:
            result.iso_data = array(original.iso_data)

    def clone(original, *, require_gate=False, width=None):
        """Return the snapshot of one stored operation, copying each object at most once.

        An annotation copies its modifiers and then its gate base, whose
        width is the stored instruction width minus the added controls. Any
        other operation is checked by ``admit_operation``. An immutable one
        is shared as is. A mutable one is shallow-copied, which keeps its Qiskit
        class and scalar flags, and every mutable payload that the supported
        block-encoding families use is replaced by a checked copy without
        ``__deepcopy__``. A plain Gate has no hidden execution payload beyond
        its parameters and stored definition. Before a mutable operation
        other than a standard gate copies its data, the estimate adds its
        ``data_size``. A standard gate has at most four scalar parameters,
        which the instruction unit covers.

        The recursive calls, for an annotation base, a controlled base, an
        Isometry's cached inverse and a stored definition, stay in this
        function and ``definition``. Moving them into further helpers would
        add Python stack frames per nesting level and lower the nesting depth
        of a supplied circuit that can be copied.
        """
        identity = id(original)
        if identity in active:
            raise ValueError("recursive native gate definitions or annotations are unsupported")
        if type(original) is AnnotatedOperation:
            if identity in memo:
                if width is not None and memo[identity][1].num_qubits != width:
                    raise ValueError("native annotation width differs from its stored instruction")
                return memo[identity][1]
            if type(original.modifiers) is not list:
                raise ValueError("native annotation modifiers require an ordinary stored list")
            active.add(identity)
            try:
                # Recurse before reading SDK width/parameter properties, which
                # themselves recurse on base_op and cannot guard malformed cycles.
                modifiers = [modifier(value) for value in original.modifiers]
                controls = sum(value.num_ctrl_qubits for value in modifiers if type(value) is ControlModifier)
                if width is not None and controls > width:
                    raise ValueError("native annotation controls exceed its stored instruction width")
                base = clone(original.base_op, require_gate=True,
                             width=None if width is None else width-controls)
                result = AnnotatedOperation(base, modifiers)
                memo[identity] = (original, result)
            finally:
                active.remove(identity)
            return result
        plain, stored = admit_operation(original, require_gate=require_gate, width=width)
        if identity in memo:
            return memo[identity][1]
        if not original.mutable:
            memo[identity] = (original, original)
            return original
        if original.base_class not in standard_classes:
            charge(data_size(original))
        active.add(identity)
        try:
            params = copied_parameters(original)
            result = copy(original)
            result._params = params
            if plain and not original.name.startswith("nwqlib_custom_"):
                # SDK control and translation dispatch on primitive names. A
                # plain Gate's stored definition owns its action, even when a
                # caller names it "x", "u", or "barrier". Keep native classes
                # intact and give admitted custom definitions a distinct name.
                result.name = "nwqlib_custom_" + original.name
            if isinstance(original, ControlledGate):
                result.base_gate = clone(original.base_gate)
            copy_class_data(original, result)
            if type(original) is Isometry:
                result._inverse = None if original._inverse is None else clone(original._inverse)
            result._definition = None if stored is None else definition(stored)
            memo[identity] = (original, result)
        finally:
            active.remove(identity)
        return result

    snapshot = definition(circuit, top_level=True)
    return snapshot, data_bytes
