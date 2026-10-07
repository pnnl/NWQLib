"""Incrementally admitted execution-only snapshots of supported native gates."""

from copy import copy
from math import isfinite

import numpy as np
try:
    from qiskit import QuantumCircuit
    from qiskit.circuit import (AnnotatedOperation, Barrier, ControlledGate, ControlModifier,
                               Gate, Instruction, InverseModifier, PowerModifier)
    from qiskit.circuit.library import (DiagonalGate, Isometry, QFTGate, StatePreparation,
        UCGate, UCRXGate, UCRYGate, UCRZGate, UniformSuperpositionGate, UnitaryGate)
    from qiskit.circuit.library.generalized_gates.mcg_up_to_diagonal import MCGupDiag
except ModuleNotFoundError as error:
    if error.name == "qiskit":
        error.add_note("Native circuit intake requires its extra: pip install 'nwqlib[qiskit]'.")
    raise

from nwqlib.backends.resources import _standard_gate_classes
from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes


def snapshot_circuit(circuit, *, max_bytes=DEFAULT_INPUT_BYTES):
    """Return a private copy of a supplied circuit and its stored-data byte census.

    The copy isolates the selected action from later changes the caller makes
    to its own circuit, gates or parameter arrays. One memo preserves acyclic
    sharing, so a definition or array referenced twice is copied once and
    stays shared. Active ancestors detect recursion before memo reuse.
    Input-circuit metadata and extra attributes are omitted. Supported SDK
    operation objects keep their SDK class and scalar execution flags.
    Supported immutable SDK singletons are shared under their SDK contract,
    so their global internals are outside this isolation. Top-level
    operations must be gates or gate-based annotations, because the consumer
    inverts and controls the copy. Barriers carry no action and are omitted
    at every stored level.

    Every stored object is charged against ``max_bytes`` before it is copied
    or shared. The per-object allowances are registered in
    ENGINEERING_CONSTANTS.md, "Supplied circuit snapshot allowances". The
    census counts this logical copy, not native RSS or synthesis work.

    Args:
        circuit: Ordinary QuantumCircuit without classical bits.
        max_bytes: Limit on the cumulative census.

    Returns:
        ``(snapshot, data_bytes)``: the copied circuit and the census in bytes.
    """
    # CircuitInstruction.operation may materialize a temporary SDK wrapper.
    # Keep each source alive with its clone so a later wrapper cannot reuse its
    # id. This applies to every stored payload, not just top-level gates.
    memo, active = {}, set()
    data_bytes = 0

    def reserve(size):
        """Add ``size`` bytes to the census, rejecting before the copy that needs them."""
        nonlocal data_bytes
        _check_bytes(data_bytes + size, max_bytes, "circuit snapshot")
        data_bytes += size
    native_classes = set(_standard_gate_classes())
    native_classes.update((DiagonalGate, QFTGate, StatePreparation, UCGate, UCRXGate,
                           UCRYGate, UCRZGate, UniformSuperpositionGate, UnitaryGate,
                           MCGupDiag))

    def array(value):
        """Copy one finite real or complex ndarray parameter once, so a shared array stays shared."""
        if type(value) is not np.ndarray or value.dtype.kind not in "fc":
            raise ValueError("native parameters require ordinary real/complex ndarrays")
        if id(value) in memo:
            return memo[id(value)][1]
        # The census grows by the kept copy, value.nbytes. The one-byte-per-entry
        # np.isfinite mask is freed before tobytes makes that copy, and
        # value.nbytes >= value.size for every real or complex dtype, so this
        # reservation also bounds the moment when the mask is alive.
        reserve(value.nbytes)
        if not np.isfinite(value).all():
            raise ValueError("native array parameters must be finite")
        result = np.frombuffer(value.tobytes(order="C"), dtype=value.dtype).reshape(value.shape)
        memo[id(value)] = (value, result)
        return result

    def parameter(value):
        """Admit one bound numerical gate parameter: an ndarray or a finite int, float or complex."""
        if type(value) is np.ndarray:
            return array(value)
        if type(value) in (int, float, complex) or isinstance(value, np.number):
            value = value.item() if isinstance(value, np.number) else value
            # One complex128 (16 bytes) per scalar, or the byte length of a larger integer.
            reserve(max(16, (value.bit_length()+7)//8) if type(value) is int else 16)
            if not (isfinite(value.real) and isfinite(value.imag)):
                raise ValueError("native scalar parameters must be finite")
            return value
        raise ValueError("native gate parameters must be bound numerical values")

    def definition(original, *, top_level=False):
        """Copy one definition circuit instruction by instruction, once per shared circuit.

        The census charges 32 bytes plus 16 per qubit for the new circuit
        and 16 bytes plus 8 per qubit for every stored instruction, including
        omitted barriers. ``top_level`` marks the supplied circuit, whose
        operations must be gates.
        """
        if type(original) is not QuantumCircuit or original.num_clbits:
            raise ValueError("native definitions require ordinary circuits without classical bits")
        identity = id(original)
        if identity in active:
            raise ValueError("recursive native gate definitions are unsupported")
        if identity in memo:
            return memo[identity][1]
        reserve(32 + 16*original.num_qubits)
        phase = parameter(original.global_phase)
        result = QuantumCircuit(original.num_qubits, global_phase=phase)
        memo[identity] = (original, result)
        active.add(identity)
        try:
            for instruction in original.data:
                reserve(16 + 8*len(instruction.qubits))
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
        modifier costs 32 bytes, and a control modifier also its control-bit
        storage.
        """
        if type(original) not in (ControlModifier, InverseModifier, PowerModifier):
            raise ValueError("unsupported native annotation modifier")
        if id(original) in memo:
            return memo[id(original)][1]
        reserve(32)
        if type(original) is PowerModifier:
            power = original.power
            if type(power) not in (int, float) and not isinstance(power, (np.integer, np.floating)):
                raise ValueError("native annotation power must be a finite real scalar")
            result = PowerModifier(parameter(power))
        elif type(original) is ControlModifier:
            count, state = original.num_ctrl_qubits, original.ctrl_state
            count = count.item() if isinstance(count, np.integer) else count
            state = state.item() if isinstance(state, np.integer) else state
            if type(count) is not int or count < 0:
                raise ValueError("native annotation controls require a nonnegative integer width")
            # The SDK constructor may form 2**count. Charge the bytes of a
            # count-bit control-state integer before that allocation, even when
            # a caller has mutated the modifier's data.
            reserve(max(1, (count+7)//8))
            if state is not None:
                if type(state) is str:
                    if len(state) != count or not state or set(state) - {"0", "1"}:
                        raise ValueError("native annotation control state must match its width")
                elif type(state) is not int or state < 0 or state.bit_length() > count:
                    raise ValueError("native annotation control state must match its width")
            result = ControlModifier(count, state)
        else:
            result = InverseModifier()
        memo[id(original)] = (original, result)
        return result

    def admit_operation(original, *, require_gate, width):
        """Check that a non-annotated operation is supported and has the stored width.

        Supported operations are SDK gates whose base class is a standard or
        listed library gate, Isometry, and plain operations. A plain operation
        has type exactly Gate, ControlledGate or Instruction, and its already
        stored definition gives its action. ``require_gate`` also rejects any
        operation that is not a Gate. Returns ``(plain, stored)``, whether the
        operation is plain and its stored definition or None.
        """
        if require_gate and not isinstance(original, Gate):
            raise ValueError("top-level native operations must be gates or gate-based annotations")
        module = type(original).__module__
        sdk = (module or "").startswith("qiskit.") or (
            module is None and type(original).__name__.startswith("_Singleton"))
        # Support is checked before the immutable fast path, so custom mutable
        # or copy properties cannot bypass admission by imitating a singleton.
        plain = type(original) in (Gate, ControlledGate, Instruction)
        supported = ((sdk and isinstance(original, Gate) and original.base_class in native_classes)
            or plain or type(original) is Isometry)
        if not supported or original.num_clbits:
            raise ValueError("unsupported native instruction type")
        if width is not None and original.num_qubits != width:
            raise ValueError("native gate width differs from its stored instruction")
        stored = getattr(original, "_definition", None)
        if plain and stored is None:
            raise ValueError("plain native gates require an already stored definition")
        return plain, stored

    def copied_parameters(original):
        """Return admitted copies of an operation's parameters.

        A label-form StatePreparation keeps its label characters, each one of
        ``0 1 + - r l`` and charged one byte. Amplitudes that are all Python
        floats or complex numbers, the form Qiskit stores, are admitted
        together: one census charge of 16 bytes per amplitude, as
        ``parameter`` charges each, and one finiteness check of their
        complex128 array. The kept values are the same objects. Every other
        parameter, including a Python int, is admitted by ``parameter``.
        """
        values = original.params
        if isinstance(original, StatePreparation) and original._from_label:
            reserve(len(values))
            if any(type(value) is not str or value not in ("0", "1", "+", "-", "r", "l")
                   for value in values):
                raise ValueError("StatePreparation labels require 0, 1, +, -, r or l")
            return list(values)
        if (isinstance(original, StatePreparation) and not original._from_int
                and set(map(type, values)) <= {float, complex}):
            reserve(16 * len(values))
            if not np.isfinite(np.asarray(values, dtype=np.complex128)).all():
                raise ValueError("native scalar parameters must be finite")
            return list(values)
        return [parameter(value) for value in values]

    def copy_class_data(original, result):
        """Replace the UCGate, StatePreparation and Isometry data of ``result`` by admitted copies.

        None of this data refers to another operation, so nothing here
        recurses into ``clone``.
        """
        if isinstance(original, UCGate):
            simplify, controls = original.simp_contr
            reserve(8*len(controls))
            if any(type(bit) is not int for bit in controls):
                raise ValueError("UCGate control selection requires integer indices")
            result.simp_contr = (simplify, set(controls))
        if isinstance(original, StatePreparation):
            # Its inverse() rebuilds the gate from this argument rather than
            # from _params, so bind that call to the admitted current amplitudes,
            # sharing the admitted list itself. The census keeps its charge for
            # this argument.
            reserve(16*len(result._params))
            result._params_arg = ("".join(result._params) if original._from_label else
                int(original._params_arg) if original._from_int else result._params)
        if type(original) is Isometry:
            result.iso_data = array(original.iso_data)

    def clone(original, *, require_gate=False, width=None):
        """Return the snapshot of one stored operation, copying each object at most once.

        An annotation copies its modifiers and then its gate base, whose
        width is the stored instruction width minus the added controls. It
        costs 32 bytes plus 8 per modifier slot. Any other operation is
        checked by ``admit_operation`` and costs 32 bytes. An immutable one
        is shared as is. A mutable one is shallow-copied, which keeps its SDK
        class and scalar flags, and every mutable payload that the supported
        block-encoding families use is replaced by an admitted copy without
        ``__deepcopy__``. A plain Gate has no hidden execution payload beyond
        its parameters and stored definition.

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
            count = len(original.modifiers)
            reserve(32+8*count)
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
        reserve(32)
        if not original.mutable:
            memo[identity] = (original, original)
            return original
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
