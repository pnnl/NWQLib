"""Operation counts and native two-qubit depth of circuits already prepared for a backend.

Compilation runs only on request (``inspect_circuit_resources``). The native
depth (``inspect_native_depth``) and the check of a requested depth limit
before submission (``check_native_depth_cap``) never compile.
"""

import os
import sys
from inspect import signature
from math import isfinite

from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes


def _options_snapshot(value, *, max_bytes):
    """Copy portable options under one byte/depth bound, before compilation.

    Sixty-four container levels leave headroom below Python recursion limits.
    SDK objects are outside this portable scalar, list and dict interface.

    The counted size bounds the length of the options written as UTF-8 JSON
    (``json.dumps(..., ensure_ascii=False)``), the form that the run log and
    record content hashes use. The copy shares the caller's strings and
    numbers and creates only new containers, so the counted size measures the
    options' content, not Python object memory. Counted sizes:

    - A string, ``6*len + 2``, where ``len`` counts code points. The longest
      UTF-8 JSON form of one code point is the six-byte ``\\u00XX`` escape of
      a control character. A code point outside the Basic Multilingual Plane
      takes four bytes, and the 2 covers the quotes. ASCII-escaped JSON
      (``ensure_ascii=True``, the ``json.dumps`` default) writes such a code
      point as a 12-byte surrogate pair, so this count does not bound that
      form.
    - An integer, ``max(8, bit_length + 2)``. Its decimal text has at most
      bit_length digits (one digit for zero), and the 2 covers a sign.
    - None, a Boolean or a float, 32 bytes. The longest shortest-round-trip
      binary64 text has 24 characters.
    - A list, tuple or dictionary, ``32 + 16*len`` for its brackets and the
      separators of each entry. Dictionary keys are counted as strings.

    docs/ENGINEERING_CONSTANTS.md ("Resource forecast and explicit inspection
    bounds") registers these allowances.
    """
    active = set()
    size = 0

    def reserve(amount):
        """Count ``amount`` more bytes, rejecting a total above ``max_bytes`` before it changes."""
        nonlocal size
        _check_bytes(size + amount, max_bytes, "native inspection options")
        size += amount

    def copy(value, depth):
        """Copy one acyclic portable options subtree while counting its serialized size and
        depth.
        """
        if depth > 64:
            raise ValueError("transpile options exceed 64 container levels")
        if value is None or type(value) in (bool, int, float, str):
            if type(value) is float and not isfinite(value):
                raise ValueError("transpile options require finite numerical values")
            reserve(
                6 * len(value) + 2
                if type(value) is str
                else max(8, value.bit_length() + 2)
                if type(value) is int
                else 32
            )
            if type(value) is str:
                try:
                    value.encode("utf-8")
                except UnicodeEncodeError as error:
                    raise ValueError(
                        "transpile option strings must have valid UTF-8 encoding"
                    ) from error
            return value
        if type(value) not in (dict, tuple, list) or id(value) in active:
            raise ValueError("transpile options require finite acyclic scalar/list/dict settings")
        reserve(32 + 16 * len(value))
        active.add(id(value))
        try:
            if type(value) is dict:
                if any(type(key) is not str for key in value):
                    raise ValueError("transpile option keys must be strings")
                return {
                    copy(key, depth + 1): copy(child, depth + 1) for key, child in value.items()
                }
            return [copy(child, depth + 1) for child in value]
        finally:
            active.remove(id(value))

    return copy(value, 0)


def inspect_circuit_resources(
    circuit,
    *,
    native_basis,
    label,
    transpile_options=None,
    basis_label="native",
    max_operations=100_000,
    max_bytes=DEFAULT_INPUT_BYTES,
):
    """Count one circuit's top-level operations by name, and compile only on request.

    Each top-level ``circuit.data`` entry counts once under its operation name,
    including measurements, barriers, simulator saves, ``Clifford`` objects and
    user-defined gates. Definitions and control-flow bodies are not expanded,
    and names are reported as they are, without interpretation. A nonempty
    ``transpile_options`` counts the operations of one transpiled copy instead,
    with simulator saves removed before compilation. The executable circuit is
    never copied for the count without compilation, mutated or run.

    Returns a mapping with ``circuit`` (what was counted), ``basis``,
    ``compiler`` (Qiskit version and resolved options, or ``None``),
    ``operations`` (name to count), ``total_operations``, ``num_qubits``,
    ``num_clbits`` and ``depth`` (Qiskit's default depth, which skips
    directives such as barriers and simulator saves). ``max_operations`` covers the input and a
    compiled copy. ``max_bytes`` covers the counted size of the options and of
    the collections the inspection builds, not compiler expansion, SDK
    workspace or process RSS.
    """
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("max_operations must be a positive integer")
    _check_bytes(0, max_bytes, "native inspection")
    from qiskit import QuantumCircuit, transpile, __version__ as qiskit_version

    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("resource inspection requires an actual selected native circuit")
    if any(type(text) is not str or not text for text in (label, basis_label)):
        raise ValueError("resource inspection requires labels for the circuit and its basis")
    size, width = len(circuit.data), circuit.num_qubits + circuit.num_clbits
    if size > max_operations:
        raise ValueError("native inspection input exceeds max_operations")
    # Registered inventory envelope: 64 bytes per operation and 16 per quantum
    # or classical wire. The compiled-copy check below uses 96 per operation.
    _check_bytes(64 * size + 16 * width, max_bytes, "native inspection collections")
    options = _options_snapshot(
        {} if transpile_options is None else transpile_options, max_bytes=max_bytes
    )
    if type(options) is not dict:
        raise ValueError("transpile_options must be a mapping of keyword settings")
    if type(native_basis) not in (tuple, list) or any(
        type(name) is not str for name in native_basis
    ):
        raise ValueError("native basis requires a finite sequence of gate names")
    # 16 bytes per basis entry plus each name at 6 bytes per code point, the
    # UTF-8 JSON bound of _options_snapshot.
    _check_bytes(
        16 * len(native_basis) + 6 * sum(len(name) for name in native_basis),
        max_bytes,
        "native basis snapshot",
    )
    if options:
        from qiskit.compiler.transpiler import transpile as compiler_transpile
        from qiskit import user_config

        parameters = signature(compiler_transpile).parameters
        unknown = options.keys() - parameters.keys()
        if unknown or "circuits" in options:
            raise ValueError(
                "unsupported transpile options: "
                + ", ".join(sorted(unknown | ({"circuits"} & options.keys())))
            )
        # Match the installed transpile defaults explicitly, including user
        # configuration/environment seed selection, and save the resolved values.
        configuration = user_config.get_config()
        if options.get("optimization_level") is None:
            options["optimization_level"] = configuration.get("transpile_optimization_level", 2)
        if options.get("seed_transpiler") is None:
            seed = os.getenv("QISKIT_TRANSPILER_SEED")
            options["seed_transpiler"] = (
                int(seed) if seed is not None else configuration.get("transpiler_seed")
            )
        for name in (
            "approximation_degree",
            "unitary_synthesis_method",
            "qubits_initially_zero",
            "ignore_backend_supplied_default_methods",
        ):
            options.setdefault(name, parameters[name].default)
        if not options.get("basis_gates"):
            from qiskit.circuit.library import get_standard_gate_name_mapping

            standard = get_standard_gate_name_mapping()
            options["basis_gates"] = [name for name in native_basis if name in standard]
        options = _options_snapshot(options, max_bytes=max_bytes)
        _check_bytes(96 * size + 16 * width, max_bytes, "auxiliary inspection copy")
        # Aer save instructions are simulator requests the transpiler cannot
        # lower, so the compiled copy omits them.
        save_module = sys.modules.get("qiskit_aer.library.save_instructions.save_data")
        save_types = () if save_module is None else (save_module.SaveData,)
        selected = circuit.copy_empty_like(vars_mode="drop")
        selected.metadata = {}
        selected.data = [
            item for item in circuit.data if not isinstance(item.operation, save_types)
        ]
        counted = transpile(selected, **options)
        if len(counted.data) > max_operations:
            raise ValueError("auxiliary compiler output exceeds max_operations")
        _check_bytes(
            64 * len(counted.data) + 16 * (counted.num_qubits + counted.num_clbits),
            max_bytes,
            "auxiliary inspection output",
        )
        described = (
            f"auxiliary transpiled copy of {label} without simulator saves; "
            "original unchanged; not executed"
        )
        basis = "transpiled:" + ",".join(options["basis_gates"])
        compiler = {"name": "qiskit.transpile", "version": qiskit_version, "options": options}
    else:
        counted = circuit
        described = label + "; not executed by inspection"
        basis = f"{basis_label}:{','.join(native_basis)}" if native_basis else basis_label
        compiler = None
    operations = {}
    for item in counted.data:
        name = item.operation.name
        operations[name] = operations.get(name, 0) + 1
    # Per distinct name: 64 bytes for its entry and count, plus the name at 6
    # bytes per code point (the UTF-8 JSON bound of _options_snapshot), and
    # 256 bytes for the fixed result fields.
    _check_bytes(
        sum(64 + 6 * len(name) for name in operations) + 256,
        max_bytes,
        "native operation inventory",
    )
    return {
        "circuit": described,
        "basis": basis,
        "compiler": compiler,
        "operations": operations,
        "total_operations": len(counted.data),
        "num_qubits": counted.num_qubits,
        "num_clbits": counted.num_clbits,
        "depth": counted.depth(),
    }


def inspect_native_depth(circuit, target, *, max_operations=100_000, max_bytes=DEFAULT_INPUT_BYTES):
    """Compute the native two-qubit depth of one prepared circuit on its backend target, without changing or running it.

    The circuit must be the one that executes, already compiled for
    `target`. A transpiled or decomposed copy would describe another circuit,
    so none is made. The result describes only this circuit and reads no
    execution options. Whether the provider executes the circuit unchanged,
    without gate folding, twirling or dynamical decoupling for example, is
    decided by the backend's preparation, whose record holds an unqualified
    depth when the backend cannot establish it (`NativeDepth`). The depth is
    qualified only when every operation of the circuit has a known native
    meaning on the target:

    - a gate on one or two qubits whose name and class match an operation of
      `target` that the target supports on those physical qubits with those
      parameters. A native gate for which Qiskit also has a definition, such
      as ECR, stays native,
    - a measurement into one classical bit, a reset, or a delay with a
      numeric duration, on one qubit and supported by the target on that
      qubit, or
    - a barrier on any number of qubits.

    Any other operation leaves the depth unqualified, with the reason:
    an operation that the target does not support on its qubits, a gate whose
    name matches a native gate but whose class does not, an opaque or composite
    gate, a simulator save or other directive, an operation on more than two
    qubits other than a barrier, control flow (if, while, for, switch or box),
    a classical store, a delay whose duration is a classical expression,
    classical variables or stretches, and unbound parameters. Qiskit 2.6.0
    attaches a classical condition to an operation only through control flow,
    so control flow covers condition-controlled operations. A circuit above `max_operations`, or whose inspection needs
    more than `max_bytes` (64 bytes per operation and 16 per qubit or classical
    bit, the bound of `inspect_circuit_resources`), is not scanned and is also
    unqualified. Unqualified means unknown, never a depth of zero.

    For a qualified circuit, the native two-qubit depth is the largest sum of
    weights along a dependency path, with weight 1 for each two-qubit gate and
    0 for every other operation. Each qubit and classical bit starts at depth
    0. In execution order, each operation takes the largest depth among the
    qubits and classical bits it touches, adds its weight and writes the
    result back to all of them. By induction over the operations, the depth
    held by a wire is the largest weight of a path ending at the last
    operation on that wire. Every path into an operation arrives through the
    last earlier operation on one of its wires, so the value it writes is the
    largest weight of a path ending at it. A zero-weight barrier or
    measurement therefore still orders the gates on its wires: CX(0,1), a
    barrier on (1,2), then CX(2,3) has depth 2, while the same two gates
    without the barrier have depth 1. Two measurements into one classical bit
    order their qubits in the same way.
    The empty circuit has depth 0. The total native depth, reported only,
    uses the same recurrence with weight 1 for every qualified gate,
    measurement, reset and delay and weight 0 for a barrier. The width is the
    circuit's `num_qubits`, idle qubit positions included and classical bits
    excluded, and it is recorded also for an unqualified depth. Qiskit
    2.6.0's `QuantumCircuit.depth(filter_function)` computes this recurrence,
    writing back the largest depth also for an operation that the filter
    excludes, so it is reused after the scan above has qualified every
    operation.

    Args:
        circuit (QuantumCircuit): The prepared circuit that executes.
        target (Target): The Qiskit `Target` of the backend that compiled it,
            whose qubit indices are the circuit's qubit positions.
        max_operations (int): Default `100_000`. Largest number of operations
            of the circuit that is scanned.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000`). Largest
            size of the known inspection data. It does not bound Qiskit's own
            memory.

    Returns:
        depth (NativeDepth): The qualified depth, total depth, two-qubit gate
            count and width, or the reason the depth is unavailable.

    Raises:
        ValueError: If `circuit` is not a Qiskit circuit, `target` is not a
            Qiskit `Target`, or `max_operations` or `max_bytes` is not a
            positive integer.
    """
    if type(max_operations) is not int or max_operations < 1:
        raise ValueError("max_operations must be a positive integer")
    _check_bytes(0, max_bytes, "native depth inspection")
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, ControlFlowOp, Delay, Gate, Measure, Reset, Store
    from qiskit.circuit.classical.expr import Expr
    from qiskit.transpiler import Target
    from nwqlib.execution import NativeDepth

    if not isinstance(circuit, QuantumCircuit) or not isinstance(target, Target):
        raise ValueError("native depth requires an actual prepared circuit and the Target of its backend")
    width = circuit.num_qubits

    def unqualified(reason):
        """The unqualified result for this circuit, without depth values, recording ``reason``."""
        return NativeDepth(qualified=False, reason=reason, width=width)

    size = len(circuit.data)
    if size > max_operations:
        return unqualified(f"the circuit has {size} operations, more than max_operations={max_operations}")
    needed = 64 * size + 16 * (width + circuit.num_clbits)
    if needed > max_bytes:
        return unqualified(f"inspecting the circuit needs {needed} data bytes, more than max_bytes={max_bytes}")
    if circuit.num_parameters:
        return unqualified("the circuit has unbound parameters")
    if circuit.num_vars or circuit.num_stretches:
        return unqualified("the circuit has classical variables or stretches")
    two_qubit_gates = 0
    for item in circuit.data:
        operation = item.operation
        kind = getattr(operation, "base_class", None)
        if kind is Barrier:
            continue
        if isinstance(operation, ControlFlowOp):
            return unqualified(f"control flow ({operation.name}) has no static depth")
        if isinstance(operation, Store):
            return unqualified("a classical store has no native depth")
        if isinstance(operation, Delay) and isinstance(operation.duration, Expr):
            return unqualified("a delay whose duration is a classical expression has no static native depth")
        arity = len(item.qubits)
        if arity > 2:
            return unqualified(f"{operation.name} acts on {arity} qubits")
        qubits = tuple(circuit.find_bit(qubit).index for qubit in item.qubits)
        native = target.operation_from_name(operation.name) if operation.name in target.operation_names else None
        # A target lists an operation by an instance, or a variadic one by its class.
        expected = native if isinstance(native, type) else getattr(native, "base_class", None)
        allowed = (isinstance(operation, Gate) and arity in (1, 2)
                   or isinstance(operation, (Measure, Reset, Delay)) and arity == 1)
        if (not allowed or kind is None or kind is not expected
                or not target.instruction_supported(operation_name=operation.name, qargs=qubits,
                                                    parameters=list(operation.params))):
            return unqualified(f"{operation.name} on qubits {qubits} is not a native operation of the target")
        two_qubit_gates += int(arity == 2)
    # Every remaining two-qubit operation other than a barrier is a native gate.
    two_qubit_depth = circuit.depth(lambda item: len(item.qubits) == 2 and item.operation.base_class is not Barrier)
    total_depth = circuit.depth(lambda item: item.operation.base_class is not Barrier)
    return NativeDepth(qualified=True, native_two_qubit_depth=two_qubit_depth, total_native_depth=total_depth,
                       native_two_qubit_gates=two_qubit_gates, width=width)


class NativeDepthCapRefusal(ValueError):
    """A requested limit on the native two-qubit depth refuses a prepared circuit before its submission.

    `check_native_depth_cap` raises it from a Method's `before_submit`, so the
    Run records no submission and spends no shots on the refused circuit.
    Circuits submitted earlier, their data and their counted work stay in the
    Run. A controller can turn the refusal into a stop reason from the
    attributes below.

    Attributes:
        experiment: Name of the Plan experiment of the refused circuit.
        prepared_id: Content hash of the circuit's preparation record
            (`PreparedArtifact`).
        cap: The requested largest native two-qubit depth.
        native_depth: The circuit's `NativeDepth` from its preparation
            record, unchanged, or `None` when the backend does not compute one.
        depth: The native two-qubit depth computed at preparation, larger
            than `cap`, or `None` when the depth is unavailable.
        reason: Why the native depth is unavailable: the backend computes
            none, the circuit's content, the backend's options, the installed
            SDK or a provider estimate prevent qualification (the reason in
            `native_depth`), or the depth is qualified but the version of a
            package recorded with the preparation differs from the installed
            one or cannot be read. `None` when the depth exceeds `cap`.
        provenance: For a qualified depth refused because a package version
            recorded with its preparation differs from the installed one or
            cannot be read, the description of that mismatch, the same text
            as `reason`. `None` for the other refusals. This refusal depends
            on the packages installed where the Run is read, not on the
            circuit. Reopened where its recorded versions are installed, the
            Run checks the same preparation record against the limit again.
            A new Run prepared in the installed environment records a depth
            of its own, which then decides the limit.
    """

    def __init__(self, *, experiment, prepared_id, cap, native_depth, provenance=None):
        # The four refusals need different remedies: another backend, another
        # circuit, other backend options or counts instead of a provider
        # estimate, the recorded package versions or a new Run in the
        # installed environment, or a shallower circuit or larger limit.
        # provenance is the package mismatch of a qualified depth
        # (check_native_depth_cap).
        self.experiment, self.prepared_id, self.cap = experiment, prepared_id, cap
        self.native_depth, self.provenance = native_depth, provenance
        circuit = f"prepared circuit {prepared_id} of experiment {experiment!r}"
        if provenance is not None:
            self.depth, self.reason = None, provenance
            message = (f"native-depth qualification unavailable for {circuit} in this environment: "
                       f"{provenance}. Its preparation record keeps the depth, but a depth qualified with "
                       f"other package versions cannot allow a new submission under the requested native "
                       f"two-qubit depth limit {cap}. Reopen the Run where its recorded package versions are "
                       "installed, prepare the circuit in a new Run in the installed environment, whose "
                       "preparation record then decides the limit, or request execution without the limit")
        elif native_depth is None:
            self.depth, self.reason = None, "the backend does not compute a native depth for its circuits"
            message = (f"native-depth qualification unavailable for {circuit}: {self.reason}, so the requested "
                       f"native two-qubit depth limit {cap} refuses its submission. Choose a backend whose "
                       "preparation computes the native depth, such as IBM Runtime, or request execution "
                       "without the limit")
        elif not native_depth.qualified:
            self.depth, self.reason = None, native_depth.reason
            message = (f"native-depth qualification unavailable for {circuit}: {self.reason}. Without a "
                       f"qualified native depth, the requested native two-qubit depth limit {cap} refuses its "
                       "submission. Remove the cause named above, or request execution without the limit")
        else:
            self.depth, self.reason = native_depth.native_two_qubit_depth, None
            message = (f"{circuit} has native two-qubit depth {self.depth}, above the requested limit {cap}, so "
                       "its submission is refused. Choose a shallower circuit or request a larger limit")
        super().__init__(message)


def check_native_depth_cap(prepared, cap):
    """Refuse the prepared circuits of one submission when a requested native two-qubit depth limit is not met.

    A Method's `before_submit` calls it with the prepared circuits of the
    submission and the limit the user requested, before the Run records the
    submission. With `cap=None` it returns without reading anything, not even
    the installed package versions. With an integer limit, a prepared circuit
    is accepted when three conditions hold:

    - its preparation computed a qualified native depth
      (`PreparedArtifact.native_depth`, from `inspect_native_depth`),
    - each package version recorded with the preparation
      (`PreparedArtifact.environment`, the Sources whose reference is
      `importlib.metadata`; for IBM Runtime these are nwqlib, Qiskit and
      qiskit-ibm-runtime) equals the installed one, and
    - that depth is at most the limit. A depth equal to the limit is
      accepted.

    The preparation record is read, also after the Run is reopened, so nothing
    is compiled or computed again, and the record is never changed. A missing
    or unqualified depth refuses the circuit, because an unknown depth cannot
    meet the limit. The installed Qiskit reads the saved circuit, the
    installed NWQLib restores it, and the qualification was decided for the
    SDK installed at preparation, so a depth qualified with other versions,
    as in a Run reopened in another environment, refuses the circuit too. A
    recorded package that is not installed, or whose version cannot be read,
    counts as different. The recorded Python version is not compared.
    Classical routines (host kernels) have no circuit and are skipped. Only a
    new submission is checked. Results of a job already submitted are
    retrieved without this check.

    Args:
        prepared (tuple[PreparedHandle, ...]): The prepared work of the
            submission, as `before_submit` receives it.
        cap (int | None): The largest native two-qubit depth allowed, a
            nonnegative integer, or `None` for no limit.

    Raises:
        NativeDepthCapRefusal: For the first prepared circuit whose depth
            exceeds `cap`, is unavailable, or was qualified with package
            versions other than the installed ones.
        ValueError: If `cap` is neither `None` nor a nonnegative integer.
    """
    if cap is None:
        return
    if type(cap) is not int or cap < 0:
        raise ValueError("a native two-qubit depth limit must be a nonnegative integer or None")
    from importlib.metadata import version
    for handle in prepared:
        record = handle.record
        if record.execution != "quantum_circuit":
            continue
        native = record.native_depth
        refusal = dict(experiment=handle.realization.experiment, prepared_id=record.content_id, cap=cap,
                       native_depth=native)
        if native is None or not native.qualified:
            raise NativeDepthCapRefusal(**refusal)
        for source in record.environment:
            if source.reference != "importlib.metadata":
                continue
            try:
                installed = version(source.name)
            except Exception as error:
                # Any failed lookup leaves the installed version unknown,
                # which cannot equal the recorded one.
                raise NativeDepthCapRefusal(**refusal, provenance=(
                    f"its preparation recorded {source.name} {source.version}, and the installed version of "
                    f"{source.name} cannot be read ({type(error).__name__}: {error})")) from error
            if installed != source.version:
                raise NativeDepthCapRefusal(**refusal, provenance=(
                    f"its preparation recorded {source.name} {source.version}, but {source.name} "
                    f"{installed} is installed"))
        if native.native_two_qubit_depth > cap:
            raise NativeDepthCapRefusal(**refusal)
