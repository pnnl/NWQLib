"""Explicit operation inventory of existing native circuits; compile only on request."""

import os
import sys
from inspect import signature
from math import isfinite

from nwqlib.operators.access import DEFAULT_INPUT_BYTES, _check_bytes


def _options_snapshot(value, *, max_bytes):
    """Copy portable options under one byte/depth bound, before compilation.

    Sixty-four container levels leave headroom below Python recursion limits.
    Native SDK object payloads are outside this portable scalar/list/dict
    interface.

    The charge bounds the length of the options written as UTF-8 JSON
    (``json.dumps(..., ensure_ascii=False)``), the form that the Run journal
    and record content identities use. The copy shares the caller's strings
    and numbers and creates only new containers, so the charge measures the
    options' content, not Python object memory. Charged sizes:

    - A string, ``6*len + 2``, where ``len`` counts code points. The longest
      UTF-8 JSON form of one code point is the six-byte ``\\u00XX`` escape of
      a control character. A code point outside the Basic Multilingual Plane
      takes four bytes, and the 2 covers the quotes. ASCII-escaped JSON
      (``ensure_ascii=True``, the ``json.dumps`` default) writes such a code
      point as a 12-byte surrogate pair, so this charge does not bound that
      form.
    - An integer, ``max(8, bit_length + 2)``. Its decimal text has at most
      bit_length digits (one digit for zero), and the 2 covers a sign.
    - None, a Boolean or a float, 32 bytes. The longest shortest-round-trip
      binary64 text has 24 characters.
    - A list, tuple or dictionary, ``32 + 16*len`` for its brackets and the
      separators of each entry. Dictionary keys are charged as strings.

    docs/ENGINEERING_CONSTANTS.md ("Resource forecast and explicit inspection
    bounds") registers these allowances.
    """
    active = set()
    size = 0

    def reserve(amount):
        """Charge ``amount`` more bytes, rejecting the total above ``max_bytes`` before it changes."""
        nonlocal size
        _check_bytes(size + amount, max_bytes, "native inspection options")
        size += amount

    def copy(value, depth):
        """Copy one acyclic portable options subtree while charging its serialized size and
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
    """Count one circuit's top-level operations by name; compile only on request.

    Each top-level ``circuit.data`` entry counts once under its operation name,
    including measurements, barriers, simulator saves, ``Clifford`` objects and
    user-defined gates. Definitions and control-flow bodies are not expanded,
    and names are reported as they are, without interpretation. A nonempty
    ``transpile_options`` inventories one auxiliary transpiled copy instead,
    with simulator saves removed before compilation. The executable circuit is
    never copied for the raw inventory, mutated or run.

    Returns a mapping with ``circuit`` (what was counted), ``basis``,
    ``compiler`` (Qiskit version and resolved options, or ``None``),
    ``operations`` (name to count), ``total_operations``, ``num_qubits``,
    ``num_clbits`` and ``depth`` (Qiskit's default depth, which skips
    directives such as barriers and simulator saves). ``max_operations`` covers the input and a
    compiled copy. ``max_bytes`` covers known inspection collections and
    options, not compiler expansion, vendor workspace or process RSS.
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
        # lower; the compiled copy omits them.
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
