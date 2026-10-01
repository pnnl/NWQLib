"""Explicit released-NWQEC compilation of an already selected static body.

Compilation is an auxiliary operation. It reads an existing prepared circuit or
lowered block, runs NWQEC 0.1.2 once in a bounded child process
(``_auxiliary_process``) and stores a LogicalCompilation. It never replaces the
selected construction or its readout. docs/fault-tolerant-resources.md defines
the targets, stages and limitations.
"""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

from pathlib import Path
import hashlib
import json
import sys
from typing import Annotated, Literal

from pydantic import Field, StrictBool, model_validator

from nwqlib.core.records import (
    ContentID,
    Nonnegative,
    Rational,
    Real,
    Record,
    Scope,
    Source,
    Text,
    Unit,
)
from nwqlib.core.planning import ObservationSpec
from nwqlib.evidence import Evidence, Fact
from nwqlib.execution import LogicalPreparationReceipt, RegisterMap
from nwqlib.ir.expressions import NonnegativeInt
from nwqlib.resources import ResourceQuantity
from nwqlib._validation import boolean, finite_real, integer
from nwqlib.operators.access import _check_bytes

from nwqlib.io.qasm import export_qasm

Target = Literal["clifford_t", "pbc", "pbc_tfuse", "clifford_reduction"]
ErrorPolicy = Literal["per-gate", "total", "relative"]
Positive = Annotated[Real, Field(gt=0)]


class LogicalCompilation(Record):
    """Auxiliary compiler output, never a phase-faithful replacement for its source.

    Original readout and source phase describe the input, not acquired data or
    recovered compiler phase. Raw PBC operation names/angles stay in the stored
    output QASM. Resource quantities concern only the stated compiler stage.
    """

    source_kind: Literal["prepared", "logical_circuit"]
    source_id: ContentID
    quantum_layout: tuple[RegisterMap, ...]
    measurement_layout: tuple[RegisterMap, ...]
    lowering: LogicalPreparationReceipt | None
    original_readout: ObservationSpec | None
    original_population: Text
    original_native_basis: tuple[Text, ...]
    terminal_measurements: tuple[tuple[NonnegativeInt, NonnegativeInt], ...]
    simulator_saves: tuple[tuple[Text, str | None, tuple[NonnegativeInt, ...], Text], ...]
    source_global_phase: Real
    compiler_version: Literal["0.1.2"]
    compiler_module_file: Text
    target: Target
    rz_err: ErrorPolicy
    requested_epsilon: Positive | None
    effective_epsilon: Positive
    keep_ccx: Literal[False] = False
    keep_cx: Literal[False] = False
    count_only: StrictBool
    stage: Literal["synthesis_before_fusion", "final_artifact"]
    input_qasm_digest: ContentID
    input_operations: NonnegativeInt
    width: NonnegativeInt
    raw_counts: tuple[tuple[Text, NonnegativeInt], ...]
    quantities: tuple[ResourceQuantity, ...]
    output_qasm: str | None
    elapsed_seconds: Nonnegative
    max_operations: Annotated[NonnegativeInt, Field(gt=0)]
    max_bytes: Annotated[NonnegativeInt, Field(gt=0)]
    timeout_seconds: Positive
    limitations: tuple[Text, ...]

    @model_validator(mode="after")
    def _stage(self):
        """Keep the stage, target and output consistent with ``count_only`` and the recorded limits.

        A count-only run stops at Clifford+T synthesis before fusion and has no
        output circuit. A full run stores its final QASM. Raw gate names are
        distinct, their total respects ``max_operations``, and measured qubits
        lie inside the compiled width.
        """
        if self.count_only != (self.stage == "synthesis_before_fusion"):
            raise ValueError("compiler stage must match count_only")
        if self.count_only and self.target != "clifford_t":
            raise ValueError("count_only supports only clifford_t")
        if (self.output_qasm is None) != self.count_only:
            raise ValueError("full compilation requires its output artifact")
        if len(dict(self.raw_counts)) != len(self.raw_counts):
            raise ValueError("raw gate names must be distinct")
        if sum(value for _, value in self.raw_counts) > self.max_operations:
            raise ValueError("compiled output exceeds max_operations")
        if any(q >= self.width for q, _ in self.terminal_measurements):
            raise ValueError("terminal measurement lies outside the compiled width")
        return self

    def save(self, path, *, max_bytes=DEFAULT_MAX_BYTES):
        """Save this stored record through the existing bounded JSON writer."""
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        ArchiveFiles(path.parent, max_bytes).write_json(path.name, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path, *, max_bytes=DEFAULT_MAX_BYTES):
        """Read current-format stored compilation data without importing NWQEC."""
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        return cls.model_validate(ArchiveFiles(path.parent, max_bytes).read_json(path.name))

    def __str__(self):
        """Short summary: compiler, target, stage, source, requested epsilon, quantities and limitations."""
        lines = [
            f"NWQEC {self.compiler_version}: {self.target}, {self.stage}",
            f"source={self.source_id}; requested RZ epsilon={self.effective_epsilon:g}",
        ]
        lines.extend(str(quantity) for quantity in self.quantities)
        lines.extend(self.limitations)
        return "\n".join(lines)


def _source(source, index):
    """Read the existing selected native owner without copying or preparing it.

    A Prepared source uses the circuit index order of ``Prepared.circuits``
    (``Prepared._circuit_handles``). A handle that ``Run.release_native``
    released is reloaded from its saved QPY by ``PreparedHandle._restore_native``,
    as ``Prepared.inspect_resources`` reloads it, without lowering or
    transpiling again.
    """
    from nwqlib._prepared_execution import Prepared
    from nwqlib.blocks.lowering import LogicalCircuit

    if isinstance(source, Prepared):
        handles = source._circuit_handles()
        if index >= len(handles):
            raise ValueError("no prepared quantum circuit at the selected index")
        handle = handles[index]
        record = handle.record
        return handle._restore_native().circuit, dict(
            source_kind="prepared",
            source_id=record.content_id,
            quantum_layout=record.native_quantum_layout or record.quantum_layout,
            measurement_layout=record.native_classical_layout or record.classical_layout,
            lowering=record.logical,
            original_readout=record.observation,
            original_population=record.population,
            original_native_basis=record.native_basis,
        )
    if isinstance(source, LogicalCircuit):
        if index:
            raise ValueError("LogicalCircuit requires index=0")
        return source.circuit, dict(
            source_kind="logical_circuit",
            source_id=source.construction_id,
            quantum_layout=tuple(RegisterMap(name=n, bits=b) for n, b in source.quantum_layout),
            measurement_layout=tuple(
                RegisterMap(name=n, bits=b) for n, b in source.measurement_layout
            ),
            lowering=LogicalPreparationReceipt(
                dynamic_visits=source.dynamic_visits,
                reserved_work=source.construction_work,
                defined_selections=source.defined_selections,
            ),
            original_readout=None,
            original_population="selected static block body",
            original_native_basis=(),
        )
    raise TypeError("compile_logical requires Prepared or LogicalCircuit")


def _static_body(circuit, *, max_operations, max_bytes):
    """Separate terminal readout from a static Gate-only compiler input.

    NWQEC compiles unitary bodies, so terminal measurements and recognized Aer
    saves become source metadata and are removed from the compiler input.
    Anything nonunitary before them, a classical condition, or a measurement
    followed by more gates is rejected. The body is exchanged as QASM2
    because NWQEC 0.1.2 reads QASM files. Returns the QASM2 text, the
    (qubit, clbit) terminal measurements, the saved readout settings and the
    source global phase.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Barrier, Gate, Measure

    if not isinstance(circuit, QuantumCircuit):
        raise ValueError("selected artifact has no Qiskit circuit")
    size = len(circuit.data)
    if size > max_operations:
        raise ValueError("compiler input exceeds max_operations")
    # The inspection inventory envelope: 64 bytes per operation plus 16 per
    # quantum or classical wire for the Python collections built below
    # (docs/ENGINEERING_CONSTANTS.md, "Resource forecast and explicit
    # inspection bounds").
    _check_bytes(
        64 * size + 16 * (circuit.num_qubits + circuit.num_clbits),
        max_bytes,
        "compiler input collections",
    )
    phase = finite_real(float(circuit.global_phase), "source global phase")
    save_module = sys.modules.get("qiskit_aer.library.save_instructions.save_data")
    save_types = () if save_module is None else (save_module.SaveData,)
    known_saves = {
        "save_amplitudes",
        "save_amplitudes_sq",
        "save_density_matrix",
        "save_expval",
        "save_expval_var",
        "save_probabilities",
        "save_probabilities_dict",
        "save_state",
        "save_statevector",
        "save_statevector_dict",
    }
    body = QuantumCircuit(circuit.num_qubits)
    measurements, saves = [], []
    terminal = False
    for instruction in circuit.data:
        op = instruction.operation
        qubits = tuple(circuit.find_bit(q).index for q in instruction.qubits)
        if getattr(op, "condition", None) is not None:
            raise ValueError("NWQEC requires a static body without classical conditions")
        if isinstance(op, Barrier):
            continue
        if isinstance(op, Measure):
            terminal = True
            measurements.append((qubits[0], circuit.find_bit(instruction.clbits[0]).index))
        elif isinstance(op, save_types) and op.name in known_saves:
            from nwqlib._choice_archive import _json_scalar

            # Aer readout coefficients may be NumPy real scalars. Preserve
            # their actual values using the archive encoder, with a byte cap
            # during encoding rather than a truncated repr of the parameters.
            encoder = json.JSONEncoder(default=_json_scalar, allow_nan=False, separators=(",", ":"))
            parts, used = [], 0
            for part in encoder.iterencode(
                {"params": op.params, "subtype": getattr(op, "_subtype", None)}
            ):
                used += len(part.encode("utf-8"))
                _check_bytes(used, max_bytes, "simulator readout metadata")
                parts.append(part)
            terminal = True
            saves.append((op.name, op.label, qubits, "".join(parts)))
        elif terminal:
            raise ValueError("NWQEC does not accept intermediate measurement/save declarations")
        elif not isinstance(op, Gate) or instruction.clbits:
            raise ValueError(
                f"NWQEC static body rejects nonunitary/control instruction {op.name!r}"
            )
        else:
            body.append(op, qubits, copy=False)
    # Reuse native intake for bounded copies, nested-definition admission and
    # custom primitive-name collision handling. It reads stored definitions;
    # it does not synthesize a new native realization or apply an operator.
    from nwqlib.blocks._qiskit_intake import snapshot_circuit

    body, _ = snapshot_circuit(body, max_bytes=max_bytes)
    try:
        qasm = export_qasm(body, format="qasm2")
    except Exception as error:
        raise ValueError(f"selected static body cannot be exported as QASM2: {error}") from error
    # Encoded once; the caller reuses these bytes for its size checks, its
    # digest and the compiler's body.qasm.
    data = qasm.encode("utf-8")
    del qasm
    _check_bytes(len(data), max_bytes, "compiler QASM2 input")
    return data, tuple(measurements), tuple(saves), phase


def compile_logical(
    source,
    *,
    index=0,
    target="clifford_t",
    rz_err="per-gate",
    epsilon=None,
    count_only=False,
    max_operations=100_000,
    max_bytes=DEFAULT_MAX_BYTES,
    timeout_seconds=60,
):
    """Compile one existing selected static body in a bounded local child.

    Limits cover known input/output operations and transport bytes, not native
    compiler RSS. Requested synthesis precision is recorded, never certified.
    Terminal source readout is metadata and is not compiled into the body.

    Args:
        source: A live ``Prepared`` or a ``LogicalCircuit`` from ``lower_qiskit``.
        index: Position among the prepared quantum handles. It must be 0 for a
            LogicalCircuit.
        target: "clifford_t", "pbc", "pbc_tfuse" or "clifford_reduction".
        rz_err: NWQEC RZ synthesis error policy, "per-gate", "total" or "relative".
        epsilon: Requested RZ synthesis precision for that policy. None resolves
            to 1e-10 for "per-gate" and 1e-2 otherwise.
        count_only: Record Clifford+T counts before final fusion, without an
            output circuit. Valid only for "clifford_t".
        max_operations: Cap on source instructions and compiled output gates.
        max_bytes: Cap on transport, capture and publication bytes.
        timeout_seconds: Wall-clock deadline for the compiler child. The
            defaults of the three limits are registered in
            docs/ENGINEERING_CONSTANTS.md ("Explicit logical compilation and
            physical projection").

    Returns:
        A LogicalCompilation for this source and stage.
    """
    from ._auxiliary_process import run_auxiliary

    index = integer(index, "index", 0)
    max_operations = integer(max_operations, "max_operations", 1)
    max_bytes = integer(max_bytes, "max_bytes", 1)
    timeout_seconds = finite_real(timeout_seconds, "timeout_seconds")
    count_only = boolean(count_only, "count_only")
    if target not in {"clifford_t", "pbc", "pbc_tfuse", "clifford_reduction"}:
        raise ValueError("unsupported NWQEC target")
    if rz_err not in {"per-gate", "total", "relative"}:
        raise ValueError("unsupported NWQEC RZ error policy")
    if count_only and target != "clifford_t":
        raise ValueError("count_only supports only clifford_t")
    requested = None if epsilon is None else finite_real(epsilon, "epsilon")
    # An omitted epsilon resolves here rather than inside NWQEC, so the stored
    # effective value is exactly the value sent to the compiler. These are
    # NWQLib's documented defaults (docs/fault-tolerant-resources.md), untuned
    # requests registered in docs/ENGINEERING_CONSTANTS.md ("Provider and
    # compiler policies"). Revisit for the intended synthesis accuracy or
    # compiler policy. The requested epsilon is not a certified bound on the
    # compiled circuit's error.
    effective = (1e-10 if rz_err == "per-gate" else 1e-2) if requested is None else requested
    if effective <= 0 or timeout_seconds <= 0:
        raise ValueError("epsilon and timeout_seconds must be positive")
    circuit, binding = _source(source, index)
    qasm, measurements, saves, phase = _static_body(
        circuit, max_operations=max_operations, max_bytes=max_bytes
    )
    # Admit the source/readout metadata before spending a compiler invocation.
    # Core records contain portable values; this streams a size check without
    # retaining another complete serialized source alongside the final receipt.
    # qasm holds the UTF-8 bytes of the QASM2 body, encoded once.
    metadata_bytes = len(qasm)
    encoder = json.JSONEncoder(default=lambda value: value.model_dump(mode="json"), allow_nan=False)
    for part in encoder.iterencode((binding, measurements, saves)):
        metadata_bytes += len(part.encode("utf-8"))
        _check_bytes(metadata_bytes, max_bytes, "compiler source and readout metadata")
    result, elapsed = run_auxiliary(
        "nwqec",
        dict(
            target=target,
            rz_err=rz_err,
            epsilon=effective,
            count_only=count_only,
            max_operations=max_operations,
        ),
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        files=(("body.qasm", qasm),),
    )
    if result["width"] != circuit.num_qubits:
        raise ValueError("NWQEC changed the source width; original layout cannot be preserved")
    stage = "synthesis_before_fusion" if count_only else "final_artifact"
    counts = result["counts"]
    compiler = Source(
        name="NWQEC",
        version=result["version"],
        domain=f"{target}/{stage}",
        reference=result["module_file"],
    )
    scope = Scope(domain=f"one explicitly compiled static body: {target}/{stage}")
    output_identity = (
        "sha256:"
        + hashlib.sha256(
            (
                result["qasm"] if result["qasm"] is not None else json.dumps(counts, sort_keys=True)
            ).encode()
        ).hexdigest()
    )
    evidence = Evidence(
        kind="observed",
        source=compiler,
        status="witnessed",
        artifact=output_identity,
        witnessed_scope=scope,
        subject_id=binding["source_id"],
    )
    values = {
        "operations": sum(counts.values()),
        "logical_width": result["width"],
        "logical_depth": result["depth"],
    }
    pbc = target in {"pbc", "pbc_tfuse"}
    if not pbc:
        values.update(t=counts.get("t", 0) + counts.get("tdg", 0), cx=counts.get("cx", 0))
    quantities = tuple(
        ResourceQuantity(
            metric=metric,
            fact=Fact(
                quantity=metric,
                unit=Unit(symbol="count", dimension="count"),
                scope=scope,
                availability="unknown" if value is None else "concrete",
                value=None if value is None else Rational(numerator=value, denominator=1),
                reason="depth is unavailable for this compiler stage/representation"
                if value is None
                else None,
                evidence=None if value is None else evidence,
            ),
            population=f"one static compiler body; {stage}",
            lifecycle="prepared",
            basis=f"nwqec:{target}",
            interpretation="unavailable" if value is None else "exact",
            sources=(binding["source_id"], compiler.content_id),
        )
        for metric, value in values.items()
    )
    limitations = (
        "Requested RZ epsilon is not verified achieved precision or total algorithm error.",
        "QASM2/compiler phase loss is not recovered; output is not a phase-faithful reusable block.",
        "Original terminal readout is source metadata; no observations were acquired.",
        "Operation/transport caps and timeout do not bound native internal RSS or synthesis work.",
    ) + (
        (
            "PBC Pauli rotations/terminal all-qubit frame are not native T counts or original readout.",
        )
        if pbc
        else ()
    )
    compiled = LogicalCompilation(
        **binding,
        terminal_measurements=measurements,
        simulator_saves=saves,
        source_global_phase=phase,
        compiler_version=result["version"],
        compiler_module_file=result["module_file"],
        target=target,
        rz_err=rz_err,
        requested_epsilon=requested,
        effective_epsilon=effective,
        count_only=count_only,
        stage=stage,
        input_qasm_digest="sha256:" + hashlib.sha256(qasm).hexdigest(),
        input_operations=len(circuit.data),
        width=result["width"],
        raw_counts=tuple(sorted(counts.items())),
        quantities=quantities,
        output_qasm=result["qasm"],
        elapsed_seconds=elapsed,
        max_operations=max_operations,
        max_bytes=max_bytes,
        timeout_seconds=timeout_seconds,
        limitations=limitations,
    )
    _check_bytes(len(compiled.model_dump_json().encode()), max_bytes, "logical compilation record")
    return compiled
