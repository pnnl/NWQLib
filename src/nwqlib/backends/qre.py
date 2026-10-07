"""One explicit QDK counts-model projection of a stored Clifford+T compilation.

The physical model is the one implemented by the pinned QDK 1.32.3 modules.
``qdk.qre.models.qubits.GateBased`` supplies gate and measurement errors and
times. ``qdk.qre.models.qec.SurfaceCode`` supplies the rotated surface code
with ``2*d*d - 1`` physical qubits per patch (d*d data and d*d - 1 syndrome
qubits) and a syndrome cycle of one one-qubit gate time, four two-qubit gate
times and one measurement time, repeated d times per logical cycle. The same
module sets the error rate of one logical lattice-surgery step per patch to
``0.03 * (p/0.01)**((d+1)//2)``, with p the largest of the H, CNOT and
measurement error rates. NWQLib takes the patch size, the cycle and the error
formula from the pinned QDK code and does not evaluate them itself.

The QDK source comments cite three papers without versions. Read against the
versions below, the papers support these parts of the model, and the rest is
QDK's own modeling choice:

- Patch size. Horsman et al., arXiv:1111.4022v3, Sec. 7.1 (same section in
  v1 and v2), shows that the rotated lattice has d*d data qubits and, in its
  d = 5 and d = 3 examples, d*d - 1 independent stabilizers. One syndrome
  qubit per stabilizer gives ``2*d*d - 1``. The paper also reaches 13 qubits
  at d = 3 by reusing the four central syndrome qubits, which QDK does not
  model.
- Syndrome cycle. Wang, Fowler and Hollenberg, arXiv:1009.3686v1, Fig. 1(b)
  orders the four CNOT layers, and Fig. 2 measures one stabilizer with four
  CNOTs between two syndrome measurements, without initialization gates. The
  one-qubit gate time in QDK's cycle, which QDK calls ancilla preparation, is
  not in those figures. The d cycles per logical step match
  arXiv:1111.4022v3, Sec. 6, which requires d rounds of error correction
  after each operation.
- Error formula. Fowler et al., arXiv:1208.0928v2, Sec. VII, Eqs. (10) and
  (11) (same numbers in v1), give the empirical approximation
  P_L ~ 0.03 (p/p_th)**d_e, with d_e = (d+1)/2 for odd d and d/2 for even d,
  which is ``(d+1)//2``. There P_L is the rate of logical X errors per
  surface-code cycle, and the fit uses p_th = 0.57% for that paper's circuits
  and error model. Its footnote 14 says logical Z errors occur at about the
  same rate. QDK keeps the 0.03 prefactor, uses p_th = 0.01 instead and
  charges the result per lattice-surgery step of d cycles.
- Threshold 0.01. arXiv:1009.3686v1 reports thresholds of 1.1% to 1.4%
  depending on the error model (abstract, and 1.1% for its standard model on
  p. 3). QDK's 0.01 lies below that range. The paper gives no threshold of
  exactly 1%.

``qdk.qre.models.factories.Litinski19Factory`` supplies the published
distillation table (Litinski, arXiv:1905.06903v3). NWQLib fixes the
remaining choices listed in
ENGINEERING_CONSTANTS.md, "Explicit logical compilation and physical
projection", and docs/fault-tolerant-resources.md.
"""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from nwqlib._validation import finite_real, integer
from nwqlib.core.records import ContentID, Nonnegative, Record, Source, Text
from nwqlib.ir.expressions import NonnegativeInt
from .nwqec import LogicalCompilation, Positive


class PhysicalProjectionEntry(Record):
    """One feasible modeled configuration returned by the fixed QDK query.

    configuration_json records its actual code/factory instruction provenance,
    factory copies/runs and dependency properties. Runtime is seconds, not the
    nanoseconds used by the underlying model. Error excludes synthesis fidelity.
    """

    physical_qubits: NonnegativeInt
    runtime_seconds: Nonnegative
    modeled_error: Annotated[Nonnegative, Field(le=1)]
    configuration_json: Text


class PhysicalProjection(Record):
    """Stored conditional projection, never actual hardware execution evidence."""

    compilation_id: ContentID
    qdk_version: Literal["1.32.3"]
    qdk_module_file: Text
    model_source: Source
    logical_counts: tuple[tuple[Text, NonnegativeInt], ...]
    measurement_source: Literal["original_terminal_readout", "supplied_model_assumption"]
    model_json: Text
    requested_synthesis_epsilon: Positive
    synthesis_error_policy: Text
    entries: tuple[PhysicalProjectionEntry, ...]
    statistics: tuple[tuple[Text, NonnegativeInt], ...]
    elapsed_seconds: Nonnegative
    timeout_seconds: Positive
    diagnostics: str
    limitations: tuple[Text, ...]

    def save(self, path, *, max_bytes=DEFAULT_MAX_BYTES):
        """Store the existing projection using bounded archive JSON."""
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        ArchiveFiles(path.parent, max_bytes).write_json(path.name, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path, *, max_bytes=DEFAULT_MAX_BYTES):
        """Read current-format projection data without importing QDK."""
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        return cls.model_validate(ArchiveFiles(path.parent, max_bytes).read_json(path.name))

    def __str__(self):
        """One line per feasible configuration (qubits, seconds, modeled error), then limitations and diagnostics."""
        lines = ["Physical projection under QDK's serial logical-counts model:"]
        if not self.entries:
            lines.append("No feasible result in the fixed distance/factory set.")
        for entry in self.entries:
            lines.append(
                f"{entry.physical_qubits} physical qubits; {entry.runtime_seconds:.8g} s; "
                f"modeled error={entry.modeled_error:.8g}"
            )
        lines.extend(self.limitations)
        if self.diagnostics:
            lines.append(self.diagnostics)
        return "\n".join(lines)


def estimate_physical(
    compilation,
    *,
    measurement_count=None,
    max_error=0.01,
    gate_time_ns=100,
    measurement_time_ns=500,
    timeout_seconds=60,
):
    """Project one full Clifford+T body through the fixed SurfaceCode/factory model.

    An unmeasured body requires an explicit terminal measurement assumption.
    The query uses distances 3, 5 and 7 and the finite published Litinski19
    table. It never broadens an empty result set or changes the selected
    compilation.

    The logical counts are width, T count (T plus TDG) and measurement count.
    Rotation, CCZ and CCiX counts are zero because the admitted gate set
    contains none of them, not because missing quantities are treated as zero.

    Args:
        compilation: A full (not count-only) "clifford_t" LogicalCompilation.
        measurement_count: Assumed terminal measurements for a body without
            its own terminal readout. None is required when the body has one.
        max_error: Total modeled error budget, strictly between 0 and 1.
        gate_time_ns: One- and two-qubit gate time in nanoseconds.
        measurement_time_ns: Measurement time in nanoseconds.
        timeout_seconds: Wall-clock deadline for the QDK child.

    Returns:
        A PhysicalProjection whose entries report runtime in seconds.
    """
    from ._auxiliary_process import run_auxiliary

    if not isinstance(compilation, LogicalCompilation):
        raise TypeError("estimate_physical requires a LogicalCompilation")
    if compilation.target != "clifford_t" or compilation.count_only:
        raise ValueError("physical projection requires a full clifford_t compilation")
    counts = dict(compilation.raw_counts)
    supported = {"id", "h", "x", "y", "z", "s", "sdg", "t", "tdg", "cx", "cz", "swap"}
    if set(counts) - supported:
        raise ValueError("residual unsupported gates prevent Clifford+T counts projection")
    if compilation.width < 1:
        raise ValueError("QDK logical-counts model requires positive width")
    if compilation.terminal_measurements:
        if measurement_count is not None:
            raise ValueError("original terminal measurement mapping cannot be overridden")
        measurements = len(compilation.terminal_measurements)
        measurement_source = "original_terminal_readout"
    else:
        if measurement_count is None:
            raise ValueError(
                "unmeasured body requires explicit measurement_count; simulator saves are not hardware readout"
            )
        measurements = integer(measurement_count, "measurement_count", 0)
        measurement_source = "supplied_model_assumption"
    if measurements > compilation.max_operations:
        raise ValueError("measurement_count exceeds the explicit operation envelope")
    max_error = finite_real(max_error, "max_error")
    timeout_seconds = finite_real(timeout_seconds, "timeout_seconds")
    gate_time_ns = integer(gate_time_ns, "gate_time_ns", 1)
    measurement_time_ns = integer(measurement_time_ns, "measurement_time_ns", 1)
    if not 0 < max_error < 1 or timeout_seconds <= 0:
        raise ValueError("max_error must be in (0,1) and timeout_seconds positive")
    logical = dict(
        numQubits=compilation.width,
        tCount=counts.get("t", 0) + counts.get("tdg", 0),
        measurementCount=measurements,
        rotationCount=0,
        rotationDepth=0,
        cczCount=0,
        ccixCount=0,
    )
    request = dict(
        counts=logical,
        max_error=max_error,
        gate_time_ns=gate_time_ns,
        measurement_time_ns=measurement_time_ns,
    )
    result, elapsed = run_auxiliary(
        "qre", request, max_bytes=DEFAULT_MAX_BYTES, timeout_seconds=timeout_seconds
    )
    return PhysicalProjection(
        compilation_id=compilation.content_id,
        qdk_version=result["version"],
        qdk_module_file=result["module_file"],
        model_source=Source(
            name="QDK QRE",
            version=result["version"],
            domain="conditional GateBased/SurfaceCode/Litinski19 logical-counts model",
            reference=result["module_file"],
        ),
        logical_counts=tuple(sorted(logical.items())),
        measurement_source=measurement_source,
        model_json=json.dumps(result["model"], sort_keys=True, separators=(",", ":")),
        requested_synthesis_epsilon=compilation.effective_epsilon,
        synthesis_error_policy=compilation.rz_err,
        entries=tuple(PhysicalProjectionEntry(**entry) for entry in result["entries"]),
        statistics=tuple(sorted(result["statistics"].items())),
        elapsed_seconds=elapsed,
        timeout_seconds=timeout_seconds,
        diagnostics=result["diagnostics"],
        limitations=(
            "Conditional serial-T/measurement counts model; original Clifford schedule and T depth are omitted.",
            "GateBased error 1e-4 and supplied timings are model assumptions, not measured calibration.",
            "Only distances 3, 5 and 7 and the finite published factory table were considered.",
            "Modeled execution error excludes unverified NWQEC synthesis error and total scientific error.",
        ),
    )


def _estimate_qdk(request):
    """Child-only direct QDK API call, with caches and telemetry disabled."""
    from dataclasses import asdict
    from importlib.metadata import version
    from nwqlib._optional import optional_import

    qdk = optional_import("qdk", extra="qre")
    from qdk.estimator import LogicalCounts
    from qdk.qre import (
        estimate,
        ErrorComposition,
        PSSPC,
        LatticeSurgery,
        instruction_name,
        property_name,
    )
    from qdk.qre.application import QSharpApplication
    from qdk.qre.models.qubits import GateBased
    from qdk.qre.models.qec import SurfaceCode
    from qdk.qre.models.factories import Litinski19Factory

    installed = version("qdk")
    if installed != "1.32.3":
        raise RuntimeError(f"requires qdk[qre]==1.32.3; found {installed}")
    # Fixed model (ENGINEERING_CONSTANTS.md, "Explicit logical compilation and
    # physical projection"). 1e-4 is GateBased's default error rate. PSSPC's
    # num_ts_per_rotation=20 and ccx_magic_states=False equal QDK's defaults
    # and only change rotation and CCX handling, which is empty here. The
    # distances 3, 5 and 7 are a fixed, disclosed subset of the odd distances
    # 3 to 25 that SurfaceCode admits.
    architecture = GateBased(
        error_rate=1e-4,
        gate_time=request["gate_time_ns"],
        two_qubit_gate_time=request["gate_time_ns"],
        measurement_time=request["measurement_time_ns"],
    )
    app = QSharpApplication(
        LogicalCounts(request["counts"]), use_cache=False, use_trace_backend=False
    )
    table = estimate(
        app,
        architecture,
        SurfaceCode.q(distance=[3, 5, 7]) * Litinski19Factory.q(),
        PSSPC.q(num_ts_per_rotation=20, ccx_magic_states=False)
        * LatticeSurgery.q(slow_down_factor=1.0),
        max_error=request["max_error"],
        composition=ErrorComposition.UnionBound,
        use_graph=False,
        post_process=False,
    )
    entries = []
    for row in table:
        # Keep actual instruction provenance without expanding its graph into
        # a repeated tree. Each node keeps its original child indices.
        provenance = dict(
            roots=row.source.roots,
            nodes=[
                dict(
                    instruction=instruction_name(node.instruction.id),
                    transform=None if node.transform is None else type(node.transform).__name__,
                    parameters=None if node.transform is None else asdict(node.transform),
                    children=node.children,
                )
                for node in row.source.nodes
            ],
            factories={
                instruction_name(key): dict(
                    copies=value.copies,
                    runs=value.runs,
                    states=value.states,
                    error_rate=value.error_rate,
                    physical_qubits=row.source[key].instruction.expect_space(),
                    runtime_ns=row.source[key].instruction.expect_time(),
                )
                for key, value in row.factories.items()
            },
            properties={
                property_name(key) or str(key): value for key, value in row.properties.items()
            },
        )
        entries.append(
            dict(
                physical_qubits=row.qubits,
                # QDK reports runtime in nanoseconds. Records store seconds.
                runtime_seconds=row.runtime * 1e-9,
                modeled_error=row.error,
                configuration_json=json.dumps(provenance, sort_keys=True, separators=(",", ":")),
            )
        )
    model = dict(
        architecture={"name": "GateBased", **asdict(architecture)},
        surface_code={
            "name": "SurfaceCode",
            **asdict(SurfaceCode(distance=3)),
            "distance": [3, 5, 7],
        },
        factory="Litinski19Factory published table (Litinski arXiv:1905.06903v3)",
        psspc=asdict(PSSPC(num_ts_per_rotation=20, ccx_magic_states=False)),
        lattice_surgery=asdict(LatticeSurgery(slow_down_factor=1.0)),
        max_error=request["max_error"],
        composition="UnionBound",
        use_graph=False,
        post_process=False,
        use_cache=False,
        use_trace_backend=False,
    )
    return dict(
        version=installed,
        module_file=qdk.__file__,
        model=model,
        entries=entries,
        statistics=asdict(table.stats),
    )
