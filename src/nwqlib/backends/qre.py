"""Projection, on request, of the Clifford+T counts of one circuit execution to physical resources with one QDK counts model.

The counts come from a saved full Clifford+T compilation (``LogicalCompilation``)
or from a resource estimate in the ``clifford_t`` basis (``WorkloadEstimate``).

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
  applies the result per lattice-surgery step of d cycles.
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

import json
from fractions import Fraction
from pathlib import Path
from typing import Annotated, Literal

from pydantic import Field

from nwqlib._validation import finite_real, integer, round_up
from nwqlib.core.records import ContentID, Float64, Nonnegative, Rational, Record, Source, Text
from nwqlib.ir.expressions import NonnegativeInt
from nwqlib.resources.records import WorkloadEstimate
from .nwqec import LogicalCompilation, Positive


class PhysicalProjectionEntry(Record):
    """One feasible modeled configuration returned by the fixed QDK query.

    configuration_json records the code and factory instructions it came from,
    the factory copies and runs, and the dependency properties. Runtime is seconds, not the
    nanoseconds used by the underlying model. Error excludes synthesis fidelity.
    """

    physical_qubits: NonnegativeInt
    runtime_seconds: Nonnegative
    modeled_error: Annotated[Nonnegative, Field(le=1)]
    configuration_json: Text


class PhysicalProjection(Record):
    """A saved conditional projection, never evidence of execution on hardware.

    `count_source` is `"logical_compilation"` for the exact counts of a
    `LogicalCompilation` and `"resource_estimate"` for the counts of a
    `WorkloadEstimate`, and `source_id` is that record's content hash.
    `count_population` states what the counts describe, always one circuit
    execution, and `t_interpretation` how to read the T count (`"exact"` for
    a compilation, the estimate's label otherwise). `requested_synthesis_epsilon`
    is the requested rotation precision, None for an estimate without one.
    `synthesis_error_policy` is the compilation's `rz_err`, or
    `"resource_context_per_rotation"` for an estimate, whose precision applies
    to each rotation. An estimate also records its `synthesis_source`, its
    count of arbitrary rotations (`rotation_count`) and, when both are
    known, `rotation_error_sum`, the upward-rounded product of the two, which
    bounds the rotations' summed synthesis error only if each rotation meets
    the precision and which `modeled_error` excludes.
    """

    count_source: Literal["logical_compilation", "resource_estimate"]
    source_id: ContentID
    count_population: Text
    t_interpretation: Text
    qdk_version: Literal["1.32.3"]
    qdk_module_file: Text
    model_source: Source
    logical_counts: tuple[tuple[Text, NonnegativeInt], ...]
    measurement_source: Literal["original_terminal_readout", "supplied_model_assumption"]
    model_json: Text
    requested_synthesis_epsilon: Positive | None
    synthesis_error_policy: Text
    synthesis_source: Source | None = None
    rotation_count: NonnegativeInt | None = None
    rotation_error_sum: Nonnegative | None = None
    entries: tuple[PhysicalProjectionEntry, ...]
    statistics: tuple[tuple[Text, NonnegativeInt], ...]
    elapsed_seconds: Nonnegative
    timeout_seconds: Positive | None
    diagnostics: str
    limitations: tuple[Text, ...]

    def save(self, path, *, max_bytes=None):
        """Save this projection as JSON, with at most ``max_bytes`` written bytes, or no limit when it is None."""
        from nwqlib._choice_archive import ArchiveFiles

        path = Path(path)
        ArchiveFiles(path.parent, max_bytes).write_json(path.name, self.model_dump(mode="json"))

    @classmethod
    def load(cls, path, *, max_bytes=None):
        """Read saved projection data in the current format without importing QDK.

        ``max_bytes`` bounds the bytes read, and None sets no limit.
        """
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


def _whole(quantity, *, exact=False):
    """The integer value of a resource quantity, or None when it is unavailable, symbolic, fractional or, with ``exact``, not exact."""
    value = quantity.fact.value
    if exact and quantity.interpretation != "exact":
        return None
    if isinstance(value, Rational) and value.denominator == 1:
        return value.numerator
    if isinstance(value, Float64) and value.value.is_integer():
        return int(value.value)
    return None


def _compiled_counts(compilation):
    """Width, T count, terminal measurement count and record fields of a full clifford_t compilation."""
    if compilation.target != "clifford_t" or compilation.count_only:
        raise ValueError("physical projection requires a full clifford_t compilation")
    counts = dict(compilation.raw_counts)
    # QDK's counts have no Clifford field, so each admitted label other than t and tdg adds nothing. SX = H S H and
    # SXdg = H Sdg H, phase included, are Clifford. A controlled SX is not, and this list reads no angles, so a rotation
    # label stays refused even at a Clifford angle. Admitting a non-Clifford gate would understate the T count.
    supported = {"id", "h", "x", "y", "z", "s", "sdg", "sx", "sxdg", "t", "tdg", "cx", "cz", "swap"}
    if set(counts) - supported:
        raise ValueError("residual unsupported gates prevent Clifford+T counts projection")
    return (compilation.width, counts.get("t", 0) + counts.get("tdg", 0), len(compilation.terminal_measurements),
            dict(count_source="logical_compilation", source_id=compilation.content_id,
                 count_population="one static compiler body", t_interpretation="exact",
                 requested_synthesis_epsilon=compilation.effective_epsilon,
                 synthesis_error_policy=compilation.rz_err))


def _estimated_counts(workload):
    """Width, T count, measurement count and record fields of a clifford_t estimate of one circuit execution.

    QDK's counts describe one execution of one circuit, while an estimate's
    counts total its whole workload. The estimate counts one execution when
    its ``settings``, ``shots`` plus ``exact_evaluations`` and
    ``adaptive_rounds`` are exact and equal (0, 0, 0), a construction without
    a terminal measurement batch, or (1, 1, 0), one setting run once.
    """
    if workload.context.basis != "clifford_t":
        raise ValueError("physical projection of a resource estimate requires counts in the clifford_t basis")
    settings, shots, exact, rounds = (_whole(workload.quantity(metric), exact=True)
                                      for metric in ("settings", "shots", "exact_evaluations", "adaptive_rounds"))
    population = None if None in (settings, shots, exact, rounds) else (settings, shots + exact, rounds)
    if population not in {(0, 0, 0), (1, 1, 0)}:
        if settings == 1 and rounds == 0:
            remedy = "Plan the circuit with shots=1 or with exact readout, or compile it with compile_logical"
        else:
            remedy = (f"This estimate has settings={settings} and adaptive_rounds={rounds}, where None is not an "
                      "exact count, so changing its shots or readout cannot make it one execution. Compile each "
                      "prepared circuit with compile_logical(prepared, index=...) to count it on its own")
        raise ValueError("physical projection needs the counts of one circuit execution: the estimate's settings, "
                         "shots plus exact_evaluations and adaptive_rounds must be exact and equal 0, 0 and 0, or "
                         f"1, 1 and 0. {remedy}")
    t = workload.quantity("t")
    t_count = _whole(t)
    if t_count is None:
        raise ValueError(f"the estimate's T count is not a known integer ({t.fact.reason or t.interpretation}); "
                         "compile the circuit with compile_logical")
    width = _whole(workload.quantity("logical_width", location="logical_device"))
    if width is None:
        raise ValueError("the estimate's logical width is not a known integer")
    measured = _whole(workload.quantity("measurements"))
    if measured is None:
        raise ValueError("the estimate's measurement count is not a known integer")
    rotations = _whole(workload.quantity("arbitrary_rotations"))
    precision = workload.context.precision
    epsilon = None if precision is None else precision.value
    return width, t_count, measured, dict(
        count_source="resource_estimate", source_id=workload.content_id,
        count_population=f"one circuit execution (settings={settings}, acquisitions={shots + exact}, "
                         "adaptive_rounds=0)",
        t_interpretation=t.interpretation, requested_synthesis_epsilon=epsilon,
        synthesis_error_policy="resource_context_per_rotation", synthesis_source=workload.context.synthesis,
        rotation_count=rotations,
        rotation_error_sum=None if rotations is None or epsilon is None else round_up(
            Fraction(rotations) * Fraction(epsilon)))


def estimate_physical(
    compilation,
    *,
    measurement_count=None,
    max_error=0.01,
    gate_time_ns=100,
    measurement_time_ns=500,
    timeout_seconds=None,
):
    """Project the Clifford+T counts of one circuit execution through the fixed SurfaceCode/factory model.

    The counts come from a full Clifford+T compilation, exactly, or from a
    resource estimate in the `clifford_t` basis without compiling. An
    unmeasured body requires an explicit terminal measurement assumption.
    The query uses distances 3, 5 and 7 and the finite published Litinski19
    table. It never broadens an empty result set or changes `compilation`.

    The logical counts are width, T count and measurement count. For a
    compilation the T count is T plus TDG. Rotation, CCZ and CCiX counts are
    zero: a compilation's accepted gate set contains none of them, and an
    estimate's rotations reach QDK as T gates, because QDK's `PSSPC`
    transform uses a fixed 20 T states per rotation whatever the precision.
    Missing quantities are refused, not treated as zero.

    Args:
        compilation: A full (not count-only) `"clifford_t"`
            `LogicalCompilation`, or a `WorkloadEstimate` counted with
            `ResourceContext(basis="clifford_t")` that describes one circuit
            execution. Its `settings`, `shots` plus `exact_evaluations` and
            `adaptive_rounds` must be exact and equal 0, 0 and 0 or 1, 1 and
            0, and its T count, logical width and measurement count must be
            known integers. A branch inside the execution keeps these exact
            when its alternatives agree on them.
        measurement_count: Assumed terminal measurements for a body without
            its own terminal readout (no terminal measurements of a
            compilation, an estimated measurement count of 0). None is
            required when the body has one.
        max_error: Total modeled error budget, strictly between 0 and 1.
        gate_time_ns: One- and two-qubit gate time in nanoseconds.
        measurement_time_ns: Measurement time in nanoseconds.
        timeout_seconds: Default None, no deadline. Positive wall-clock
            deadline for the QDK child. The buffers that this process holds
            for the child, its request, its captured diagnostics with their
            decoded text and its result, are limited to the physical memory
            of this computer, with no parameter to change it.

    Returns:
        A PhysicalProjection whose entries report runtime in seconds.

    Raises:
        TypeError: If `compilation` is neither record.
        ValueError: If a compilation is count-only, not `clifford_t` or has
            gates outside Clifford+T, if an estimate is in another basis,
            counts more than one execution or has an unknown T count, width
            or measurement count, if the width is not positive, or if
            `measurement_count` is missing for an unmeasured body or given
            for a measured one.
    """
    from ._auxiliary_process import optional_timeout, run_auxiliary

    if isinstance(compilation, LogicalCompilation):
        width, t_count, measured, fields = _compiled_counts(compilation)
    elif isinstance(compilation, WorkloadEstimate):
        width, t_count, measured, fields = _estimated_counts(compilation)
    else:
        raise TypeError("estimate_physical requires a LogicalCompilation or a WorkloadEstimate")
    if width < 1:
        raise ValueError("QDK logical-counts model requires positive width")
    if measured:
        if measurement_count is not None:
            raise ValueError("original terminal measurement mapping cannot be overridden")
        measurements = measured
        measurement_source = "original_terminal_readout"
    else:
        if measurement_count is None:
            raise ValueError(
                "unmeasured body requires explicit measurement_count; simulator saves are not hardware readout"
            )
        measurements = integer(measurement_count, "measurement_count", 0)
        measurement_source = "supplied_model_assumption"
    max_error = finite_real(max_error, "max_error")
    timeout_seconds = optional_timeout(timeout_seconds)
    gate_time_ns = integer(gate_time_ns, "gate_time_ns", 1)
    measurement_time_ns = integer(measurement_time_ns, "measurement_time_ns", 1)
    if not 0 < max_error < 1:
        raise ValueError("max_error must be in (0,1)")
    logical = dict(
        numQubits=width,
        tCount=t_count,
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
        "qre", request, max_bytes=None, timeout_seconds=timeout_seconds
    )
    limitations = (
        "Conditional serial-T/measurement counts model; original Clifford schedule and T depth are omitted.",
        "GateBased error 1e-4 and supplied timings are model assumptions, not measured calibration.",
        "Only distances 3, 5 and 7 and the finite published factory table were considered.",
        "Modeled execution error excludes rotation synthesis error and total scientific error.",
    )
    if fields["count_source"] == "resource_estimate":
        limitations += (f"Counts of one circuit execution from a resource estimate; T count "
                        f"{fields['t_interpretation']}.",)
        if fields["rotation_error_sum"] is not None and fields["rotation_count"]:
            limitations += (f"If each of the {fields['rotation_count']} arbitrary rotations meets precision "
                            f"{fields['requested_synthesis_epsilon']:.8g}, their synthesis errors add to at most "
                            f"{fields['rotation_error_sum']:.8g}, outside modeled error.",)
    return PhysicalProjection(
        **fields,
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
        entries=tuple(PhysicalProjectionEntry(**entry) for entry in result["entries"]),
        statistics=tuple(sorted(result["statistics"].items())),
        elapsed_seconds=elapsed,
        timeout_seconds=timeout_seconds,
        diagnostics=result["diagnostics"],
        limitations=limitations,
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
