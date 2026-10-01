"""Portable resource facts and scoped declarations; no SDK or executable laws."""

import math
from typing import Literal

from pydantic import StrictBool, model_validator

from nwqlib.core.records import ContentID, Float64, Limit, Rational, Record, Source, Text
from nwqlib.evidence import Evidence, Fact
from nwqlib.ir.expressions import (
    AdmissionLimits,
    Binary,
    Binding,
    Constant,
    Expression,
    NonnegativeInt,
    Parameter,
    ParameterRef,
    number,
    topological,
)

Metric = Literal[
    "operations",
    "single_qubit",
    "two_qubit",
    "controlled",
    "global_phases",
    "clifford",
    "t",
    "toffoli",
    "ccz",
    "arbitrary_rotations",
    "cx",
    "logical_depth",
    "t_depth",
    "non_clifford_depth",
    "calls",
    "measurements",
    "resets",
    "settings",
    "shots",
    "exact_evaluations",
    "unique_settings",
    "root_setting_declarations",
    "root_repetitions",
    "adaptive_rounds",
    "expected_operations",
    "classical_work",
    "construction_work",
    "preparation_components",
    "logical_width",
    "system",
    "clean_ancilla",
    "dirty_ancilla",
    "classical_registers",
    "classical_bits",
    "memory",
    "known_memory",
    "input_bytes",
    "analysis_bytes",
    "io_bytes",
    "materialization_bytes",
    "stored_bytes",
]
# Count metrics that the fold carries through every step of the Program traversal.
GATES = ("operations", "single_qubit", "two_qubit", "controlled", "global_phases", "clifford",
         "t", "toffoli", "ccz", "arbitrary_rotations", "cx", "logical_depth", "t_depth", "non_clifford_depth")
COUNTS = GATES + ("calls", "measurements", "resets", "settings", "shots", "exact_evaluations", "adaptive_rounds",
                  "classical_work", "preparation_components")
DEPTHS = frozenset(("logical_depth", "t_depth", "non_clifford_depth"))
# The fold repeats the admitted Program traversal and carries one cost per count
# metric, so its work limit is one traversal unit plus one unit per metric for
# each unit of Program admission work. A fold that exceeds it has grown faster
# than the Program and is rejected.
FOLD_WORK_PER_ADMISSION_STEP = 1 + len(COUNTS)
# exact: an exact count, and upper_bound: a ceiling, each only as strong as the
# evidence kind recorded with it (an asserted exact law stays an assertion).
# estimate: derived from observed, empirically predicted or numerically
# estimated evidence. conditional: holds only under a stated premise, such as
# a serial schedule. unavailable: no applicable law, which is never read as zero.
Interpretation = Literal["exact", "upper_bound", "estimate", "conditional", "unavailable"]
# Alternative cost questions about one construction. Quantities in different
# bases are never converted into or added to each other.
MetricBasis = Literal["selected_logical", "cx", "clifford_t", "toffoli"]
# The stored conventional angles that name a one-qubit phase or single-axis
# rotation (docs/resources.md, "Selected laws and logical bases"). A phase or
# Z rotation whose angle equals one of CLIFFORD_ANGLES is the identity, S, Z or
# S-inverse up to global phase, and one whose angle equals one of T_ANGLES is
# T or T-inverse. A rotation about another Pauli axis is a Clifford conjugate
# of the Z rotation by the same angle, so for it an angle in CLIFFORD_ANGLES
# names a Clifford gate and one in T_ANGLES names one T gate up to Clifford
# gates. The comparison is exact on the binary64 value, with no reduction
# modulo 2 pi and no tolerance, so a nearby arbitrary angle is never renamed.
# Every other nonzero angle is an arbitrary rotation. The resource fold's
# primitive recipes (fold._Fold.primitive, phases only) and the QHD rotation
# laws (algorithms.qhd.resources.rotation_population for Rz, Ry and phases,
# and _binary_population for Rz and phases) classify with these sets.
CLIFFORD_ANGLES = frozenset((0.0, math.pi / 2, -math.pi / 2, math.pi, -math.pi,
                             3 * math.pi / 2, -3 * math.pi / 2, 2 * math.pi, -2 * math.pi))
T_ANGLES = frozenset((math.pi / 4, -math.pi / 4))


class ResourceContext(Record):
    """One operation's metric basis, synthesis choice and resident payload scope.

    resident stays live throughout this workload. capacities are declarations,
    kept for a later assessment; the fold never allocates or queries devices.
    batch_schedule selects serial independent acquisitions or leaves their
    concurrency unspecified; it is a planning condition for the consumer.

    Attributes:
        basis: Metric basis in which gate metrics are counted.
        precision: Rotation synthesis precision, required before arbitrary rotations get a T count.
        synthesis: Selected synthesis law. A ResourceLaw applies only under the same choice.
        resident: Payloads live for the whole workload, added to every memory peak.
        capacities: Device capacity declarations kept for a later assessment.
        batch_schedule: ``serial`` discharges the schedule condition on batch peaks.
    """

    basis: MetricBasis = "selected_logical"
    precision: Float64 | None = None
    synthesis: Source | None = None
    resident: tuple["Workspace", ...] = ()
    capacities: tuple[Limit, ...] = ()
    batch_schedule: Literal["unspecified", "serial"] = "unspecified"

    @model_validator(mode="after")
    def _precision(self):
        if self.precision is not None and self.precision.value <= 0:
            raise ValueError("rotation precision must be positive")
        return self


class ResourceLaw(Record):
    """A selected per-call scalar law, conditional on an exact bounded context.

    bindings fix selected cost_parameters and call arguments exactly, except
    unbound_parameters explicitly declares coverage over those formal parameters'
    whole admitted domains. Empty coverage keeps exact-point matching. No
    arbitrary formula parsing or executable callable is admitted. Different
    bases describe alternatives, not additions.

    A law is a scalar per call rather than a formula so that the fold can
    check it, compare it with a known recipe and keep its evidence without
    evaluating caller code.

    Attributes:
        metric: The count metric this law supplies.
        basis: Metric basis the value is counted in.
        value: Per-call value. An exact law requires an integer.
        interpretation: Exact count, upper bound or estimate.
        evidence: Where the value comes from. Observed, empirically predicted or numerically estimated evidence makes it an estimate.
        bindings: Cost parameters and call arguments the law is valid for.
        unbound_parameters: Formal parameters whose whole admitted domain the law covers.
        controlled: Whether the law describes the controlled selection.
        adjoint: Whether the law describes the adjoint selection.
        precision: Rotation precision the law assumes, or None.
        synthesis: Synthesis choice the law assumes, or None.
        assumptions: Conditions carried into every quantity derived from the law.
    """

    metric: Metric
    basis: MetricBasis
    value: NonnegativeInt | Float64
    interpretation: Literal["exact", "upper_bound", "estimate"]
    evidence: Evidence
    bindings: tuple[Binding, ...] = ()
    unbound_parameters: tuple[Text, ...] = ()
    controlled: StrictBool = False
    adjoint: StrictBool = False
    precision: Float64 | None = None
    synthesis: Source | None = None
    assumptions: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _value(self):
        if isinstance(self.value, Float64) and self.value.value < 0:
            raise ValueError("resource laws must be nonnegative")
        if self.interpretation == "exact" and isinstance(self.value, Float64):
            raise ValueError("exact resource inventory requires an integer")
        if len({item.parameter for item in self.bindings}) != len(self.bindings):
            raise ValueError("duplicate resource law binding")
        covered = set(self.unbound_parameters)
        if len(covered) != len(self.unbound_parameters) or covered & {
            item.parameter for item in self.bindings
        }:
            raise ValueError("resource law parameter coverage must be distinct from fixed bindings")
        return self


class Workspace(Record):
    """Declared simultaneous bytes at a location; None preserves missing workspace.

    A selected definition's workspace lasts for each invocation; resident
    payloads in ResourceContext last for the workload. Sizes are declarations,
    not observed allocations, and quantum widths are separately counted.
    """

    location: Text
    purpose: Literal[
        "input", "preparation", "analysis", "io", "materialization", "stored", "workspace"
    ]
    bytes: NonnegativeInt | None
    source: Source


ResourceContext.model_rebuild()


class ResourceQuantity(Record):
    """A scoped metric with per-metric availability and original evidence.

    derivation_sources excludes ancillary conformance from the value's basis.
    required_schedule must be honored before using a batch peak for capacity.

    Attributes:
        metric: Quantity name, such as ``cx``, ``shots`` or ``logical_width``.
        fact: Value or unavailability reason, with unit, scope, conditions and derived evidence kind.
        population: What is counted, for example dynamic workload or simultaneous live footprint.
        lifecycle: Workflow stage of the quantity (planned, prepared, submitted or observed). The fold emits only ``planned``.
        basis: Metric basis of the producing context.
        interpretation: Exact, upper bound, estimate, conditional or unavailable.
        location: Footprint location for located metrics, None for workload totals.
        sources: Evidence identities attached to the value, including conformance checks.
        derivation_sources: Evidence identities that determined the value.
        required_schedule: Schedule the consumer must follow for this peak to hold, or None.
    """

    schema_version: Literal[2] = 2
    metric: Metric
    fact: Fact
    population: Text
    lifecycle: Literal["planned", "prepared", "submitted", "observed"] = "planned"
    basis: Text
    interpretation: Interpretation
    location: Text | None = None
    sources: tuple[ContentID, ...] = ()
    derivation_sources: tuple[ContentID, ...] = ()
    required_schedule: Literal["serial_acquisitions"] | None = None

    def __str__(self):
        """Show this stored quantity and its scope without evaluating a law."""
        fact = self.fact
        if isinstance(fact.value, Rational):
            value = str(fact.value.numerator)
            if fact.value.denominator != 1:
                value += f"/{fact.value.denominator}"
        elif isinstance(fact.value, Float64):
            value = format(fact.value.value, ".8g")
        elif fact.symbol is not None:
            value = f"symbolic ({fact.symbol.name})"
        else:
            value = f"{fact.availability}: {fact.reason}"
        location = "" if self.location is None else f" at {self.location}"
        lines = [
            f"{self.metric}{location}: {value} {fact.unit.symbol} [{self.interpretation}]",
            f"  basis={self.basis}; lifecycle={self.lifecycle}; population={self.population}",
        ]
        if self.required_schedule:
            lines.append(f"  required schedule: {self.required_schedule}")
        if fact.assumptions:
            lines.append("  conditions: " + "; ".join(fact.assumptions))
        return "\n".join(lines)

    @model_validator(mode="after")
    def _availability(self):
        """Keep the quantity's value, unit and evidence consistent with its meaning.

        The interpretation is ``unavailable`` exactly when the Fact is
        unknown or not applicable. A concrete value must be a real Rational
        or Float64 and nonnegative, and an exact value must be an integer
        cardinality, except ``expected_operations``, which may be fractional.
        Byte metrics use the byte unit and all others the count unit. The
        Fact names this metric, and every derivation source is also a kept
        source.
        """
        unavailable = self.fact.availability in {"unknown", "not_applicable"}
        if unavailable != (self.interpretation == "unavailable"):
            raise ValueError("resource interpretation must agree with fact availability")
        if self.fact.availability == "concrete":
            value = self.fact.value
            if not isinstance(value, (Rational, Float64)):
                raise ValueError("resource quantities must be real scalars")
            if (value.numerator if isinstance(value, Rational) else value.value) < 0:
                raise ValueError("resource quantities must be nonnegative")
            integral = (
                value.denominator == 1 if isinstance(value, Rational) else value.value.is_integer()
            )
            if (
                self.interpretation == "exact"
                and self.metric != "expected_operations"
                and not integral
            ):
                raise ValueError("exact resource cardinalities must be integral")
        byte_metric = self.metric.endswith("bytes") or self.metric in {"memory", "known_memory"}
        if (self.fact.unit.symbol, self.fact.unit.dimension) != (
            ("byte", "bytes") if byte_metric else ("count", "count")
        ):
            raise ValueError("resource metric/unit mismatch")
        if self.fact.quantity != self.metric:
            raise ValueError("Fact quantity must match resource metric")
        if not set(self.derivation_sources) <= set(self.sources):
            raise ValueError("derivation sources must be included in the kept source references")
        return self


class WorkloadEstimate(Record):
    """Fold of one selected construction, with bounded shared symbolic definitions.

    sources preserve original evidence declarations; arithmetic does not witness
    them. defined_selections is the reachable potential inventory, not a native
    construction receipt. work_units charges kept fold contexts and outputs,
    not runtime. expressions use local refs and never expand Repeat or sharing.
    Terminal exact_evaluations counts grouped statistic requests, not returned
    labels/bins or simulator trajectories. Direct non-batch programs encode no
    acquisition population: zero shots/settings does not imply a free native run.

    Attributes:
        construction_id: Identity of the folded SelectedConstruction.
        context: The ResourceContext that fixed basis, precision, synthesis and schedule.
        quantities: One quantity per (metric, location), including unavailable ones.
        expressions: Local symbolic cost definitions referenced by symbolic facts.
        parameters: Program parameters those expressions may reference.
        sources: Original evidence records referenced by the quantities.
        defined_selections: Identities of potentially reached selected definitions and host kernels.
        work_units: Fold bookkeeping work, bounded by the Program's admission limits.
        evaluated_contexts: Distinct node and call contexts folded.
        limits: AdmissionLimits of the folded Program.
        assumptions: Program premises, readiness blockers and the fold's own scope conditions.
    """

    schema_version: Literal[2] = 2
    construction_id: ContentID
    context: ResourceContext
    quantities: tuple[ResourceQuantity, ...]
    expressions: tuple[Expression, ...] = ()
    parameters: tuple[Parameter, ...] = ()
    sources: tuple[Evidence, ...] = ()
    defined_selections: tuple[ContentID, ...] = ()
    work_units: NonnegativeInt
    evaluated_contexts: NonnegativeInt
    limits: AdmissionLimits
    assumptions: tuple[Text, ...] = ()

    @model_validator(mode="after")
    def _references(self):
        """Check a loaded estimate's bookkeeping and that every reference resolves inside it.

        work_units may not exceed ``FOLD_WORK_PER_ADMISSION_STEP * max_steps``,
        the largest fold ceiling an admitted Program can reach, and
        evaluated_contexts may not exceed work_units. The expression table
        must be an acyclic graph within the admission limits, with declared
        parameters and integer constants within ``max_integer_bits``. Each
        (metric, location) pair appears once, every source
        identity names a stored Evidence record and every symbolic value names
        a stored expression.
        """
        if (self.work_units > FOLD_WORK_PER_ADMISSION_STEP * self.limits.max_steps
                or self.evaluated_contexts > self.work_units):
            raise ValueError("resource work/context count exceeds its admission envelope")
        table = {item.id: item.value for item in self.expressions}
        params = {item.name for item in self.parameters}
        sources = {item.content_id for item in self.sources}
        if len(table) != len(self.expressions) or len(params) != len(self.parameters):
            raise ValueError("duplicate resource expression/parameter")
        if len(table) + len(params) > self.limits.max_definitions:
            raise ValueError("resource definitions exceed admission limit")
        topological(
            table,
            lambda node: (node.left, node.right) if isinstance(node, Binary) else (),
            self.limits,
        )
        for node in table.values():
            if isinstance(node, ParameterRef) and node.parameter not in params:
                raise ValueError("undeclared resource parameter")
            if isinstance(node, Constant):
                value = number(node.value)
                if type(value) is int and value.bit_length() > self.limits.max_integer_bits:
                    raise ValueError("resource constant exceeds integer limit")
        identities = set()
        for quantity in self.quantities:
            key = (quantity.metric, quantity.location)
            if key in identities:
                raise ValueError("duplicate resource metric/location")
            identities.add(key)
            if not set(quantity.sources) <= sources:
                raise ValueError("unresolved resource evidence reference")
            if quantity.fact.symbol is not None and quantity.fact.symbol.name not in table:
                raise ValueError("unresolved resource expression reference")
        return self

    def quantity(self, metric: str, *, location: str | None = None) -> ResourceQuantity:
        """Read one metric without recomputation, synthesis or serialization."""
        matches = [
            item for item in self.quantities if item.metric == metric and item.location == location
        ]
        if len(matches) != 1:
            nearby = []
            for item in self.quantities:
                if item.metric == metric:
                    nearby.append(repr(item.location)[:80])
                    if len(nearby) == 8:
                        break
            raise ValueError(f"no resource quantity for metric={metric!r}, location={location!r}; "
                             f"available locations for this metric (up to 8): {', '.join(nearby) or 'none'}; "
                             "inspect .quantities for declared metric/location pairs")
        return matches[0]

    def __str__(self):
        """Display the existing fold, including unavailable costs and conditions."""
        lines = [f"Selected resources ({self.context.basis}):"]
        lines.extend(str(quantity) for quantity in self.quantities)
        if self.assumptions:
            lines.append("Conditions: " + "; ".join(self.assumptions))
        return "\n".join(lines)
