"""Portable resource facts and scoped declarations, without SDK objects or executable laws."""

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
# GATES are the gate-count and depth metrics that a recipe or a per-call law
# supplies. COUNTS adds the other count metrics, and the fold carries every
# COUNTS metric through each step of the Program traversal.
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
# rotation (docs/development/execution.md, "Resource estimate bookkeeping"). A phase or
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
    """Settings that fix how a resource estimate counts: gate basis, rotation precision, synthesis, resident data and schedule.

    Build it with keyword arguments, for example `ResourceContext(basis="cx")`,
    and pass it as `context=` to [`nwqlib.estimate`][nwqlib.scientist.estimate]
    or [`estimate`][nwqlib.resources.fold.estimate]. Every argument is
    optional, and `ResourceContext()` counts in the `"selected_logical"`
    basis.

    Attributes:
        basis: Default `"selected_logical"`, which counts the planned
            construction's own operations. Basis in which gate metrics are
            counted: `"selected_logical"`, `"cx"`, `"clifford_t"` or
            `"toffoli"`. Counts in different bases answer different
            questions and are never converted into or added to each other.
        precision: Default `None`. Positive rotation synthesis precision, in
            the sense that `synthesis` defines. An arbitrary rotation gets a T
            count only when the precision and the synthesis are known, and a
            block's `ResourceLaw` applies only when its `precision` equals
            this value.
        synthesis: Default `None`. Source naming the synthesis method. A
            block's [`ResourceLaw`][nwqlib.resources.records.ResourceLaw]
            applies only under the same choice.
        resident: Default `()`. [`Workspace`][nwqlib.resources.records.Workspace]
            entries that stay in memory for the whole workload and are added
            to every memory peak.
        capacities: Default `()`. Device capacities as `Limit` records, kept
            for a later device assessment. The estimate itself allocates
            nothing and queries no device.
        batch_schedule: Default `"unspecified"`. `"serial"` declares that
            independent circuit batches run one at a time, so a batch memory
            peak holds without condition. With `"unspecified"` such a peak is
            labeled `conditional`. The required schedule stays recorded for
            the execution to follow.

    Raises:
        ValueError: If `precision` is not positive.
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
    """The cost of one call of a circuit block in one metric, as the block declares it.

    A circuit block of a Plan can declare cost rules for its metrics. The
    estimate uses a rule only for a call whose control and adjoint flags, cost
    parameters and call arguments, basis, precision and synthesis all match.
    The value is a number per call, not a formula, so the estimate can check
    it against a known gate recipe and keep its source without running
    caller code. Rules in different bases are alternatives, not additions.
    Build it with keyword arguments when you write a block (see [Compose
    blocks](../blocks.md)). `metric`, `basis`, `value`, `interpretation` and
    `evidence` are required.

    Attributes:
        metric: Required. The count metric the rule supplies.
        basis: Required. Basis the value is counted in.
        value: Required. Nonnegative per-call value, an integer when
            `interpretation` is `"exact"`.
        interpretation: Required. `"exact"`, `"upper_bound"` or
            `"estimate"`.
        evidence: Required. Where the value comes from. Observed,
            empirically predicted or numerically estimated evidence makes
            every quantity derived from it an estimate.
        bindings: Default `()`. Cost parameters and call arguments that must
            match exactly for the rule to apply.
        unbound_parameters: Default `()`. Formal parameters whose whole
            accepted range the rule covers instead of one value. Empty means
            exact matching only.
        controlled: Default `False`. Whether the rule is for the controlled
            block.
        adjoint: Default `False`. Whether the rule is for the adjoint block.
        precision: Default `None`. Rotation precision the rule assumes.
        synthesis: Default `None`. Synthesis method the rule assumes.
        assumptions: Default `()`. Conditions carried into every quantity
            derived from the rule.

    Raises:
        ValueError: If the value is negative, an exact value is not an
            integer, a binding is repeated, or a parameter is both bound and
            listed in `unbound_parameters`.
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
    """Bytes needed at the same time at one location, as declared by a block or the caller.

    A block's workspace lasts for each call. Entries in
    `ResourceContext.resident` last for the whole workload. Sizes are
    declarations, not observed allocations, and quantum widths are counted
    separately. Build it with keyword arguments. All four fields are
    required.

    Attributes:
        location: Where the bytes live, for example `"host"`.
        purpose: `"input"`, `"preparation"`, `"analysis"`, `"io"`,
            `"materialization"`, `"stored"` or `"workspace"`.
        bytes: Nonnegative byte count, or `None` when unknown. An unknown
            size makes the memory peak at that location unavailable, never
            zero.
        source: Source of the declaration.
    """

    location: Text
    purpose: Literal[
        "input", "preparation", "analysis", "io", "materialization", "stored", "workspace"
    ]
    bytes: NonnegativeInt | None
    source: Source


ResourceContext.model_rebuild()


class ResourceQuantity(Record):
    """One count or peak of a resource estimate, with its value, label and sources.

    [`WorkloadEstimate.quantity`][nwqlib.resources.records.WorkloadEstimate.quantity]
    returns it. The value is `fact.value`, an exact `Rational` (read
    `numerator` and `denominator`) or a `Float64`, in the unit `fact.unit`:
    `count`, or `byte` for byte metrics. A symbolic value is
    `fact.symbol`, and an unavailable one has `fact.reason` and no value.
    `interpretation` says how to read the value (see [Read a
    quantity](resources.md#read-a-quantity)). `print(quantity)` shows the value, label
    and conditions. The fields below are read-only.

    Attributes:
        metric: Name of the quantity, such as `cx`, `shots` or
            `logical_width`.
        fact: The value, or the reason it is unavailable, with its unit,
            scope, conditions (`fact.assumptions`) and the weakest evidence
            kind among the sources that determined it.
        population: What is counted, for example `"dynamic workload"` for a
            total over the whole run or `"simultaneous live footprint"` for a
            peak.
        lifecycle: Workflow stage of the quantity: `"planned"`,
            `"prepared"`, `"submitted"` or `"observed"`. A planning estimate
            gives only `"planned"`.
        basis: Gate basis of the estimate.
        interpretation: `"exact"`, `"upper_bound"`, `"estimate"`,
            `"conditional"` or `"unavailable"`.
        location: Location of a width or byte peak, `None` for a total.
        sources: Content hashes of all evidence attached to the value,
            including evidence used only to check a gate recipe against a
            declared rule.
        derivation_sources: Content hashes of the evidence that determined the
            value, a subset of `sources`.
        required_schedule: `"serial_acquisitions"` when this peak holds only
            if independent circuit batches run one at a time, otherwise
            `None`. Follow it before comparing the peak with a device
            capacity.
        schema_version: Format version of the record, 2.

    Raises:
        ValueError: When a record is loaded whose label disagrees with its
            availability, whose value is negative, not real or, when exact,
            not an integer (`expected_operations` may be fractional), whose
            unit does not match the metric, or whose derivation sources are
            not among its sources.
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
    """Resource estimate of a planned computation: one quantity per metric and location.

    [`nwqlib.estimate(plan)`][nwqlib.scientist.estimate] returns it when no
    device profile is given, and so does [`estimate`][nwqlib.resources.fold.estimate].
    Read one metric with `quantity(metric)`, or a width or byte peak with
    `quantity(metric, location=...)`, and list everything with `quantities`
    or `print(workload)`. A metric without a cost rule is present and
    labeled `unavailable`, never zero. The estimate is of the logical
    circuit description, before compilation, routing or error correction.

    `exact_evaluations` counts grouped exact-statistic requests, not returned
    labels, bins or simulator trajectories. A circuit with no terminal
    measurement batch declares no measurements, so zero `shots` and
    `settings` there do not mean that running it is free. The fields below
    are read-only.

    Attributes:
        construction_id: Content hash of the planned circuit description
            that was estimated.
        context: The [`ResourceContext`][nwqlib.resources.records.ResourceContext]
            that fixed basis, precision, synthesis and schedule.
        quantities: One [`ResourceQuantity`][nwqlib.resources.records.ResourceQuantity]
            per (metric, location), including unavailable ones.
        expressions: Symbolic cost expressions that symbolic quantities
            name. Repeats and shared parts are not expanded.
        parameters: Circuit parameters those expressions may use.
        sources: The original evidence records of the cost rules. The
            arithmetic of the estimate does not make them verified.
        defined_selections: Content hashes of the circuit blocks and classical
            kernels the run may reach, not a record of built circuits.
        work_units: Bookkeeping work of the estimate, not run time, at most
            `24 * limits.max_steps`.
        evaluated_contexts: Distinct node and call contexts evaluated.
        limits: The size limits of the planned circuit description.
        assumptions: Conditions of the whole estimate, such as
            `"logical dependency schedule; no physical QEC/routing/factory projection"`.
        schema_version: Format version of the record, 2.

    Examples:
        A two-dimensional Lanczos Plan, estimated in the CX basis, has an
        exact width of two qubits and an upper bound of three CX gates. No
        rule gives its total operation count:

        >>> import nwqlib
        >>> from nwqlib import Eigenproblem
        >>> from nwqlib.algorithms import Lanczos
        >>> from nwqlib.resources import ResourceContext
        >>> problem = Eigenproblem(A=[[1.5, -1], [-1, 0.5]])
        >>> method = Lanczos(initial_state=[1, 0], krylov_dimension=2)
        >>> selected = nwqlib.plan(problem, method=method, seed=7)
        >>> context = ResourceContext(basis="cx")
        >>> workload = nwqlib.estimate(selected, context=context)
        >>> cx = workload.quantity("cx")
        >>> cx.interpretation, cx.fact.value.numerator
        ('upper_bound', 3)
        >>> width = workload.quantity("logical_width", location="logical_device")
        >>> width.interpretation, width.fact.value.numerator
        ('exact', 2)
        >>> workload.quantity("operations").interpretation
        'unavailable'
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
        """Return the quantity of one metric at one location, without recomputing anything.

        Args:
            metric (str): Metric name, such as `"cx"` or `"logical_width"`.
            location (str | None): Location of a width or byte peak, for
                example `"logical_device"` or `"host"`. `None`, the default,
                selects a total.

        Returns:
            quantity (ResourceQuantity): The stored quantity.

        Raises:
            ValueError: If no quantity has this metric and location. The
                message lists up to eight locations of the metric.
        """
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
