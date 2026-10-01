"""Read-only assessment of one selected point against supplied device facts.

The shared resource fold owns inventories. This module compares those facts to
allocation stocks and evaluates only the finite models in profiles.py. It never
selects a recipe, prepares a circuit, reads a payload or refreshes a profile.

One assessment resolves the Plan's selected construction for one Realization,
folds it once (``_select_workload``) and reports five independent axes.
Applicability reads the planner's own recorded premise. Capability compares
the selected artifact, readout, Program nodes and exact instructions with the
stored target. Capacity compares folded byte peaks and lower requirements with
each granted location's stock. Time evaluates the supplied linear time models
on counted acquisition features. Accuracy is the Plan's ErrorModel assessment
of the selected criterion. Capability, capacity and time all read the same
fold. Each conclusion keeps its scope and assumptions, and an axis status
summarizes only its own details (``_outcome``), so a memory failure cannot
hide a supported error bound. Time predictions stay conditional because they
rest on supplied coefficients, so none becomes an execution guarantee.
docs/profiles.md defines these axes and the time-model scopes.
"""

from datetime import datetime, timezone
from fractions import Fraction
from math import fsum, isfinite
from typing import Literal, NamedTuple

from pydantic import PrivateAttr, model_validator

from nwqlib.backends.capabilities import BackendCapability
from nwqlib.backends.profiles import (
    Allocation,
    DeviceProfile,
    ModelUncertainty,
    TimeModel,
    TimeScope,
    Timestamp,
)
from nwqlib.blocks.records import SelectedConstruction
from nwqlib.core.planning import Experiment, Plan, Realization, RuntimeOptions, readout_shape
from nwqlib.core.records import (
    ContentID,
    Float64,
    Limit,
    Rational,
    Record,
    Scope,
    Source,
    Text,
    Unit,
)
from nwqlib.evidence.records import Evidence, Fact
from nwqlib.evidence.error_model import (
    AssessmentContext,
    ClaimAssessment,
    ErrorFrame,
    FramedFact,
    TargetReference,
    predicate_value,
)
from nwqlib.ir import MeasurementBatch, Readiness
from nwqlib.ir.expressions import number
from nwqlib.ir.validation import _Admission, children
from nwqlib.problems.records import Accuracy
from nwqlib.resources import ResourceContext, ResourceQuantity, WorkloadEstimate, estimate

Outcome = Literal["feasible", "infeasible", "conditional", "unknown"]
Axis = Literal[
    "scientific_applicability",
    "execution_capability",
    "capacity_materialization",
    "time_cost",
    "accuracy_evidence",
]

_ASSESSMENT_SOURCE = Source(
    name="stored_profile_assessment",
    version="1",
    domain="bounded metadata comparisons at one selected Plan/Realization",
    reference="nwqlib.backends.assessment",
)


class AssessmentDetail(Record):
    """One scoped conclusion of a device forecast, with the value and limit it compares.

    Read it from `AxisAssessment.details`. The fields below are read-only. A
    detail holds either a resource value (`fact`) or a method statement
    (`statement`), never both, because they are different quantities.

    Attributes:
        quantity: The quantity compared, such as a memory peak.
        scope: Where the quantity applies, such as one granted location.
        status: `"feasible"`, `"infeasible"`, `"conditional"` or `"unknown"`.
        reason: Why the detail has its status.
        fact: The resource value compared, or `None`.
        resource_id: Content hash of the resource quantity in the estimate, or
            `None`.
        prediction_ids: Content hashes of the time predictions this detail
            summarizes.
        limit: The supplied `Limit` it was compared with, unchanged, or `None`.
        evidence_ids: Content hashes of the declarations and records used.
        assumptions: Assumptions the conclusion rests on.
        statement: The method's complete stated assumption or claim, or `None`.
    """

    quantity: Text
    scope: Text
    status: Outcome
    reason: Text
    fact: Fact | None = None
    resource_id: ContentID | None = None
    prediction_ids: tuple[ContentID, ...] = ()
    limit: Limit | None = None
    evidence_ids: tuple[ContentID, ...] = ()
    assumptions: tuple[Text, ...] = ()
    statement: FramedFact | None = None

    @model_validator(mode="after")
    def _fact_owner(self):
        """Allow a planner statement or a resource fact, never both, since they are different quantities."""
        if self.statement is not None and self.fact is not None:
            raise ValueError("a detail keeps one scientific statement or one resource fact")
        return self


class AxisAssessment(Record):
    """One axis of a device forecast: its status and every fact behind it.

    Read it from the `applicability`, `capability`, `capacity`, `time` and
    `accuracy` fields of a
    [`ProfileAssessment`][nwqlib.backends.assessment.ProfileAssessment]. The fields
    below are read-only. The status summarizes the details of this axis only. A
    known failure (`"infeasible"`) decides the axis. Otherwise any
    `"conditional"` detail makes it conditional, then any `"unknown"` detail
    makes it unknown, and it is `"feasible"` only when every detail is.

    Attributes:
        axis: `"scientific_applicability"`, `"execution_capability"`,
            `"capacity_materialization"`, `"time_cost"` or `"accuracy_evidence"`.
        status: `"feasible"`, `"infeasible"`, `"conditional"` or `"unknown"`.
        details: The [`AssessmentDetail`][nwqlib.backends.assessment.AssessmentDetail]
            records, partial knowledge included.
    """

    axis: Axis
    status: Outcome
    details: tuple[AssessmentDetail, ...]

    @model_validator(mode="after")
    def _status(self):
        """Require the stored status to be ``_outcome`` of the stored details, so it cannot be edited alone."""
        if not self.details or self.status != _outcome(self.details):
            raise ValueError("axis status must summarize its nonempty scoped details")
        return self


class TimePrediction(Record):
    """The prediction of one time model for one point, in seconds, with its interval and the feature counts it used.

    Read it from `ProfileAssessment.predictions`. The fields below are read-only.
    The answer is `seconds.value`, a prediction for the model's `scope`, not an
    observed elapsed time and not permission to run. When an input is missing,
    `seconds` is `None` and `reasons` says why. An interval keeps the kind and
    coverage of the model's uncertainty, so a confidence interval for a mean
    never becomes a bound on a single run.

    Attributes:
        model_id: Content hash of the time model.
        point_id: Content hash of the assessed point.
        form: `"acquisition_linear/1"`.
        scope: The model's time scope, copied from the model.
        seconds: Nonnegative predicted seconds, or `None` when unavailable.
        unit: Seconds.
        lower_seconds: Lower end of the interval, at least zero, or `None`.
        upper_seconds: Upper end of the interval, or `None`.
        uncertainty: The model's `ModelUncertainty`, or `None`.
        features: The feature counts the prediction used.
        evidence: Evidence that names the model's source.
        reasons: Why the prediction is unavailable. Empty when it is available.
        assumptions: Assumptions the prediction rests on.
    """

    model_id: ContentID
    point_id: ContentID
    form: Literal["acquisition_linear/1"]
    scope: TimeScope
    seconds: Float64 | None
    unit: Unit
    lower_seconds: Float64 | None = None
    upper_seconds: Float64 | None = None
    uncertainty: ModelUncertainty | None = None
    features: tuple[Fact, ...]
    evidence: Evidence
    reasons: tuple[Text, ...]
    assumptions: tuple[Text, ...]

    @model_validator(mode="after")
    def _prediction(self):
        """Keep an available prediction and an unavailable one apart.

        The unit is seconds. An unavailable prediction (``seconds`` None) has
        reasons and no interval or uncertainty. An available one is nonnegative
        and has no reasons. An interval has both endpoints, a stated
        uncertainty, a nonnegative lower end, and contains the central value.
        """
        if (self.unit.symbol, self.unit.dimension) != ("s", "time"):
            raise ValueError("time prediction requires seconds")
        if self.seconds is None:
            if (
                not self.reasons
                or self.lower_seconds is not None
                or self.upper_seconds is not None
                or self.uncertainty is not None
            ):
                raise ValueError("unavailable prediction requires reasons and no numeric interval")
        elif self.reasons or self.seconds.value < 0:
            raise ValueError("available prediction is nonnegative and has no missing-input reasons")
        if (self.lower_seconds is None) != (self.upper_seconds is None):
            raise ValueError("prediction interval requires both endpoints")
        if self.lower_seconds is not None and (
            self.uncertainty is None
            or self.lower_seconds.value < 0
            or not self.lower_seconds.value <= self.seconds.value <= self.upper_seconds.value
        ):
            raise ValueError("prediction interval must contain the central prediction")
        return self


class AssessmentPoint(Record):
    """Actual immutable input identities for this read-only evaluation."""

    problem_id: ContentID
    plan_id: ContentID
    realization_id: ContentID
    base_construction_id: ContentID
    construction_id: ContentID
    resource_estimate_id: ContentID
    resource_context_id: ContentID
    profile_id: ContentID
    allocation_id: ContentID
    runtime_options_id: ContentID | None
    runtime_source_id: ContentID
    compiler_source_id: ContentID | None
    assessed_at: Timestamp


class _Workload(NamedTuple):
    """The single selected fold and features that every axis of one assessment reads.

    readout_items is the output cardinality, or None when a register width is
    unresolved (shape_known is then False). features holds the counted time
    model inputs as Facts: invocations, sampled_shots, exact_evaluations,
    logical_operations and, for amplitudes, return and publication sizes.
    """

    construction: SelectedConstruction
    estimate: WorkloadEstimate
    experiment: Experiment
    readout: str | None
    readout_items: int | None
    features: tuple[Fact, ...]
    node_kinds: frozenset[str]
    program_admission: _Admission
    readiness: Readiness
    shape_known: bool


def _outcome(details):
    """Summarize one axis. Infeasible wins, then conditional, then unknown, then feasible.

    One proved failure is decisive for its axis. Otherwise any condition or
    missing fact prevents an unconditional feasible summary.
    """
    outcomes = {detail.status for detail in details}
    return next(
        status
        for status in ("infeasible", "conditional", "unknown", "feasible")
        if status in outcomes
    )


def _axis(axis, details):
    """Build one axis whose status is the ``_outcome`` of its details."""
    return AxisAssessment(axis=axis, status=_outcome(details), details=details)


def _value(fact):
    """Return a concrete numeric fact value exactly (int, Fraction or float), else None."""
    if fact.availability != "concrete" or isinstance(fact.value, bool):
        return None
    if isinstance(fact.value, Rational):
        if fact.value.denominator == 1:
            return fact.value.numerator
        return Fraction(fact.value.numerator, fact.value.denominator)
    if isinstance(fact.value, Float64):
        return fact.value.value
    return None


def _integer(quantity):
    """The concrete integer value of a ResourceQuantity, or None when it is unknown or not an integer."""
    fact = quantity.fact
    if (
        fact.availability == "concrete"
        and isinstance(fact.value, Rational)
        and fact.value.denominator == 1
    ):
        return fact.value.numerator
    return None


def _fresh(record, at):
    """Whether a dated record is in force at the assessment time, not at the time of reading."""
    return record.recorded_at <= at <= record.valid_until


def _count_fact(name, value, scope, subject, *, reason=None):
    """Build an exact count fact derived from the selection, or an unknown fact with its reason."""
    args = dict(
        quantity=name,
        unit=Unit(symbol="count", dimension="count"),
        scope=Scope(domain=scope),
    )
    if value is None:
        return Fact(
            **args, availability="unknown", reason=reason or "selected feature is unavailable"
        )
    return Fact(
        **args,
        availability="concrete",
        value=Rational(numerator=value, denominator=1),
        evidence=Evidence(
            kind="proved_relation",
            source=_ASSESSMENT_SOURCE,
            status="witnessed",
            subject_id=subject,
            artifact=subject,
            witnessed_scope=args["scope"],
        ),
    )


def _select_workload(plan, realization, context, memo=None):
    """Fold the Realization's selected construction once and derive its acquisition features.

    ``memo`` is the caller's fold memo (``resources.estimate``): points of one
    ``estimate_plan`` call that select an equal construction under the same
    context share one fold, including the base fold when a point selects the
    Plan's own construction.

    The construction is the one native preparation would use for this
    Realization, so the forecast and a later receipt describe the same
    circuit. Features depend on the Experiment's shape.

    - A direct observation is one invocation. Its sampled shots are the
      requested shots, unknown for a provider-managed estimate, and it has one
      exact evaluation for an exact readout. The raw Program fold carries no
      acquisition population for a direct observation, so these counts come
      from the observation itself.
    - A measurement batch takes shots and exact evaluations from the fold,
      with the fold's own interpretation and assumptions. It counts as one
      invocation only when no other measurement batch is reachable from it
      and its population resolves to one request (positive counts, one
      grouped exact evaluation or one estimate repetition). Otherwise the
      invocation count stays unknown rather than a guessed job count.

    logical_operations always comes from the fold. Amplitude readouts add the
    native return size and the stored vector size. Readout cardinality is
    checked against the Program's integer-size limit by bit length before
    ``readout_shape`` forms it. Unresolved register widths leave the readout
    shape unknown instead of guessing.
    """
    # Consume the same effective pair boundary as native preparation. The fold
    # remains the sole population and workspace engine for this selected graph.
    experiment, construction = realization._selected_construction(plan)
    program_admission = _Admission(construction.program)
    readiness = program_admission.check()
    values = program_admission.expressions(
        program_admission.binding_map(construction.program.bindings)
    )

    def total_width(registers):
        """Sum resolved register widths with bounded integer growth, or return None if any is symbolic."""
        total = 0
        for register in registers:
            width = program_admission.integer(register.width, values, "readout register width")
            if width is None:
                return None
            if total == 0:
                total = width
                continue
            if width == 0:
                continue

            total = program_admission.bounded(total + width)
        return total

    width = total_width(construction.program.registers)
    classical_width = total_width(
        tuple(r for r in construction.program.classical if r.dtype == "bits")
    )
    resources = estimate(construction, context=context, memo=memo)
    nodes = {item.id: item.node for item in construction.program.definitions}
    reachable = set()
    pending = [construction.program.root]
    while pending:
        name = pending.pop()
        if name not in reachable:
            reachable.add(name)
            pending.extend(children(nodes[name]))
    node_kinds = frozenset(nodes[name].kind for name in reachable)
    if experiment.batch is None:
        observation = experiment.observation
        readout = observation.kind
        repetitions = observation.shots
        details = observation
        # Direct Program resources contain no acquisition population. These are
        # requested invocations/evaluations, never claimed simulator trajectories.
        counts = {
            "invocations": 1,
            "sampled_shots": None if readout == "estimated_observable" else observation.shots,
            "exact_evaluations": int(
                readout not in {"counts", "host_scalars", "estimated_observable"}
            ),
        }
        features = [
            _count_fact(
                name,
                value,
                "selected explicit native acquisition",
                realization.content_id,
                reason="provider-managed sampling cost is unavailable" if value is None else None,
            )
            for name, value in counts.items()
        ]
    else:
        batch = nodes[construction.program.root]
        readout = batch.observation_kind
        details = experiment.readout
        if readout == "probabilities" and details.amplitudes is not None:
            # A probability batch with an amplitude declaration returns kept
            # amplitudes of the selected construction. Resolve it as
            # preparation, restoration and analysis do. Other batches keep
            # their declared kind, which may be unknown before preparation.
            _, details = experiment._resolved_observation(
                program_admission, construction_id=construction.content_id
            )
            readout = details.kind
        # A selected terminal counts batch is one submitted request. A terminal
        # exact batch is one only when its grouped evaluation count is one.
        # Nested/zero/unknown groups do not invent an orchestration/job count.
        terminal = not any(
            isinstance(nodes[name], MeasurementBatch)
            for name in reachable
            if name != construction.program.root
        )
        shots = resources.quantity("shots")
        exact = resources.quantity("exact_evaluations")
        shot_count, exact_count = _integer(shots), _integer(exact)
        repetitions = shot_count
        invocation = (
            1
            if terminal
            and (
                readout == "counts"
                and shot_count is not None
                and (shot_count > 0)
                or (
                    readout in {"pauli_expectation", "probabilities", "amplitudes", "trajectory"}
                    and exact_count == 1
                )
                or (
                    readout == "estimated_observable"
                    and batch.repetitions is not None
                    and (
                        program_admission.integer(
                            batch.repetitions, values, "estimated invocations"
                        )
                        == 1
                    )
                )
            )
            else None
        )
        features = [
            _count_fact(
                "invocations",
                invocation,
                "selected batch native request",
                realization.content_id,
                reason="native invocation count requires a supported terminal acquisition",
            )
        ]
        for name, quantity in (("sampled_shots", shots), ("exact_evaluations", exact)):
            # Preserve the fold's source and assumptions, including symbolic data.
            features.append(quantity.fact.revise(quantity=name))
    operations = resources.quantity("operations")
    features.append(operations.fact.revise(quantity="logical_operations"))
    if readout == "amplitudes":
        declaration = details.amplitudes
        # The largest value below is 16 * 2**width bytes, which has width + 5 bits.
        if declaration.width + 5 > construction.program.limits.max_integer_bits:
            raise ValueError(
                "native amplitude return cardinality exceeds the admitted integer size"
            )
        for name, value in (
            ("native_return_items", 1 << declaration.width),
            ("native_return_logical_bytes", 16 * (1 << declaration.width)),
            ("stored_vector_items", declaration.output.basis.dimension),
            ("stored_vector_logical_bytes", declaration.output.data_bytes),
        ):
            features.append(
                _count_fact(
                    name,
                    value,
                    "selected amplitude return/publication; simulator RSS unknown",
                    realization.content_id,
                )
            )
    features = tuple(features)
    # This is a scalar cardinality, not allocation of that many statistic slots.
    # Bound its integer growth without substituting metadata max_items for the
    # separate eventual acquisition's population cap.
    bits = (
        1 + len(details.qubits)
        if readout == "probabilities"
        else classical_width + 1
        if readout == "counts"
        and classical_width is not None
        and repetitions is not None
        and classical_width < repetitions.bit_length()
        else None
    )
    if bits is not None:
        if bits > construction.program.limits.max_integer_bits:
            raise ValueError("readout cardinality exceeds the admitted integer-output size")

    readout_items = readout_shape(
        kind=readout,
        details=details,
        width=width,
        classical_width=classical_width,
        repetitions=repetitions,
        # A trajectory counts all of its points (``point_items``) against the
        # same integer-size limit, by bit length before forming a marginal.
        max_items=(1 << construction.program.limits.max_integer_bits) - 1 if readout == "trajectory" else None,
        max_items_source="the Program's admitted integer-output size" if readout == "trajectory" else None,
    )
    shape_known = width is not None and (readout != "counts" or classical_width is not None)
    if not shape_known:
        readout_items = None
    return _Workload(
        construction,
        resources,
        experiment,
        readout,
        readout_items,
        features,
        node_kinds,
        program_admission,
        readiness,
        shape_known,
    )


def _assess_applicability(plan, context):
    """Evaluate associated planner premises separately from unresolved assumptions and descriptor
    matching.

    Exactly one framed ``scientific_applicability`` statement must be present,
    and its evidence must name this Method's source, this Problem and this
    construction. A Method descriptor that merely matches the Problem type
    supplies no mathematical premise, so without such a receipt the premise
    detail is unknown. Plan assumptions and requirements add a separate
    conditional detail. The builtin Methods record no such statement, so their
    axis is unknown, or conditional when the Plan lists assumptions or
    requirements. A feasible axis needs a proved-relation statement without
    assumptions, an admitted Program, a True predicate value and no Plan
    assumptions or requirements. Such a statement with a False value makes
    the axis infeasible.
    """
    obligations = []
    if plan.assumptions or plan.requirements:
        obligations.append(
            AssessmentDetail(
                quantity="unresolved planner obligations",
                scope="selected Plan",
                status="conditional",
                reason="; ".join((*plan.assumptions, *plan.requirements)),
            )
        )
    candidates = [
        statement
        for statement in plan.facts
        if statement.fact.quantity == "scientific_applicability"
    ]
    if len(candidates) != 1:
        return _axis(
            "scientific_applicability",
            [
                AssessmentDetail(
                    quantity="method premises",
                    scope="this Problem and method-selected construction",
                    status="unknown",
                    reason="actual planner applicability receipt is absent or ambiguous; descriptor matching is insufficient",
                ),
                *obligations,
            ],
        )
    statement = candidates[0]
    fact = statement.fact
    evidence = fact.evidence
    associated = (
        evidence is not None
        and evidence.source == plan.method.descriptor.source
        and evidence.status == "witnessed"
        and evidence.subject_id == plan.construction.content_id
        and evidence.artifact == plan.problem.content_id
        and evidence.witnessed_scope == fact.scope
        and fact.scope == plan.output.frame(plan.problem).scope
        and (fact.unit.symbol, fact.unit.dimension) == ("1", "dimensionless")
        and context.plan_id == plan.content_id
        and context.problem_id == plan.problem.content_id
        and context.base_construction_id == plan.construction.content_id
    )
    status = "unknown"
    reason = "planner receipt is unavailable or does not bind this Problem/method/construction"
    if associated and isinstance(fact.value, bool):
        value = predicate_value(
            statement, frame=ErrorFrame.from_output(plan.problem, plan.output), context=context
        )
        if evidence.kind != "proved_relation" or fact.assumptions:
            status, reason = (
                "conditional",
                "planner applicability keeps an assertion or unresolved assumptions",
            )
        elif not context.admitted:
            status, reason = "conditional", "selected Program admission remains unresolved"
        elif value is True:
            status, reason = (
                "feasible",
                "actual planner admission supports its recorded method premises",
            )
        elif value is False:
            status, reason = (
                "infeasible",
                "actual planner admission records a failed necessary premise",
            )
        else:
            status, reason = (
                "conditional",
                "planner predicate keeps unsupported or unavailable premises",
            )
    details = [
        AssessmentDetail(
            quantity=fact.quantity,
            scope="recorded planner admission",
            status=status,
            reason=reason,
            statement=statement,
            evidence_ids=(statement.content_id,),
            assumptions=fact.assumptions,
        )
    ]
    details.extend(obligations)
    return _axis("scientific_applicability", details)


def _assess_capability(workload, profile, at):

    """Compare the selected workload with the fresh supplied target subset and transform-aware
    instruction widths.

    A coarse capability such as STATEVECTOR does not certify a readout,
    Program node or instruction that the target does not list, and an
    undeclared subset is unknown rather than supported. Each selected
    implementation must be covered exactly, or through every primitive of its
    decomposition at the added control width. An expired target
    specification makes these comparisons unknown, except that an unsupported
    instruction on a potential control-flow path is reported as conditional.
    """
    target = profile.configuration.target
    evidence_ids = (profile.evidence.content_id, target.content_id)
    details = []
    if not workload.shape_known:
        details.append(
            AssessmentDetail(
                quantity="readout shape",
                scope="selected circuit registers",
                status="unknown",
                reason="selected readout dimensions/layout are unresolved",
            )
        )

    def requirement(quantity, actual, supported):
        """Record whether the required set lies inside a declared target subset (None is unknown)."""
        if supported is None:
            status, reason = "unknown", "target does not declare this concrete subset"
        elif actual <= set(supported):
            status, reason = "feasible", "selected requirement is within the stored target subset"
        else:
            status, reason = (
                "infeasible",
                "unsupported: " + ", ".join(sorted(actual - set(supported))),
            )
        if not _fresh(profile, at):
            status, reason = "unknown", "target specification is expired or not yet effective"
        details.append(
            AssessmentDetail(
                quantity=quantity,
                scope="selected logical artifact/readout",
                status=status,
                reason=reason,
                evidence_ids=evidence_ids,
            )
        )

    requirement("artifact", {"selected_construction"}, target.artifacts)
    if workload.readout is None:
        details.append(
            AssessmentDetail(
                quantity="readout",
                scope="selected acquisition",
                status="unknown",
                reason="selected IR acquisition kind is unknown",
            )
        )
    else:
        # A trajectory requires the schedule and each point's readout kind.
        schedule = workload.experiment.observation or workload.experiment.readout
        requirement("readout", {workload.readout} | {point.kind for point in schedule.positions
                                                    if point.kind != "reduction"}, target.readouts)
        needed = (
            BackendCapability.COUNTS
            if workload.readout == "counts"
            else BackendCapability.STATEVECTOR
        )
        if workload.readout == "host_scalars":
            needed = BackendCapability.HOST_KERNEL
            requirement(
                "execution representation", {"classical"}, (profile.configuration.representation,)
            )
            requirement(
                "host dependencies",
                {
                    dependency
                    for kernel in workload.construction.kernels
                    for dependency in kernel.dependencies
                },
                target.host_dependencies,
            )
        if workload.readout == "pauli_expectation" and BackendCapability.EXPECTATION in (
            target.capabilities or ()
        ):
            needed = BackendCapability.EXPECTATION
        if workload.readout == "estimated_observable":
            needed = BackendCapability.ESTIMATED_OBSERVABLE
        requirement(
            "coarse execution capability",
            {needed.value},
            None if target.capabilities is None else {item.value for item in target.capabilities},
        )
    requirement("Program nodes", workload.node_kinds, target.program_nodes)
    selected_ids = set(workload.estimate.defined_selections)
    instructions = target.instructions
    # A support lookup is linear in the stored catalogue plus selected recipe,
    # rather than multiplying every selected primitive by every declaration.
    supported_widths = {}
    for item in instructions or ():
        key = (
            item.primitive,
            None if item.implementation is None else item.implementation.content_id,
            item.controlled,
            item.adjoint,
        )
        supported_widths[key] = max(supported_widths.get(key, -1), item.max_qubits)
    # Check either the exact selected implementation or every primitive in
    # its decomposition, including extra control width and adjoint support.
    for selected in workload.construction.selections:
        if selected.content_id not in selected_ids:
            continue
        width = (
            sum(port.width for port in selected.signature.quantum)
            if all(type(port.width) is int for port in selected.signature.quantum)
            else None
        )
        key = (None, selected.implementation.content_id, selected.controlled, selected.adjoint)
        supported = width is not None and supported_widths.get(key, -1) >= width
        if not supported and selected.decomposition is not None:
            supported = all(
                supported_widths.get(
                    (primitive.gate, None, selected.controlled, selected.adjoint), -1
                )
                >= len(primitive.qubits) + int(selected.controlled)
                for primitive in selected.decomposition
            )
        status = "feasible" if supported else "infeasible"
        reason = (
            "stored subset covers the exact selected implementation/primitive transform"
            if supported
            else (
                "stored subset does not cover this selected implementation, transform or instruction width"
            )
        )
        if instructions is None or width is None or not _fresh(profile, at):
            status, reason = (
                "unknown",
                "instruction subset, width or current target specification is unavailable",
            )
        if not supported and workload.node_kinds & {"branch", "adaptive_loop"}:
            status, reason = (
                "conditional",
                "a potential control-flow path needs an unsupported instruction",
            )
        details.append(
            AssessmentDetail(
                quantity=selected.signature.name,
                scope="selected instruction support",
                status=status,
                reason=reason,
                evidence_ids=(*evidence_ids, selected.content_id),
            )
        )
    return _axis("execution_capability", details)


def _capacity_detail(quantity: ResourceQuantity, limit, allocation, context, at, *, complete):
    """Compare a scoped peak/subtotal, not an upper envelope to a lower bound.

    An exact byte requirement above the stock proves infeasibility even when
    other workspace is unknown. Feasibility needs a complete peak (``complete``)
    that is exact or an upper bound, fits the stock and, for per-acquisition
    reuse, has the serial schedule it requires. An upper envelope above the
    stock only makes the comparison conditional, because an envelope is not a
    lower requirement.
    """
    value = _integer(quantity)
    status, reason = (
        "unknown",
        "matching execution memory stock or concrete byte requirement is unavailable",
    )
    applicable = (
        quantity.lifecycle == "planned"
        and quantity.basis == context.basis
        and quantity.population
        in {
            "simultaneous live footprint",
            "whole-workload peak under serial independent acquisitions",
        }
    )
    scheduled = quantity.required_schedule is None or context.batch_schedule == "serial"
    if not applicable:
        reason = (
            "resource lifecycle/basis/population does not match this planned capacity comparison"
        )
    elif limit is not None and value is not None:
        if quantity.interpretation == "exact" and value > limit.value:
            # A known subtotal is a lower requirement even if other workspace is
            # missing. Unlike an upper envelope, it can disprove a fit.
            status, reason = (
                "infeasible",
                "known byte requirement exceeds this location's granted stock",
            )
        elif (
            scheduled
            and complete
            and quantity.interpretation in {"exact", "upper_bound"}
            and value <= limit.value
        ):
            status, reason = (
                "feasible",
                "complete declared peak fits this location's stock in its stated scope",
            )
        elif not scheduled:
            status, reason = (
                "conditional",
                "per-acquisition reuse requires the unestablished serial schedule",
            )
        elif not complete:
            status, reason = (
                "conditional",
                "known subtotal fits or is only an envelope; missing workspace still matters",
            )
        elif quantity.interpretation == "upper_bound" and value > limit.value:
            status, reason = (
                "conditional",
                "upper envelope exceeds stock; it is not a proved lower requirement",
            )
        else:
            status, reason = (
                "conditional",
                "estimate/condition does not establish an unconditional capacity comparison",
            )
    if not _fresh(allocation, at):
        status, reason = (
            "unknown",
            "resource grant is expired or not yet effective; kept stocks are historical",
        )
    return AssessmentDetail(
        quantity=quantity.metric,
        scope=f"{quantity.location}: {quantity.population}",
        status=status,
        reason=reason,
        fact=quantity.fact,
        resource_id=quantity.content_id,
        limit=limit,
        evidence_ids=(*quantity.sources, allocation.evidence.content_id),
        assumptions=quantity.fact.assumptions,
    )


def _state_body_detail(workload, profile, allocation, stocks, at):
    """Compare powers by bit length before constructing any integer storage law.

    An unpartitioned statevector body on q qubits needs ``b * 2**q`` bytes,
    and a density matrix ``b * 4**q``, with b = 16 for complex128 and 8 for
    complex64. This is a necessary lower requirement for the body alone, not
    a sufficient memory estimate for the simulator. It applies only to one
    exact simultaneous width on a single-device allocation. When the total
    width is not exact, the exact system-register width still gives a valid
    lower requirement.
    """

    configuration = profile.configuration
    representation = configuration.representation
    if representation not in {"statevector", "density_matrix"}:
        return AssessmentDetail(
            quantity="native representation memory",
            scope="native state and workspace",
            status="unknown",
            reason="tensor memory needs contraction/truncation features, not qubit count"
            if representation == "tensor"
            else "no native state-memory law applies to the declared representation",
        )
    widths = [item for item in workload.estimate.quantities if item.metric == "logical_width"]
    if (
        allocation.topology != "single_device"
        or len(widths) != 1
        or widths[0].location != allocation.locations[0]
    ):
        return AssessmentDetail(
            quantity="native state body",
            scope="declared physical placement",
            status="unknown",
            reason="single-state storage law requires one matching location; no distributed/pooled inference",
        )
    width = widths[0]
    if width.interpretation != "exact" or _integer(width) is None:
        # Unknown extra ancillas do not erase a known system-register lower
        # requirement. Consume the separately kept role peak from the fold.
        system = next(
            (
                item
                for item in workload.estimate.quantities
                if item.metric == "system" and item.location == width.location
            ),
            None,
        )
        if system is not None and system.interpretation == "exact" and _integer(system) is not None:
            width = system
    q = _integer(width)
    limit = stocks.get((width.location, "memory"))
    if (
        q is None
        or width.interpretation != "exact"
        or configuration.precision not in {"complex64", "complex128"}
    ):
        return AssessmentDetail(
            quantity="native state body",
            scope=width.location,
            status="unknown",
            reason="body lower requirement needs an exact simultaneous width and complex precision",
            resource_id=width.content_id,
        )
    item_bytes = 16 if configuration.precision == "complex128" else 8
    exponent = q if representation == "statevector" else 2 * q
    # The representation identity and exact width justify a necessary body size,
    # not sufficiency for the native implementation's other allocations.
    args = dict(
        quantity="state_body_memory_lower_bound",
        unit=Unit(symbol="byte", dimension="bytes"),
        scope=Scope(
            kind="model",
            domain=f"{configuration.precision} {representation} lower memory requirement at {width.location}",
        ),
        assumptions=(
            f"one unpartitioned body: {item_bytes} * 2**{exponent} bytes",
            f"width lower requirement comes from the exact {width.metric} resource quantity",
            "workspace, buffers, copies and readout storage are additional unresolved requirements",
        ),
    )
    body = None
    if exponent + item_bytes.bit_length() <= workload.construction.program.limits.max_integer_bits:
        body = item_bytes << exponent
    if body is None:
        fact = Fact(
            **args,
            availability="unknown",
            reason="exact body value exceeds the admitted integer-output size; comparison uses bit lengths",
        )
    else:
        fact = Fact(
            **args,
            availability="concrete",
            value=Rational(numerator=body, denominator=1),
            evidence=Evidence(kind="proved_relation", source=_ASSESSMENT_SOURCE),
        )
    status, reason = "unknown", "matching memory grant is unavailable"
    if limit is not None:
        # b * 2**e > L exactly when 2**e > L // b, and for m >= 0 the smallest
        # k with 2**k > m is m.bit_length(). The comparison never forms 2**e.
        exceeds = limit.value < item_bytes or exponent >= (limit.value // item_bytes).bit_length()
        status = "infeasible" if exceeds else "conditional"
        reason = (
            "representation-specific body lower requirement exceeds this location's stock"
            if exceeds
            else "body fits; native workspace/buffers/readout and their overlap are not bounded by this law"
        )
    if not _fresh(allocation, at) or not _fresh(profile, at):
        status, reason = (
            "unknown",
            "current representation or allocation specification is unavailable",
        )
    return AssessmentDetail(
        quantity="native state body",
        scope=width.location,
        status=status,
        reason=reason,
        fact=fact,
        limit=limit,
        resource_id=width.content_id,
        evidence_ids=(
            profile.evidence.content_id,
            allocation.evidence.content_id,
            width.content_id,
        ),
        assumptions=args["assumptions"],
    )


def _assess_capacity(workload, profile, allocation, at):
    """Compare folded byte requirements with granted per-location stocks.

    Stocks are never pooled across locations. A 10-GiB peak at one location
    is compared with that location's 8-GiB stock, never with the 16-GiB total
    of two devices. The comparisons, each kept as its own detail:

    - The allocation must belong to the profile's configuration, and any
      capacity declared in the fold context must be one of the granted Limit
      records. Otherwise the axis reports that mismatch and leaves every
      stock unknown.
    - Each execution stock is compared with the matching folded peak. When
      the total memory peak is unknown, the known subtotal is compared too,
      because it can still prove a failure.
    - A memory population without a matching stock is reported as unknown.
    - Exact resident input, analysis, I/O, materialization and stored bytes
      are simultaneous for the whole workload, so each alone can refute a
      stock regardless of the batch schedule. Declared resident bytes are
      subtracted from the remaining stock without forming a possibly oversized
      sum.
    - A grant larger than the profile's matching hard stock is infeasible.
    - A width above the target's qubit ceiling is infeasible when exact.
    - For any representation other than "classical", the state-body lower
      requirement of ``_state_body_detail`` is added.

    Every supplied stock that no comparison covers is listed as unknown, so a
    missing comparison is visible rather than read as a fit.
    """

    resources = workload.estimate
    supplied_stocks = tuple(
        limit
        for limit in (*allocation.limits, *profile.limits)
        if limit.kind == "capacity_stock"
    )

    def finish(details):
        """Add an unknown detail for every supplied stock that no comparison covered."""
        covered = {d.limit.content_id for d in details if d.limit is not None}
        for limit in supplied_stocks:
            if limit.content_id not in covered:
                details.append(
                    AssessmentDetail(
                        quantity=limit.metric,
                        scope=limit.scope,
                        limit=limit,
                        status="unknown",
                        reason="no justified matching lifecycle/location/population comparison covers this stock",
                    )
                )
                covered.add(limit.content_id)
        return _axis("capacity_materialization", details)

    if allocation.configuration_id != profile.configuration.content_id:
        return finish(
            [
                AssessmentDetail(
                    quantity="allocation",
                    scope="intended target/configuration",
                    status="unknown",
                    reason="grant belongs to another hardware/runtime/build/precision configuration",
                    evidence_ids=(allocation.content_id, profile.configuration.content_id),
                )
            ]
        )
    # The fold may keep a supplied stock declaration, but it is not a second
    # grant. Reject contradictory declarations without silently replacing either.
    if resources.context.capacities:
        known = {item.content_id for item in allocation.limits}
        if any(item.content_id not in known for item in resources.context.capacities):
            return finish(
                [
                    AssessmentDetail(
                        quantity="capacity declarations",
                        scope="ResourceContext and Allocation",
                        status="unknown",
                        reason="fold context capacities do not match the actual granted Limit records",
                    )
                ]
            )
    details = []
    quantities = {(item.metric, item.location): item for item in resources.quantities}
    stocks = {
        (item.scope, item.metric): item
        for item in allocation.limits
        if item.stage == "execution" and item.kind == "capacity_stock"
    }
    # Each supplied execution stock drives its own scoped comparison. A kept
    # subtotal can refute its stock independently of unknown total workspace.
    for owner in (allocation, profile):
        for limit in owner.limits:
            if limit.stage != "execution" or limit.kind != "capacity_stock":
                continue
            metric = {"memory": "memory", "stored": "stored_bytes"}.get(limit.metric)
            quantity = quantities.get((metric, limit.scope))
            if quantity is None:
                continue
            details.append(
                _capacity_detail(
                    quantity, limit, owner, resources.context, at, complete=metric == "memory"
                )
            )
            known = quantities.get(("known_memory", limit.scope))
            if (
                metric == "memory"
                and known is not None
                and quantity.fact.availability != "concrete"
            ):
                details.append(
                    _capacity_detail(known, limit, owner, resources.context, at, complete=False)
                )
    # Missing stock is itself useful evidence for each actual memory population.
    for quantity in resources.quantities:
        if quantity.metric == "memory" and (quantity.location, "memory") not in stocks:
            details.append(
                _capacity_detail(quantity, None, allocation, resources.context, at, complete=True)
            )
    # Invariant resident lower requirements can reject independently of an
    # unknown/conditional batch peak. The fold preserves these purpose quantities.
    for quantity in resources.quantities:
        if (
            quantity.metric
            in {
                "input_bytes",
                "analysis_bytes",
                "io_bytes",
                "materialization_bytes",
                "stored_bytes",
            }
            and quantity.interpretation == "exact"
            and quantity.required_schedule is None
        ):
            limit = stocks.get((quantity.location, "memory"))
            value = _integer(quantity)
            if limit is not None and value is not None and value > limit.value:
                details.append(
                    _capacity_detail(
                        quantity, limit, allocation, resources.context, at, complete=False
                    )
                )
    # A resident declaration is simultaneous for the entire workload, independent
    # of any batch schedule. Compare by remaining stock, without constructing a
    # potentially oversized sum or attempting to infer a variable-workspace peak.
    remaining = {
        location: limit.value for (location, metric), limit in stocks.items() if metric == "memory"
    }
    resident_failures = set()
    for resident in resources.context.resident:
        location = resident.location
        if resident.bytes is None or location not in remaining or location in resident_failures:
            continue
        if resident.bytes > remaining[location]:
            resident_failures.add(location)
        else:
            remaining[location] -= resident.bytes
    if _fresh(allocation, at):
        for location in sorted(resident_failures):
            details.append(
                AssessmentDetail(
                    quantity="resident memory lower requirement",
                    scope=location,
                    status="infeasible",
                    reason="simultaneous declared resident bytes alone exceed this location's stock, regardless of schedule",
                    limit=stocks[location, "memory"],
                    evidence_ids=(resources.context.content_id, allocation.evidence.content_id),
                )
            )
    allocated = {(item.stage, item.metric, item.scope): item for item in allocation.limits}
    for advertised in profile.limits:
        matching = allocated.get((advertised.stage, advertised.metric, advertised.scope))
        if (
            matching is not None
            and advertised.kind == "capacity_stock"
            and matching.value > advertised.value
            and _fresh(profile, at)
            and _fresh(allocation, at)
        ):
            details.append(
                AssessmentDetail(
                    quantity="allocation stock",
                    scope=advertised.scope,
                    status="infeasible",
                    reason="requested grant exceeds the stored device's matching hard stock ceiling",
                    limit=advertised,
                    evidence_ids=(allocation.content_id, profile.evidence.content_id),
                )
            )
    if profile.configuration.target.max_qubits is not None:
        widths = [item for item in resources.quantities if item.metric == "logical_width"]
        for width in widths:
            value = _integer(width)
            if value is not None and value > profile.configuration.target.max_qubits:
                status = "infeasible" if width.interpretation == "exact" else "conditional"
                if not _fresh(profile, at):
                    status = "unknown"
                details.append(
                    AssessmentDetail(
                        quantity="logical_width",
                        scope=f"{width.location}: target width ceiling",
                        status=status,
                        reason="selected width exceeds the target's declared qubit ceiling",
                        fact=width.fact,
                        resource_id=width.content_id,
                        evidence_ids=(profile.configuration.target.content_id,),
                    )
                )
    if profile.configuration.representation != "classical":
        details.append(_state_body_detail(workload, profile, allocation, stocks, at))
    if not details:
        details.append(
            AssessmentDetail(
                quantity="memory",
                scope="selected workload",
                status="unknown",
                reason="no complete matching workspace/capacity inventory is available",
            )
        )
    return finish(details)


def _model_reasons(model, workload, profile, allocation, at, runtime):
    """List every reason this workload lies outside the model's validated domain.

    A time model is evaluated only inside the configuration, allocation,
    runtime, acquisition kind, resource basis, readout, population, parameter
    point, width, operation count, non-gate populations, shot and evaluation
    counts and operation mix it was supplied for. An unavailable feature is
    outside the domain, never zero. An upper-bound estimate of a non-gate
    population, and any concrete shot or evaluation count, is admitted when it
    lies inside the maximum. ``_predict_time`` separately refuses a batch
    envelope that a nonzero coefficient would multiply. All reasons are kept so the
    unavailable prediction explains itself. An empty list permits evaluation.
    """

    domain = model.domain
    configuration = profile.configuration
    reasons = []
    if domain.configuration_id != configuration.content_id:
        reasons.append(
            "model belongs to another hardware/runtime/build/precision/target configuration"
        )
    if (
        domain.allocation_id != allocation.content_id
        or allocation.configuration_id != configuration.content_id
    ):
        reasons.append("model allocation locations/topology/resources do not match")
    if domain.runtime != "seed_independent" and domain.runtime != runtime:
        reasons.append("model runtime options do not match the requested runtime")
    acquisition = "direct_observation" if workload.experiment.batch is None else "measurement_batch"
    if domain.acquisition != acquisition:
        reasons.append(
            "model operation/acquisition population differs from the selected Experiment"
        )
    if (
        configuration.precision is None
        or configuration.representation is None
        or configuration.compiler is None
    ):
        reasons.append("runtime precision/representation/compiler domain is incomplete")
    context = workload.estimate.context
    if (domain.basis, domain.rotation_precision, domain.synthesis, domain.batch_schedule) != (
        context.basis,
        context.precision,
        context.synthesis,
        context.batch_schedule,
    ):
        reasons.append("resource basis/synthesis/precision/schedule differs from the model domain")
    for name, record in (("model", model), ("profile", profile), ("allocation", allocation)):
        if not _fresh(record, at):
            reasons.append(
                f"{name} is expired or not yet effective at the supplied assessment time"
            )
    if workload.readout not in domain.readouts:
        reasons.append("readout is missing or outside the model domain")
    observation = workload.experiment.observation or workload.experiment.readout
    if domain.population != observation.population:
        reasons.append("model domain describes another readout population")
    point = {
        binding.parameter: number(binding.value)
        for binding in workload.construction.program.bindings
    }
    if any(
        (
            binding.parameter not in point or number(binding.value) != point[binding.parameter]
            for binding in domain.bindings
        )
    ):
        reasons.append("model domain is restricted to another effective parameter point")
    if workload.readout_items is None or workload.readout_items > domain.max_readout_items:
        reasons.append("readout output size is unavailable or outside the model domain")
    widths = [item for item in workload.estimate.quantities if item.metric == "logical_width"]
    if len(widths) != 1 or widths[0].interpretation != "exact" or _integer(widths[0]) is None:
        reasons.append("model requires one exact simultaneous logical width")
    elif not domain.min_qubits <= _integer(widths[0]) <= domain.max_qubits:
        reasons.append("logical width lies outside the model scale domain")
    # quantity() scans the kept tuple. Five requests below are five scans,
    # charged anew for each actual supplied model, independently of output size.

    operations = workload.estimate.quantity("operations")
    count = _integer(operations)
    if count is None or operations.interpretation != "exact":
        reasons.append("exact selected logical-operation feature is unavailable")
    elif not domain.min_operations <= count <= domain.max_operations:
        reasons.append("logical operation count lies outside the model scale domain")
    for metric in ("resets", "measurements", "classical_work", "adaptive_rounds"):
        quantity = workload.estimate.quantity(metric)
        value = _integer(quantity)
        # A finite [0, maximum] range admits an exact count or a proved upper
        # envelope inside it. An unsupported/nonexact estimate is not zero.
        if (
            value is None
            or quantity.interpretation not in {"exact", "upper_bound"}
            or value > getattr(domain, "max_" + metric)
        ):
            reasons.append(f"{metric} is unavailable or outside the model operation domain")
    values = {fact.quantity: _value(fact) for fact in workload.features}
    for name, maximum in (
        ("sampled_shots", domain.max_shots),
        ("exact_evaluations", domain.max_exact_evaluations),
    ):
        if values[name] is None or values[name] > maximum:
            reasons.append(f"{name} is unavailable or outside the model domain")
    selected_ids = set(workload.estimate.defined_selections)
    allowed_selections = set(domain.selection_ids)
    allowed_primitives = set(domain.primitive_gates)
    for selected in workload.construction.selections:
        if selected.content_id not in selected_ids:
            continue
        if selected.controlled or selected.adjoint or selected.decomposition is None:
            if selected.content_id not in allowed_selections:
                reasons.append(
                    f"selected kernel/transform {selected.signature.name} is outside the model operation mix"
                )
        elif not {item.gate for item in selected.decomposition} <= allowed_primitives:
            reasons.append(
                f"selected primitive recipe {selected.signature.name} is outside the model operation mix"
            )
    return reasons


def _predict_time(model: TimeModel, workload, profile, allocation, at, runtime, point_id):

    """Evaluate an applicable supplied time model on exact selected workload features.

    The ``acquisition_linear/1`` prediction is ``sum(c_f * x_f)`` seconds over
    the model's nonzero coefficients c_f and the exact features x_f. A supplied
    residual interval gives ``[max(0, s + lower), s + upper]`` around the
    central value s, keeping the interval's kind and coverage unchanged. An
    unmet domain condition, a missing required feature, a batch feature that
    is only an envelope, or binary64 overflow leaves the prediction
    unavailable with its reasons instead of a clipped number.
    """
    reasons = _model_reasons(model, workload, profile, allocation, at, runtime)
    features = {fact.quantity: fact for fact in workload.features}
    values = {name: _value(fact) for name, fact in features.items()}
    for coefficient in model.coefficients:
        if coefficient.seconds_per_unit and values[coefficient.feature] is None:
            reasons.append(f"missing required {coefficient.feature} feature")
    if workload.experiment.batch is not None:
        for name, metric in (
            ("sampled_shots", "shots"),
            ("exact_evaluations", "exact_evaluations"),
        ):
            if workload.estimate.quantity(metric).interpretation != "exact" and any(
                coefficient.feature == name and coefficient.seconds_per_unit
                for coefficient in model.coefficients
            ):
                reasons.append(
                    f"{name} is an envelope, not an exact feature for this central prediction"
                )
    # Evaluate a central prediction only when its domain and required features
    # are available. A resource envelope is not an exact feature value.
    seconds = lower = upper = None
    if not reasons:
        try:
            terms = [
                coefficient.seconds_per_unit * float(values[coefficient.feature])
                for coefficient in model.coefficients
                if coefficient.seconds_per_unit
            ]
            central = fsum(terms)
            if not isfinite(central):
                raise OverflowError("nonfinite model sum")
            seconds = Float64(value=central)
            if model.uncertainty is not None:
                lo = max(0.0, central + model.uncertainty.lower_residual_seconds)
                hi = central + model.uncertainty.upper_residual_seconds
                if not isfinite(hi):
                    raise OverflowError("nonfinite model interval")
                lower, upper = Float64(value=lo), Float64(value=hi)
        except OverflowError:
            reasons.append(
                "finite model arithmetic exceeds binary64 range; no clipped prediction is supplied"
            )
            seconds = lower = upper = None
    return TimePrediction(
        model_id=model.content_id,
        point_id=point_id,
        form=model.form,
        scope=model.scope,
        seconds=seconds,
        unit=Unit(symbol="s", dimension="time"),
        lower_seconds=lower,
        upper_seconds=upper,
        uncertainty=model.uncertainty if seconds is not None else None,
        features=workload.features,
        evidence=model.evidence,
        reasons=reasons,
        assumptions=model.assumptions
        + (
            "prediction is conditional on the supplied domain, coefficients and source evidence",
            "validity is evaluated at the original assessment time; calibration validity at execution time is not established",
            "any residual interval is intersected with nonnegative physical elapsed time",
            "native-call wall scope includes waiting/extraction; excludes preparation, publication and whole-run cost"
            if model.scope == "native_call_wall"
            else "scope excludes planning, compilation, queue delay, analysis and whole-run cost",
        ),
    )


def _assess_time(workload, profile, allocation, at, *, runtime, point_id):

    """Compare scoped model predictions and actual acquisition counts with matching supplied
    limits.

    Every model prediction appears as a conditional or unknown detail, never
    as feasible, because it rests on supplied coefficients. Job, shot and
    exact-evaluation limits are decided from exact counts. A wall-time limit is
    compared only with predictions of the same scope, and the comparison stays
    conditional even when the central value fits. Scopes are never summed.
    """
    predictions = tuple(
        _predict_time(model, workload, profile, allocation, at, runtime, point_id)
        for model in profile.models
    )
    details = []
    for prediction in predictions:
        details.append(
            AssessmentDetail(
                quantity="wall_time",
                scope=prediction.scope,
                status="conditional" if prediction.seconds is not None else "unknown",
                reason=f"{prediction.form} predicts {prediction.seconds.value:g} s under the supplied model"
                if prediction.seconds is not None
                else "; ".join(prediction.reasons),
                prediction_ids=(prediction.content_id,),
                evidence_ids=(prediction.model_id, prediction.evidence.content_id),
                assumptions=prediction.assumptions,
            )
        )
    # Count limits can be decided from exact populations. A model-based
    # deadline comparison stays conditional even when its central estimate fits.
    features = {fact.quantity: fact for fact in workload.features}
    allocation_limit_ids = {limit.content_id for limit in allocation.limits}
    profile_limit_ids = {limit.content_id for limit in profile.limits}
    for limit in (*allocation.limits, *profile.limits):
        if limit.stage != "execution" or limit.kind == "capacity_stock":
            continue
        feature = {
            ("jobs", "selected_acquisition"): "invocations",
            ("shots", "selected_acquisition"): "sampled_shots",
            ("evaluations", "selected_exact_evaluations"): "exact_evaluations",
        }.get((limit.metric, limit.scope))
        status, reason = "unknown", "no matching metric/unit/population model covers this limit"
        fact = None
        prediction_ids = ()
        if feature is not None:
            fact = features[feature]
            value = _value(fact)
            exact = (
                workload.experiment.batch is None
                or feature == "invocations"
                or workload.estimate.quantity(
                    "shots" if feature == "sampled_shots" else "exact_evaluations"
                ).interpretation
                == "exact"
            )
            if value is not None and exact:
                status = "feasible" if value <= limit.value else "infeasible"
                reason = (
                    "requested acquisition count fits the matching limit"
                    if status == "feasible"
                    else ("requested acquisition count exceeds the matching limit")
                )
        elif limit.metric == "wall_time":
            matched = [p for p in predictions if p.scope == limit.scope and p.seconds is not None]
            if matched:
                prediction_ids = tuple(p.content_id for p in matched)
                status = "conditional"
                statements = []
                for prediction in matched:
                    comparison = "within" if prediction.seconds.value <= limit.value else "above"
                    statements.append(
                        f"model central prediction is {comparison} the scoped deadline"
                    )
                    if prediction.uncertainty is not None:
                        statements.append(
                            f"interval kind remains {prediction.uncertainty.kind}; no hard execution guarantee"
                        )
                reason = "; ".join(statements)
        if limit.content_id in allocation_limit_ids and not _fresh(allocation, at):
            status, reason = "unknown", "allocation limit is expired or not yet effective"
        if limit.content_id in profile_limit_ids and not _fresh(profile, at):
            status, reason = "unknown", "profile limit is expired or not yet effective"
        details.append(
            AssessmentDetail(
                quantity=limit.metric,
                scope=limit.scope,
                status=status,
                reason=reason,
                fact=fact,
                limit=limit,
                prediction_ids=prediction_ids,
            )
        )
    if not details:
        details.append(
            AssessmentDetail(
                quantity="time/cost",
                scope="selected acquisition",
                status="unknown",
                reason="no applicable supplied named time model or matching consumption limit",
            )
        )
    return _axis("time_cost", details), predictions


def _assessment_time(value):
    """Read the clock once for a new assessment; stored views never call this."""
    value = datetime.now(timezone.utc) if value is None else value
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("assessment time must be timezone-aware")
    return value.astimezone(timezone.utc)


def _assessment_inputs(plan, realization, profile, allocation, context, runtime, assessed_at, memo=None):
    """Resolve one real selected point and use the existing contextual fold (``memo``: ``_select_workload``)."""
    if not isinstance(plan, Plan) or not isinstance(realization, Realization):
        raise TypeError("assessment requires the actual Plan and its Realization")
    if not isinstance(profile, DeviceProfile) or not isinstance(allocation, Allocation):
        raise TypeError("device assessment requires an explicit profile and Allocation")
    if runtime is not None and not isinstance(runtime, RuntimeOptions):
        raise TypeError("runtime must be an actual prepared RuntimeOptions or None")
    workload = _select_workload(plan, realization, context, memo)
    point = AssessmentPoint(
        problem_id=plan.problem.content_id,
        plan_id=plan.content_id,
        realization_id=realization.content_id,
        base_construction_id=plan.construction.content_id,
        construction_id=workload.construction.content_id,
        resource_estimate_id=workload.estimate.content_id,
        resource_context_id=context.content_id,
        profile_id=profile.content_id,
        allocation_id=allocation.content_id,
        runtime_options_id=None if runtime is None else runtime.content_id,
        runtime_source_id=profile.configuration.runtime.content_id,
        compiler_source_id=None
        if profile.configuration.compiler is None
        else profile.configuration.compiler.content_id,
        assessed_at=assessed_at,
    )
    return point, workload


def _accuracy_detail_fields(error, frame):
    """Summarize a ClaimAssessment as the single accuracy detail.

    PASS supports the sufficient criterion. Remaining contributions or a
    absent criterion stay unknown. Any other status is conditional,
    because failing a sufficient bound does not show that the actual error is
    too large.
    """
    if error is None:
        return dict(
            quantity=frame.quantity,
            scope=frame.scope.domain,
            status="unknown",
            reason="no applicable accuracy criterion was assessed; resource predictions remain independent",
        )
    if error.status == "PASS":
        status, reason = (
            "feasible",
            "common ErrorModel supports the selected sufficient accuracy criterion",
        )
    elif error.remaining:
        status, reason = (
            "unknown",
            "common ErrorModel has unavailable accuracy contributions or applicability",
        )
    else:
        status, reason = (
            "conditional",
            "a sufficient criterion is unresolved; this does not refute actual accuracy",
        )
    return dict(
        quantity=error.frame.quantity,
        scope=error.frame.scope.domain,
        status=status,
        reason=reason,
        evidence_ids=(error.content_id, error.model_id),
        assumptions=error.prerequisites,
    )


class ProfileAssessment(Record):
    """The forecast of one resolved Plan point on a device profile: five independent axes and the time predictions.

    [`assess`][nwqlib.backends.assessment.assess] returns it, and
    [`PlanEstimate.assessments`][nwqlib.backends.assessment.PlanEstimate] holds
    one per point. The fields below are read-only. Each axis is an
    [`AxisAssessment`][nwqlib.backends.assessment.AxisAssessment] whose `status`
    summarizes its own details only, so a memory failure cannot hide a supported
    error bound. No conclusion grants permission to run, configures a backend or
    changes the computation. An excessive sufficient error bound stays
    `INCONCLUSIVE` in `error` rather than proving that the actual error is too
    large.

    Attributes:
        point: Content hashes of the inputs: problem, Plan, point, base and
            selected construction, resource estimate and context, profile,
            allocation, expected runtime and compiler sources, known runtime seed,
            and the assessment time.
        realization: The resolved point, with its parameter bindings.
        resources: Resource estimate of this point. Capability, capacity and time
            read the same counts.
        applicability: Whether a stated assumption of the method holds for this
            problem. The built-in Methods record no such statement, so this axis
            is unknown, or conditional when the Plan lists assumptions or
            requirements.
        capability: Whether the target supports the point's circuit form, Program
            nodes, readout, instructions, host kernel and control or adjoint use.
        capacity: Whether each granted location holds the point's simultaneous
            memory and storage peaks and its known lower requirements.
        time: The time-model predictions and the matching time limits.
        accuracy: The Plan's error-model assessment of `criterion`.
        criterion: The [`Accuracy`][nwqlib.problems.records.Accuracy] criterion
            assessed, or `None`.
        error: The complete `ClaimAssessment` of `criterion`, or `None` without a
            criterion or error model. Before execution it names no observation or
            result.
        predictions: One [`TimePrediction`][nwqlib.backends.assessment.TimePrediction]
            per time model of the profile, in the profile's order.
    """

    point: AssessmentPoint
    realization: Realization
    resources: WorkloadEstimate
    applicability: AxisAssessment
    capability: AxisAssessment
    capacity: AxisAssessment
    time: AxisAssessment
    accuracy: AxisAssessment
    frame: ErrorFrame
    criterion: Accuracy | None = None
    error: ClaimAssessment | None = None
    predictions: tuple[TimePrediction, ...]

    @model_validator(mode="after")
    def _identities(self):
        """Require all five axes, predictions and error summaries to refer to the same selected
        point.
        """
        if (
            self.resources.content_id != self.point.resource_estimate_id
            or self.resources.construction_id != self.point.construction_id
            or self.resources.context.content_id != self.point.resource_context_id
            or self.realization.content_id != self.point.realization_id
            or self.realization.plan_id != self.point.plan_id
        ):
            raise ValueError("assessment must preserve its actual selected point and fold")
        if self.error is not None:
            context = self.error.context
            if (
                self.error.accuracy != self.criterion
                or self.error.frame != self.frame
                or context.plan_id != self.point.plan_id
                or context.problem_id != self.point.problem_id
                or context.base_construction_id != self.point.base_construction_id
                or context.selected_construction_id != self.point.construction_id
                or context.bindings != self.realization.bindings
                or context.observation_id is not None
                or context.contribution_ids
                or context.result_id is not None
            ):
                raise ValueError(
                    "accuracy assessment belongs to another scientific point or criterion"
                )
        for field, axis in (
            ("applicability", "scientific_applicability"),
            ("capability", "execution_capability"),
            ("capacity", "capacity_materialization"),
            ("time", "time_cost"),
            ("accuracy", "accuracy_evidence"),
        ):
            if getattr(self, field).axis != axis:
                raise ValueError("assessment axis is attached to the wrong field")
        expected = _accuracy_detail_fields(self.error, self.frame)
        if len(self.accuracy.details) != 1 or any(
            getattr(self.accuracy.details[0], key) != value for key, value in expected.items()
        ):
            raise ValueError("accuracy summary must describe the actual common assessment")
        if any(p.point_id != self.point.content_id for p in self.predictions):
            raise ValueError("prediction belongs to another assessment point")
        if len({p.model_id for p in self.predictions}) != len(self.predictions):
            raise ValueError("duplicate prediction model")
        prediction_ids = {p.content_id for p in self.predictions}
        if any(not set(d.prediction_ids) <= prediction_ids for d in self.time.details):
            raise ValueError("time detail references another prediction")
        summaries = [d for d in self.time.details if d.prediction_ids and d.limit is None]
        if [d.prediction_ids for d in summaries] != [(p.content_id,) for p in self.predictions]:
            raise ValueError("time summaries must follow actual ordered predictions")
        return self

    def validate_context(self, plan, profile, allocation, runtime=None, *, frame=None):
        """Match stored identities and source declarations without re-evaluation.

        ``frame`` is ``ErrorFrame.from_output(plan.problem, plan.output)`` when
        the caller already formed it for several assessments of one Plan.
        """
        if self.error is not None:
            self.error.validate_plan(plan)
        if (
            self.point.plan_id != plan.content_id
            or self.point.problem_id != plan.problem.content_id
            or self.point.profile_id != profile.content_id
            or self.point.allocation_id != allocation.content_id
            or self.point.runtime_options_id != (None if runtime is None else runtime.content_id)
            or self.point.runtime_source_id != profile.configuration.runtime.content_id
            or self.point.compiler_source_id
            != (
                None
                if profile.configuration.compiler is None
                else profile.configuration.compiler.content_id
            )
            or self.frame != (ErrorFrame.from_output(plan.problem, plan.output) if frame is None else frame)
        ):
            raise ValueError("stored assessment belongs to another Plan/profile/allocation/runtime")
        if [p.model_id for p in self.predictions] != [m.content_id for m in profile.models]:
            raise ValueError("prediction model membership/order differs from the supplied profile")
        for prediction, model in zip(self.predictions, profile.models, strict=True):
            if (
                prediction.evidence != model.evidence
                or prediction.form != model.form
                or prediction.scope != model.scope
                or prediction.assumptions[: len(model.assumptions)] != model.assumptions
                or prediction.uncertainty
                != (model.uncertainty if prediction.seconds is not None else None)
            ):
                raise ValueError("prediction changed its original model evidence/provenance")
        return self

    def validate_prepared(self, plan, realization, receipt, *, observation=None):
        """Associate a saved forecast with its actual acquisition, never reassess.

        Only identities and receipt-specific fields are compared. The
        Plan-level checks, the error claim and its frame included, are
        ``validate_context``'s. ``PlanEstimate.validate_plan`` runs them once
        per assessment, and ``PlanEstimate.assessment_for`` runs that once per
        Plan. ``observation`` is the readout that ``realization`` resolves to
        in ``plan``, when the caller already resolved it; otherwise it is
        resolved here.
        """
        if observation is None:
            observation = realization.resolved_observation(plan)[1]
        if (
            self.point.plan_id != plan.content_id
            or self.point.problem_id != plan.problem.content_id
            or self.realization != realization
            or self.point.realization_id != receipt.realization_id
            or receipt.plan_id != plan.content_id
            or self.point.construction_id != receipt.construction_id
            or (
                self.point.runtime_options_id is not None
                and self.point.runtime_options_id != receipt.runtime.content_id
            )
            or receipt.observation != observation
        ):
            raise ValueError(
                "assessment differs from the actual prepared Plan/Realization/readout/runtime"
            )
        return self


def assess(
    plan: Plan,
    realization: Realization,
    *,
    profile: DeviceProfile,
    allocation: Allocation,
    context: ResourceContext | None = None,
    runtime: RuntimeOptions | None = None,
    assessed_at: datetime | None = None,
    accuracy: Accuracy | None = None,
    facts: tuple[FramedFact, ...] = (),
    reference: TargetReference | None = None,
    max_assessments=4096,
) -> ProfileAssessment:
    """Forecast one resolved point of a Plan on a device profile and allocation, without running anything.

    `nwqlib.estimate(plan, profile=..., allocation=...)` assesses every point of
    the Plan whose parameters are fixed. Call `assess` for one point, for
    example the point that `plan.resolve(experiment_name)` returns. The result
    reports five independent axes: applicability, capability, capacity, time and
    accuracy. Each axis has the status `"feasible"`, `"infeasible"`,
    `"conditional"` or `"unknown"`, and a failure on one axis, such as memory,
    does not hide what another supports, such as an error bound. Time
    predictions rest on the supplied coefficients, so none is an execution
    guarantee. No pilot run, profile refresh, circuit compilation or replay of the
    computation takes place. The [profiles guide](../profiles.md) defines the axes, and
    [Forecast cost and feasibility](backends.md#forecast-cost-and-feasibility) assesses a
    one-qubit Plan.

    Args:
        plan (Plan): The Plan whose point is assessed.
        realization (Realization): One resolved point of `plan`, such as
            `plan.resolve(experiment_name)`.
        profile (DeviceProfile): The device profile.
        allocation (Allocation): The resources granted to the workload.
        context (ResourceContext | None): Resource-counting context, such as the
            gate basis. `None` uses `ResourceContext()`.
        runtime (RuntimeOptions | None): Backend seed of an already known
            preparation. With `None`, only seed-independent models apply. Seeds are
            never guessed.
        assessed_at (datetime | None): Time zone-aware assessment time, against
            which the validity of the profile, allocation and models is checked.
            `None` records the current UTC time once.
        accuracy (Accuracy | None): Accuracy criterion to assess. `None` uses the
            Plan's `selection_accuracy`, and with neither the accuracy axis is
            unknown. Another criterion changes only this assessment, never the
            Plan's shots or selection.
        facts (tuple[FramedFact, ...]): Supplied error facts, passed unchanged to
            the Plan's error model. A fact for one point cannot cover another.
        reference (TargetReference | None): Supplied reference value, passed
            unchanged to the Plan's error model.
        max_assessments (int): Positive limit on assessment and
            prediction rows, here one plus the number of time models, checked
            before any model is evaluated.

    Returns:
        assessment (ProfileAssessment): The forecast.
            Its `predictions` hold the time forecasts in seconds, its `capacity` the
            memory comparison and its `error` the complete accuracy assessment.

    Raises:
        TypeError: If `profile` is not a `DeviceProfile` or `accuracy` is not an
            `Accuracy`.
        ValueError: If the rows exceed `max_assessments`, or the stored records
            do not match the Plan, profile and allocation.
    """
    return _assess(plan, realization, profile=profile, allocation=allocation, context=context,
                   runtime=runtime, assessed_at=assessed_at, accuracy=accuracy, facts=facts,
                   reference=reference, max_assessments=max_assessments
                   ).validate_context(plan, profile, allocation, runtime)


def _assess(plan, realization, *, profile, allocation, context, runtime, assessed_at, accuracy,
            facts, reference, max_assessments, memo=None):
    """Build one point's assessment (``assess``) without its final ``validate_context``.

    ``assess`` validates the result, and ``estimate_plan`` validates each
    assessment once through ``PlanEstimate.validate_plan``. ``memo`` is the
    caller's fold memo (``resources.estimate``).
    """
    if not isinstance(profile, DeviceProfile):
        raise TypeError("assessment requires a supplied DeviceProfile")
    _forecast_rows(1, profile, max_assessments)
    at = _assessment_time(assessed_at)
    context = ResourceContext() if context is None else context
    point, workload = _assessment_inputs(
        plan, realization, profile, allocation, context, runtime, at, memo
    )
    accuracy = plan.selection_accuracy if accuracy is None else accuracy
    frame = ErrorFrame.from_output(plan.problem, plan.output)
    scientific = AssessmentContext(
        problem_id=plan.problem.content_id,
        plan_id=plan.content_id,
        base_construction_id=plan.construction.content_id,
        selected_construction_id=workload.construction.content_id,
        bindings=realization.bindings,
        admitted=workload.readiness.ready,
    )
    if accuracy is not None and not isinstance(accuracy, Accuracy):
        raise TypeError("accuracy requires its explicit Accuracy criterion")
    error = None
    if accuracy is not None and plan.error_model is not None:
        error = plan.error_model.assess(
            accuracy,
            context=scientific,
            facts=facts,
            reference=reference,
            max_integer_bits=plan.construction.program.limits.max_integer_bits,
        )
    time, predictions = _assess_time(
        workload, profile, allocation, at, runtime=runtime, point_id=point.content_id
    )
    result = ProfileAssessment(
        point=point,
        realization=realization,
        resources=workload.estimate,
        applicability=_assess_applicability(plan, scientific),
        capability=_assess_capability(workload, profile, at),
        capacity=_assess_capacity(workload, profile, allocation, at),
        time=time,
        accuracy=_axis(
            "accuracy_evidence", [AssessmentDetail(**_accuracy_detail_fields(error, frame))]
        ),
        frame=frame,
        criterion=accuracy,
        error=error,
        predictions=predictions,
    )
    return result


class PlanEstimate(Record):
    """The resource estimate of a whole Plan with the device forecasts of its resolved points.

    `nwqlib.estimate(plan, profile=profile, allocation=allocation)` returns it.
    Without a profile and allocation, `estimate` returns the
    [`WorkloadEstimate`](resources.md) alone. The
    fields below are read-only. All assessments share one assessment time.
    Points that an adaptive method has not chosen yet, and points along a range
    axis that is not expanded, are listed in `unpredicted` instead of being
    guessed. When a Run is prepared with this estimate, each submission looks up
    the forecast of its own point, and later timings can be compared with it by
    [`align_telemetry`][nwqlib.backends.telemetry.align_telemetry].

    Attributes:
        plan_id: Content hash of the Plan.
        resources: Resource estimate of the whole Plan, independent of any
            profile.
        assessments: One
            [`ProfileAssessment`][nwqlib.backends.assessment.ProfileAssessment]
            per resolved point.
        profile: The [`DeviceProfile`][nwqlib.backends.profiles.DeviceProfile],
            or `None` when only an allocation was supplied.
        allocation: The [`Allocation`][nwqlib.backends.profiles.Allocation], or
            `None`.
        assessed_at: The assessment time, in UTC.
        unpredicted: Why each point without a forecast has none.
    """

    plan_id: ContentID
    resources: WorkloadEstimate
    assessments: tuple[ProfileAssessment, ...]
    profile: DeviceProfile | None
    allocation: Allocation | None
    assessed_at: Timestamp
    unpredicted: tuple[Text, ...] = ()
    _point_map: dict = PrivateAttr(default_factory=dict)
    _plan_validated: bool = PrivateAttr(default=False)
    _observations: dict = PrivateAttr(default_factory=dict)

    @model_validator(mode="after")
    def _sources(self):
        """Tie every assessment to this estimate and index it by Realization.

        Each assessment must name this Plan, its base construction, the stored
        profile and allocation, the one assessment time and the same resource
        context. At most one assessment exists per Realization, so
        ``assessment_for`` finds a receipt's forecast by a dictionary lookup.
        """
        if self.assessments and (self.profile is None or self.allocation is None):
            raise ValueError("device assessments require their actual profile and Allocation")
        self._point_map = {}
        for assessment in self.assessments:
            point = assessment.point
            if (
                point.plan_id != self.plan_id
                or point.base_construction_id != self.resources.construction_id
                or point.profile_id != self.profile.content_id
                or point.allocation_id != self.allocation.content_id
                or point.assessed_at != self.assessed_at
                or point.realization_id in self._point_map
                or assessment.resources.context != self.resources.context
            ):
                raise ValueError(
                    "Plan estimate must preserve each original forecast point and its model context"
                )
            self._point_map[point.realization_id] = assessment
        return self

    def validate_plan(self, plan):
        """Check that every stored assessment belongs to this Plan and carries no invented runtime seed."""
        if (
            self.plan_id != plan.content_id
            or self.resources.construction_id != plan.construction.content_id
        ):
            raise ValueError("estimate belongs to another selected Plan")
        frame = ErrorFrame.from_output(plan.problem, plan.output) if self.assessments else None
        for assessment in self.assessments:
            if assessment.point.runtime_options_id is not None:
                raise ValueError("Plan-wide estimates cannot invent future backend seeds")
            assessment.validate_context(plan, self.profile, self.allocation, frame=frame)
        self._plan_validated = True
        return self

    def assessment_for(self, plan, receipt):
        """Return the stored forecast of a prepared circuit's point, or None, without evaluating anything.

        The forecast is checked against the Plan once per estimate, and against the
        preparation record each time.

        Args:
            plan (Plan): The Plan this estimate belongs to.
            receipt (PreparedArtifact): The preparation record of one prepared circuit.

        Returns:
            assessment (ProfileAssessment | None): The forecast of that circuit's point, or `None`
                when the estimate has no forecast for it.

        Raises:
            ValueError: If the estimate belongs to another Plan, or the forecast does
                not match the preparation record.
        """
        if (
            self.plan_id != plan.content_id
            or self.resources.construction_id != plan.construction.content_id
        ):
            raise ValueError("estimate belongs to another selected Plan")
        assessment = self._point_map.get(receipt.realization_id)
        if assessment is None:
            return None
        if not self._plan_validated:
            self.validate_plan(plan)
        observation = self._observations.get(receipt.realization_id)
        if observation is None:
            observation = assessment.realization.resolved_observation(plan)[1]
            self._observations[receipt.realization_id] = observation
        return assessment.validate_prepared(plan, receipt.realization, receipt, observation=observation)


def _forecast_rows(points, profile, max_assessments):
    """Bound the actual assessment and prediction records before model evaluation."""
    if type(max_assessments) is not int or max_assessments < 1:
        raise ValueError("max_assessments must be a positive integer")
    required = points * (1 + len(profile.models))
    if required > max_assessments:
        raise ValueError(
            f"profile forecast requires {required} assessment/prediction rows; max_assessments={max_assessments}"
        )
    return required


def _profile_points(plan):
    """Resolve the Plan's Experiments whose next parameter point is already fixed.

    A controller's future point or an unexpanded range axis is listed as
    unpredicted instead of being guessed. A point with remaining symbolic
    parameters is assessed but also listed, because its forecast does not
    cover a later bound acquisition.
    """
    points = []
    unpredicted = []
    names = {p.name for p in plan.construction.program.parameters}
    nodes = {d.id: d.node for d in plan.construction.program.definitions}
    for experiment in plan.experiments:
        known = set(plan._binding_map)
        if experiment.batch is not None:
            batch = nodes[experiment.batch]
            known.update(b.parameter for b in batch.settings[experiment.setting_index].bindings)
            if any(axis.parameter not in known for axis in batch.axes):
                unpredicted.append(
                    f"{experiment.name}: future range-axis points are not expanded for forecasting"
                )
                continue
        missing = names - known
        try:
            point = plan.resolve(experiment.name)
        except KeyError as error:
            if error.args and error.args[0] in missing:
                unpredicted.append(
                    f"{experiment.name}: the controller has not selected its next parameter point"
                )
                continue
            raise
        if missing:
            unpredicted.append(
                f"{experiment.name}: unresolved parameters remain symbolic; no forecast covers a later bound acquisition"
            )
        points.append(point)
    return tuple(points), tuple(unpredicted)


def estimate_plan(
    plan,
    *,
    context=None,
    profile=None,
    allocation=None,
    assessed_at=None,
    accuracy=None,
    facts=(),
    reference=None,
    max_assessments=4096,
    profile_points=None,
):
    """Read the whole selected body plus supplied device models, never execute.

    max_assessments counts actual point-assessment and model-prediction rows.
    The entire requested expansion is checked before the first model evaluation.
    Profile-free logical folding is independent of this forecast-output cap.
    Its default of 4096 rows is an output-size control registered in
    docs/ENGINEERING_CONSTANTS.md ("Resource forecast and explicit inspection
    bounds").

    One fold memo (``resources.estimate``) lives for this call: the base
    construction folds first, and every point whose selected construction and
    context equal an already folded pair reuses that fold, so each distinct
    pair folds once. ``profile_points`` is the ``(points, unpredicted)`` pair
    that ``_profile_points(plan)`` returned to a caller that already needed it
    (``scientist.compare``); None evaluates it here.
    """
    if not isinstance(plan, Plan):
        raise TypeError("estimate_plan requires an actual Plan")
    if type(facts) is not tuple or any(not isinstance(f, FramedFact) for f in facts):
        raise TypeError("forecast facts must be immutable framed evidence")
    if reference is not None and not isinstance(reference, TargetReference):
        raise TypeError("forecast reference must be a framed TargetReference")
    if facts or reference is not None:
        if profile is None or plan.error_model is None or (accuracy is None and plan.selection_accuracy is None):
            raise ValueError("supplied accuracy evidence requires a profile, selected error model and Accuracy criterion")
    context = ResourceContext() if context is None else context
    at = _assessment_time(assessed_at)
    points, unpredicted = (), ()
    if profile is not None:
        if not isinstance(profile, DeviceProfile) or not isinstance(allocation, Allocation):
            raise ValueError(
                "device predictions require a supplied DeviceProfile and explicit Allocation"
            )
        points, unpredicted = _profile_points(plan) if profile_points is None else profile_points
        _forecast_rows(len(points), profile, max_assessments)
    elif allocation is not None:
        if not isinstance(allocation, Allocation):
            raise TypeError("allocation must be a supplied Allocation")
        unpredicted = (
            "no device profile supplied; Allocation is kept without invented device predictions",
        )
    memo = {}
    resources = estimate(plan.construction, context=context, memo=memo)
    # validate_plan below validates each assessment once.
    assessments = tuple(
        _assess(
            plan,
            point,
            profile=profile,
            allocation=allocation,
            context=context,
            runtime=None,
            assessed_at=at,
            accuracy=accuracy,
            facts=facts,
            reference=reference,
            max_assessments=max_assessments,
            memo=memo,
        )
        for point in points
    )
    return PlanEstimate(
        plan_id=plan.content_id,
        resources=resources,
        assessments=assessments,
        profile=profile,
        allocation=allocation,
        assessed_at=at,
        unpredicted=unpredicted,
    ).validate_plan(plan)
