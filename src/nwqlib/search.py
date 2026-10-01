"""Finite ranking of explicitly selected science and its original resource evidence."""

from dataclasses import dataclass, field
from hashlib import sha256
import json
from typing import Literal

from pydantic import model_validator

from nwqlib.backends.assessment import PlanEstimate, estimate_plan
from nwqlib.backends.profiles import Allocation, TimeScope
from nwqlib.core.planning import Plan
from nwqlib.core.records import ContentID, Rational, Record, Text, Unit
from nwqlib.evidence import Fact
from nwqlib.evidence._work import ExactArithmetic
from nwqlib.evidence.error_model import FramedFact, TargetReference
from nwqlib.operators.access import Count
from nwqlib.resources import ResourceContext, WorkloadEstimate
from nwqlib.scientist import Comparison, ComparisonRow

__all__ = [
    "Candidate",
    "Objective",
    "ObjectiveValue",
    "SearchWork",
    "SearchSelection",
    "SearchResult",
    "scan",
]


@dataclass(frozen=True)
class Candidate:
    """An existing Plan and supplied context; no acquisition overrides.

    Empty prior_work means no historical work is recorded, never zero cost.
    Occurrences in a supplied prior_work tuple are preserved without deduplication.

    Attributes:
        plan: Explicit existing selected Plan; the scan does not invent or optimize a Method configuration.
        allocation: Optional device allocation for assessment of this candidate.
        facts: Existing framed evidence supplied for the candidate's assessment.
        reference: Optional existing target reference, without another reference computation.
        prior_work: Optional previous scoped work used by assessment accounting.
    """

    plan: Plan
    allocation: Allocation | None = field(default=None, kw_only=True)
    facts: tuple[FramedFact, ...] = field(default=(), kw_only=True)
    reference: TargetReference | None = field(default=None, kw_only=True)
    prior_work: tuple[Fact, ...] = field(default=(), kw_only=True)

    def __post_init__(self):
        """Require the declared record types, so a candidate carries only existing selections and evidence."""
        if not isinstance(self.plan, Plan):
            raise TypeError("Candidate requires an already selected Plan")
        if self.allocation is not None and not isinstance(self.allocation, Allocation):
            raise TypeError("Candidate allocation must be an explicit Allocation")
        if type(self.facts) is not tuple or any(
            not isinstance(fact, FramedFact) for fact in self.facts
        ):
            raise TypeError("Candidate facts must be a finite tuple of framed evidence")
        if type(self.prior_work) is not tuple or any(
            not isinstance(fact, Fact) for fact in self.prior_work
        ):
            raise TypeError(
                "Candidate prior_work must preserve a finite tuple of supplied occurrences"
            )
        if self.reference is not None and not isinstance(self.reference, TargetReference):
            raise TypeError("Candidate reference must be an existing framed TargetReference")


class Objective(Record):
    """One minimized scoped law; model_id/scope belong only to predicted_seconds.

    requested_shots sums actual selected readouts. logical_width takes the
    largest exact per-location width of an individual acquisition, not a sum
    of simultaneous jobs. predicted_seconds sums one model/scope under the
    explicitly serial ResourceContext, keeping conditional prediction meaning.

    Attributes:
        kind: ``requested_shots``, ``logical_width`` or ``predicted_seconds``, each minimized.
        model_id: Timing model whose predictions are summed, for ``predicted_seconds`` only.
        scope: Timing scope of those predictions, for ``predicted_seconds`` only.
    """

    kind: Literal["requested_shots", "logical_width", "predicted_seconds"]
    model_id: ContentID | None = None
    scope: TimeScope | None = None

    @model_validator(mode="after")
    def _model(self):
        if self.kind == "predicted_seconds":
            if self.model_id is None or self.scope is None:
                raise ValueError("predicted time objective requires one explicit model and scope")
        elif self.model_id is not None or self.scope is not None:
            raise ValueError("model/scope belong only to the predicted time objective")
        return self


class ObjectiveValue(Record):
    """objective_id binds value/unit; sources and assumptions keep its actual basis.

    status is exact, conditional or unavailable; reason is required only for
    unavailable values. Exact sums of binary64 predictions remain conditional
    predictions, not certified elapsed-time bounds or simultaneous coverage.

    Attributes:
        objective_id: Content identity of the Objective this value answers.
        value: Exact rational value, or None when unavailable.
        unit: ``count`` for shots and width, ``s`` for predicted seconds.
        status: ``exact``, ``conditional`` or ``unavailable``.
        sources: Identities of the stored quantities, experiments or predictions the value reads.
        assumptions: Conditions inherited from those sources.
        reason: Why the value is unavailable, or None.
    """

    objective_id: ContentID
    value: Rational | None
    unit: Unit
    status: Literal["exact", "conditional", "unavailable"]
    sources: tuple[ContentID, ...]
    assumptions: tuple[Text, ...] = ()
    reason: Text | None = None

    @model_validator(mode="after")
    def _value(self):
        if (self.value is None) != (self.status == "unavailable"):
            raise ValueError("objective value must preserve its availability")
        if self.value is None:
            if self.reason is None:
                raise ValueError("unavailable objective requires its reason")
        elif self.value.numerator < 0 or self.reason is not None or not self.sources:
            raise ValueError(
                "available objective requires nonnegative value, sources and no missing reason"
            )
        return self


class SearchWork(Record):
    """Forecast and fold work done or reused by one scan. No CPU time is measured.

    Attributes:
        origin: ``evaluated_here`` for new candidates, ``stored_comparison`` for a rescored Comparison.
        requested_points: Forecast points the rows need, equal to assessed plus reused points.
        assessed_points: Forecast points evaluated by this call.
        reused_points: Forecast points taken from an identical earlier row or the stored Comparison.
        base_folds: Forecasts or resource folds computed by this call.
        reused_folds: Rows whose forecast or fold came from an identical earlier row or the stored Comparison.
    """

    origin: Literal["evaluated_here", "stored_comparison"]
    requested_points: Count
    assessed_points: Count
    reused_points: Count
    base_folds: Count
    reused_folds: Count

    @model_validator(mode="after")
    def _counts(self):
        if self.requested_points != self.assessed_points + self.reused_points:
            raise ValueError(
                "requested forecast points must partition into evaluated and reused points"
            )
        if self.origin == "stored_comparison" and (self.assessed_points or self.base_folds):
            raise ValueError("stored comparison cannot claim new model or fold work")
        return self


class SearchSelection(Record):
    """Scoped strict Pareto frontier, including every original unavailable row.

    Row i dominates row j when every objective of i is at most that of j and
    at least one is smaller. The frontier is every fully available row that
    no other fully available row dominates, so equal rows all stay. A row
    with any unavailable objective is listed as incomparable instead of being
    ranked as zero or infinity. The validator recomputes both lists from the
    stored values with exact rational arithmetic and rejects supplied lists
    that differ, so a saved selection cannot claim a frontier its values do
    not support.

    Attributes:
        comparison_id: Identity of the Comparison these rows belong to.
        objectives: Ordered minimized objectives.
        values: One ObjectiveValue per row and objective, in row order.
        nondominated: Frontier row indices, recomputed on validation.
        incomparable: Indices of rows with an unavailable objective, recomputed on validation.
        prior_work: Supplied prior-work facts per row, kept with their multiplicity.
        work: Evaluation and reuse counts of the scan that produced this selection.
        max_pair_comparisons: Admission cap on ordered coordinate comparisons.
        max_values: Admission cap on row-objective cells.
        max_integer_bits: Exact-arithmetic integer bit limit.
    """

    comparison_id: ContentID
    objectives: tuple[Objective, ...]
    values: tuple[tuple[ObjectiveValue, ...], ...]
    nondominated: tuple[Count, ...] | None = None
    incomparable: tuple[Count, ...] | None = None
    prior_work: tuple[tuple[Fact, ...], ...]
    work: SearchWork
    max_pair_comparisons: Count = 1_000_000
    max_values: Count = 4096
    max_integer_bits: Count = 4096

    @model_validator(mode="after")
    def _rows(self):
        """Check the stored values and recompute the frontier and incomparable rows from them.

        Every row must hold one value per objective in objective order, with
        the objective's unit, integer counts for shots and width, and a
        predicted duration that is never marked exact. The size limits are
        checked before any comparison. The frontier then follows the class
        definition with exact rational comparisons. Supplied lists that
        differ reject, and absent lists are filled in.
        """
        if not self.values or not self.objectives or len(self.prior_work) != len(self.values):
            raise ValueError("search requires nonempty aligned original rows and objectives")
        _frontier_size(
            len(self.values), len(self.objectives), self.max_pair_comparisons, self.max_values
        )
        ids = tuple(o.content_id for o in self.objectives)
        if len(set(ids)) != len(ids):
            raise ValueError("search objectives must be distinct")
        for row in self.values:
            if tuple(value.objective_id for value in row) != ids:
                raise ValueError("objective values differ from their ordered laws")
            for objective, value in zip(self.objectives, row, strict=True):
                time = objective.kind == "predicted_seconds"
                if (value.unit.symbol, value.unit.dimension) != (
                    ("s", "time") if time else ("count", "count")
                ):
                    raise ValueError("objective unit differs from its law")
                if not time and value.value is not None and value.value.denominator != 1:
                    raise ValueError("shot and logical-width objectives require integer counts")
                if time and value.status == "exact":
                    raise ValueError("predicted duration is not an exact measured time")
        arithmetic = ExactArithmetic(max_integer_bits=self.max_integer_bits)
        numeric = tuple(
            tuple(None if v.value is None else arithmetic.fraction(v.value) for v in row)
            for row in self.values
        )
        comparable = tuple(i for i, row in enumerate(numeric) if None not in row)
        # a strictly dominates b iff every a_j <= b_j and at least one is <.
        # ponytail: bounded quadratic frontier; use a specialized skyline
        # algorithm if explicitly requested larger finite populations need it.
        frontier = tuple(
            i
            for i in comparable
            if not any(
                j != i
                and numeric[j] != numeric[i]
                and all(arithmetic.le(a, b) for a, b in zip(numeric[j], numeric[i], strict=True))
                for j in comparable
            )
        )
        missing = tuple(i for i, row in enumerate(numeric) if None in row)
        if self.nondominated is not None and self.nondominated != frontier:
            raise ValueError("frontier differs from the complete strict dominance relation")
        if self.incomparable is not None and self.incomparable != missing:
            raise ValueError("incomparable rows differ from unavailable objective values")
        object.__setattr__(self, "nondominated", frontier)
        object.__setattr__(self, "incomparable", missing)
        return self

    def validate_comparison(self, comparison):
        """Check that this selection was computed for exactly this Comparison's rows."""
        if self.comparison_id != _comparison_identity(comparison) or len(self.values) != len(
            comparison.rows
        ):
            raise ValueError("search selection belongs to another original comparison")
        return self


@dataclass(frozen=True)
class SearchResult:
    """The Comparison that a scan ranked and its SearchSelection.

    Construction checks that the selection was computed for exactly this
    Comparison. ``select`` returns an original row unchanged.

    Attributes:
        comparison: Actual assessed candidate rows, including unsuccessful or unresolved rows.
        selection: Rule outcome naming eligible or selected rows and unresolved conditions.
    """

    comparison: Comparison
    selection: SearchSelection

    def __post_init__(self):
        self.selection.validate_comparison(self.comparison)

    def select(self, index):
        """Return the original row, including a dominated or blocked one. Execution rejects a blocked row."""
        return self.comparison.select(index)


def _comparison_identity(comparison):
    """Hash existing immutable identifiers; no Plan serialization or hydration.

    The Comparison and its rows are frozen, so the identity is stored on the
    Comparison object at the first call (outside its dataclass fields, so
    equality ignores it) and later calls, such as ``SearchResult``
    validation, return it.
    """
    cached = comparison.__dict__.get("_identity")
    if cached is not None:
        return cached
    rows = tuple(
        (
            row.method.content_id,
            None if row.plan is None else row.plan.content_id,
            row.reason,
            None if row.estimate is None else row.estimate.content_id,
            None if row.allocation is None else row.allocation.content_id,
            tuple(f.content_id for f in row.facts),
            None if row.reference is None else row.reference.content_id,
            tuple(f.content_id for f in row.prior_work),
        )
        for row in comparison.rows
    )
    encoded = json.dumps(
        ("nwqlib.comparison/1", comparison.problem.content_id, rows), separators=(",", ":")
    ).encode()
    identity = "sha256:" + sha256(encoded).hexdigest()
    object.__setattr__(comparison, "_identity", identity)
    return identity


def _frontier_size(count, width, limit, max_values):
    """Admit N rows and k objectives before any value is computed.

    The table has N*k cells, at most ``max_values``. The frontier compares
    every ordered pair of distinct rows on every objective, N*(N-1)*k
    coordinate comparisons, at most ``limit``.
    """
    if type(max_values) is not int or max_values < 1:
        raise ValueError("max_values must be a positive integer")
    if count * width > max_values:
        raise ValueError("candidate/objective output exceeds max_values before evaluation")
    if type(limit) is not int or limit < 0:
        raise ValueError("max_pair_comparisons must be a nonnegative integer")
    if count * max(0, count - 1) * width > limit:
        raise ValueError("finite Pareto comparisons exceed max_pair_comparisons before evaluation")


def _objective_value(row, objective, arithmetic, profile_points):
    """Read one objective for one comparison row from its stored selection and evidence.

    Nothing is planned, folded or predicted here. The value comes from the
    row's Plan and its original WorkloadEstimate or PlanEstimate, which is why
    rescoring a stored Comparison costs no new model or fold work. Whenever
    the stored evidence does not fix the value, the result is unavailable
    with a reason, so an incomplete row cannot rank as if it cost nothing.

    - ``requested_shots``: planned terminal-batch shots from the fold plus
      the raw shots of direct (non-batch) Experiments, which the fold does
      not see. Adaptive Programs, provider-managed estimates and query
      templates without a complete population are unavailable.
    - ``logical_width``: the largest exact per-location logical width. A
      batch peak that holds only under a serial schedule counts only when
      the stored context declares ``batch_schedule="serial"``.
    - ``predicted_seconds``: under an explicit serial schedule, the sum of
      exactly one matching prediction (model_id and scope) per assessed
      acquisition point. It remains conditional as a model prediction.

    Sums and maxima use exact rational arithmetic within the scan's
    integer-bit limit.

    ``profile_points`` returns ``backends.assessment._profile_points`` of a
    Plan, evaluated once per Plan for the whole scan.
    """
    time = objective.kind == "predicted_seconds"
    unit = Unit(symbol="s" if time else "count", dimension="time" if time else "count")
    sources = []
    assumptions = []
    status = "conditional" if time else "exact"

    def result(value=None, reason=None):
        scalar = (
            None
            if value is None
            else Rational(numerator=value.numerator, denominator=value.denominator)
        )
        return ObjectiveValue(
            objective_id=objective.content_id,
            value=scalar,
            unit=unit,
            status="unavailable" if value is None else status,
            sources=tuple(dict.fromkeys(sources)),
            assumptions=tuple(dict.fromkeys(assumptions)),
            reason=reason,
        )

    if row.plan is None:
        return result(reason=row.reason or "blocked row has no selected workload")
    direct_shots = arithmetic.fraction(0)
    if objective.kind == "requested_shots":
        from nwqlib.ir import AdaptiveLoop

        direct = tuple(e for e in row.plan.experiments if e.batch is None)
        _, unresolved = profile_points(row.plan)
        if unresolved:
            return result(
                reason="selected query templates do not give a complete shot population: "
                + "; ".join(unresolved)
            )
        if any(isinstance(d.node, AdaptiveLoop) for d in row.plan.construction.program.definitions):
            return result(
                reason="direct readouts do not specify the complete adaptive acquisition population"
            )
        # The construction fold deliberately counts MeasurementBatch work only.
        # Direct Experiment readouts are separate actual Plan declarations;
        # their population must not disappear into that fold's structural zero.
        for experiment in direct:
            if experiment.observation.kind == "estimated_observable":
                return result(reason="provider-managed estimation has no selected raw shot count")
            direct_shots = arithmetic.add(
                direct_shots, arithmetic.fraction(experiment.observation.shots or 0)
            )
            sources.append(experiment.content_id)
        if all(e.batch is None for e in row.plan.experiments):
            sources.append(row.plan.content_id)
            return result(direct_shots)
    estimate = row.estimate
    resources = estimate.resources if isinstance(estimate, PlanEstimate) else estimate
    if not isinstance(resources, WorkloadEstimate):
        return result(reason="no original logical estimate is stored for this row")
    total = direct_shots if objective.kind == "requested_shots" else arithmetic.fraction(0)
    if not time:
        metric = "shots" if objective.kind == "requested_shots" else "logical_width"
        quantities = tuple(q for q in resources.quantities if q.metric == metric)
        if not quantities:
            return result(reason="selected resources lack the requested scoped quantity")
        for quantity in quantities:
            fact = quantity.fact
            conditional_width = (
                metric == "logical_width"
                and quantity.interpretation == "conditional"
                and quantity.required_schedule == "serial_acquisitions"
                and resources.context.batch_schedule == "serial"
            )
            if (
                fact.availability != "concrete"
                or quantity.interpretation != "exact"
                and not conditional_width
                or not isinstance(fact.value, Rational)
            ):
                return result(reason="objective requires exact concrete selected workload evidence")
            if (
                quantity.required_schedule is not None
                and resources.context.batch_schedule != "serial"
            ):
                return result(reason="quantity requires its declared serial acquisition schedule")
            value = arithmetic.fraction(fact.value)
            total = (
                arithmetic.add(total, value)
                if metric == "shots"
                else value
                if arithmetic.le(total, value)
                else total
            )
            sources.append(quantity.content_id)
            assumptions.extend(fact.assumptions)
            if fact.assumptions:
                status = "conditional"
        return result(total)
    if not isinstance(estimate, PlanEstimate) or not estimate.assessments:
        return result(reason="no original point forecasts exist for the requested model")
    estimate.validate_plan(row.plan)
    if resources.context.batch_schedule != "serial":
        return result(reason="duration summation requires an explicit serial acquisition schedule")
    if estimate.unpredicted:
        return result(
            reason="forecast does not cover the complete selected workload: "
            + "; ".join(estimate.unpredicted)
        )
    for assessment in estimate.assessments:
        matches = tuple(
            p
            for p in assessment.predictions
            if p.model_id == objective.model_id and p.scope == objective.scope
        )
        if len(matches) != 1:
            return result(reason="selected acquisition lacks one matching model/scope prediction")
        prediction = matches[0]
        sources.append(prediction.content_id)
        assumptions.extend(prediction.assumptions)
        if prediction.seconds is None:
            return result(
                reason="matching prediction is unavailable: " + "; ".join(prediction.reasons)
            )
        total = arithmetic.add(total, arithmetic.fraction(prediction.seconds))
    return result(total)


def scan(
    candidates,
    *,
    objectives,
    profile=None,
    context=None,
    assessed_at=None,
    max_candidates=256,
    max_assessments=4096,
    max_pair_comparisons=1_000_000,
    max_values=4096,
    max_integer_bits=4096,
):
    """Rank finite existing Plans, or rescore an unchanged original Comparison.

    Candidate evaluation folds selected laws and supplied models. Rescoring a
    Comparison reads its existing quantities only; no planning/folding/model
    call occurs. Ranking is conditional when its inputs are conditional.

    Every candidate is an existing Plan, and ``SearchResult.select`` returns
    one of these original rows. A different shot count or Method setting is a
    different Plan, supplied as another candidate. Candidate, cell and pair
    limits are checked before any fold, model evaluation or frontier
    comparison, and a profile's complete forecast expansion is checked
    before its first model evaluation.

    Args:
        candidates (tuple[Candidate, ...] | Comparison): Candidate rows for one Problem, or an existing Comparison to rescore.
        objectives (tuple[Objective, ...]): Distinct minimized objectives.
        profile (DeviceProfile | None): Optional device profile. It requires an Allocation on every candidate.
        context (ResourceContext | None): Context for new folds. Not accepted when rescoring.
        assessed_at (datetime | None): Timezone-aware assessment time for new forecasts. None uses the current time. Not accepted when rescoring.
        max_candidates (int): Limit on the number of rows.
        max_assessments (int): Limit on new or traversed stored forecast rows.
        max_pair_comparisons (int): Limit on ordered coordinate comparisons of the frontier.
        max_values (int): Limit on row-objective cells.
        max_integer_bits (int): Exact-arithmetic integer bit limit.

    Returns:
        result (SearchResult): The Comparison and its SearchSelection.
    """
    if (
        type(objectives) is not tuple
        or not objectives
        or any(not isinstance(o, Objective) for o in objectives)
    ):
        raise TypeError("objectives must be a nonempty finite tuple of Objective records")
    if len({o.content_id for o in objectives}) != len(objectives):
        raise ValueError("search objectives must be distinct")
    if type(max_candidates) is not int or max_candidates < 1:
        raise ValueError("max_candidates must be a positive integer")
    if type(max_assessments) is not int or max_assessments < 1:
        raise ValueError("max_assessments must be a positive integer")
    if not isinstance(candidates, Comparison) and (
        type(candidates) is not tuple
        or not candidates
        or any(not isinstance(c, Candidate) for c in candidates)
    ):
        raise TypeError("scan requires a finite Candidate tuple or existing Comparison")
    count = len(candidates.rows) if isinstance(candidates, Comparison) else len(candidates)
    if not count or count > max_candidates:
        raise ValueError("finite candidate count exceeds max_candidates")
    _frontier_size(count, len(objectives), max_pair_comparisons, max_values)
    arithmetic = ExactArithmetic(max_integer_bits=max_integer_bits)
    # One _profile_points evaluation per Plan object for this scan; the rows
    # keep their Plans alive, so the object id stays unique during the call.
    profiled = {}

    def profile_points(plan):
        if id(plan) not in profiled:
            from nwqlib.backends.assessment import _profile_points
            profiled[id(plan)] = _profile_points(plan)
        return profiled[id(plan)]

    if isinstance(candidates, Comparison):
        if any(v is not None for v in (profile, context, assessed_at)):
            raise ValueError(
                "stored comparison supplies its original contexts; rescoring cannot replace them"
            )
        comparison = candidates
        seen = set()
        stored = 0
        for row in comparison.rows:
            forecast = row.estimate
            if isinstance(forecast, PlanEstimate) and id(forecast) not in seen:
                seen.add(id(forecast))
                stored += len(forecast.assessments)
                if stored > max_assessments:
                    raise ValueError("stored forecast traversal exceeds max_assessments")
                stored += sum(len(a.predictions) for a in forecast.assessments)
                if stored > max_assessments:
                    raise ValueError("stored forecast traversal exceeds max_assessments")
        requested = sum(
            len(r.estimate.assessments)
            for r in comparison.rows
            if isinstance(r.estimate, PlanEstimate)
        )
        work = SearchWork(
            origin="stored_comparison",
            requested_points=requested,
            assessed_points=0,
            reused_points=requested,
            base_folds=0,
            reused_folds=sum(
                r.plan is not None and r.estimate is not None for r in comparison.rows
            ),
        )
    else:
        from nwqlib.backends.assessment import _assessment_time, _forecast_rows
        from nwqlib.resources import estimate as fold

        context = ResourceContext() if context is None else context
        problem = candidates[0].plan.problem
        if any(c.plan.problem.content_id != problem.content_id for c in candidates):
            raise ValueError(
                "finite method comparison requires the same original scientific Problem"
            )
        keys = tuple(
            (
                c.plan.content_id,
                None if c.allocation is None else c.allocation.content_id,
                tuple(f.content_id for f in c.facts),
                None if c.reference is None else c.reference.content_id,
            )
            for c in candidates
        )
        if profile is not None:
            from nwqlib.backends.profiles import DeviceProfile

            if not isinstance(profile, DeviceProfile) or any(
                c.allocation is None for c in candidates
            ):
                raise ValueError(
                    "model evaluation requires its actual DeviceProfile and every explicit Allocation"
                )
            unique = {key: c for key, c in zip(keys, candidates, strict=True)}
            points = sum(len(profile_points(c.plan)[0]) for c in unique.values())
            _forecast_rows(points, profile, max_assessments)
        at = _assessment_time(assessed_at)
        forecasts = {}
        folds = {}
        rows = []
        requested = assessed = reused = base_folds = reused_folds = 0
        for key, candidate in zip(keys, candidates, strict=True):
            selected = candidate.plan
            if profile is not None or candidate.allocation is not None:
                if key in forecasts:
                    estimate = forecasts[key]
                    reused += len(estimate.assessments)
                    reused_folds += 1
                else:
                    estimate = estimate_plan(
                        selected,
                        profile=profile,
                        allocation=candidate.allocation,
                        context=context,
                        assessed_at=at,
                        facts=candidate.facts,
                        reference=candidate.reference,
                        max_assessments=max_assessments,
                    )
                    forecasts[key] = estimate
                    assessed += len(estimate.assessments)
                    base_folds += 1
                requested += len(estimate.assessments)
            else:
                fold_key = selected.construction.content_id
                if fold_key in folds:
                    estimate = folds[fold_key]
                    reused_folds += 1
                else:
                    estimate = fold(selected.construction, context=context)
                    folds[fold_key] = estimate
                    base_folds += 1
            rows.append(
                ComparisonRow(
                    selected.method,
                    selected,
                    estimate=estimate,
                    allocation=candidate.allocation,
                    facts=candidate.facts,
                    reference=candidate.reference,
                    prior_work=candidate.prior_work,
                )
            )
        comparison = Comparison(problem, tuple(rows))
        work = SearchWork(
            origin="evaluated_here",
            requested_points=requested,
            assessed_points=assessed,
            reused_points=reused,
            base_folds=base_folds,
            reused_folds=reused_folds,
        )
    values = tuple(
        tuple(_objective_value(row, o, arithmetic, profile_points) for o in objectives)
        for row in comparison.rows
    )
    selection = SearchSelection(
        comparison_id=_comparison_identity(comparison),
        objectives=objectives,
        values=values,
        prior_work=tuple(row.prior_work for row in comparison.rows),
        work=work,
        max_pair_comparisons=max_pair_comparisons,
        max_values=max_values,
        max_integer_bits=max_integer_bits,
    )
    return SearchResult(comparison, selection)
