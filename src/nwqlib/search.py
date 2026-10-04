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
    """A Plan to rank with [`scan`][nwqlib.search.scan], with its optional device context.

    Build it with the Plan and optional keyword arguments, for example
    `Candidate(plan, allocation=allocation)`, and pass a tuple of candidates
    to `scan`. A Candidate changes nothing in its Plan. A different Method
    setting or shot count is a different Plan and a separate candidate, and
    `scan` neither creates nor tunes Method configurations.

    Args:
        plan: Required, positional. A Plan from [`plan`][nwqlib.scientist.plan].
        allocation: The devices granted to a run of this Plan
            (`Allocation`), needed for device forecasts.
        facts: Accuracy evidence (`FramedFact` records) to assess with this
            candidate's forecast.
        reference: Target reference (`TargetReference`) to assess with it.
            No reference value is computed.
        prior_work: Records (`Fact`) of work done earlier for this candidate.
            Each supplied record is kept, repeated ones included. An empty
            tuple means that no earlier work is recorded, not that it cost
            nothing.

    """

    plan: Plan
    allocation: Allocation | None = field(default=None, kw_only=True)
    facts: tuple[FramedFact, ...] = field(default=(), kw_only=True)
    reference: TargetReference | None = field(default=None, kw_only=True)
    prior_work: tuple[Fact, ...] = field(default=(), kw_only=True)

class Objective(Record):
    """A quantity that [`scan`][nwqlib.search.scan] minimizes over its candidates.

    Build it with keyword arguments, for example
    `Objective(kind="requested_shots")`, and pass a tuple of objectives to
    `scan`. `kind` is the only argument required. `model_id` and `scope` are
    given for `predicted_seconds` only, and both are then required.

    - `requested_shots` sums the shots that the Plan's readouts request.
    - `logical_width` is the largest width, in qubits, of one execution at
      one location. It is not a sum over executions that run at the same
      time.
    - `predicted_seconds` sums the predictions of one device timing model
      and scope over the Plan's executions, under a serial schedule declared
      with `ResourceContext(batch_schedule="serial")`. The sum stays a
      prediction that holds only under its model.

    Attributes:
        kind: Required. `"requested_shots"`, `"logical_width"` or
            `"predicted_seconds"`.
        model_id: Default `None`. Content hash of the timing model whose
            predictions are summed, required for `predicted_seconds`.
        scope: Default `None`. Timing scope of those predictions:
            `"selected_acquisition"`, `"acquisition_overhead"` or
            `"native_call_wall"`. Required for `predicted_seconds`.

    Raises:
        ValueError: If `predicted_seconds` lacks `model_id` or `scope`, or
            another kind has either.
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
    """The value of one objective for one candidate, with what it rests on.

    [`scan`][nwqlib.search.scan] computes these values, one per candidate and
    objective, in `SearchSelection.values`. A value is `exact`, `conditional`
    on the assumptions it lists, or `unavailable` with a `reason`. A sum of
    binary64 time predictions is computed exactly but stays a conditional
    prediction. It does not bound the elapsed time or state that the
    predictions hold together. The fields below are read-only.

    Attributes:
        objective_id: Content hash of the [`Objective`][nwqlib.search.Objective]
            this value answers.
        value: Exact nonnegative rational value (`Rational`), or `None` when
            unavailable.
        unit: `count` for shots and width, `s` for predicted seconds.
        status: `"exact"`, `"conditional"` or `"unavailable"`.
        sources: Content hashes of the stored resource quantities, experiments
            or predictions that the value is computed from. Nonempty for an
            available value.
        assumptions: Conditions taken over from those sources.
        reason: Why the value is unavailable, or `None` for an available
            value.
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
    """How many resource estimates and device forecasts one scan computed or reused.

    [`scan`][nwqlib.search.scan] records it in `SearchSelection.work`. It
    counts estimates and forecast points, not CPU time. The fields below are
    read-only.

    Attributes:
        origin: `"evaluated_here"` when `scan` received candidates,
            `"stored_comparison"` when it rescored a Comparison.
        requested_points: Forecast points that the rows need, equal to
            `assessed_points + reused_points`.
        assessed_points: Forecast points evaluated by this call. Zero when
            rescoring.
        reused_points: Forecast points taken from an identical earlier row
            or from the stored Comparison.
        base_folds: Resource estimates or forecasts computed by this call.
            Zero when rescoring.
        reused_folds: Rows whose estimate or forecast came from an identical
            earlier row or from the stored Comparison.
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
    """The objective values of every row and the rows on the Pareto front.

    [`scan`][nwqlib.search.scan] returns it in `SearchResult.selection`. Row
    i dominates row j when every objective of i is at most that of j and at
    least one is smaller. The Pareto front, `nondominated`, is every row
    whose objectives are all available and that no other such row
    dominates, so equal rows all stay. A row with any unavailable objective
    is listed in `incomparable` instead of being ranked as zero or infinity.
    The front holds only within the rows given. It is not a feasibility
    filter or a global optimum. Validation recomputes both lists from the
    stored values with exact rational arithmetic. The fields below are read-only.

    Attributes:
        comparison_id: Content hash of the Comparison that this selection
            ranks, computed from its Problem and rows.
        objectives: The minimized objectives, in the order given.
        values: One [`ObjectiveValue`][nwqlib.search.ObjectiveValue] per row
            and objective, in row order.
        nondominated: Indices of the rows on the Pareto front, recomputed on
            validation.
        incomparable: Indices of the rows with an unavailable objective,
            recomputed on validation.
        prior_work: The supplied records of earlier work of each row, each
            record kept as often as it was supplied.
        work: The [`SearchWork`][nwqlib.search.SearchWork] of the scan that
            produced this selection.
        max_pair_comparisons: Default `1_000_000`. Largest number of
            coordinate comparisons of the front, `N*(N-1)*k` for N rows and
            k objectives, checked before any comparison.
        max_values: Default `4096`. Largest number of row and objective
            values, `N*k`, checked before any comparison.
        max_integer_bits: Default `4096`. Largest bit length of an integer in
            the exact rational arithmetic.
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
        definition with exact rational comparisons, which determine both lists.
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
        object.__setattr__(self, "nondominated", frontier)
        object.__setattr__(self, "incomparable", missing)
        return self

    def validate_comparison(self, comparison):
        """Check that this selection was computed for exactly the rows of `comparison`.

        Use it after loading a saved selection, for example with
        `SearchSelection.model_validate_json(...)`, to check that it belongs
        to a Comparison.

        Args:
            comparison (Comparison): The Comparison to check against.

        Returns:
            selection (SearchSelection): This selection, unchanged.

        Raises:
            ValueError: If the selection belongs to another Comparison or has
                another number of rows.
        """
        if self.comparison_id != _comparison_identity(comparison) or len(self.values) != len(
            comparison.rows
        ):
            raise ValueError("search selection belongs to another original comparison")
        return self


@dataclass(frozen=True)
class SearchResult:
    """The rows that a scan ranked and their Pareto front.

    [`scan`][nwqlib.search.scan] returns it. Read the front in
    `selection.nondominated`, then pass `select(index)` to
    [`solve`][nwqlib.scientist.solve] or
    [`prepare`][nwqlib.scientist.prepare] to run one row. Construction checks
    that the selection was computed for exactly this Comparison. The fields
    below are read-only.

    Attributes:
        comparison: The [`Comparison`][nwqlib.scientist.Comparison] of every
            candidate, including rows without a Plan or with unavailable
            values.
        selection: The [`SearchSelection`][nwqlib.search.SearchSelection]
            with every objective value, the Pareto front and the
            incomparable rows.
    """

    comparison: Comparison
    selection: SearchSelection

    def __post_init__(self):
        self.selection.validate_comparison(self.comparison)

    def select(self, index):
        """Return the row at `index`, unchanged, as `Comparison.select` does.

        Any row can be selected, including a dominated row or a row without a
        Plan. Running a row without a Plan raises `ApplicabilityError`.

        Args:
            index (int): Row position, from 0 to the number of rows minus 1.

        Returns:
            row (ComparisonRow): The row at that position.

        Raises:
            IndexError: If `index` is not an integer in that range.
        """
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
    """Rank Plans of one Problem by resource objectives and return their Pareto front.

    For each candidate, `scan` estimates the resources of its Plan, as
    [`estimate`][nwqlib.scientist.estimate] does, evaluates each objective
    and returns a [`SearchResult`][nwqlib.search.SearchResult]. Its
    `selection.nondominated` lists the rows that no other row beats on every
    objective, and `select(index)` returns a row to run. A ranking that rests
    on conditional values, such as model predictions, is conditional too.
    Passing an existing [`Comparison`][nwqlib.scientist.Comparison] rescores
    its stored estimates and forecasts, and then nothing is planned,
    estimated or predicted. Every row is an existing Plan. A different shot
    count or Method setting is a different Plan, supplied as another
    candidate. The candidate, value and comparison limits are checked before
    any estimate, model evaluation or comparison, and a profile's complete
    set of forecasts is checked before its first model evaluation. The
    [Rank candidate plans](../search.md) guide describes the objectives.

    Args:
        candidates (tuple[Candidate, ...] | Comparison): The
            [`Candidate`][nwqlib.search.Candidate] records of Plans of one
            Problem, or a Comparison to rescore.
        objectives (tuple[Objective, ...]): Distinct objectives, all
            minimized.
        profile (DeviceProfile | None): Device models for time forecasts.
            Every candidate then needs an Allocation. Not accepted when
            rescoring.
        context (ResourceContext | None): Counting options for the resource
            estimates. Omitted, `ResourceContext()` applies. Not accepted when
            rescoring.
        assessed_at (datetime | None): Time-zone-aware time of new
            forecasts. Omitted, the current time is used. Not accepted when
            rescoring.
        max_candidates (int): Default `256`. Largest number of rows.
        max_assessments (int): Default `4096`. Largest number of forecast
            rows evaluated, or read from a stored Comparison.
        max_pair_comparisons (int): Default `1_000_000`. Largest number of
            coordinate comparisons of the Pareto front, `N*(N-1)*k` for N
            rows and k objectives.
        max_values (int): Default `4096`. Largest number of row and
            objective values, `N*k`.
        max_integer_bits (int): Default `4096`. Largest bit length of an
            integer in the exact rational arithmetic.

    Returns:
        result (SearchResult): The ranked Comparison and its
            [`SearchSelection`][nwqlib.search.SearchSelection].

    Raises:
        TypeError: If `objectives` is not a nonempty tuple of Objective
            records, or `candidates` is neither a nonempty tuple of Candidate
            records nor a Comparison.
        ValueError: If the objectives are not distinct, a limit has an
            invalid value or is exceeded, the candidates plan different
            Problems, a profile is given without an Allocation on every
            candidate, or a Comparison is rescored with `profile`, `context`
            or `assessed_at`.

    Examples:
        Two Plans that differ only in their shots, ranked by requested shots:

        >>> from nwqlib import Expectation, plan, scan
        >>> from nwqlib.algorithms import ExpectationMethod
        >>> from nwqlib.search import Candidate, Objective
        >>> problem = Expectation(state=[1.0, 0.0],
        ...                       observable=[[1.0, 0.0], [0.0, -1.0]])
        >>> plans = tuple(plan(problem, method=ExpectationMethod(), shots=n,
        ...                    seed=7) for n in (8, 32))
        >>> search = scan(tuple(Candidate(p) for p in plans),
        ...               objectives=(Objective(kind="requested_shots"),))
        >>> print(search.selection.nondominated)
        (0,)
        >>> print(search.select(0).plan.shots)
        8
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
