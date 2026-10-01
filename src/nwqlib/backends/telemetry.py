"""Bounded projection of existing forecasts and execution records; no acquisition.

The execution trace is the event owner. This module neither appends events nor
replays assessment, folds resources, fits models, or interprets scientific output.

A residual ``observed - predicted`` is formed only for the assessment bound to
an attempt before it started, a completed acquisition and an exact timing of
the forecast's scope. Every other pairing is kept with its reason, so
failures and censored timings stay visible. docs/profiles.md, "Compare
forecasts with measured times", defines this contract.
"""

from math import isfinite

from pydantic import model_validator

from nwqlib.backends.assessment import ProfileAssessment, TimePrediction
from nwqlib.core.analysis import Result, RunData
from nwqlib.core.records import ContentID, Real, Record, Text
from nwqlib.execution import (
    ConsumptionEvent,
    ExecutionTrace,
    ObservationView,
    PreparedArtifact,
    TimingObservation,
)


class AttemptTiming(Record):
    """A timing that the caller measured for one attempt of a Run, to pair with its forecast.

    Build it with keyword arguments, for example
    `AttemptTiming(run_id=..., attempt=..., prepared_id=..., timing=...)`,
    and pass it in `timings=` of
    [`align_telemetry`][nwqlib.backends.telemetry.align_telemetry]. Every field is
    required. The timing is either an exact duration or a right-censored lower
    bound, with its source. Distinct attempts stay distinct even when their
    timings agree.

    Attributes:
        run_id: Required. ID of the Run.
        attempt: Required. ID of the attempt in the Run's execution record.
        prepared_id: Required. Content hash of the preparation record the attempt
            used.
        timing: Required. The `TimingObservation`: scope, nonnegative seconds,
            `"exact"` or `"right_censored"`, and source. A right-censored timing
            needs the reason for its cutoff.
    """

    run_id: Text
    attempt: Text
    prepared_id: ContentID
    timing: TimingObservation


class TimingComparison(Record):
    """One original forecast and matching timing, or explicit missing evidence.

    residual_seconds is signed observed minus predicted seconds, without a
    coverage claim. reasons explains every unavailable residual. timing_id None
    means no measurement of the forecast's scope was supplied for this attempt.
    """

    prediction_id: ContentID
    timing_id: ContentID | None
    residual_seconds: Real | None
    reasons: tuple[Text, ...]

    @model_validator(mode="after")
    def _availability(self):
        if (self.residual_seconds is None) != bool(self.reasons):
            raise ValueError("unavailable residual requires reasons; available residual has none")
        return self


class TelemetryRow(Record):
    """Projection of one source event without histogram or native payload copies.

    event keeps reservations, exposure and terminal outcome unchanged. timings
    combines its native measurement with supplied scoped measurements. collected
    IDs name actual supplied chunks; contribution IDs name their use by the
    separately supplied result. Neither population proves output completeness.
    comparisons keep predictions even when timing cannot support a residual.
    """

    event: ConsumptionEvent
    timings: tuple[TimingObservation, ...]
    collected_observation_ids: tuple[ContentID, ...]
    contribution_ids: tuple[ContentID, ...]
    comparisons: tuple[TimingComparison, ...]
    reasons: tuple[Text, ...]

    @model_validator(mode="after")
    def _membership(self):
        """Keep the row tied to its one event.

        Timings are distinct and include the event's own native timing when it
        has one.
        Collected observations can only be the event's own observation, and
        result contributions are a subset of them, so a row cannot borrow
        another attempt's data. Each (prediction, timing) pair appears once.
        """
        if len({t.content_id for t in self.timings}) != len(self.timings):
            raise ValueError("duplicate timing measurement")
        if self.event.timing is not None and self.event.timing not in self.timings:
            raise ValueError("source event timing must remain kept")
        if (
            len(set(self.collected_observation_ids)) != len(self.collected_observation_ids)
            or len(set(self.contribution_ids)) != len(self.contribution_ids)
            or not set(self.contribution_ids) <= set(self.collected_observation_ids)
            or any(value != self.event.observation_id for value in self.collected_observation_ids)
        ):
            raise ValueError(
                "collected/result contributions must belong to the event's actual receipt"
            )
        keys = [(c.prediction_id, c.timing_id) for c in self.comparisons]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate timing comparison")
        return self


class PredictionLedger(Record):
    """The forecasts of one Run paired with its observed timings, attempt by attempt.

    [`align_telemetry`][nwqlib.backends.telemetry.align_telemetry] returns it. The
    fields below are read-only. Each row holds one attempt's execution event, its
    timings, the observations it collected and its forecast comparisons, where
    `residual_seconds` is the signed `observed - predicted` seconds or `None`
    with the reasons. Collected observations and the contributions of the
    Result stay separate lists. Use `revise` to record a new version of supplied
    data. It never fits coefficients or changes the original
    assessments, model domains, calibration status or coverage. A forecast is
    valid at its original assessment time, and a historical residual does not
    establish that calibration was valid when the job ran.

    Attributes:
        trace_id: Content hash of the Run's execution record.
        run_id: ID of the Run.
        plan_id: Content hash of the Plan.
        assessment_ids: Content hashes of the forecasts used.
        predictions: The original
            [`TimePrediction`][nwqlib.backends.assessment.TimePrediction] records.
        rows: One row per attempt, failures, partial collections and
            right-censored timings included.
        result_id: Content hash of the analysis Result, or `None`. It does not
            imply that the Result is scientifically valid.
    """

    trace_id: ContentID
    run_id: Text
    plan_id: ContentID
    assessment_ids: tuple[ContentID, ...]
    predictions: tuple[TimePrediction, ...]
    rows: tuple[TelemetryRow, ...]
    result_id: ContentID | None

    @model_validator(mode="after")
    def _relations(self):
        """Accept a residual only for its completed, pre-bound forecast and an exact timing of the same scope."""
        if (
            len(set(self.assessment_ids)) != len(self.assessment_ids)
            or len({p.content_id for p in self.predictions}) != len(self.predictions)
            or len({row.event.attempt for row in self.rows}) != len(self.rows)
        ):
            raise ValueError("ledger source identities must be distinct")
        predictions = {p.content_id: p for p in self.predictions}
        for row in self.rows:
            timings = {t.content_id: t for t in row.timings}
            if self.result_id is None and row.contribution_ids:
                raise ValueError("result contributions require their actual result identity")
            for comparison in row.comparisons:
                prediction = predictions.get(comparison.prediction_id)
                timing = timings.get(comparison.timing_id)
                if prediction is None or (comparison.timing_id is not None and timing is None):
                    raise ValueError("comparison references unavailable prediction/timing")
                if comparison.residual_seconds is not None and (
                    row.reasons
                    or row.event.assessment_id not in self.assessment_ids
                    or row.event.status != "completed"
                    or timing is None
                    or timing.censoring != "exact"
                    or timing.scope != prediction.scope
                    or prediction.seconds is None
                    or comparison.residual_seconds != timing.seconds - prediction.seconds.value
                ):
                    raise ValueError(
                        "residual must match its original completed forecast/timing association"
                    )
        return self

    def validate_context(self, *, trace, assessments):
        """Check that a saved `PredictionLedger` still matches its execution record and forecasts, without recomputing anything.

        Args:
            trace (ExecutionTrace): The Run's execution record.
            assessments (tuple[ProfileAssessment, ...]): The forecasts the record
                used.

        Returns:
            record (PredictionLedger): This record.

        Raises:
            ValueError: If the record differs from the execution record, uses a
                forecast of another Plan or attempt, or changed its predictions.
        """
        if (
            self.trace_id != trace.content_id
            or self.plan_id != trace.plan_id
            or self.run_id != trace.run_id
            or tuple(row.event for row in self.rows) != trace.events
        ):
            raise ValueError("ledger differs from its original execution trace")
        supplied = {a.content_id: a for a in assessments}
        forecasts, used = {}, {}
        for row in self.rows:
            assessment = supplied.get(row.event.assessment_id)
            if assessment is None:
                if row.comparisons:
                    raise ValueError("ledger requires its originally bound assessment")
                continue
            if assessment.point.plan_id != self.plan_id:
                raise ValueError("ledger assessment belongs to another Plan")
            used[assessment.content_id] = assessment
            members = {p.content_id: p for p in assessment.predictions}
            forecasts.update(members)
            if any(c.prediction_id not in members for c in row.comparisons):
                raise ValueError("comparison uses another attempt's forecast")
        if tuple(used) != self.assessment_ids or tuple(forecasts.values()) != self.predictions:
            raise ValueError("ledger changed its original model predictions or source membership")
        return self


def align_telemetry(
    run_or_result=None,
    *,
    trace: ExecutionTrace | None = None,
    assessments: tuple[ProfileAssessment, ...] = (),
    receipts: tuple[PreparedArtifact, ...] = (),
    observations: ObservationView | None = None,
    timings: tuple[AttemptTiming, ...] = (),
    result: Record | None = None,
    # Default registered in docs/ENGINEERING_CONSTANTS.md ("Resource forecast and
    # explicit inspection bounds"), an output-size control, not a model domain.
    max_comparisons: int = 100_000,
) -> PredictionLedger:
    """Pair the forecasts of a Run with its observed timings and return the signed residuals.

    Pass a Run or a completed Result, for example `align_telemetry(run)`, to read
    its execution record, preparation records, observations and the forecast it
    was prepared with. The returned `PredictionLedger` keeps every attempt, failures and
    partial collections included. A residual `observed_seconds - predicted_seconds`
    is formed only when the attempt was bound to its forecast before it started,
    its data collection completed, and an exact timing of the forecast's scope
    exists. Every other attempt keeps its forecast with the reason it has no
    residual, so a runtime or compiler mismatch, a missing scope or preparation
    record, a failed collection or a right-censored timing stays visible. A
    right-censored timing is a lower bound, never an exact residual. Nothing is
    executed, assessed again or fitted, no provider is contacted, and no
    histogram or backend payload is copied into it. A residual makes no
    coverage claim, and a historical residual does not establish that the device
    calibration was valid when the job ran. The
    [telemetry section of the profiles guide](../profiles.md#original-forecasts-and-observed-telemetry)
    describes the pairing rules.

    Args:
        run_or_result (Run | Result | None): The Run or Result to read. With
            `None`, pass `trace` and the other records explicitly.
        trace (ExecutionTrace | None): Execution record, when no Run or Result is
            given.
        assessments (tuple[ProfileAssessment, ...]): Forecasts to pair. Empty
            uses the forecast the Run was prepared with.
        receipts (tuple[PreparedArtifact, ...]): Preparation records, when no Run
            or Result is given.
        observations (ObservationView | None): Observations, when no Run or Result
            is given.
        timings (tuple[AttemptTiming, ...]): Additional supplied timings. Each
            must belong to an attempt of this run. Identical timings of one
            attempt count once, and different attempts stay distinct.
        result (Record | None): The analysis Result whose contributions are
            recorded, when no Run or Result is given.
        max_comparisons (int): Positive limit on the forecast
            and timing pairs formed.

    Returns:
        record (PredictionLedger): The forecasts paired with the timings.

    Raises:
        TypeError: If no execution record or `RunData` is available.
        ValueError: If a Run or Result is mixed with explicit records, a timing
            or record belongs to another run, attempt or Plan, identities
            conflict, or the pairs exceed `max_comparisons`.
    """
    if type(max_comparisons) is not int or max_comparisons < 1:
        raise ValueError("max_comparisons must be a positive integer")
    if run_or_result is not None:
        if trace is not None or receipts or observations is not None or result is not None:
            raise ValueError(
                "a Run/Result cannot be mixed with a different trace or acquisition data"
            )
        data = run_or_result.data
        if not isinstance(data, RunData):
            raise TypeError("telemetry requires actual RunData")
        trace, receipts, observations = data.trace, data.receipts, data.observations
        if not assessments and data.forecast is not None:
            assessments = data.forecast.assessments
        result = run_or_result if isinstance(run_or_result, Result) else run_or_result.result
    if not isinstance(trace, ExecutionTrace):
        raise TypeError("telemetry requires an actual execution trace")
    observations = ObservationView() if observations is None else observations
    comparison_count = 0
    assessment_map = {a.content_id: a for a in assessments}
    receipt_map = {r.content_id: r for r in receipts}
    events = {event.attempt: event for event in trace.events}
    if len(events) != len(trace.events):
        raise ValueError("trace has conflicting or duplicate attempt identities")
    if any(a.point.plan_id != trace.plan_id for a in assessment_map.values()):
        raise ValueError("supplied assessment belongs to another Plan")
    if any(r.plan_id != trace.plan_id for r in receipt_map.values()):
        raise ValueError("supplied preparation belongs to another Plan")
    # Admit timing and observation identities before associating them with
    # a forecast, so unrelated runs cannot produce plausible-looking residuals.
    supplied = {}
    for sample in timings:
        event = events.get(sample.attempt)
        if (
            sample.run_id != trace.run_id
            or event is None
            or sample.prepared_id != event.prepared_id
        ):
            raise ValueError("supplied timing belongs to another run/attempt/prepared receipt")
        supplied.setdefault(sample.attempt, {})[sample.timing.content_id] = sample.timing
    # A completed event names its observation: one chunk's identity, or for
    # a trajectory the identity of its point chunks in schedule order
    # (ObservationView(chunks=ordered)). Point chunks join their receipt
    # through declares_readout_of and are grouped by attempt first.
    collected, points = {}, {}
    for chunk in observations.chunks:
        trace.validate_observation(chunk)
        event = events.get(chunk.attempt)
        receipt = receipt_map.get(chunk.prepared_id)
        if (
            chunk.run_id != trace.run_id
            or chunk.plan_id != trace.plan_id
            or event is None
            or event.prepared_id != chunk.prepared_id
            or (chunk.point is None and event.observation_id != chunk.content_id)
            or event.returned_shots != chunk.returned_shots
        ):
            raise ValueError("supplied observation differs from its actual event")
        if receipt is not None and (
            receipt.realization_id != chunk.realization_id
            or not chunk.declares_readout_of(receipt)
            or receipt.population != chunk.population
        ):
            raise ValueError(
                "supplied observation differs from its prepared selection/readout/population"
            )
        if chunk.point is None:
            collected[chunk.content_id] = chunk.attempt
        else:
            points.setdefault(chunk.attempt, []).append(chunk)
    for attempt, chunks in points.items():
        receipt = receipt_map.get(chunks[0].prepared_id)
        if receipt is None:
            # Without the receipt the schedule order is unknown, so the points
            # are not joined; the row keeps its missing-receipt reason.
            continue
        ordered = [None] * len(receipt.observation.positions)
        for chunk in chunks:
            slot = receipt.observation.point_index(chunk.point)
            if ordered[slot] is not None:
                raise ValueError("supplied trajectory repeats a declared point")
            ordered[slot] = chunk
        if None in ordered:
            continue  # A partly collected trajectory is not its event's observation.
        identity = ObservationView(chunks=tuple(ordered)).content_id
        if events[attempt].observation_id != identity:
            raise ValueError("supplied observation differs from its actual event")
        collected[identity] = attempt
    contributions = ()
    if result is not None:
        if not isinstance(result, Result) or result.data.trace.run_id != trace.run_id:
            raise ValueError("result must belong to this actual run")
        if getattr(result, "plan_id", None) != trace.plan_id or not hasattr(
            result, "contribution_ids"
        ):
            raise ValueError("result must name this Plan and its actual contribution identities")
        contributions = result.contribution_ids
        if (
            len(set(contributions)) != len(contributions)
            or not set(contributions) <= collected.keys()
        ):
            raise ValueError("result contributions require supplied actual collected observations")
    contribution_ids = set(contributions)
    # Keep every attempted event, including missing evidence. Only an
    # assessment bound before that attempt is eligible for a forecast comparison.
    rows, forecasts, used_assessments = [], {}, {}
    for event in trace.events:
        reasons = []
        assessment = assessment_map.get(event.assessment_id)
        receipt = receipt_map.get(event.prepared_id)
        if event.assessment_id is None:
            reasons.append("no assessment was bound before this attempt")
        elif assessment is None:
            reasons.append("the originally bound assessment was not supplied")
        if receipt is None:
            reasons.append("the actual prepared receipt was not supplied")
        if assessment is not None:
            used_assessments[event.assessment_id] = assessment
            if receipt is not None and (
                assessment.point.realization_id != receipt.realization_id
                or assessment.point.construction_id != receipt.construction_id
                or (
                    assessment.point.runtime_options_id is not None
                    and assessment.point.runtime_options_id != receipt.runtime.content_id
                )
            ):
                reasons.append(
                    "assessment differs from the actual prepared Realization/construction/runtime"
                )
            if receipt is not None and (
                assessment.point.runtime_source_id != receipt.target.content_id
                or assessment.point.compiler_source_id != receipt.compiler.content_id
            ):
                reasons.append(
                    "actual runtime/compiler differs from the supplied forecast configuration"
                )
        measurements = dict(supplied.get(event.attempt, {}))
        if event.timing is not None:
            measurements[event.timing.content_id] = event.timing
        predictions = () if assessment is None else assessment.predictions
        by_scope = {}
        for timing in measurements.values():
            by_scope.setdefault(timing.scope, []).append(timing)
        # Match timing scope and censoring before subtraction. An interrupted
        # or right-censored duration cannot be treated as an exact timing residual.
        comparisons = []
        for prediction in predictions:
            forecasts[prediction.content_id] = prediction
            matched = by_scope.get(prediction.scope, ())
            comparison_count += max(1, len(matched))
            if comparison_count > max_comparisons:
                raise ValueError("telemetry forecast/timing pairs exceed max_comparisons")
            for timing in matched or [None]:
                excluded = list(reasons)
                if event.status != "completed":
                    excluded.append("attempt has no completed acquisition receipt")
                if prediction.seconds is None:
                    excluded.append("original forecast is unavailable")
                if timing is None:
                    excluded.append("no timing of the forecast's scope was supplied")
                elif timing.censoring != "exact":
                    excluded.append("right-censored timing is a lower bound, not an exact residual")
                residual = None
                if not excluded:
                    residual = timing.seconds - prediction.seconds.value
                    if not isfinite(residual):
                        residual = None
                        excluded.append("residual exceeds finite binary64 range")
                comparisons.append(
                    TimingComparison(
                        prediction_id=prediction.content_id,
                        timing_id=None if timing is None else timing.content_id,
                        residual_seconds=residual,
                        reasons=tuple(excluded),
                    )
                )
        observed = (event.observation_id,) if event.observation_id in collected else ()
        rows.append(
            TelemetryRow(
                event=event,
                timings=tuple(measurements.values()),
                collected_observation_ids=observed,
                contribution_ids=tuple((value for value in observed if value in contribution_ids)),
                comparisons=tuple(comparisons),
                reasons=tuple(reasons),
            )
        )
    return PredictionLedger(
        trace_id=trace.content_id,
        run_id=trace.run_id,
        plan_id=trace.plan_id,
        assessment_ids=tuple(used_assessments),
        predictions=tuple(forecasts.values()),
        rows=tuple(rows),
        result_id=None if result is None else result.content_id,
    )
