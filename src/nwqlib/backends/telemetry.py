"""Bounded projection of existing forecasts and execution records; no acquisition.

The execution trace is the event owner. This module neither appends events nor
replays assessment, folds resources, fits models, or interprets scientific output.

A residual ``observed - predicted`` is formed only for the assessment bound to
an attempt before it started, a completed acquisition and an exact timing of
the forecast's scope. Every other pairing is kept with its reason, so
failures and censored timings stay visible. docs/profiles.md, "Original
forecasts and observed telemetry", defines this contract.
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
    """A supplied measurement associated with one actual run/attempt/receipt.

    timing preserves its exact duration or right-censored lower bound and source.
    Distinct attempts remain distinct even when timing values happen to agree.
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
    """Immutable bounded data snapshot, referencing the single execution history.

    trace_id/run_id/plan_id identify its source. assessment_ids and predictions
    keep original forecast identities/values; rows keep every attempted event
    including failures, partial collections and censored timing. result_id names
    the optional actual analysis record, without inferring scientific validity.
    Use Record.revise for new supplied data versions; this never fits coefficients
    or modifies old assessments, model domains, calibration status or coverage.
    Forecast validity concerns its original assessment time. A historical
    residual does not establish calibration validity at execution time.
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
        """Validate source membership after interchange, without replay or fitting."""
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
    """Align existing bounded evidence without assessment, execution or fitting.

    Only a pre-attempt assessment_id permits forecast residuals. Missing receipts
    or assessments keep the event with reasons. Foreign supplied run/attempt
    timings and conflicting identities are rejected instead of silently dropped.
    max_comparisons bounds this projection's actual forecast/timing pairs.
    Native payloads and histograms are not copied into the returned ledger.
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
