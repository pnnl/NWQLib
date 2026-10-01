"""Actual public forecast arithmetic, source association and expansion bounds."""

from dataclasses import dataclass
from datetime import timedelta

import pytest

from _profile_fixtures import stored_inputs
from nwqlib import compare, estimate
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.backends.assessment import PlanEstimate, estimate_plan


def test_public_estimate_preserves_hand_forecast_without_native_work(monkeypatch):
    plan, _, profile, allocation, context, _, at = stored_inputs(calibrated=True)
    import nwqlib.backends.assessment as owner

    calls = []

    def now(value):
        calls.append(value)
        return at if value is None else value

    monkeypatch.setattr(owner, "_assessment_time", now)
    result = estimate(plan, profile=profile, allocation=allocation, context=context)
    assert calls.count(None) == 1
    assert result.plan_id == plan.content_id
    assert result.allocation == allocation and result.profile == profile
    (assessment,) = result.assessments
    (prediction,) = assessment.predictions
    assert prediction.seconds.value == 4.5  # 1 invocation + 2 exact + 3 * .5 gates.
    assert (prediction.lower_seconds.value, prediction.upper_seconds.value) == (4.25, 5.0)
    assert assessment.point.runtime_options_id is None
    assert assessment.applicability.status == assessment.accuracy.status == "unknown"
    assert assessment.point.assessed_at == result.assessed_at == at

    def forbidden(*args, **kwargs):
        raise AssertionError("loading a forecast must not fold or evaluate its model")

    monkeypatch.setattr(owner, "estimate", forbidden)
    monkeypatch.setattr(owner, "_predict_time", forbidden)
    loaded = PlanEstimate.model_validate_json(result.model_dump_json())
    loaded.validate_plan(plan)
    assert loaded.assessments == result.assessments
    with pytest.raises(ValueError, match="another selected Plan"):
        loaded.validate_plan(plan.revise(assumptions=plan.assumptions + ("different selection",)))


def test_whole_expansion_rejects_before_model_work(monkeypatch):
    plan, _, profile, allocation, _, _, at = stored_inputs()
    import nwqlib.backends.assessment as owner

    def forbidden(*args, **kwargs):
        raise AssertionError("oversized forecast must reject before folding or model work")

    monkeypatch.setattr(owner, "estimate", forbidden)
    monkeypatch.setattr(owner, "_predict_time", forbidden)
    with pytest.raises(ValueError, match="requires 2.*max_assessments=1"):
        estimate_plan(
            plan, profile=profile, allocation=allocation, assessed_at=at, max_assessments=1
        )


def test_comparison_keeps_blocked_allocation_and_one_timestamp(monkeypatch):
    plan, _, profile, allocation, _, _, at = stored_inputs()
    import nwqlib.backends.assessment as owner

    calls = []

    def now(value):
        calls.append(value)
        return at if value is None else value

    monkeypatch.setattr(owner, "_assessment_time", now)

    @dataclass(frozen=True)
    class Unsupported(ExpectationMethod):
        def plan(self, *args, **kwargs):
            raise ApplicabilityError("declared unsupported preparation")

    comparison = compare(
        plan.problem,
        methods=(Unsupported(), ExpectationMethod(preparation_choice="hzh")),
        profile=profile,
        allocation=allocation,
        seed=7,
    )
    assert calls.count(None) == 1
    blocked, available = comparison.rows
    assert blocked.plan is None and blocked.allocation is allocation
    assert blocked.reason == "declared unsupported preparation"
    assert available.allocation is allocation
    assert available.estimate.assessed_at == at
    assert available.estimate.assessments[0].predictions[0].seconds.value == 1.125


def test_selected_sampling_criterion_is_preserved_without_claiming_total_accuracy():
    from nwqlib import Accuracy, plan as select_plan
    from nwqlib.backends import assess

    original, _, profile, allocation, _, _, at = stored_inputs()
    sampling = Accuracy(absolute_tolerance=0.5, component="sampling", confidence=0.9)
    plan = select_plan(original.problem, method=ExpectationMethod(), accuracy=sampling, seed=7)
    plan_id, shots = plan.content_id, plan.shots
    default = estimate_plan(plan, profile=profile, allocation=allocation, assessed_at=at)
    assert all(
        a.criterion == sampling and a.error.accuracy.confidence == 0.9 for a in default.assessments
    )
    assert all(a.error.remaining == ("sampling",) for a in default.assessments)
    total = Accuracy(absolute_tolerance=0.5, confidence=0.8)
    changed = assess(
        plan,
        plan.resolve(plan.experiments[0].name),
        profile=profile,
        allocation=allocation,
        assessed_at=at,
        accuracy=total,
    )
    assert changed.criterion == total
    assert {"sampling", "native_preparation", "physical_model"} <= set(changed.error.remaining)
    assert changed.error.status == "INCONCLUSIVE"
    assert (
        plan.content_id == plan_id and plan.shots == shots and plan.selection_accuracy == sampling
    )


def test_adapt_future_points_remain_unpredicted_without_invented_queries():
    from test_adapt_primary import plan_for

    plan = plan_for()
    _, _, profile, allocation, _, _, at = stored_inputs()
    forecast = estimate_plan(plan, profile=profile, allocation=allocation, assessed_at=at)
    assert not forecast.assessments
    assert forecast.unpredicted
    assert all("controller has not selected" in reason for reason in forecast.unpredicted)
    assert forecast.resources.construction_id == plan.construction.content_id


def test_public_timestamp_and_whole_comparison_cap_precede_model_evaluation(monkeypatch):
    plan, _, profile, allocation, context, _, at = stored_inputs(calibrated=True)
    import nwqlib.backends.assessment as owner

    first = estimate(plan, profile=profile, allocation=allocation, context=context, assessed_at=at)
    assert first.assessed_at == at and first.assessments[0].point.assessed_at == at
    expired = estimate(plan, profile=profile, allocation=allocation,
                       assessed_at=profile.valid_until + timedelta(seconds=1))
    (prediction,) = expired.assessments[0].predictions
    assert prediction.seconds is None and any("expired" in reason for reason in prediction.reasons)
    # Expiry removes the time prediction, not the selected logical cost.
    assert expired.resources.quantity("operations").fact.value.numerator == 3
    calls = []
    predict = owner._predict_time
    monkeypatch.setattr(owner, "_predict_time", lambda *a, **k: calls.append(1) or predict(*a, **k))
    with pytest.raises(ValueError, match="requires 4.*max_assessments=3"):
        compare(plan.problem, methods=(ExpectationMethod(), ExpectationMethod(preparation_choice="hzh")),
                profile=profile, allocation=allocation, assessed_at=at, max_assessments=3, seed=7)
    assert calls == []
    result = compare(plan.problem, methods=(ExpectationMethod(), ExpectationMethod(preparation_choice="hzh")),
                     profile=profile, allocation=allocation, assessed_at=at, max_assessments=4, seed=7)
    assert len(calls) == 2
    for row in result.rows:
        assert row.estimate.assessed_at == at
        assert row.estimate.plan_id == row.plan.content_id
        assert all(a.point.assessed_at == at for a in row.estimate.assessments)


def test_public_forecasts_keep_framed_evidence_at_each_actual_point(monkeypatch):
    from test_profile_assessment import _scalar_plan
    from nwqlib import Accuracy
    from nwqlib.core import Rational
    from nwqlib.evidence import TargetReference
    from nwqlib.ir import Binding
    import nwqlib.backends.assessment as owner
    import nwqlib.scientist as scientist

    base, _, profile, allocation, _, _, at = stored_inputs()
    selected, _, facts = _scalar_plan(base)
    selected = selected.revise(selection_accuracy=Accuracy(absolute_tolerance=.375))
    reference = TargetReference(relation="exact_target", failure_probability=0.,
        fact=facts[0].revise(bindings=(), fact=facts[0].fact.revise(
            quantity=selected.error_model.frame.quantity, value=Rational(numerator=2, denominator=1))))
    # This analytical fixture has a declared parameter; explicitly choose its
    # existing n=4 point, without introducing a native or reference computation.
    monkeypatch.setattr(owner, "_profile_points", lambda p: (
        (p.resolve("scalar", bindings=(Binding(parameter="n", value=4),)),), ()))
    result = estimate(selected, profile=profile, allocation=allocation, assessed_at=at,
                      facts=facts, reference=reference)
    assessment = result.assessments[0]
    assert assessment.error.status == "PASS" and assessment.error.reference == reference
    assert assessment.error.covered_subtotal == Rational(numerator=3, denominator=8)
    monkeypatch.setattr(scientist, "_plan_with_streams", lambda *a, **k: selected)
    comparison = compare(selected.problem, methods=(selected.method, selected.method),
        profile=profile, allocation=allocation, assessed_at=at, facts=facts, reference=reference)
    assert all(row.facts == facts and row.reference == reference and
               row.estimate.assessments[0].error.status == "PASS" for row in comparison.rows)
    # The same supplied n=4 statement cannot be relabeled as n=1 evidence.
    monkeypatch.setattr(owner, "_profile_points", lambda p: (
        (p.resolve("scalar", bindings=(Binding(parameter="n", value=1),)),), ()))
    with pytest.raises(ValueError, match="conflicting evidence parameter restriction"):
        compare(selected.problem, methods=(selected.method,), profile=profile, allocation=allocation,
                assessed_at=at, facts=facts, reference=reference)
    with pytest.raises(ValueError, match="logical folding has no assessment"):
        estimate(selected, facts=facts)
    with pytest.raises(ValueError, match="requires a profile assessment"):
        compare(selected.problem, methods=(selected.method,), facts=facts)
