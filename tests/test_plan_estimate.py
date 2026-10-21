"""Actual public forecast arithmetic, source association and expansion bounds."""


import pytest

from _profile_fixtures import stored_inputs
from nwqlib import estimate
from nwqlib.algorithms.expectation import ExpectationMethod
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
