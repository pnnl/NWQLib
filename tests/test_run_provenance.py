"""Original resource forecasts survive actual attempts and saved continuations."""

from unittest.mock import ANY

import pytest

import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.backends.assessment import PlanEstimate
from _profile_fixtures import original_forecast
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def forbid_replanning(monkeypatch):
    import nwqlib.backends.assessment as assessment
    import nwqlib.resources as resources

    def forbidden(*args, **kwargs):
        pytest.fail("execution or reopening recomputed the selected forecast")

    monkeypatch.setattr(assessment, "estimate_plan", forbidden)
    monkeypatch.setattr(assessment, "assess", forbidden)
    monkeypatch.setattr(assessment, "_assess", forbidden)
    monkeypatch.setattr(assessment, "estimate", forbidden)
    monkeypatch.setattr(resources, "estimate", forbidden)
    monkeypatch.setattr(CountsMethod, "plan", forbidden)


def test_original_forecast_and_allocation_survive_actual_submission_and_reopen(tmp_path, monkeypatch):
    plan = selected()
    forecast, allocation = original_forecast(plan)
    # Independent scalar oracle: one invocation costs 2 s and seven shots cost 3.5 s.
    assert forecast.assessments[0].predictions[0].seconds is not None, forecast.assessments[0].predictions[0].reasons
    assert forecast.assessments[0].predictions[0].seconds.value == 5.5
    original_id = forecast.assessments[0].content_id
    forbid_replanning(monkeypatch)
    InjectedBackend.supports_synchronous = False
    run = Run(plan, backend=InjectedBackend(), directory=tmp_path / "run",
              forecast=forecast, allocation=allocation)
    assert run.forecast is forecast and run.allocation is allocation
    run.resume()
    assert run.trace.events[0].assessment_id == original_id
    assert sum(event.shots for event in run.trace.events) == 7 and run.trace.jobs == 1
    assert run.data.forecast is forecast and run.data.allocation is allocation
    rng = run.rng.snapshot()
    path = run.save(tmp_path / "copy")
    run.close()
    InjectedBackend.status = "completed"
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.forecast == forecast
        assert restored.forecast.model_dump(mode="json") == forecast.model_dump(mode="json")
        assert restored.allocation == allocation
        assert restored.rng.snapshot() == rng
        result = restored.wait(timeout=1, poll_interval=0)
        assert result.counts == 7 and restored.trace.jobs == 1
        assert result.data.forecast is restored.forecast
        assert result.data.allocation is restored.allocation
        assert result.data.trace.events[0].assessment_id == original_id
        result.save(tmp_path / "result")
    saved = nwqlib.load_result(tmp_path / "result", method=CountsMethod)
    assert saved.data.forecast == forecast
    assert saved.data.forecast != PlanEstimate.model_validate({
        **{name: getattr(forecast, name) for name in type(forecast).model_fields},
        "unpredicted": ("another forecast",),
    })
    assert saved.data.forecast.model_dump(mode="json") == forecast.model_dump(mode="json")
    # An operand of another type gets its own comparison.
    assert saved.data.forecast == ANY and forecast.__eq__(object()) is NotImplemented
    assert saved.data.allocation == allocation
    assert saved.data.forecast.assessments[0].predictions[0].seconds.value == 5.5
    assert saved.analyze().data is saved.data
    assert [name for name, _ in InjectedBackend.calls].count("prepare") == 1
    assert [name for name, _ in InjectedBackend.calls].count("launch") == 1
