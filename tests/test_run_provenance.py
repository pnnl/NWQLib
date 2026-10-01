"""Original resource forecasts survive actual attempts and saved continuations."""

import json

import pytest

import nwqlib
from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
from nwqlib.backends.assessment import PlanEstimate, estimate_plan
from _profile_fixtures import original_forecast
from test_run_archive import classical_plan
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


@pytest.mark.parametrize("synchronous", (False, True))
def test_original_forecast_and_allocation_survive_actual_submission_and_reopen(
    tmp_path, monkeypatch, synchronous
):
    plan = selected()
    forecast, allocation = original_forecast(plan)
    # Independent scalar oracle: one invocation costs 2 s and seven shots cost 3.5 s.
    assert forecast.assessments[0].predictions[0].seconds is not None, forecast.assessments[0].predictions[0].reasons
    assert forecast.assessments[0].predictions[0].seconds.value == 5.5
    original_id = forecast.assessments[0].content_id
    forbid_replanning(monkeypatch)
    InjectedBackend.supports_synchronous = synchronous
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
        assert restored.forecast == forecast and restored.allocation == allocation
        assert restored.rng.snapshot() == rng
        result = restored.wait(timeout=1, poll_interval=0)
        assert result.counts == 7 and restored.trace.jobs == 1
        assert result.data.forecast is restored.forecast
        assert result.data.allocation is restored.allocation
        assert result.data.trace.events[0].assessment_id == original_id
        result.save(tmp_path / "result")
    saved = nwqlib.load_result(tmp_path / "result", method=CountsMethod)
    assert saved.data.forecast == forecast and saved.data.allocation == allocation
    assert saved.data.forecast.assessments[0].predictions[0].seconds.value == 5.5
    assert saved.analyze().data is saved.data
    assert [name for name, _ in InjectedBackend.calls].count("prepare") == 1
    assert [name for name, _ in InjectedBackend.calls].count("launch") == 1


def test_host_intent_binds_the_original_forecast_without_changing_kernels(tmp_path, monkeypatch):
    plan = classical_plan("qpe")
    forecast, allocation = original_forecast(plan, model=False)
    forbid_replanning(monkeypatch)
    with Run(plan, forecast=forecast, allocation=allocation) as run:
        result = run.wait()
        assert run.trace.jobs == 0 and run.trace.host_invocations > 0
        run.extend_limits(max_total_circuits=run.limits.max_total_circuits + 1)
        amendment, = run.limit_amendments
        assert (amendment.circuit_preparations, amendment.circuit_attempts, amendment.raw_shots) == (0, 0, 0)
        assert result.data.trace.limit_amendments == ()
        by_id = {a.point.realization_id: a.content_id for a in forecast.assessments}
        assert all(event.assessment_id == by_id[receipt.realization_id]
                   for event, receipt in zip(run.trace.events, run.prepared_artifacts, strict=True))
        result.save(tmp_path / "host")
    saved = nwqlib.load_result(tmp_path / "host")
    assert saved.data.forecast == forecast and saved.data.allocation == allocation


def test_other_plan_or_allocation_is_rejected_before_preparation(tmp_path):
    plan = selected()
    forecast, allocation = original_forecast(plan)
    with pytest.raises(ValueError, match="another selected Plan"):
        Run(selected(shots=8), backend=InjectedBackend(), directory=tmp_path / "wrong-plan",
            forecast=forecast, allocation=allocation)
    with pytest.raises(ValueError, match="allocation differs"):
        Run(plan, backend=InjectedBackend(), directory=tmp_path / "wrong-grant",
            forecast=forecast, allocation=allocation.revise(name="different grant"))
    assert not InjectedBackend.calls and not tuple(tmp_path.iterdir())


def test_forecast_association_is_validated_before_new_intent(tmp_path, monkeypatch):
    plan = selected()
    forecast, allocation = original_forecast(plan)
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "run",
             forecast=forecast, allocation=allocation) as run:
        prepared = prepare_experiment(plan.resolve("counts"), run=run)
        # Keep the handle's native/backend identity, falsify only its actual readout receipt.
        changed = type(prepared)._make(prepared.record.revise(
            observation=prepared.record.observation.revise(shots=8)), prepared.realization,
            prepared._native, prepared._items, prepared._setting, prepared._bindings)
        with pytest.raises(ValueError, match="actual prepared"):
            submit_experiment(changed, run=run)
        assert run.trace.jobs == 0 and not run.trace.events
        assert [name for name, _ in InjectedBackend.calls] == ["prepare"]
        submit_experiment(prepared, run=run)
        assert run.trace.jobs == 1 and sum(event.shots for event in run.trace.events) == 7


def test_saved_event_cannot_lose_its_forecast_association(tmp_path):
    plan = selected()
    forecast, allocation = original_forecast(plan)
    InjectedBackend.status = "completed"
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "run",
             forecast=forecast, allocation=allocation) as run:
        result = run.wait(timeout=1, poll_interval=0)
        path = result.save(tmp_path / "result")
        event = result.data.trace.events[0].revise(assessment_id=None)
        trace = result.data.trace.revise(events=(event,))
    file = path / "result.json"
    data = json.loads(file.read_text())
    data["data"]["trace"] = trace.model_dump(mode="json")
    file.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="original forecast association"):
        nwqlib.load_result(path, method=CountsMethod)


def test_profile_free_forecast_preserves_allocation_without_inventing_events(tmp_path):
    plan = selected()
    _, allocation = original_forecast(plan)
    forecast = estimate_plan(plan, allocation=allocation)
    assert not forecast.assessments and forecast.unpredicted
    restored = PlanEstimate.model_validate(forecast.model_dump(mode="json"))
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "run",
             forecast=restored, allocation=allocation) as run:
        run.resume()
        assert run.trace.events[0].assessment_id is None
        assert run.forecast == forecast and run.allocation is allocation


def test_reopened_run_checks_each_forecast_claim_once_and_each_receipt_by_its_own_fields(tmp_path, monkeypatch):
    """Reopening validates the forecast's Plan-level checks once per assessment, not again per saved event."""
    from nwqlib import Accuracy
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.backends.assessment import ProfileAssessment
    from nwqlib.evidence.error_model import ClaimAssessment, ErrorFrame
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import ingest_occupation

    problem = Expectation(state=ingest_occupation("00", num_qubits=2),
                          observable=ingest_pauli((("XX", 1.0), ("YZ", 1.0), ("ZY", 1.0), ("ZZ", 1.0)), num_qubits=2))
    plan = nwqlib.plan(problem, method=ExpectationMethod(), seed=7,
                       accuracy=Accuracy(component="sampling", absolute_tolerance=2.0, confidence=0.9))
    forecast, allocation = original_forecast(plan)
    points = len(forecast.assessments)
    assert points == 4 and all(assessment.error is not None for assessment in forecast.assessments)
    with Run(plan, backend=AerBackend(), directory=tmp_path / "run", forecast=forecast, allocation=allocation) as run:
        run.resume()
        ids = [event.assessment_id for event in run.trace.events]
        path = run.save(tmp_path / "copy")
    assert len(ids) == points and set(ids) == {assessment.content_id for assessment in forecast.assessments}
    calls = {"context": 0, "claim": 0, "frame": 0}
    context = ProfileAssessment.validate_context
    claim, frame = ClaimAssessment.validate_plan, ErrorFrame.from_output.__func__

    def counted_context(self, *args, **kwargs):
        calls["context"] += 1
        return context(self, *args, **kwargs)

    def counted_claim(self, plan):
        calls["claim"] += 1
        return claim(self, plan)

    def counted_frame(cls, *args):
        calls["frame"] += 1
        return frame(cls, *args)

    monkeypatch.setattr(ProfileAssessment, "validate_context", counted_context)
    monkeypatch.setattr(ClaimAssessment, "validate_plan", counted_claim)
    monkeypatch.setattr(ErrorFrame, "from_output", classmethod(counted_frame))
    with nwqlib.load_run(path, backend=AerBackend(), progress=False) as restored:
        assert [event.assessment_id for event in restored.trace.events] == ids
    # The Plan-level checks run once per assessment (validate_context and its error claim, whose
    # own check forms one frame) plus one Plan frame, and are not repeated for each receipt.
    assert calls == {"context": points, "claim": points, "frame": points + 1}
