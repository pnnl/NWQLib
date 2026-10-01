"""Original forecasts aligned to actual offline Run attempts, without reevaluation."""

from types import SimpleNamespace

import pytest

from nwqlib._prepared_execution import Run, prepare_experiment
from nwqlib.backends import align_telemetry, assess
from nwqlib.backends.assessment import estimate_plan
from nwqlib.backends.telemetry import AttemptTiming, PredictionLedger
from nwqlib.execution import TimingObservation
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend
from _profile_fixtures import original_forecast


def forecast(plan, scope="native_call_wall", *, compiler=None):
    original, allocation = original_forecast(plan)
    profile = original.profile
    configuration = (
        profile.configuration
        if compiler is None
        else profile.configuration.revise(compiler=compiler)
    )
    if compiler is not None:
        allocation = allocation.revise(configuration_id=configuration.content_id)
    model = profile.models[0].revise(
        scope=scope,
        domain=profile.models[0].domain.revise(
            configuration_id=configuration.content_id, allocation_id=allocation.content_id
        ),
    )
    profile = profile.revise(configuration=configuration, models=(model,))
    return estimate_plan(
        plan, profile=profile, allocation=allocation, assessed_at=original.assessed_at
    )


def no_replay(monkeypatch):
    import nwqlib.backends.assessment as owner
    import nwqlib.resources as resources

    def forbidden(*args, **kwargs):
        raise AssertionError("telemetry repeated model, native or provider work")

    for obj, name in (
        (owner, "assess"),
        (owner, "_assess"),
        (owner, "_select_workload"),
        (owner, "_predict_time"),
        (resources, "estimate"),
        (InjectedBackend, "refresh"),
        (InjectedBackend, "submit"),
    ):
        monkeypatch.setattr(obj, name, forbidden)


def test_original_prediction_residual_and_saved_result_versions(tmp_path, monkeypatch):
    from nwqlib import load_result

    plan = selected()
    original = forecast(plan)
    (assessment,) = original.assessments
    (prediction,) = assessment.predictions
    assert prediction.seconds.value == 5.5  # 2 seconds + seven shots * .5.
    old_json = original.model_dump_json()
    ticks = iter((10.0, 16.5))
    monkeypatch.setattr("nwqlib._prepared_execution.perf_counter", lambda: next(ticks))
    InjectedBackend.supports_synchronous = True
    with Run(
        plan,
        backend=InjectedBackend(),
        forecast=original,
        allocation=original.allocation,
        directory=tmp_path / "run",
    ) as run:
        result = run.wait(timeout=1, poll_interval=0)
        (event,) = run.trace.events
        assert event.assessment_id == assessment.content_id and event.timing.seconds == 6.5
        sample = AttemptTiming(
            run_id=run.run_id,
            attempt=event.attempt,
            prepared_id=event.prepared_id,
            timing=TimingObservation(
                scope="native_call_wall",
                seconds=3.0,
                censoring="right_censored",
                reason="supplied continuing target stopped at three seconds",
                source=prediction.evidence.source,
            ),
        )
        path = result.save(tmp_path / "result")
        # A new supplied model changes future forecasts only.
        model = original.profile.models[0]
        changed = model.revise(
            coefficients=(
                model.coefficients[0].revise(seconds_per_unit=3.0),
                *model.coefficients[1:],
            )
        )
        updated = estimate_plan(
            plan,
            profile=original.profile.revise(models=(changed,)),
            allocation=original.allocation,
            assessed_at=original.assessed_at,
        )
        assert updated.assessments[0].predictions[0].seconds.value == 6.5
        no_replay(monkeypatch)
        ledger = align_telemetry(run, timings=(sample, sample))
        (row,) = ledger.rows
        assert [c.residual_seconds for c in row.comparisons] == [None, 1.0]
        assert "right-censored" in row.comparisons[0].reasons[-1]
        assert len(row.timings) == 2
        assert row.collected_observation_ids == row.contribution_ids == result.contribution_ids
        loaded_ledger = PredictionLedger.model_validate_json(ledger.model_dump_json())
        loaded_ledger.validate_context(trace=run.trace, assessments=original.assessments)
        with pytest.raises(ValueError, match="assessment"):
            loaded_ledger.validate_context(trace=run.trace, assessments=updated.assessments)
        with pytest.raises(ValueError, match="max_comparisons"):
            align_telemetry(run, timings=(sample,), max_comparisons=1)
        assert original.model_dump_json() == old_json
    saved = load_result(path, method=CountsMethod)
    assert align_telemetry(saved).rows[0].comparisons[0].residual_seconds == 1.0
    assert saved.data.forecast.assessments[0].predictions[0] == prediction


@pytest.mark.parametrize("outcome", ("failure", "partial"))
def test_failure_and_partial_collection_keep_actual_exposure(tmp_path, monkeypatch, outcome):
    plan = selected()
    original = forecast(plan)
    InjectedBackend.supports_synchronous = True
    if outcome == "failure":
        InjectedBackend.fail = True
    else:
        monkeypatch.setattr(
            InjectedBackend,
            "_result",
            lambda self, native: SimpleNamespace(
                raw_output={"counts": {}}, metadata={"native_job_id": "original-job"}
            ),
        )
    ticks = iter((10.0, 12.0))
    monkeypatch.setattr("nwqlib._prepared_execution.perf_counter", lambda: next(ticks))
    with Run(
        plan,
        backend=InjectedBackend(),
        forecast=original,
        allocation=original.allocation,
        directory=tmp_path / "run",
    ) as run:
        if outcome == "failure":
            with pytest.raises(RuntimeError, match="native read failed"):
                run.resume()
        else:
            result = run.wait(timeout=1, poll_interval=0)
            assert result.counts == 0 and run.observations.chunks[0].returned_shots == 0
        no_replay(monkeypatch)
        (row,) = align_telemetry(run).rows
        assert run.trace.jobs == 1 and row.event.shots == 7
        assert row.event.timing.seconds == 2.0
        if outcome == "failure":
            assert row.event.status == "uncertain" and not row.collected_observation_ids
            assert row.comparisons[0].residual_seconds is None
        else:
            assert row.collected_observation_ids == tuple(
                c.content_id for c in run.observations.chunks
            )
            # The timing describes the completed native call, independently of
            # whether all requested scientific samples were returned.
            assert row.comparisons[0].residual_seconds == -3.5


def test_scope_configuration_and_original_intent_association(tmp_path, monkeypatch):
    """Only a matching pre-attempt forecast and timing scope permit a residual, excluding foreign
    compiler/run evidence.
    """
    plan = selected()
    original = forecast(plan, "selected_acquisition")
    InjectedBackend.supports_synchronous = True
    with Run(
        plan,
        backend=InjectedBackend(),
        forecast=original,
        allocation=original.allocation,
        directory=tmp_path / "scope",
    ) as run:
        handle = prepare_experiment(plan.resolve("counts"), run=run)
        wrong = assess(
            plan,
            handle.realization,
            profile=original.profile,
            allocation=original.allocation,
            runtime=handle.record.runtime.revise(seed=handle.record.runtime.seed + 1),
            assessed_at=original.assessed_at,
        )
        with pytest.raises(ValueError, match="runtime"):
            wrong.validate_prepared(plan, handle.realization, handle.record)
        result = run.wait(timeout=1, poll_interval=0)
        (event,) = run.trace.events
        supplied = AttemptTiming(
            run_id=run.run_id,
            attempt=event.attempt,
            prepared_id=event.prepared_id,
            timing=TimingObservation(
                scope="selected_acquisition",
                seconds=4.5,
                source=original.profile.models[0].evidence.source,
            ),
        )
        assert align_telemetry(result).rows[0].comparisons[0].residual_seconds is None
        assert (
            align_telemetry(result, timings=(supplied,)).rows[0].comparisons[0].residual_seconds
            == -1.0
        )
        with pytest.raises(ValueError, match="another run"):
            align_telemetry(run, timings=(supplied.revise(run_id="foreign"),))
    foreign = forecast(
        plan, compiler=original.profile.configuration.compiler.revise(version="foreign")
    )
    with Run(
        plan,
        backend=InjectedBackend(),
        forecast=foreign,
        allocation=foreign.allocation,
        directory=tmp_path / "compiler",
    ) as run:
        run.wait(timeout=1, poll_interval=0)
        (row,) = align_telemetry(run).rows
        assert row.comparisons[0].residual_seconds is None
        assert any("runtime/compiler" in reason for reason in row.reasons)
        assert run.forecast.assessments[0].predictions[0].seconds.value == 5.5
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "unbound") as run:
        run.wait(timeout=1, poll_interval=0)
        (row,) = align_telemetry(run, assessments=original.assessments).rows
        assert row.event.assessment_id is None and not row.comparisons
