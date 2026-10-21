"""Static preparation keeps its original backend seed, and seed-specific calibrations keep their domains."""

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.core.planning import RandomStreams, RuntimeOptions
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend


def backend_draws(entropy, count):
    """Independent uint32 oracle: low then high halves of PCG64 raw words."""
    _, backend = np.random.SeedSequence(entropy).spawn(2)
    words = np.random.PCG64(backend).random_raw((count + 1) // 2)
    return [part for word in words for part in (int(word) & 0xFFFFFFFF, int(word) >> 32)][:count]


def forbidden(*args, **kwargs):
    raise AssertionError("static read/resume replayed native or numerical work")


def test_failed_static_preparation_keeps_original_seed_and_rejects_implicit_rebuild(
    tmp_path, monkeypatch, prepared_stubs
):
    plan = selected()
    expected = backend_draws(31, 1)[0]
    calls = []

    def fail(self, circuit, **kwargs):
        calls.append(kwargs["runtime"].seed)
        raise RuntimeError("interrupted native preparation")

    monkeypatch.setattr(InjectedBackend, "prepare", fail)
    path = tmp_path / "failed-preparation"
    with Run(plan, backend=InjectedBackend(), directory=path) as run:
        with pytest.raises(RuntimeError, match="interrupted native"):
            plan.method.prepare(plan, run=run)
        state = run.rng.snapshot()
        assert calls == [expected] and run.trace.preparations == 1
        assert run.prepared_artifacts == () and run.trace.events == ()
        with pytest.raises(RuntimeError, match="no replacement"):
            run.resume()
        assert calls == [expected] and run.rng.snapshot() == state
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.rng.snapshot() == state and restored.trace.preparations == 1
        with pytest.raises(RuntimeError, match="no replacement"):
            restored.resume()
        assert calls == [expected] and not restored.trace.events


def test_fixed_seed_calibration_and_seed_independent_forecast_keep_their_domains(monkeypatch):
    from _profile_fixtures import stored_inputs
    from nwqlib.backends.assessment import assess, estimate_plan

    plan, point, profile, allocation, context, bare, at = stored_inputs(calibrated=True)
    state = plan.randomness
    monkeypatch.setattr(RandomStreams, "next_seed", forbidden)
    independent = profile.models[0]
    seven, eight = RuntimeOptions(seed=7), RuntimeOptions(seed=8)
    models = tuple(
        independent.revise(name=name, domain=independent.domain.revise(runtime=runtime))
        for name, runtime in (("seed seven", seven), ("seed eight", eight))
    ) + (independent,)
    profile = profile.revise(models=models)
    for runtime, expected in (
        (seven, (True, False, True)),
        (eight, (False, True, True)),
        (None, (False, False, True)),
    ):
        assessment = assess(
            plan,
            point,
            profile=profile,
            allocation=allocation,
            context=context,
            runtime=runtime,
            assessed_at=at,
        )
        assert assessment.point.runtime_options_id == (
            None if runtime is None else runtime.content_id
        )
        assert (
            tuple(prediction.seconds is not None for prediction in assessment.predictions)
            == expected
        )
        # Supplied model arithmetic: 1 invocation +2 exact evaluation +3*.5 gates.
        assert all(
            prediction.seconds.value == 4.5
            for prediction in assessment.predictions
            if prediction.seconds is not None
        )
    forecast = estimate_plan(
        plan, profile=profile, allocation=allocation, context=context, assessed_at=at
    )
    assert forecast.assessments[0].point.runtime_options_id is None
    assert tuple(p.seconds is not None for p in forecast.assessments[0].predictions) == (
        False,
        False,
        True,
    )
    assert tuple(model.domain.runtime for model in profile.models) == (
        seven,
        eight,
        "seed_independent",
    )
    assert plan.randomness == state
