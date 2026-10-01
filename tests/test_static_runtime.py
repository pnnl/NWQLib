"""Actual static streams and preparation history through the shared Run.

The prior base+ordinal stream/overflow and report-position machinery is retired:
backend draws follow recorded SeedSequence/PCG64 state. Original forecast joins,
pending atomicity and encoding byte admission remain covered by
test_run_provenance/test_run_journal; selected ComparisonRow routing is covered
by test_search.
"""

from types import SimpleNamespace

import numpy as np
import pytest

import nwqlib
from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
from nwqlib.algorithms.gcim import FixedGCIM
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


def test_static_settings_use_actual_backend_draws_and_repeat_only_in_a_new_run(
    monkeypatch, prepared_stubs
):
    from nwqlib.algorithms.gcim import fixed_basis
    from nwqlib.backends import qiskit_aer as aer
    from _profile_fixtures import original_forecast

    plan = nwqlib.plan(
        nwqlib.Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]]),
        method=FixedGCIM(basis=([1, 0], [0, 1])),
        shots=4,
        seed=7,
    )
    # One QWC group (Z): b**2 G = 4 grouped settings (fixed_basis.sampled_settings).
    settings = [s[0] for s in fixed_basis.sampled_settings(2, plan.reconstruction.groups)]
    assert [e.name for e in plan.experiments] == settings and len(settings) == 4
    original = plan.randomness
    expected = backend_draws(7, len(settings))
    forecast, allocation = original_forecast(plan, model=False)
    monkeypatch.setattr(
        fixed_basis,
        "_solve_projected_pencil",
        lambda *a, **k: (None, None, "injected unavailable projected result", None),
    )
    monkeypatch.setattr(
        aer,
        "_submit_aer_execution",
        lambda native: SimpleNamespace(
            # Keys are system_bits then phase; bit 0 stays zero, as a diagonal requires.
            raw_output={"counts": {"00": 2, "10": 2}}, metadata={"native_job_id": "static"}
        ),
    )
    # Each Run restores the recorded post-selection streams. Re-deriving them
    # from the root seed would keep these backend seeds but lose the method state.
    monkeypatch.setattr(np.random, "SeedSequence", forbidden)
    for repeated in (False, True):
        with Run(plan, forecast=forecast, allocation=allocation) as run:
            # Method-side draws cannot change the backend stream.
            if repeated:
                run.rng.method.random(11)
            plan.method.prepare(plan, run=run)
            assert [r.runtime.seed for r in run.prepared_artifacts] == expected[:1]
            assert run.trace.events == ()
            result = run.wait()
            assert [r.runtime.seed for r in result.data.receipts] == expected
            assert tuple(c.experiment for c in result.data.observations.chunks) == tuple(
                e.name for e in plan.experiments
            )
            assert result.data.forecast is forecast and result.data.allocation is allocation
            for event, receipt in zip(result.data.trace.events, result.data.receipts, strict=True):
                assert event.prepared_id == receipt.content_id
                assert event.assessment_id == forecast.assessment_for(plan, receipt).content_id
            state = run.rng.snapshot()
            assert run.wait() is result
            assert result.analyze().data is result.data
            assert len(result.report()["trace"]["submissions"]) == len(settings)
            assert run.rng.snapshot() == state and plan.randomness == original
    assert [options["seed"] for _, options in prepared_stubs.prepared] == expected + expected


def test_unused_local_and_imported_preparations_keep_distinct_streams_and_journal_order(
    tmp_path, prepared_stubs
):
    """Imported prepared work consumes no local RNG draw, and unused local preparations keep
    their original ordering.
    """
    plan = selected()
    expected = backend_draws(31, 3)
    InjectedBackend.supports_synchronous = True
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "source") as source:
        source_state = source.rng.snapshot()
        imported = prepare_experiment(
            plan.resolve("counts"), run=source, runtime=RuntimeOptions(seed=99)
        )
        assert source.rng.snapshot() == source_state  # An explicit seed draws no replacement.
    with Run(plan, backend=InjectedBackend(), directory=tmp_path / "destination") as run:
        first = prepare_experiment(plan.resolve("counts"), run=run)
        state = run.rng.snapshot()
        chunk = submit_experiment(imported, run=run)
        assert run.collect(chunk) and run.rng.snapshot() == state
        second = prepare_experiment(plan.resolve("counts"), run=run)
        assert [r.runtime.seed for r in run.prepared_artifacts] == [expected[0], 99, expected[1]]
        assert run.trace.preparations == 2 and len(run.trace.events) == 1
        result = run.wait()
        receipts = result.data.receipts
        assert [r.runtime.seed for r in receipts] == [expected[0], 99, expected[1], expected[2]]
        assert result.counts == 14  # One explicit imported acquisition and one static acquisition.
        assert run.trace.local_prepared_ids == (
            first.record.content_id,
            second.record.content_id,
            receipts[-1].content_id,
        )
        assert tuple(event.prepared_id for event in run.trace.events) == (
            imported.record.content_id,
            receipts[-1].content_id,
        )
        state, trace = run.rng.snapshot(), run.trace
    with nwqlib.load_run(
        tmp_path / "destination", backend=InjectedBackend(), method=CountsMethod
    ) as restored:
        assert restored.prepared_artifacts == receipts and restored.trace == trace
        assert restored.rng.snapshot() == state
        assert restored.wait().counts == 14 and restored.rng.snapshot() == state
    assert sum(name == "prepare" for name, _ in InjectedBackend.calls) == 4
    assert sum(name == "launch" for name, _ in InjectedBackend.calls) == 2


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


def test_uncertain_static_attempt_retrieves_its_seed_after_unused_preparation(
    tmp_path, prepared_stubs
):
    plan = selected()
    expected = backend_draws(31, 2)
    path = tmp_path / "uncertain"
    InjectedBackend.supports_synchronous = True
    InjectedBackend.fail = True
    with Run(plan, backend=InjectedBackend(), directory=path) as run:
        unused = prepare_experiment(plan.resolve("counts"), run=run)
        with pytest.raises(RuntimeError, match="native read failed"):
            run.resume()
        state = run.rng.snapshot()
        assert [r.runtime.seed for r in run.prepared_artifacts] == expected
        assert run.trace.preparations == 2 and run.trace.events[0].status == "uncertain"
        assert unused.record.content_id not in {event.prepared_id for event in run.trace.events}
        assert run.exposure["uncertain"]["shots"] == 7
    InjectedBackend.status = "completed"
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        result = restored.wait(timeout=1, poll_interval=0)
        assert result.counts == 7 and restored.rng.snapshot() == state
        assert [r.runtime.seed for r in result.data.receipts] == expected
        assert len(result.data.trace.events) == 1 and result.data.trace.jobs == 1
    assert sum(name == "prepare" for name, _ in InjectedBackend.calls) == 2
    assert sum(name == "launch" for name, _ in InjectedBackend.calls) == 1


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


def test_receipt_lookup_shares_the_saved_record(
    tmp_path, monkeypatch, prepared_stubs
):
    with Run(selected(), backend=InjectedBackend(), directory=tmp_path / "receipt") as run:
        handle = prepare_experiment(run.plan.resolve("counts"), run=run)
        receipt = handle.record
        with monkeypatch.context() as patch:
            patch.setattr(Run, "prepared_artifacts", property(forbidden))
            assert run.prepared_artifact(receipt.content_id) is receipt
            with pytest.raises(ValueError, match="unknown prepared receipt"):
                run.prepared_artifact("sha256:" + "0" * 64)
        snapshot = run.data
        assert snapshot.receipts[0] is receipt
    assert run.prepared_artifact(receipt.content_id) is receipt and snapshot.receipts[0] is receipt
