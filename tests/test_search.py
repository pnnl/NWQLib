"""Finite selected-Plan ranking, real execution and independent scoped evidence."""

from fractions import Fraction

import pytest

from nwqlib import Expectation, plan, prepare, scan, solve, submit
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.core.records import Limit, Rational, Unit
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.resources import ResourceContext, Workspace
from nwqlib.scientist import Comparison, ComparisonRow
from nwqlib.search import Candidate, Objective, SearchSelection
from test_run_lifecycle import InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend
from _profile_fixtures import original_forecast
from test_run_provenance import forbid_replanning


def choices(shots=(8, 32)):
    problem = Expectation(state=[1.0, 0.0], observable=ingest_pauli((("Z", 1.0),), num_qubits=1))
    return tuple(plan(problem, method=ExpectationMethod(), shots=n, seed=7) for n in shots)


def number(value):
    return Fraction(value.value.numerator, value.value.denominator)


def test_selected_shots_reach_actual_prepare_submit_and_report(monkeypatch):
    plans = choices()
    result = scan(
        tuple(Candidate(p) for p in plans), objectives=(Objective(kind="requested_shots"),)
    )
    assert [number(row[0]) for row in result.selection.values] == [8, 32]
    assert result.selection.nondominated == (0,)
    first = result.select(0)
    assert first is result.comparison.rows[0] and first.plan is plans[0]

    def forbidden(*args, **kwargs):
        pytest.fail("selected-row execution repeated planning or estimation")

    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    monkeypatch.setattr("nwqlib.scientist.estimate", forbidden)
    monkeypatch.setattr("nwqlib.resources.estimate", forbidden)
    solved = solve(first)
    assert solved.value == 1.0
    assert solved.plan is plans[0]
    assert sum(event.shots for event in solved.data.trace.events) == 8
    assert solved.report()["plan"]["shots"] == 8
    assert result.select(1).plan is plans[1]  # Deliberate dominated choice stays legal.


def test_selected_row_keeps_original_forecast_through_repeated_runs_and_archive(
    tmp_path, monkeypatch
):
    from pathlib import Path

    from nwqlib import load_result

    chosen = selected()
    forecast, allocation = original_forecast(chosen)
    row = ComparisonRow(chosen.method, chosen, estimate=forecast, allocation=allocation)
    forbid_replanning(monkeypatch)
    InjectedBackend.supports_synchronous = True
    # Each detached Run, including the one a repeated submit creates, uses the
    # default run directory; keep it in tmp_path.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    prepared = prepare(row, backend=InjectedBackend())
    results = (submit(prepared).wait(), submit(prepared).wait())
    for result in results:
        assert result.plan is chosen
        assert result.data.forecast is forecast and result.data.allocation is allocation
        assert result.data.trace.events[0].assessment_id == forecast.assessments[0].content_id
        assert sum(event.shots for event in result.data.trace.events) == 7
    restored = load_result(results[0].save(tmp_path / "original"), method=chosen.method)
    assert restored.data.forecast.model_dump(mode="json") == forecast.model_dump(mode="json")
    assert restored.data.allocation == allocation
    with pytest.raises(ApplicabilityError, match="unsupported"):
        prepare(
            ComparisonRow(
                chosen.method, None, reason="unsupported scientific domain", allocation=allocation
            )
        )


def test_unknown_provider_and_controller_populations_are_not_zero_or_one_query():
    from nwqlib import Eigenproblem
    from nwqlib.algorithms import ADAPT, Lanczos

    exact = plan(Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)),
                 method=Lanczos(initial_state=[1., 1.], krylov_dimension=2), seed=7)
    exact_ranked = scan((Candidate(exact),), objectives=(Objective(kind="requested_shots"),))
    assert number(exact_ranked.selection.values[0][0]) == 0
    forecast, _ = original_forecast(exact, model=False)
    # T2(Z)=I is algebraically known; T1 and T3 are two points of one exact
    # trajectory. This complete static population has zero sampled shots and
    # no future stage.
    assert len(forecast.assessments) == 1 and forecast.unpredicted == ()

    fixed = choices((8,))[0]
    managed = plan(fixed.problem, method=ExpectationMethod(estimate_precision=0.1), seed=7)
    compared = scan(
        (Candidate(fixed), Candidate(managed)), objectives=(Objective(kind="requested_shots"),)
    )
    assert number(compared.selection.values[0][0]) == 8
    assert compared.selection.values[1][0].value is None
    assert "provider-managed" in compared.selection.values[1][0].reason
    assert compared.selection.nondominated == (0,) and compared.selection.incomparable == (1,)

    # ADAPT declares reusable query laws. Two declarations at eight shots
    # each do not mean its data-dependent trajectory consumes sixteen shots.
    adaptive = plan(
        Eigenproblem(A=ingest_pauli((("Z", 1.0),), num_qubits=1)),
        method=ADAPT(
            initial_state=[1.0, 1.0],
            pool=(ingest_pauli((("Y", 1j),), num_qubits=1),),
            max_iterations=1,
        ),
        shots=8,
        seed=7,
    )
    ranked = scan((Candidate(adaptive),), objectives=(Objective(kind="requested_shots"),))
    assert ranked.selection.values[0][0].value is None
    assert "query templates" in ranked.selection.values[0][0].reason
    assert ranked.selection.nondominated == () and ranked.selection.incomparable == (0,)


def test_cached_folds_and_supplied_prior_occurrences_survive_rescoring(monkeypatch):
    """Duplicate Plans reuse folds, but identical prior-work records still represent separate
    supplied occurrences.
    """
    from nwqlib.evidence import Evidence, Fact
    from nwqlib.core.records import Scope, Source
    import nwqlib.resources as resources

    plans = choices((8, 8, 32))
    source = Source(
        name="prior factorization",
        version="1",
        domain="supplied work",
        reference="one five-unit occurrence",
    )
    prior = Fact(
        quantity="factorization_work",
        unit=Unit(symbol="work_unit", dimension="count"),
        scope=Scope(domain="prior occurrence"),
        availability="concrete",
        value=Rational(numerator=5, denominator=1),
        evidence=Evidence(kind="user_assertion", source=source),
    )
    calls = []
    original = resources.estimate

    def record(*args, **kwargs):
        calls.append(args[0].content_id)
        return original(*args, **kwargs)

    monkeypatch.setattr(resources, "estimate", record)
    result = scan(
        (Candidate(plans[0], prior_work=(prior, prior)), Candidate(plans[1]), Candidate(plans[2])),
        objectives=(Objective(kind="requested_shots"),),
    )
    assert len(calls) == 2
    assert result.selection.work.base_folds == 2 and result.selection.work.reused_folds == 1
    assert result.selection.prior_work[0] == (prior, prior)
    assert result.selection.nondominated == (0, 1)
    blocked = ComparisonRow(plans[0].method, None, reason="known unsupported population")
    comparison = Comparison(plans[0].problem, (*result.comparison.rows, blocked))

    def forbidden(*args, **kwargs):
        pytest.fail("stored comparison reran folding, planning or model evaluation")

    monkeypatch.setattr(resources, "estimate", forbidden)
    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    monkeypatch.setattr("nwqlib.search.estimate_plan", forbidden)
    rescored = scan(comparison, objectives=result.selection.objectives)
    assert rescored.comparison is comparison and rescored.select(3) is blocked
    assert rescored.selection.nondominated == (0, 1) and rescored.selection.incomparable == (3,)
    assert rescored.selection.values[3][0].reason == blocked.reason
    assert rescored.selection.prior_work[0] == (prior, prior)
    assert rescored.selection.work.base_folds == rescored.selection.work.assessed_points == 0


def test_serial_model_scope_and_unknown_forecast_remain_conditional(monkeypatch):
    """Hand-computed serial model values stay conditional, and a mismatched prediction scope
    becomes incomparable.
    """
    plans = (selected(3), selected(7))
    original, allocation = original_forecast(plans[0])
    model = original.profile.models[0]
    model = model.revise(domain=model.domain.revise(batch_schedule="serial"))
    other = model.revise(name="overlapping wall scope", scope="native_call_wall")
    profile = original.profile.revise(models=(model, other))
    timed = Objective(kind="predicted_seconds", model_id=model.content_id, scope=model.scope)
    context = ResourceContext(batch_schedule="serial")
    result = scan(
        tuple(Candidate(p, allocation=allocation) for p in plans),
        objectives=(Objective(kind="requested_shots"), Objective(kind="logical_width"), timed),
        profile=profile,
        context=context,
        assessed_at=original.assessed_at,
    )
    assert [[number(v) for v in row] for row in result.selection.values] == [
        [3, 1, Fraction(7, 2)],
        [7, 1, Fraction(11, 2)],
    ]
    assert all(row[2].status == "conditional" for row in result.selection.values)
    assert result.selection.nondominated == (0,)

    def forbidden(*args, **kwargs):
        pytest.fail("stored model comparison evaluated or folded again")

    monkeypatch.setattr("nwqlib.search.estimate_plan", forbidden)
    monkeypatch.setattr("nwqlib.backends.assessment._predict_time", forbidden)
    monkeypatch.setattr("nwqlib.resources.estimate", forbidden)
    wrong = scan(result.comparison, objectives=(timed.revise(scope="native_call_wall"),))
    assert wrong.selection.incomparable == (0, 1) and wrong.selection.nondominated == ()
    assert all("matching" in row[0].reason for row in wrong.selection.values)
    assert (
        SearchSelection.model_validate_json(result.selection.model_dump_json()) == result.selection
    )


def test_repeated_candidate_reuses_its_forecast_without_new_model_evaluation(monkeypatch):
    """A repeated Plan/Allocation/evidence key must not evaluate the device model again."""
    import nwqlib.backends.assessment as assessment

    plans = (selected(3), selected(7))
    original, allocation = original_forecast(plans[0])
    calls = []
    predict = assessment._predict_time
    monkeypatch.setattr(
        assessment, "_predict_time", lambda *a, **k: calls.append(1) or predict(*a, **k)
    )
    result = scan(
        tuple(Candidate(p, allocation=allocation) for p in (plans[0], plans[1], plans[0])),
        objectives=(Objective(kind="requested_shots"),),
        profile=original.profile,
        assessed_at=original.assessed_at,
    )
    # One prediction per declared acquisition and model, for each distinct Plan only.
    points = [len(p.experiments) * len(original.profile.models) for p in plans]
    assert all(points)
    assert len(calls) == sum(points)
    rows = result.comparison.rows
    assert rows[2].estimate is rows[0].estimate and rows[2].plan is plans[0]
    work = result.selection.work
    assert (work.requested_points, work.assessed_points, work.reused_points) == (
        2 * points[0] + points[1], sum(points), points[0]
    )
    assert (work.base_folds, work.reused_folds) == (2, 1)
    assert result.selection.values[2] == result.selection.values[0]


def test_explicit_allocations_preserve_nonmonotone_per_location_capacity():
    """Per-device capacity is nonmonotone across explicit grants and cannot be inferred from
    their aggregate memory.
    """
    chosen = selected()
    forecast, allocation = original_forecast(chosen, model=False)
    source = forecast.profile.evidence.source

    def grant(name, gib):
        return allocation.revise(
            name=name,
            locations=("gpu0", "gpu1"),
            topology="independent_devices",
            limits=tuple(
                Limit(
                    stage="execution",
                    metric="memory",
                    unit=Unit(symbol="byte", dimension="bytes"),
                    kind="capacity_stock",
                    value=gib * 2**30,
                    scope=location,
                )
                for location in ("gpu0", "gpu1")
            ),
        )

    allocations = (
        grant("fits first", 12),
        grant("two eight-GiB devices", 8),
        grant("fits last", 16),
    )
    context = ResourceContext(
        batch_schedule="serial",
        resident=(
            Workspace(location="gpu0", purpose="workspace", bytes=10 * 2**30, source=source),
        ),
    )
    result = scan(
        tuple(Candidate(chosen, allocation=a) for a in allocations),
        objectives=(Objective(kind="requested_shots"),),
        profile=forecast.profile,
        context=context,
        assessed_at=forecast.assessed_at,
    )
    details = [
        next(
            d
            for d in row.estimate.assessments[0].capacity.details
            if d.limit is not None and d.limit.scope == "gpu0" and d.quantity == "memory"
        )
        for row in result.comparison.rows
    ]
    assert [d.status for d in details] == ["feasible", "infeasible", "feasible"]
    assert all(d.fact.value.numerator == 10 * 2**30 for d in details)
    assert result.selection.nondominated == (0, 1, 2)
    assert result.select(1).allocation is allocations[1]
    assert result.select(1).estimate.assessments[0].capacity.status == "infeasible"


def test_frontier_domain_ties_and_partial_comparability():
    result = scan(
        tuple(Candidate(p) for p in choices((4, 8, 4))),
        objectives=(Objective(kind="requested_shots"),),
    )
    selection = result.selection
    assert selection.nondominated == (0, 2)
    fractional = selection.values[0][0].revise(value=Rational(numerator=1, denominator=2))
    with pytest.raises(ValueError, match="integer count"):
        selection.revise(values=((fractional,), *selection.values[1:]))
    unknown = selection.values[2][0].revise(
        value=None, status="unavailable", reason="supplied value unavailable"
    )
    partial = selection.revise(
        values=(*selection.values[:2], (unknown,)), nondominated=None, incomparable=None
    )
    assert partial.nondominated == (0,) and partial.incomparable == (2,)
    assert SearchSelection.model_validate_json(partial.model_dump_json()) == partial
    with pytest.raises(ValueError, match="another original comparison"):
        selection.validate_comparison(
            Comparison(
                result.comparison.problem,
                (result.comparison.rows[1], result.comparison.rows[0], result.comparison.rows[2]),
            )
        )


def test_finite_caps_precede_folding_and_exact_frontier_work(monkeypatch):
    plans = choices()
    candidates = tuple(Candidate(p) for p in plans)

    def forbidden(*args, **kwargs):
        pytest.fail("inadmissible search performed work")

    monkeypatch.setattr("nwqlib.resources.estimate", forbidden)
    with pytest.raises(ValueError, match="max_candidates"):
        scan(candidates, objectives=(Objective(kind="requested_shots"),), max_candidates=1)
    with pytest.raises(ValueError, match="max_pair_comparisons"):
        scan(candidates, objectives=(Objective(kind="requested_shots"),), max_pair_comparisons=1)
    with pytest.raises(ValueError, match="max_values"):
        scan(
            candidates[:1],
            objectives=(Objective(kind="requested_shots"), Objective(kind="logical_width")),
            max_values=1,
        )
    forecast, allocation = original_forecast(selected())
    monkeypatch.setattr("nwqlib.search.estimate_plan", forbidden)
    with pytest.raises(ValueError, match="max_assessments"):
        scan(
            (Candidate(selected(), allocation=allocation),),
            objectives=(Objective(kind="requested_shots"),),
            profile=forecast.profile,
            assessed_at=forecast.assessed_at,
            max_assessments=1,
        )
