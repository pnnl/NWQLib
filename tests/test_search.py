"""Finite selected-Plan ranking, real execution and independent scoped evidence."""

from fractions import Fraction

import pytest

from nwqlib import Expectation, plan, prepare, scan, solve, submit
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.core.records import Rational, Unit
from nwqlib.operators.inputs import ingest_pauli
from nwqlib.resources import ResourceContext
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
    assert restored.data.forecast == forecast
    assert restored.data.allocation == allocation
    with pytest.raises(ApplicabilityError, match="unsupported"):
        prepare(
            ComparisonRow(
                chosen.method, None, reason="unsupported scientific domain", allocation=allocation
            )
        )


def test_unknown_provider_and_controller_populations_are_not_zero_or_one_query():
    from nwqlib import Eigenproblem, estimate
    from nwqlib.algorithms import ADAPT, Lanczos
    from nwqlib.resources import estimate as fold
    from nwqlib.resources.records import GATES

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

    # ADAPT declares reusable query laws. Its four query templates (screen,
    # energy, pair, diagonal) at eight shots each do not mean that its
    # data-dependent trajectory takes four settings or 32 shots.
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
    assert estimate(adaptive).quantity("settings").fact.value is None

    # The estimate of the whole Plan (Method.query_catalog_reason) gives a
    # count as zero only when no ADAPT query can contribute to it: exact
    # evaluations of sampled counts, shots of exact statistics, and every
    # quantum count of the classical route, which runs host computations only.
    # Every other count is unavailable with ADAPT's reason. Each width and
    # byte peak keeps the value, evidence and schedule condition of the
    # construction's own estimate as a footprint of the templates, and the
    # quantum widths stay symbolic in the register-width parameters. A forecast
    # with a profile and a scan row hold this same estimate of the whole Plan.
    problem, method = adaptive.problem, adaptive.method
    reason = method.query_catalog_reason()
    routes = {
        "sampled": (adaptive, {"exact_evaluations"}),
        "exact": (plan(problem, method=method, seed=7), {"shots"}),
        "classical": (plan(problem, method=method, execution="classical", seed=7),
                      {*GATES, "calls", "measurements", "resets", "settings", "unique_settings",
                       "preparation_components", "shots", "exact_evaluations"}),
    }
    for route, (selected_plan, zeros) in routes.items():
        workload, own = estimate(selected_plan), fold(selected_plan.construction)
        assert reason in workload.assumptions and reason not in own.assumptions
        for quantity in workload.quantities:
            fact = quantity.fact
            if quantity.location is not None:
                twin = own.quantity(quantity.metric, location=quantity.location)
                assert quantity.population == fact.scope.domain
                assert "declared footprint" in quantity.population and "whole-workload" not in quantity.population
                assert (fact.availability, fact.value, fact.symbol, fact.evidence, quantity.interpretation,
                        quantity.required_schedule) == (twin.fact.availability, twin.fact.value, twin.fact.symbol,
                                                        twin.fact.evidence, twin.interpretation,
                                                        twin.required_schedule)
            elif quantity.metric in zeros:
                assert fact.value == Rational(numerator=0, denominator=1) and quantity.interpretation == "exact"
            else:
                assert fact.availability == "unknown" and fact.reason == reason
        forecast, _ = original_forecast(selected_plan, model=False)
        assert forecast.resources == workload and forecast.assessments == ()
        width_ranked = scan((Candidate(selected_plan),), objectives=(Objective(kind="logical_width"),))
        assert width_ranked.comparison.rows[0].estimate == workload
        assert width_ranked.selection.values[0][0].value is None
        if route == "classical":
            continue
        width = workload.quantity("logical_width", location="logical_device")
        assert width.fact.availability == "symbolic"
        assert width.required_schedule == ("serial_acquisitions" if route == "sampled" else None)
        # One prepared query is a fully bound point. Its own construction
        # requests one setting of eight shots, or one exact evaluation, and
        # keeps the gate counts that no law gives unavailable.
        with prepare(selected_plan, progress=False).run as run:
            (receipt,) = run.data.receipts
        point = fold(receipt.realization.selected_construction(selected_plan))
        metric, requested = ("shots", 8) if route == "sampled" else ("exact_evaluations", 1)
        assert point.quantity("settings").fact.value == Rational(numerator=1, denominator=1)
        assert point.quantity(metric).fact.value == Rational(numerator=requested, denominator=1)
        assert point.quantity("cx").fact.availability == "unknown" and reason not in point.assumptions


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


def test_two_objective_front_keeps_both_sides_of_a_shots_width_trade_off():
    """Sampled Lanczos and FixedGCIM Plans trade shots against width, so both stay on the front.

    The counts follow the documented sampled workloads at n shots per setting.
    Lanczos with m = 2 measures the moment degrees 1 to 2m-1, three settings.
    FixedGCIM measures b(b+1)G/2 settings, here b = 3 real basis vectors and
    G = 2 qubit-wise commuting groups {XI, IX} and {ZZ}: the words have no Y
    and every coefficient and stored amplitude is real, so H and S are real
    and only the real parts are measured. The widths, 4 for Lanczos
    and 3 for FixedGCIM (two system qubits and the pair ancilla), are those
    docs/scientist.md reports for these Methods. The width of a Plan with
    several measurement runs holds only when the runs execute one after
    another, hence the serial context.
    """
    from nwqlib import Eigenproblem
    from nwqlib.algorithms import FixedGCIM, Lanczos

    problem = Eigenproblem(A=ingest_pauli((("ZZ", 1.0), ("XI", 0.5), ("IX", 0.5)), num_qubits=2))
    lanczos = Lanczos(initial_state=[1, 0, 0, 0], krylov_dimension=2)
    gcim = FixedGCIM(basis=([1, 0, 0, 0], [0, 1, 1, 0], [0, 0, 0, 1]))
    n = 64
    plans = (plan(problem, method=lanczos, shots=n, seed=7),
             plan(problem, method=gcim, shots=n, seed=7),
             plan(problem, method=gcim, shots=2 * n, seed=7))
    objectives = (Objective(kind="requested_shots"), Objective(kind="logical_width"))
    ranked = scan(tuple(Candidate(p) for p in plans), objectives=objectives,
                  context=ResourceContext(batch_schedule="serial"))
    expected = [[3 * n, 4], [3 * 4 // 2 * 2 * n, 3], [3 * 4 // 2 * 2 * 2 * n, 3]]
    assert [[number(v) for v in row] for row in ranked.selection.values] == expected
    # Row 0 has the fewest shots and row 1 the smallest width, so neither dominates
    # the other. Row 2 ties row 1 on width and needs more shots.
    assert ranked.selection.nondominated == (0, 1) and ranked.selection.incomparable == ()
    assert scan(ranked.comparison, objectives=objectives[:1]).selection.nondominated == (0,)
    assert scan(ranked.comparison, objectives=objectives[1:]).selection.nondominated == (1, 2)
    for index in (0, 1):
        result = solve(ranked.select(index))
        assert result.plan is plans[index]
        assert sum(event.shots for event in result.data.trace.events) == expected[index][0]


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
