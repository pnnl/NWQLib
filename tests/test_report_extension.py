"""Result displays and summaries from existing data, and an external PREP archive round trip."""

import pytest
import numpy as np

import nwqlib
from nwqlib.algorithms import Lanczos
from nwqlib.operators import ingest_pauli


def constant_result():
    return nwqlib.solve(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 2.0),), num_qubits=1)),
        method=Lanczos(initial_state=[1, 0], krylov_dimension=1),
        seed=7,
    )


def no_display_work(*args, **kwargs):
    pytest.fail("display/report performed new numerical, acquisition or archive work")


def test_scalar_display_and_report_keep_the_origin_without_new_work(tmp_path, monkeypatch):
    from io import StringIO
    from IPython.lib.pretty import pretty
    from nwqlib.core.planning import RandomStreams

    result = constant_result()
    assert result.eigenvalue == 2.0
    detached = result.revise(origin=None)
    with monkeypatch.context() as patch:
        patch.setattr(Lanczos, "plan", no_display_work)
        patch.setattr(Lanczos, "analyze", no_display_work)
        patch.setattr(type(result), "validate_plan", no_display_work)
        patch.setattr(RandomStreams, "next_seed", no_display_work)
        patch.setattr(np.linalg, "norm", no_display_work)
        for shown in (result, detached):
            print(shown, file=StringIO())
            repr(shown)
            pretty(shown)
        report = result.report()
        assert report["result"]["origin"] == result.origin.model_dump(mode="json")
        # Exact readout publishes only the proved zero sampling contribution.
        assert [f["fact"]["quantity"] for f in report["result"]["facts"]] == ["sampling"]
        assert report["receipts"] == report["artifacts"] == []
        assert report["forecast"] is report["allocation"] is None
        assert detached.report()["plan"] is detached.report()["observations"] is None
    # Tab completion on any record, here a Result, its Plan, its Method and an
    # option record, lists every field, None ones included, and NWQLib's own
    # methods, but not the rest of pydantic's BaseModel API. On a Method it
    # also leaves out the protocol methods that the Method base class declares,
    # which the class and help() still show.
    from pydantic import BaseModel
    from nwqlib.algorithms.lchs import LCHS, Trotter
    from nwqlib.algorithms.protocol import Method
    protocol = {name for name, value in vars(Method).items() if not name.startswith("_") and callable(value)}
    for record in (result, result.plan, result.plan.method, LCHS(), Trotter(steps=4)):
        names = set(dir(record))
        own = {name for cls in type(record).__mro__ if cls.__module__.startswith("nwqlib")
               for name in vars(cls) if not name.startswith("_")} - set(dir(BaseModel))
        hidden = protocol if isinstance(record, Method) else set()
        assert set(type(record).model_fields) | (own - hidden) | {"model_dump", "model_dump_json"} <= names
        assert not names & ({"parse_obj", "dict", "json", "copy", "schema", "model_validate", "model_fields"}
                            | hidden)
        assert hidden <= set(dir(type(record)))
    assert {"sampling_shots", "revise", "content_id"} <= set(dir(LCHS()))
    # Storage is an independent operation; it must not call the reading API.
    monkeypatch.setattr(type(result), "report", no_display_work)
    assert nwqlib.load_result(result.save(tmp_path / "independent-save")).eigenvalue == 2.0


def test_initial_hundred_qubit_output_display_is_metadata_only(monkeypatch):
    from nwqlib._prepared_execution import Run
    from nwqlib.algorithms import LCHS
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import StateInput, ingest_occupation

    state = ingest_occupation((0,) * 100, num_qubits=100)
    selected = nwqlib.plan(
        nwqlib.LinearDynamics(A=PeriodicStencil(num_qubits=100, mass=.1, diffusion=.1), initial_state=state, time=0),
        method=LCHS(), seed=7,
    )
    # This unacquired prepared-state request is intentionally never prepared:
    # its 2**100-amplitude output is metadata, not authorized native work.
    with Run(selected) as run:
        result = selected.method.analyze(selected, run.data, settings={})._attach(selected, run.data)
    assert not result.data.observations.chunks and result.artifact is None
    monkeypatch.setattr(StateInput, "physical_vector", no_display_work)
    monkeypatch.setattr(np, "zeros", no_display_work)
    monkeypatch.setattr(np, "empty", no_display_work)
    assert f"shape=({1 << 100},)" in str(result)
    assert "no acquired array" in str(result) and "no prepared target recorded" in str(result)
    assert result.report()["artifacts"] == []


def test_lchs_summary_discloses_existing_nonzero_bounds_and_psd_shift(monkeypatch):
    import scipy.linalg
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import providers
    from nwqlib.artifacts import ArtifactHandle

    result = nwqlib.solve(nwqlib.LinearDynamics(A=[[-.3,.1],[.1,.4]],
        initial_state=[1.,0.], time=.1), method=LCHS(),
        execution="classical", output=nwqlib.NormSquared())
    rec = result.plan.reconstruction
    assert rec.psd_shift > 0 and rec.kernel_approximation_bound > 0 and rec.quadrature_bound > 0
    before = result.model_dump(mode="json")
    monkeypatch.setattr(LCHS, "plan", no_display_work)
    monkeypatch.setattr(LCHS, "analyze", no_display_work)
    monkeypatch.setattr(LCHS, "verify", no_display_work)
    monkeypatch.setattr(providers, "eq7_tail_bound", no_display_work)
    monkeypatch.setattr(providers, "_ellipse_rule", no_display_work)
    monkeypatch.setattr(scipy.linalg, "expm", no_display_work)
    monkeypatch.setattr(ArtifactHandle, "array", property(no_display_work))
    monkeypatch.setattr(type(result), "validate_plan", no_display_work)
    text = str(result)
    for field in ("kernel_approximation_bound", "quadrature_bound", "psd_shift"):
        assert format(getattr(rec, field), ".8g") in text
    assert "PSD premise: numerical" in text and "not total physical-output error" in text
    assert result.model_dump(mode="json") == before


@pytest.mark.parametrize("family", ("qpe", "qhd"))
def test_actual_estimator_controller_and_candidate_summaries_keep_scope(family, monkeypatch):
    from test_run_archive import classical_plan

    selected = classical_plan(family)
    if family == "qpe":
        # Several positive powers fit this one-mode complex signal. The
        # result must explain why this acquisition has no uncertainty interval.
        selected = nwqlib.plan(selected.problem, method=selected.method.revise(max_time=.8),
                              execution="classical", seed=7)
    result = nwqlib.solve(selected)
    if family == "qpe":
        assert result.estimator_value == pytest.approx(1.0, abs=1e-5)
        assert result.interval is None and result.fit is not None
    else:
        assert result.candidate_coordinates == (0.0,)
        assert result.value == pytest.approx(.04, abs=1e-15)
    with monkeypatch.context() as patch:
        patch.setattr(type(selected.method), "execute", no_display_work)
        patch.setattr(type(selected.method), "analyze", no_display_work)
        text, report = str(result), result.report()
    if family == "qpe":
        assert "qcels" in text and "prepared population" in text and "ground identity not established" in text
        assert "uncertainty interval unavailable" in text and "effective grid" in text
        assert "unique mode" in text
    else:
        assert "Candidate (" in text and "0.04" in text and "not a global optimum" in text
        assert f"Tie representative: {result._values_text(result.most_probable_coordinates)}. Probability: " in text
    assert "accuracy not assessed" in text
    assert result.data.receipts[0].target.name in text
    assert "method_context" not in report


def test_workflow_objects_display_short_summaries_without_hashes_or_arrays():
    from IPython.lib.pretty import pretty
    from nwqlib.algorithms import LCHS, QLS

    problem = nwqlib.LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1, unit="m")
    comparison = nwqlib.compare(problem, methods=(LCHS(), QLS()))
    planned, refused = comparison.rows
    registration = next(entry for entry in nwqlib.methods() if entry.source.name == "lchs")
    shown = {"problem": problem, "plan": planned.plan, "comparison": comparison, "planned row": planned,
             "refused row": refused, "registration": registration}
    for label, value in shown.items():
        text = repr(value)
        assert text == str(value) == pretty(value), label
        assert "sha256:" not in text and "0.15" not in text and len(text.splitlines()) <= 6, label
    shown_problem = repr(problem)
    assert all(part in shown_problem for part in ("LinearDynamics", "unit: m", "A: dense 2x2", "time: 0.1"))
    state_line = next(line for line in shown_problem.splitlines() if "initial_state:" in line)
    assert "vector" in state_line and "2" in state_line
    shown_comparison = repr(comparison)
    assert "[0] LCHS: planned" in shown_comparison and refused.reason in shown_comparison.splitlines()[2]
    assert "linear_dynamics" in repr(registration) and "norm_squared" in repr(registration)
    assert repr(planned.estimate) == str(planned.estimate) == pretty(planned.estimate)
    assert "sha256:" not in repr(planned.estimate)


def test_methods_and_option_records_display_as_the_call_that_rebuilds_them():
    """A Method prints as its constructor call with the options that differ from their defaults.

    The expected relation is that the text is the Python call the user writes:
    evaluating it rebuilds an equal Method with the same content hash. A
    default Method, whose route names are stored as records, prints as its
    bare call, and an input prints as a summary without its numbers.
    """
    from IPython.lib.pretty import pretty
    from nwqlib.algorithms import LCHS, QCELS, QHD, QLS, RFE, RWPE, ExpectationMethod, Lanczos
    from nwqlib.algorithms.lchs import MPSCircuit, PTSC, Richardson, SampledPairs, Trotter, TrotterErrorBudgeted
    from nwqlib.algorithms.qhd import CubicSchedule

    for kind in (LCHS, QHD, QLS, QCELS, RFE, RWPE, ExpectationMethod, Lanczos):
        assert repr(kind()) == f"{kind.__name__}()"
    namespace = dict(LCHS=LCHS, QHD=QHD, MPSCircuit=MPSCircuit, PTSC=PTSC, Richardson=Richardson,
                     SampledPairs=SampledPairs, Trotter=Trotter, TrotterErrorBudgeted=TrotterErrorBudgeted,
                     CubicSchedule=CubicSchedule)
    methods = (
        LCHS(ham_evo="trotter"),
        LCHS(ham_evo=Trotter(steps=2),
             estimator=SampledPairs(repeats=4, product_formula="richardson")),
        LCHS(ham_evo=TrotterErrorBudgeted(order=1), lcu_prep=MPSCircuit(num_layers=1),
             lchs_kernel={"implementation": "near_optimal_eq7", "parameters": {"beta": .7}},
             estimator=SampledPairs(product_formula=PTSC(pf_allowance=.01, mu_cap=1.5, taylor_cutoff=8))),
        QHD(num_grid_points=4, encoding="binary", boundary="periodic", schedule=CubicSchedule(s=.5)),
    )
    for method in methods:
        text = repr(method)
        assert text == str(method) == pretty(method)
        assert not any(word in text for word in ("sha256:", "schema_version", "parent_id")), text
        rebuilt = eval(text, namespace)
        assert rebuilt == method and rebuilt.content_id == method.content_id, text
    # A ProviderConfig shown on its own is visibly one, not the name it shortens to inside a Method.
    from nwqlib.algorithms.lchs import ProviderConfig
    for config in (ProviderConfig(implementation="auto"),
                   ProviderConfig(implementation="near_optimal_eq7", parameters={"beta": .7})):
        text = repr(config)
        assert text.startswith("ProviderConfig(") and text == str(config) == pretty(config)
        assert eval(text, {"ProviderConfig": ProviderConfig}) == config
    shown = repr(Lanczos(initial_state=[1, .25], krylov_dimension=2))
    assert "krylov_dimension=2" in shown and ".25" not in shown and "sha256:" not in shown


def test_estimate_display_names_every_metric_and_states_a_shared_reason_once():
    from nwqlib.algorithms import LCHS

    # The quickstart input: no block of its dense construction has a gate-count
    # rule, so every gate metric is unavailable for the same blocks.
    problem = nwqlib.LinearDynamics(A=[[0.4, 0.15], [0.05, 0.25]], initial_state=[1.0, 0.0], time=0.1)
    workload = nwqlib.estimate(nwqlib.plan(problem, method=LCHS()))
    text = str(workload)
    entries = [line for line in text.splitlines()[1:] if not line.startswith(("  ", "Conditions: "))]
    named = [name for line in entries for name in line.split(": ", 1)[0].split(" at ")[0].split(", ")]
    assert sorted(named) == sorted(quantity.metric for quantity in workload.quantities)
    assert len(entries) < len(workload.quantities)
    for quantity in workload.quantities:
        if quantity.interpretation != "unavailable":
            assert str(quantity) in text
        else:
            reason = quantity.fact.reason
            assert reason in text or reason.replace(
                f"no selected {quantity.metric} law/decomposition", "no selected law/decomposition") in text
    assert text.splitlines()[-1] == "Conditions: " + "; ".join(workload.assumptions)


def test_explicit_external_alternative_prep_roundtrip_keeps_actual_recipe(tmp_path, monkeypatch):
    from _hadamard_method import HadamardPauliExpectation
    from nwqlib.problems import ingest_occupation

    problem = nwqlib.Expectation(
        state=ingest_occupation("1", num_qubits=1),
        observable=ingest_pauli((("Z", 1.0),), num_qubits=1),
    )
    result = nwqlib.solve(
        problem, method=HadamardPauliExpectation(prep_choice="hzh"), shots=8, seed=7
    )
    assert result.value == -1.0
    assert str(result).splitlines()[0] == "Normalized expectation: -1"
    path = result.save(tmp_path / "hzh")
    with pytest.raises(ValueError, match="explicit implementation"):
        nwqlib.load_result(path)
    monkeypatch.setattr(
        HadamardPauliExpectation,
        "plan",
        lambda *a, **k: pytest.fail("archive replanned external Method"),
    )
    restored = nwqlib.load_result(path, method=HadamardPauliExpectation)
    assert restored.plan.method.prep_choice == "hzh"
    assert restored.plan.blocks[0].record == result.plan.blocks[0].record
    assert restored.plan.blocks[0]._constructor is result.plan.blocks[0]._constructor
    assert restored.analyze().value == -1.0
    assert nwqlib.solve(restored.plan).value == -1.0
    assert result.data.trace.events == restored.data.trace.events
