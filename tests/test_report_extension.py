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


def test_explicit_external_alternative_prep_roundtrip_keeps_actual_recipe(tmp_path, monkeypatch):
    from _hadamard_method import HadamardPauliExpectation
    from nwqlib.problems import ingest_occupation

    problem = nwqlib.Expectation(
        state=ingest_occupation("1", num_qubits=1),
        observable=ingest_pauli((("Z", 1.0),), num_qubits=1),
    )
    result = nwqlib.solve(
        problem, method=HadamardPauliExpectation(preparation_choice="hzh"), shots=8, seed=7
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
    assert restored.plan.method.preparation_choice == "hzh"
    assert restored.plan.blocks[0].record == result.plan.blocks[0].record
    assert restored.plan.blocks[0]._constructor is result.plan.blocks[0]._constructor
    assert restored.analyze().value == -1.0
    assert nwqlib.solve(restored.plan).value == -1.0
    assert result.data.trace.events == restored.data.trace.events
