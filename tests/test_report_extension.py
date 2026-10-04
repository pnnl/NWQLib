"""Producer-only analysis origin and current saved-metadata admission."""

import json
from importlib.metadata import version
import platform
from uuid import UUID

import pytest
import numpy as np

import nwqlib
from nwqlib.algorithms import Lanczos
from nwqlib.core import Source
from nwqlib.core.analysis import AnalysisOrigin
from nwqlib.execution import ObservationView
from nwqlib.operators import ingest_pauli


def constant_result():
    return nwqlib.solve(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 2.0),), num_qubits=1)),
        method=Lanczos(initial_state=[1, 0], krylov_dimension=1),
        seed=7,
    )


def no_display_work(*args, **kwargs):
    pytest.fail("display/report performed new numerical, acquisition or archive work")


def test_scalar_display_keeps_frame_origin_and_no_criterion_without_work(tmp_path, monkeypatch):
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
        text = str(result)
        printed = StringIO()
        print(result, file=printed)
        assert printed.getvalue() == text + "\n" and repr(result) == pretty(result) == text
        assert text.splitlines()[0] == "Ritz eigenvalue: 2"
        assert result.target_identification in text and "accuracy not assessed" in text
        assert "no prepared target recorded" in text and "Aer" not in text
        report = result.report()
        assert report["result"]["origin"] == result.origin.model_dump(mode="json")
        # Exact readout publishes only the proved zero sampling contribution.
        assert [f["fact"]["quantity"] for f in report["result"]["facts"]] == ["sampling"]
        assert report["summary"] == text
        assert report["receipts"] == report["artifacts"] == []
        assert report["forecast"] is report["allocation"] is None
        assert "not attached" in str(detached) and "accuracy not assessed" in str(detached)
        assert detached.report()["plan"] is detached.report()["observations"] is None
    # Storage is an independent operation; it must not call the reading API.
    monkeypatch.setattr(type(result), "report", no_display_work)
    assert nwqlib.load_result(result.save(tmp_path / "independent-save")).eigenvalue == 2.0


@pytest.mark.parametrize("dimension", (2, 16))
def test_lchs_resident_array_preview_and_shape_do_not_repeat_work(dimension, monkeypatch):
    import scipy.linalg
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs.primary_records import LCHSAnalysis
    from nwqlib.artifacts import ArtifactHandle
    from nwqlib.problems.inputs import StateInput

    diagonal = np.arange(1, dimension + 1) / 10
    initial = np.full(dimension, 1j)
    selected = nwqlib.plan(
        nwqlib.LinearDynamics(A=1j * np.diag(diagonal), initial_state=initial, time=.1),
        method=LCHS(), execution="classical", seed=7,
    )
    # One explicit finite-quadrature host application on at most sixteen
    # coordinates; all branch actions coincide because L=0. No circuit.
    assert selected.reconstruction.mode == "quadrature"
    result = nwqlib.solve(selected)
    handle = result.data.artifact(result.artifact)
    array = handle._array
    from nwqlib.algorithms.lchs.providers import resolve_lchs_coefficient_plan
    coefficient_sum = sum(resolve_lchs_coefficient_plan(selected).coefficients)
    np.testing.assert_allclose(array, coefficient_sum * 1j * np.exp(-.1j * diagonal), rtol=0, atol=3e-15)
    first = format(array[0], ".8g")
    with monkeypatch.context() as patch:
        patch.setattr(LCHS, "analyze", no_display_work)
        patch.setattr(LCHS, "execute", no_display_work)
        patch.setattr(LCHSAnalysis, "solution", property(no_display_work))
        patch.setattr(LCHSAnalysis, "state_vector", property(no_display_work))
        patch.setattr(StateInput, "physical_vector", no_display_work)
        patch.setattr(ArtifactHandle, "array", property(no_display_work))
        for name in ("norm", "eigh", "eigvalsh", "svd"):
            patch.setattr(np.linalg, name, no_display_work)
        for name in ("array", "asarray", "copy"):
            patch.setattr(np, name, no_display_work)
        patch.setattr(scipy.linalg, "expm", no_display_work)
        text, report = str(result), result.report()
        assert result.facts and report["result"]["facts"]
        assert "accuracy not assessed" in text  # Component facts do not supply a criterion.
        assert "Selected kernel bounds per unit input before PSD growth: approximation=0; quadrature=0" in text
        assert "PSD premise: numerical; shift=0" in text
        assert "not total physical-output error" in text
        assert "Physical solution" in text and "physical" in text
        assert f"shape=({dimension},)" in text and "classically exponentiated" in text
        assert (first in text) == (dimension <= 8)
        assert report["artifacts"] == [dict(manifest=result.artifact.model_dump(mode="json"), available=True)]
        assert report["receipts"] == [receipt.model_dump(mode="json") for receipt in result.data.receipts]
        assert ObservationView.model_validate(report["observations"]) == result.data.observations
        # Each chunk keeps its identity; its stored statistics get none computed.
        chunks = report["observations"]["chunks"]
        assert [chunk["content_id"] for chunk in chunks] == [c.content_id for c in result.data.observations.chunks]
        assert chunks and all("content_id" not in value for chunk in chunks for value in chunk["values"])
        assert report["controller"] == result.data.controller
        json.dumps(report)  # No ndarray or native object in the reading projection.
    assert handle._array is array and not array.flags.writeable


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


def test_qls_solution_display_preserves_original_complex_scale(monkeypatch):
    from nwqlib.algorithms import QLS
    from nwqlib.algorithms.qls.primary_records import QLSAnalysis
    from nwqlib.artifacts import ArtifactHandle

    result = nwqlib.solve(nwqlib.LinearSystem(A=[[1, 0], [0, 1]], b=[3j, 4]),
                         method=QLS(), execution="classical", seed=7)
    array = result.data.artifact(result.artifact)._array
    # The selected inverse polynomial approximates A^-1, with the stated
    # relative tolerance here; physical RHS scale must not disappear.
    np.testing.assert_allclose(array, [3j, 4], rtol=.01, atol=0)
    with monkeypatch.context() as patch:
        patch.setattr(QLS, "analyze", no_display_work)
        patch.setattr(QLSAnalysis, "value", property(no_display_work))
        patch.setattr(ArtifactHandle, "array", property(no_display_work))
        patch.setattr(np.linalg, "norm", no_display_work)
        text = str(result)
        assert "Physical solution" in text and "physical" in text
        assert format(array[0], ".8g") in text and format(array[1], ".8g") in text
        assert "accuracy not assessed" in text
        assert result.report()["artifacts"][0]["manifest"]["output"]["frame"] == "physical"


def test_scalar_units_come_from_selected_output_frame_without_new_norm(monkeypatch):
    from nwqlib.algorithms import LCHS

    result = nwqlib.solve(nwqlib.LinearDynamics(A=[[1]], initial_state=[3j], time=0, unit="m"),
                         method=LCHS(), output=nwqlib.NormSquared(), seed=7)
    assert result.value == 9.0 and result.plan.error_model.frame.unit.symbol == "(m)^2"
    monkeypatch.setattr(np.linalg, "norm", no_display_work)
    monkeypatch.setattr(LCHS, "analyze", no_display_work)
    assert str(result).splitlines()[0] == "norm_squared: 9 (m)^2"
    assert "accuracy not assessed" in str(result)


@pytest.mark.parametrize("unit_vector", (False, True))
def test_initial_display_reads_resident_vector_with_original_scale_phase_and_units(unit_vector, monkeypatch):
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs.primary_records import LCHSAnalysis
    from nwqlib.problems.inputs import StateInput

    output = (nwqlib.StateVector(normalization="unit", global_phase="modulo_global_phase")
              if unit_vector else nwqlib.Solution())
    result = nwqlib.solve(nwqlib.LinearDynamics(A=[[1, 0], [0, 1]], initial_state=[3j, 4],
                                              time=0, unit="m"),
                         method=LCHS(), output=output, seed=7)
    state = result.plan.problem.initial_state
    array = state._direction if unit_vector else state._physical
    np.testing.assert_array_equal(array, [.6j, .8] if unit_vector else [3j, 4])
    assert result.artifact is None and not result.data.observations.chunks
    with monkeypatch.context() as patch:
        patch.setattr(StateInput, "physical_vector", no_display_work)
        patch.setattr(LCHSAnalysis, "solution", property(no_display_work))
        patch.setattr(LCHSAnalysis, "state_vector", property(no_display_work))
        for name in ("array", "asarray", "zeros", "copy"):
            patch.setattr(np, name, no_display_work)
        patch.setattr(np.linalg, "norm", no_display_work)
        text = str(result)
        assert ("[0+0.6j, 0.8+0j]" if unit_vector else "[0+3j, 4+0j]") in text
        assert "supplied initial vector" in text and "no acquired array" not in text
        assert ("unit=1" if unit_vector else "unit=m") in text
        assert ("phase=modulo_global_phase" if unit_vector else "phase=physical") in text
        assert result.report()["artifacts"] == []
    assert (state._direction if unit_vector else state._physical) is array


def test_compact_zero_array_display_uses_actual_host_acquisition_and_keeps_unit_zero_unavailable(monkeypatch):
    from nwqlib.algorithms import LCHS
    from nwqlib.algorithms.lchs import host
    from nwqlib.artifacts import ArtifactHandle
    from nwqlib.operators.inputs import PeriodicStencil
    from nwqlib.problems.inputs import ingest_product

    problem = nwqlib.LinearDynamics(A=PeriodicStencil(num_qubits=2, mass=.1, diffusion=.2),
        initial_state=ingest_product([[0, 0], [1, 1]]), time=0, unit="m")
    result = nwqlib.solve(problem, method=LCHS(), execution="quantum", seed=7)
    unit = nwqlib.solve(problem, method=LCHS(), output=nwqlib.StateVector(normalization="unit"), seed=7)
    assert result.plan.execution == "quantum"
    assert len(result.data.receipts) == 1 and result.data.receipts[0].execution == "host_kernel"
    assert result.data.observations.chunks[0].execution == "host_kernel"
    handle = result.data.artifact(result.artifact)
    array = handle._array
    np.testing.assert_array_equal(array, [0, 0, 0, 0])
    assert unit.unavailable is not None and unit.artifact is None
    with monkeypatch.context() as patch:
        patch.setattr(host, "_execute_initial_output", no_display_work)
        patch.setattr(ArtifactHandle, "array", property(no_display_work))
        for name in ("array", "asarray", "zeros", "copy"):
            patch.setattr(np, name, no_display_work)
        text = str(result)
        assert "[0+0j, 0+0j, 0+0j, 0+0j]" in text and "unit=m" in text
        assert "Array acquisition: host kernel" in text and "simulator" not in text
        assert result.report()["artifacts"][0]["available"] is True
        assert unit.unavailable in str(unit) and "no acquired array" in str(unit)
        assert "[0+0j" not in str(unit)
    assert handle._array is array


@pytest.mark.parametrize("family", ("adapt", "qpe", "qhd"))
def test_actual_estimator_controller_and_candidate_summaries_keep_scope(family, monkeypatch):
    from test_run_archive import classical_plan

    selected = classical_plan(family)
    if family == "qpe":
        # Several positive powers fit this one-mode complex signal. The
        # result must explain why this acquisition has no uncertainty interval.
        selected = nwqlib.plan(selected.problem, method=selected.method.revise(max_time=.8),
                              execution="classical", seed=7)
    result = nwqlib.solve(selected)
    if family == "adapt":
        assert result.eigenvalue == pytest.approx(-1.0, abs=1e-12)
        assert result.selected
    elif family == "qpe":
        assert result.estimator_value == pytest.approx(1.0, abs=1e-5)
        assert result.interval is None and result.fit is not None
    else:
        assert result.candidate_coordinates == (0.0,)
        assert result.value == pytest.approx(.04, abs=1e-15)
    with monkeypatch.context() as patch:
        patch.setattr(type(selected.method), "execute", no_display_work)
        patch.setattr(type(selected.method), "analyze", no_display_work)
        text, report = str(result), result.report()
    if family == "adapt":
        assert "Ritz eigenvalue: -1" in text and "not established" in text
        assert report["controller"] == result.data.controller and report["controller"] is not None
    elif family == "qpe":
        assert "qcels" in text and "prepared population" in text and "ground identity not established" in text
        assert "uncertainty interval unavailable" in text and "effective grid" in text
        assert "unique mode" in text
    else:
        assert "Candidate (" in text and "0.04" in text and "not a global optimum" in text
        assert f"Tie representative: {result._values_text(result.most_probable_coordinates)}. Probability: " in text
    assert "accuracy not assessed" in text
    assert result.data.receipts[0].target.name in text
    assert "method_context" not in report


def test_actual_analysis_origin_is_fresh_but_load_and_report_never_capture(tmp_path, monkeypatch):
    """New analysis gets a fresh origin, while saved-result loading and reporting must preserve
    the original one.
    """
    import nwqlib.core.analysis as owner

    first = constant_result()
    second = first.analyze()
    assert first.eigenvalue == second.eigenvalue == 2.0 and not first.data.trace.events
    assert first.data is second.data and first.plan is second.plan
    assert first.origin.invocation_id != second.origin.invocation_id
    assert (
        UUID(first.origin.invocation_id).version == UUID(second.origin.invocation_id).version == 4
    )
    assert first.origin.analyzer == second.origin.analyzer
    assert first.origin.method_id == first.plan.method.content_id
    environment = {entry.name: entry.version for entry in first.origin.environment}
    assert environment["python"] == platform.python_version()
    for name in ("numpy", "scipy", "pydantic"):
        assert environment[name] == version(name)
    fields = first.model_dump(exclude={"origin", "parent_id", "content_id"})
    assert fields == second.model_dump(exclude={"origin", "parent_id", "content_id"})
    folder = first.save(tmp_path / "origin")

    def forbidden(*args, **kwargs):
        pytest.fail("loading/reporting captured a new analysis invocation or environment")

    monkeypatch.setattr(owner, "uuid4", forbidden)
    monkeypatch.setattr(owner, "version", forbidden)
    monkeypatch.setattr(owner.platform, "python_version", forbidden)
    monkeypatch.setattr(Lanczos, "analyze", forbidden)
    restored = nwqlib.load_result(folder)
    assert type(restored.origin) is AnalysisOrigin and restored.origin == first.origin
    assert restored.report() == first.report()
    assert type(restored).model_validate_json(restored.model_dump_json()).origin == first.origin
    unknown = first.revise(origin=None)
    unknown.validate_plan(first.plan)
    assert type(unknown).model_validate_json(unknown.model_dump_json()).origin is None
    changed = first.revise(origin=first.origin.revise(method_id="sha256:" + "f" * 64))
    with pytest.raises(ValueError, match="origin.*Method"):
        changed._attach(first.plan, first.data)
    assert first.origin.method_id == first.plan.method.content_id


def test_analysis_origin_admits_actual_inputs_before_metadata_and_keeps_unknown_versions(
    monkeypatch,
):
    import nwqlib.core.analysis as owner

    source = Source(
        name="supplied reducer",
        version="analysis-v3",
        domain="scalar analysis",
        reference="fixture",
    )
    calls = []

    def missing(package):
        calls.append(package)
        raise owner.PackageNotFoundError(package)

    monkeypatch.setattr(owner, "version", missing)
    for fields in (
        dict(analyzer="not a Source"),
        dict(analyzer=source, method_id="invalid"),
        dict(analyzer=source, dependencies=(object(),)),
        dict(analyzer=source, dependencies=("a", "a"), max_dependencies=1),
    ):
        with pytest.raises((ValueError, TypeError)):
            owner.capture_analysis_origin(**fields)
        assert calls == []
    origin = owner.capture_analysis_origin(
        analyzer=source, dependencies=("selected_optional_kernel",), max_dependencies=1
    )
    assert calls == ["nwqlib", "pydantic", "selected_optional_kernel"]
    assert origin.unavailable_versions == tuple(calls)
    assert tuple(item.name for item in origin.environment) == ("python",)
    assert origin.analyzer == source and origin.method_id is None


def test_malformed_saved_metadata_rejects_before_method_loading(tmp_path, monkeypatch):
    import nwqlib.saved_evidence as owner
    from nwqlib._choice_archive import ArchiveFiles

    result = constant_result()
    folder = result.save(tmp_path / "metadata")
    path = folder / "result.json"
    original = path.read_text()
    calls = []

    def forbidden(*args, **kwargs):
        calls.append(True)
        pytest.fail("malformed JSON reached selected Method loading")

    with monkeypatch.context() as patch:
        patch.setattr(owner, "load_plan", forbidden)
        for malformed, reason in (
            (
                '{"format":"bad",' + original[1:],
                "duplicate",
            ),
            ('{"unselected":NaN,' + original[1:], "finite"),
            ('{"unselected":1e999,' + original[1:], "finite"),
        ):
            path.write_text(malformed)
            with pytest.raises(ValueError, match=reason):
                nwqlib.load_result(folder)
            assert not calls
    path.write_text(original)
    assert nwqlib.load_result(folder).eigenvalue == 2.0
    # The existing file-byte allowance is checked before allocating/parsing JSON.
    with monkeypatch.context() as patch:
        patch.setattr(json, "loads", forbidden)
        with pytest.raises(ValueError, match="file-byte"):
            ArchiveFiles(folder, 1).read_json("result.json")
    assert not calls
    text = {"escaped": '\\"[{]}', "normal": [0, 1.5, None, True]}
    files = ArchiveFiles(folder, 4096)
    files.write_json("ordinary.json", text)
    assert ArchiveFiles(folder, 4096).read_json("ordinary.json") == text


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
