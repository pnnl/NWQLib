"""Actual Result/Plan domains, saved pair validation and independent legal cases.

External Method execution/storage is covered by the nonconstant Hadamard author
witness in test_public_cli; retired Options/ChoiceReport identities add no relation.
"""

import json
from types import SimpleNamespace

import numpy as np
import pytest

import nwqlib
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.algorithms.gcim.fixed_basis import ProjectedPencil
from nwqlib.algorithms.lanczos import Lanczos
from nwqlib.core import Complex128
from nwqlib.operators import ingest_pauli
from nwqlib._prepared_execution import Run
from nwqlib._projected_eigensolver import DEFAULT_OVERLAP_EIGENVALUE_CUTOFF as DEFAULT_CUTOFF


def forbidden(*args, **kwargs):
    raise AssertionError("a record-domain check replayed numerical/native work")


def scalar_result(family, value):
    operator = ingest_pauli((("I", value),), num_qubits=1)
    if family == "expectation":
        return nwqlib.solve(
            nwqlib.Expectation(state=[1, 0], observable=operator), method=ExpectationMethod()
        )
    method = (
        FixedGCIM(basis=([1, 0],))
        if family == "gcim"
        else Lanczos(initial_state=[1, 0], krylov_dimension=1)
    )
    return nwqlib.solve(nwqlib.Eigenproblem(A=operator), method=method, seed=7)


@pytest.mark.parametrize("family", ["expectation", "gcim", "lanczos"])
@pytest.mark.parametrize("value", [0.0, -3.0])
def test_known_constant_pair_and_saved_load_reject_wrong_scalar(
    family, value, tmp_path, monkeypatch
):
    import scipy.linalg

    for owner in (np.linalg, scipy.linalg):
        for name in ("eigh", "eigvalsh", "svd"):
            monkeypatch.setattr(owner, name, forbidden)
    result = scalar_result(family, value)
    field = "value" if family == "expectation" else "eigenvalue"
    assert getattr(result, field) == value and result.data.trace.events == ()
    result.validate_plan(result.plan)
    changes = {field: value + 1}
    if family == "lanczos":
        changes["eigenvalues"] = (value + 1,)
    if family == "gcim":
        changes["pencil"] = pencil(result.plan, value=value + 1)
    bad = result.revise(**changes)
    with pytest.raises(ValueError, match="constant|identity"):
        bad.validate_plan(result.plan)
    with pytest.raises(ValueError, match="constant|identity"):
        bad._attach(result.plan, result.data)
    path = result.save(tmp_path / family)
    monkeypatch.setattr(type(result.plan.method), "plan", forbidden)
    monkeypatch.setattr(type(result.plan.method), "analyze", forbidden)
    loaded = nwqlib.load_result(path)
    assert loaded.report() == result.report() and getattr(loaded, field) == value
    # This is a structurally valid scalar record in the same saved selection
    # and acquisition data. The actual loader must reject the false relation.
    filename = path / "result.json"
    saved = json.loads(filename.read_text())
    saved["result"] = bad.model_dump(mode="json")
    filename.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="constant|identity"):
        nwqlib.load_result(path)


def pencil(plan, *, value=3.0, failed=False):
    def z(value):
        return Complex128(real=value, imag=0.0)

    return ProjectedPencil(
        hamiltonian=((z(value),),),
        overlap=((z(1.0),),),
        overlap_eigenvalues=(1.0,),
        eigenvalues=() if failed else (value,),
        coordinate_vectors=() if failed else ((z(1.0),),),
        kept_rank=0 if failed else 1,
        overlap_cutoff=plan.method.overlap_cutoff,
        overlap_filter="sampled_positive_subspace"
        if plan.shots is not None
        else "deterministic_gram",
        overlap_condition_number=None if failed else 1.0,
        projected_residual=None if failed else 0.0,
        projected_backward_error=None if failed else 0.0,
        overlap_normalization_error=None if failed else 0.0,
        failure_reason="no Gram eigenvalue above selected cutoff" if failed else None,
    )


def gcim_plan(*, shots=None, cutoff=1e-12):
    return nwqlib.plan(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 2.0), ("Z", 1.0)), num_qubits=1)),
        method=FixedGCIM(basis=([1, 0],), overlap_cutoff=cutoff),
        shots=shots,
        seed=7,
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("projected_residual", -1.0),
        ("projected_backward_error", -1.0),
        ("overlap_normalization_error", -1.0),
        ("overlap_condition_number", 0.5),
        ("overlap_cutoff", 0.0),
        ("overlap_cutoff", -1.0),
    ],
)
def test_gcim_intrinsic_diagnostic_and_cutoff_domains(field, value):
    with pytest.raises(ValueError):
        pencil(gcim_plan()).revise(**{field: value})


@pytest.mark.parametrize("shots", [None, 10])
def test_gcim_selected_filter_and_actual_analysis_cutoff_preserve_partial_failed_and_reanalysis(
    shots, monkeypatch, request, tmp_path
):
    from nwqlib.backends import qiskit_aer as aer

    plan = gcim_plan(shots=shots)
    with Run(plan) as run:
        partial = plan.method.analyze(plan, run.data, settings={})
        partial.validate_plan(plan)
        assert partial.pencil is None and partial.eigenvalue is None and partial.missing
    # A one-column |0> witness has S=1,H=3. The sampled mode reads it from
    # supplied counts, and the exact mode reduces the one-qubit saved state
    # on Aer. Both therefore produce the same physical Ritz value without any
    # reference.
    if shots:
        request.getfixturevalue("prepared_stubs")
        monkeypatch.setattr(
            aer,
            "_submit_aer_execution",
            lambda *a, **k: SimpleNamespace(raw_output={"counts": {"00": 10}},
                                            metadata={"native_job_id": "supplied"}),
        )
    result = nwqlib.solve(plan)
    assert result.eigenvalue == 3.0 and result.pencil.projected_residual == 0.0
    good = result.analyze(overlap_cutoff=0.25)
    assert good.plan is plan and good.data is result.data
    assert good.analysis_cutoff == good.pencil.overlap_cutoff == 0.25
    failed = result.analyze(overlap_cutoff=2.0)
    assert failed.eigenvalue is None and failed.pencil.overlap_eigenvalues == (1.0,)
    assert failed.pencil.projected_residual is None and failed.pencil.failure_reason
    assert "Ritz eigenvalue: 3" in str(good)
    assert "Ritz eigenvalue: unavailable" in str(failed)
    assert failed.pencil.failure_reason in str(failed)
    for actual in (good, failed):
        actual.validate_plan(plan)
        loaded = nwqlib.load_result(actual.save(tmp_path / str(actual.analysis_cutoff)))
        assert loaded.pencil == actual.pencil and loaded.analysis_cutoff == actual.analysis_cutoff
    for changed in (
        good.pencil.revise(overlap_cutoff=0.125),
        good.pencil.revise(
            overlap_filter="deterministic_gram" if shots else "sampled_positive_subspace"
        ),
    ):
        with pytest.raises(ValueError, match="settings|filter"):
            good.revise(pencil=changed).validate_plan(plan)


def lanczos_partial(*, scalar=False, sampled=False, requested=None):
    terms = (("I", 2.0),) if scalar else (("I", 2.0), ("Z", 1.0))
    plan = nwqlib.plan(
        nwqlib.Eigenproblem(A=ingest_pauli(terms, num_qubits=1)),
        method=Lanczos(initial_state=[1, 0], krylov_dimension=1, overlap_cutoff=requested),
        shots=2 if sampled else None,
        seed=7,
    )
    with Run(plan) as run:
        return plan, plan.method.analyze(plan, run.data, settings={})


@pytest.mark.parametrize(
    "change",
    [
        {"cutoff": 0.0},
        {"cutoff": -1.0},
        {"cutoff_source": "invented"},
        {"analysis_cutoff": 0.0},
        {"projected_backward_error": -1.0},
        {"overlap_normalization_error": -1.0},
    ],
)
def test_lanczos_intrinsic_diagnostic_domains(change):
    _, result = lanczos_partial()
    with pytest.raises(ValueError):
        result.revise(**change)


@pytest.mark.parametrize("family", ["gcim", "lanczos"])
def test_invalid_analysis_cutoff_rejects_before_statistics_or_projected_work(family, monkeypatch):
    from nwqlib.algorithms.gcim import fixed_basis
    import nwqlib.algorithms.lanczos.numerical as numerical

    plan = gcim_plan() if family == "gcim" else lanczos_partial()[0]
    with Run(plan) as run:
        monkeypatch.setattr(fixed_basis, "matched_chunks", forbidden)
        monkeypatch.setattr(numerical, "reconstruct", forbidden)
        monkeypatch.setattr(Lanczos, "statistics", forbidden)
        for cutoff in (True, False, 0.0, -1.0, float("inf")):
            with pytest.raises(ValueError, match="finite and positive"):
                plan.method.analyze(plan, run.data, settings={"overlap_cutoff": cutoff})


@pytest.mark.parametrize("scalar", [False, True])
def test_lanczos_scalar_and_incomplete_branches_keep_requested_cutoff(scalar):
    plan, result = lanczos_partial(scalar=scalar, requested=0.25)
    result.validate_plan(plan)
    assert result.cutoff == 0.25 and result.cutoff_source == (
        "scalar_operator" if scalar else "not_evaluated"
    )
    for change in ({"cutoff": 0.125}, {"cutoff_source": "user_fixed"}):
        with pytest.raises(ValueError, match="cutoff/source"):
            result.revise(**change).validate_plan(plan)
    if scalar:
        result.revise(
            moments=(1.0, None), moment_variances=(0.0, None), missing=(1,)
        ).validate_plan(plan)


@pytest.mark.parametrize("shots", [None, 1, 2])
def test_lanczos_complete_cutoff_branches_and_saved_failures(
    shots, monkeypatch, prepared_stubs, tmp_path
):
    from nwqlib.backends import qiskit_aer as aer
    import nwqlib.algorithms.lanczos.numerical as numerical

    plan = nwqlib.plan(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 2.0), ("Z", 1.0)), num_qubits=1)),
        method=Lanczos(initial_state=[1, 0], krylov_dimension=1),
        execution="classical" if shots is None else "quantum",
        shots=shots,
        seed=7,
    )
    monkeypatch.setattr(
        aer,
        "_submit_aer_execution",
        lambda *a, **k: SimpleNamespace(
            raw_output={"counts": {"0": shots}}, metadata={"native_job_id": "supplied"}
        ),
    )
    # A supplied failure after cutoff selection keeps the actual raw spectrum;
    # this test checks branch metadata, not solver numerical accuracy.
    monkeypatch.setattr(
        numerical,
        "_solve_projected_pencil",
        lambda *a, **k: (None, np.array([1.0]), "supplied projected failure", None),
    )
    result = nwqlib.solve(plan)
    result.validate_plan(plan)
    expected = (
        "deterministic_default"
        if shots is None
        else "empirical_gram_rms"
    )
    assert result.cutoff_source == expected
    # m=1 has S00=mu0=1 exactly. The sampled mu1 belongs to H only,
    # so either positive shot population leaves the numerical Gram floor.
    assert result.cutoff == DEFAULT_CUTOFF
    changed = result.analyze(overlap_cutoff=0.25)
    assert (
        changed.cutoff == changed.analysis_cutoff == 0.25 and changed.cutoff_source == "user_fixed"
    )
    assert changed.data is result.data and changed.plan is plan
    for actual in (result, changed):
        actual.validate_plan(plan)
        loaded = nwqlib.load_result(actual.save(tmp_path / actual.cutoff_source))
        assert loaded.cutoff_source == actual.cutoff_source and loaded.cutoff == actual.cutoff
    with pytest.raises(ValueError, match="cutoff|source"):
        result.revise(cutoff_source="user_fixed").validate_plan(plan)
    with pytest.raises(ValueError, match="cutoff|source"):
        changed.revise(cutoff=0.125).validate_plan(plan)
    if shots == 2:
        with pytest.raises(ValueError, match="cutoff"):
            result.revise(cutoff=DEFAULT_CUTOFF / 2).validate_plan(plan)
