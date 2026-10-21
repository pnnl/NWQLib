"""Actual Result/Plan domains, saved round trips and independent legal cases.

External Method execution/storage is covered by the nonconstant Hadamard author
witness in test_public_cli.
"""

import numpy as np
import pytest

import nwqlib
from nwqlib.algorithms.gcim import FixedGCIM
from nwqlib.algorithms.gcim.fixed_basis import ProjectedPencil
from nwqlib.algorithms.lanczos import Lanczos
from nwqlib.core import Complex128
from nwqlib.operators import ingest_pauli
from nwqlib._prepared_execution import Run


def forbidden(*args, **kwargs):
    raise AssertionError("a record-domain check replayed numerical/native work")


def scalar_result(family, value):
    operator = ingest_pauli((("I", value),), num_qubits=1)
    method = (
        FixedGCIM(basis=([1, 0],))
        if family == "gcim"
        else Lanczos(initial_state=[1, 0], krylov_dimension=1)
    )
    return nwqlib.solve(nwqlib.Eigenproblem(A=operator), method=method, seed=7)


@pytest.mark.parametrize("family", ["gcim", "lanczos"])
@pytest.mark.parametrize("value", [-3.0])
def test_known_constant_result_validates_its_plan_and_round_trips(
    family, value, tmp_path, monkeypatch
):
    import scipy.linalg

    for owner in (np.linalg, scipy.linalg):
        for name in ("eigh", "eigvalsh", "svd"):
            monkeypatch.setattr(owner, name, forbidden)
    result = scalar_result(family, value)
    field = "eigenvalue"
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


def pencil(plan, *, value=3.0):
    def z(value):
        return Complex128(real=value, imag=0.0)

    return ProjectedPencil(
        hamiltonian=((z(value),),),
        overlap=((z(1.0),),),
        overlap_eigenvalues=(1.0,),
        eigenvalues=(value,),
        coordinate_vectors=((z(1.0),),),
        kept_rank=1,
        overlap_cutoff=plan.method.overlap_cutoff,
        overlap_filter="sampled_positive_subspace"
        if plan.shots is not None
        else "deterministic_gram",
        overlap_condition_number=1.0,
        projected_residual=0.0,
        projected_backward_error=0.0,
        overlap_normalization_error=0.0,
        failure_reason=None,
    )


def gcim_plan():
    return nwqlib.plan(
        nwqlib.Eigenproblem(A=ingest_pauli((("I", 2.0), ("Z", 1.0)), num_qubits=1)),
        method=FixedGCIM(basis=([1, 0],), overlap_cutoff=1e-12),
        shots=None,
        seed=7,
    )


def test_gcim_selected_filter_and_actual_analysis_cutoff_preserve_partial_failed_and_reanalysis(tmp_path):
    plan = gcim_plan()
    with Run(plan) as run:
        partial = plan.method.analyze(plan, run.data, settings={})
        partial.validate_plan(plan)
        assert partial.pencil is None and partial.eigenvalue is None and partial.missing
    # A one-column |0> witness has S=1,H=3. The exact mode reduces the
    # one-qubit saved state on Aer and produces the physical Ritz value
    # without any reference.
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
        good.pencil.revise(overlap_filter="sampled_positive_subspace"),
    ):
        with pytest.raises(ValueError, match="settings|filter"):
            good.revise(pencil=changed).validate_plan(plan)
