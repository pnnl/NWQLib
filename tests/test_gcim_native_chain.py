"""Authorized native witness: 12 two-qubit exact evaluations and 384 sampled shots."""

import numpy as np
import pytest
from qiskit.quantum_info import Statevector

from test_gcim_production import plan_for, matrix
from nwqlib.backends import inspection, qiskit_aer as aer
from nwqlib.blocks import lowering
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.gcim.adapt_records import screen_labels
from nwqlib.operators.inputs import OperatorInput
from nwqlib.resources import estimate


def test_tiny_planned_hadamard_chain_and_definition_reuse(tmp_path, monkeypatch):
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.algorithms.gcim.fixed_basis import FixedGCIM

    native_growth = []
    original_native = aer._AerPreparation.prepare

    def observe_native(owner, circuit, decompose):
        before = len(owner._blocks)
        result = original_native(owner, circuit, decompose)
        native_growth.append(len(owner._blocks) - before)
        return result

    monkeypatch.setattr(aer._AerPreparation, "prepare", observe_native)

    def forbidden(*_args, **_kwargs):
        pytest.fail("fixed GCiM performed hidden full-state/reference/inventory work")

    monkeypatch.setattr(OperatorInput, "matvec", forbidden)
    monkeypatch.setattr(Statevector, "from_instruction", forbidden)
    monkeypatch.setattr(inspection, "inspect_circuit_resources", forbidden)
    expected_s = np.array([[1.0, 0.5 + 0.5j], [0.5 - 0.5j, 1.0]])
    expected_h = np.array([[0.0, 0.5 - 0.5j], [0.5 + 0.5j, 0.0]])
    total_evaluations = total_shots = 0
    for shift, shots in ((0.0, 0), (1.0, 0), (1.0, 64)):
        start_n = len(native_growth)
        # Planning/fold must remain metadata-only even in a process with an SDK.
        with monkeypatch.context() as planning:
            planning.setattr(lowering, "lower_qiskit", forbidden)
            planning.setattr(lowering, "_lower_qiskit", forbidden)
            plan = plan_for(A=np.diag([1 + shift, -1 + shift]), shots=shots or None)
            workload = estimate(plan.construction)
            assert workload.construction_id == plan.construction.content_id
        location = tmp_path / f"plan-{shift}-{shots}"
        location.mkdir()
        saved = plan.method.save_archive(plan, ArchiveFiles(location, 4_000_000))
        plan = FixedGCIM.load_archive(saved, ArchiveFiles(location, 4_000_000))
        run = Run(plan)
        result = run.wait(timeout=5, poll_interval=0)
        receipt_ids = [receipt.content_id for receipt in run.data.receipts]
        # Exact: one reduction per pair, (M - D) + D = 3 for b = 2, L = 1.
        # Sampled: b**2 G = 4 counts settings for G = 1 QWC group.
        acquisitions, natives = (4, 9) if shots else (3, 6)
        assert len(set(receipt_ids)) == acquisitions
        assert (
            sum(native_growth[start_n:]) == natives
        )  # Sampled: ancilla H and X, X/Y readouts, two controlled and two plain PREP, one system group readout. Exact: ancilla H and X, two controlled and two plain PREP.
        assert len(run.data.trace.events) == len(run.data.observations.chunks) == acquisitions
        total_evaluations += sum(e.evaluations for e in run.data.trace.events)
        total_shots += sum(e.shots for e in run.data.trace.events)
        with monkeypatch.context() as analysis:
            for owner, name in (
                (lowering, "lower_qiskit"),
                (lowering, "_lower_qiskit"),
                (aer, "_prepare_aer_execution"),
                (aer, "_submit_aer_execution"),
            ):
                analysis.setattr(owner, name, forbidden)
            result = result.analyze(overlap_cutoff=1e-10)
            result.model_dump_json()
        assert result.missing == ()
        if shots:
            assert all(
                c.returned_shots == 64 and c.trajectories is None
                for c in run.data.observations.chunks
            )
            assert all(e.sampled_shots == 64 for e in result.estimates)
            assert result.pencil.overlap_filter == "sampled_positive_subspace"
            # Values are the actual sufficient statistics, without a reference
            # state or matrix check altering the selected sampled pencil. Each
            # is a weighted group mean of Y = Z, so it lies in [-1, 1] here.
            assert all(-1.0 <= e.value <= 1.0 for e in result.estimates)
        else:
            assert all(
                c.trajectories == 1 and c.returned_shots is None
                for c in run.data.observations.chunks
            )
            np.testing.assert_allclose(
                matrix(result.pencil.overlap), expected_s, rtol=0, atol=3e-13
            )
            np.testing.assert_allclose(
                matrix(result.pencil.hamiltonian),
                expected_h + shift * expected_s,
                rtol=0,
                atol=3e-13,
            )
            np.testing.assert_allclose(
                result.pencil.eigenvalues, [-1 + shift, 1 + shift], rtol=0, atol=1e-12
            )
    # Sampled: four settings of 64 shots.
    assert total_evaluations == 3 + 3 and total_shots == 256


def test_tiny_actual_adapt_consumer_keeps_native_generator_basis_and_sampled_provenance():
    from qiskit import QuantumCircuit
    from test_adapt_primary import plan_for as adapt_plan
    from test_gcim_native_preparations import reference_kwargs

    reference = QuantumCircuit(1)
    reference.h(0)
    for shots in (None, 32):
        plan = adapt_plan(
            shots=shots,
            initial_state=reference_kwargs(reference)["reference"],
            theta=np.pi / 4,
            max_iterations=1,
            gradient_norm_floor=0.0,
            pool_rows=((("Y", 1j),), (("Y", 1j), ("I", 1j)), (("X", 1j),)),
        )
        result = Run(plan).wait(timeout=5, poll_interval=0)
        assert result.selected == (0,) and result.theta == (np.pi / 4,)
        # [Z,iY]=[Z,iY+iI]=2X and [Z,iX]=-2Y. Shared X is measured once;
        # the disjoint Y setting remains, independent of the pool size three.
        # Only a sampled Plan forms readout groups, as indices into the
        # sorted commutator labels.
        assert screen_labels(plan.reconstruction.pool, {}) == ("X", "Y")
        assert plan.reconstruction.groups == (((0,), (1,)) if shots else ())
        gradients = dict(result.history[0].gradients)
        assert gradients[0] == gradients[1] == pytest.approx(2, rel=0, abs=1e-12)
        assert -2 <= gradients[2] <= 2
        assert len(result.pencil.hamiltonian) == 2
        pairs = [c for c in result.data.observations.chunks if c.experiment == "pair"]
        screens = [c for c in result.data.observations.chunks if c.experiment == "screen"]
        # Exact: the diagonals come from the two shared full-chain queries, each
        # with a diagonal-reduction point and a screen point, and one pair remains.
        assert len(pairs) == (1 if shots is None else 6) and len(screens) == (4 if shots is None else 2)
        if shots is None:
            assert result.eigenvalue == pytest.approx(-1.0, rel=0.0, abs=1e-12)
        else:
            assert all(c.returned_shots == 32 for c in screens)
            assert all(sum(v.count for v in c.values) == 32 for c in pairs)
            assert all(c.returned_shots == 32 for c in pairs)
            assert set(i for row in result.history for i in row.contribution_ids) <= set(
                result.contribution_ids
            )
