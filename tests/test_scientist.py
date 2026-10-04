"""Public selection, saved PREP and caller-owned partial execution relations."""

from dataclasses import replace

import numpy as np
import pytest

import nwqlib
from nwqlib.algorithms import ExpectationMethod, FixedGCIM, Lanczos
from nwqlib.execution import ObservationView
from nwqlib.operators import ingest_pauli
from test_run_lifecycle import InjectedBackend, clear_backend as clear_backend


def forbidden(*args, **kwargs):
    raise AssertionError(
        "existing selected data triggered planning, normalization or native replay"
    )


def partial_result(plan, data):
    """Use the actual Method producer and the common attachment boundary."""
    return plan.method.analyze(plan, data, settings={})._attach(plan, data)


def test_solve_selected_plan_names_actual_overrides():
    overrides = {'seed': 7, 'shots': 20}
    chosen = nwqlib.plan(
        nwqlib.Expectation(state=[1, 0], observable=ingest_pauli((("Z", 1.0),), num_qubits=1)),
        method=ExpectationMethod(), seed=7)
    with pytest.raises(ValueError) as refused:
        nwqlib.solve(chosen, **overrides)
    message = str(refused.value)
    assert all(name in message for name in overrides)
    assert 'plan(...)' in message


def test_expectation_archive_preserves_actual_input_bindings_without_reselection(
    tmp_path, monkeypatch
):
    import nwqlib.algorithms.expectation as owner
    import nwqlib.blocks.selection as selection
    import nwqlib.problems.inputs as inputs
    from nwqlib._prepared_execution import Run

    selected = nwqlib.plan(
        nwqlib.Expectation(state=[1, 1j], observable=ingest_pauli((("Y", 1.0),), num_qubits=1)),
        method=ExpectationMethod(),
        seed=7,
    )
    run = Run(selected)
    partial = partial_result(selected, run.data)
    assert partial.value is None and partial.missing == ("Y",)
    with monkeypatch.context() as patch:
        patch.setattr(ExpectationMethod, "plan", forbidden)
        patch.setattr(owner, "select_preparation", forbidden)
        patch.setattr(selection, "select_preparation", forbidden)
        patch.setattr(inputs, "normalize_physical_vector_with_scale", forbidden)
        patch.setattr(inputs, "_digest", forbidden)
        folder = partial.save(tmp_path / "selected")
        restored = nwqlib.load_result(folder)
    np.testing.assert_array_equal(
        restored.plan.problem.state._direction, selected.problem.state._direction
    )
    assert not restored.plan.problem.state._direction.flags.writeable
    assert restored.plan.blocks[0]._payload._direction is restored.plan.problem.state._direction
    assert (
        restored.plan.construction == selected.construction
        and restored.plan.content_id == selected.content_id
    )
    assert restored.data.trace.events == ()

    assert nwqlib.load_result(folder).plan.content_id == selected.content_id
    run.close()


def test_supplied_preparation_archive_keeps_global_phase_and_original_parameters(
    tmp_path, monkeypatch
):
    """An analytic exp(i phi) Ry(theta)|0> state exposes lost supplied phase while later caller
    mutations stay isolated.
    """
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector
    from nwqlib.core import InputRef, Source
    from nwqlib.problems import bind_preparation_circuit
    from nwqlib._prepared_execution import Run

    phi, theta = 0.375, 0.8125
    circuit = QuantumCircuit(1, global_phase=phi)
    circuit.ry(theta, 0)
    observable = ingest_pauli((("Z", 1.0),), num_qubits=1)
    state = bind_preparation_circuit(
        circuit,
        basis=observable.manifest.basis,
        reference=InputRef(
            identity="supplied-phase-preparation",
            representation="circuit",
            source=Source(
                name="supplied",
                version="1",
                domain="one-qubit unitary",
                reference="exp(i phi) Ry(theta)",
            ),
        ),
    )
    selected = nwqlib.plan(
        nwqlib.Expectation(state=state, observable=observable), method=ExpectationMethod(), seed=7
    )
    from _profile_fixtures import stored_inputs
    from nwqlib.backends import InstructionSupport, assess

    _, _, profile, allocation, context, runtime, at = stored_inputs()
    (block,) = selected.blocks
    configuration = profile.configuration.revise(
        target=profile.configuration.target.revise(
            instructions=(
                InstructionSupport(implementation=block.record.implementation, max_qubits=1),
            )
        )
    )
    allocation = allocation.revise(configuration_id=configuration.content_id)
    (model,) = profile.models
    model = model.revise(
        domain=model.domain.revise(
            configuration_id=configuration.content_id, allocation_id=allocation.content_id
        )
    )
    profile = profile.revise(configuration=configuration, models=(model,))
    assessed = assess(
        selected,
        selected.resolve("expectation"),
        profile=profile,
        allocation=allocation,
        context=context,
        runtime=runtime,
        assessed_at=at,
    )
    support = next(
        item for item in assessed.capability.details if item.scope == "selected instruction support"
    )
    assert support.status == "feasible"
    (prediction,) = assessed.predictions
    assert prediction.seconds is None
    assert any("outside the model operation mix" in reason for reason in prediction.reasons)
    run = Run(selected)
    folder = partial_result(selected, run.data).save(tmp_path / "phase")
    circuit.global_phase = -0.25
    circuit.ry(0.5, 0)
    with monkeypatch.context() as patch:
        patch.setattr(ExpectationMethod, "plan", forbidden)
        patch.setattr("nwqlib.blocks.selection.select_preparation", forbidden)
        restored = nwqlib.load_result(folder)
    assert restored.plan.problem.state._native.global_phase == phi
    assert restored.plan.problem.state._native.data[0].operation.params == [theta]
    assert len(restored.plan.problem.state._native.data) == 1
    prepared = nwqlib.prepare(restored.plan)
    native = prepared.circuits[0]
    assert native.data[-1].operation.name == "save_expval"
    native.data.pop()  # Omit only the exact readout on this detached circuit view.
    # Physical amplitudes, including global phase, follow the supplied Ry action.
    expected = np.exp(1j * phi) * np.array([np.cos(theta / 2), np.sin(theta / 2)])
    np.testing.assert_allclose(
        Statevector.from_instruction(native).data, expected, atol=3e-14, rtol=0
    )
    result = nwqlib.submit(prepared).wait()
    assert result.value == pytest.approx(np.cos(theta), abs=3e-14, rel=0)
    assert restored.value is None and restored.data.trace.events == ()
    prepared.run.close()
    run.close()


def test_comparison_keeps_full_target_separate_from_method_trial_subspaces(monkeypatch):
    from nwqlib.algorithms.protocol import ApplicabilityError

    problem = nwqlib.Eigenproblem(A=ingest_pauli((("I", 1.0), ("X", 1.0)), num_qubits=1))
    methods = (
        Lanczos(initial_state=[1, 0], krylov_dimension=1),
        FixedGCIM(basis=([1, 1], [1, -1])),
    )
    monkeypatch.setattr("nwqlib.blocks.lowering._lower_qiskit", forbidden)
    compared = nwqlib.compare(problem, methods=methods, seed=7)
    assert compared.problem is problem and all(row.plan.problem is problem for row in compared.rows)
    assert all(row.plan.output == problem.default_output() for row in compared.rows)
    assert problem.subspace is None
    restricted = problem.revise(subspace=compared.rows[0].plan.reconstruction.subspace)
    projected = nwqlib.compare(restricted, methods=methods, seed=7)
    assert projected.rows[0].plan.problem is restricted
    assert projected.rows[1].plan is None and "subspace" in projected.rows[1].reason
    with pytest.raises(ApplicabilityError, match="subspace"):
        nwqlib.prepare(projected.rows[1])
    assert compared.rows[1].plan.problem.subspace is None


def test_failed_public_submission_keeps_partial_contributions_and_original_exposure(
    tmp_path, monkeypatch
):
    """Fail the second setting and preserve the first contribution alongside all charged,
    uncertain and forecast evidence.
    """
    from nwqlib.backends import align_telemetry
    from nwqlib.scientist import ComparisonRow
    from _profile_fixtures import original_forecast

    problem = nwqlib.Expectation(
        state=[1, 0], observable=ingest_pauli((("Z", 1.0), ("X", 1.0)), num_qubits=1)
    )
    selected = nwqlib.plan(problem, method=ExpectationMethod(), shots=4, seed=7)
    forecast, allocation = original_forecast(selected)
    row = ComparisonRow(selected.method, selected, estimate=forecast, allocation=allocation)
    InjectedBackend.supports_synchronous = True
    prepared = nwqlib.prepare(row, backend=InjectedBackend(), directory=tmp_path / "run")
    run = prepared.run
    assert len(prepared.circuits) == run.trace.preparations == 1
    initial = partial_result(selected, run.data)
    assert initial.value is None and initial.data.receipts == run.prepared_artifacts
    assert not initial.data.trace.events
    assert (
        nwqlib.load_result(initial.save(tmp_path / "unused")).data.receipts == initial.data.receipts
    )
    calls = []
    failure = RuntimeError("second setting failed after first immutable chunk")
    original_submit = InjectedBackend.submit

    def acquire(self, native, *, submission_id, run):
        calls.append(native)
        assert run.trace.events[-1].assessment_id in {a.content_id for a in forecast.assessments}
        if len(calls) == 2:
            raise failure
        return original_submit(self, native, submission_id=submission_id, run=run)

    monkeypatch.setattr(InjectedBackend, "submit", acquire)
    with pytest.raises(RuntimeError) as raised:
        nwqlib.submit(prepared)
    assert raised.value is failure and prepared.run is run
    assert [event.status for event in run.trace.events] == ["completed", "uncertain"]
    assert sum(event.shots for event in run.trace.events) == 8 and len(run.observations.chunks) == 1
    assert len(run.prepared_artifacts) == 2 and run.result is None
    snapshot, rng = run.data, run.rng.snapshot()
    monkeypatch.setattr(InjectedBackend, "submit", forbidden)
    monkeypatch.setattr(InjectedBackend, "refresh", forbidden)
    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    partial = partial_result(selected, snapshot)
    assert partial.value is None and partial.missing
    assert partial.contribution_ids == (snapshot.observations.chunks[0].content_id,)
    assert partial.plan.problem is problem and partial.data is snapshot
    with monkeypatch.context() as patch:
        patch.setattr(ExpectationMethod, "analyze", forbidden)
        report = partial.report()
    assert "unavailable" in report["summary"] and "Partial data" in report["summary"]
    assert "1/1 chunks used" in report["summary"] and "2 attempts, 1 uncertain" in report["summary"]
    assert ObservationView.model_validate(report["observations"]) == snapshot.observations
    assert report["receipts"] == [receipt.model_dump(mode="json") for receipt in snapshot.receipts]
    assert report["forecast"] == forecast.model_dump(mode="json")
    assert report["allocation"] == allocation.model_dump(mode="json")
    assert [event["status"] for event in report["trace"]["events"]] == ["completed", "uncertain"]
    ledger = align_telemetry(partial)
    assert len(ledger.rows) == 2 and ledger.rows[0].contribution_ids == partial.contribution_ids
    assert not ledger.rows[1].contribution_ids and ledger.rows[1].event.failure.startswith(
        "RuntimeError:"
    )
    folder = partial.save(tmp_path / "partial")
    restored = nwqlib.load_result(folder)
    assert restored.data.trace == snapshot.trace
    assert restored.data.forecast.model_dump(mode="json") == forecast.model_dump(mode="json")
    assert (
        restored.data.allocation == allocation
        and restored.contribution_ids == partial.contribution_ids
    )
    again = restored.analyze()
    assert again.value is None and again.data is restored.data
    assert len(calls) == 2 and run.rng.snapshot() == rng
    assert nwqlib.load_result(folder).data.trace.events == snapshot.trace.events
    assert run.trace.events == snapshot.trace.events
    # Collected data and charged exposure are different populations: an empty
    # reduction keeps both attempted settings and all receipts in its snapshot.
    empty_data = replace(snapshot, observations=ObservationView())
    empty = partial_result(selected, empty_data)
    assert empty.contribution_ids == () and empty.data.trace.events == snapshot.trace.events
    assert (
        nwqlib.load_result(empty.save(tmp_path / "empty")).data.trace.events
        == snapshot.trace.events
    )
    run.close()
