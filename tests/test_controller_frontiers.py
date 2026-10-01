"""Original adaptive queries through one injected durable backend; no simulation."""

import json
from types import SimpleNamespace
from typing import ClassVar, Literal

import pytest
import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.lanczos import Lanczos, SensitivitySampling
from nwqlib.algorithms.qpe import QCELS, RWPE, numerical
from nwqlib.backends.connection import BackendRefresh, NativePreparation, circuit_layout
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, JobLocator
from nwqlib.problems import Eigenproblem


class QueuedBackend(Record):
    kind: Literal["queued_fixture"] = "queued_fixture"
    supports_synchronous: ClassVar[bool] = False
    requires_prepared_payload: ClassVar[bool] = True
    data: ClassVar[object] = None

    def target_for(self, observation):
        if observation.kind != "counts":
            raise ValueError("fixture supports actual counts only")
        return SimpleNamespace(readouts=("counts",))

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        self.data.preparations.append(runtime.seed)
        source = Source(
            name="supplied queue", version="1", domain="injected integer populations", reference="test:queue"
        )
        native = SimpleNamespace(
            circuit=circuit, shots=observation.shots, seed=runtime.seed, width=circuit.num_clbits
        )
        payload = json.dumps(dict(shots=native.shots, seed=native.seed, width=native.width)).encode()
        return NativePreparation(
            native=native,
            payload=payload,
            payload_format="fixture/1",
            target=source,
            compiler=source,
            native_basis=("supplied",),
            environment=(),
            quantum_layout=circuit_layout(circuit, circuit.qregs),
            classical_layout=circuit_layout(circuit, circuit.cregs),
            logical_to_native=tuple(range(circuit.num_qubits)),
            operations=len(circuit.data),
            population="unconditional",
            counts_sampling=CountsSampling(kind="fresh"),
            transformation="actual selected circuit with injected fresh counts",
        )

    def restore_native(self, receipt, payload=None, *, run):
        return SimpleNamespace(
            circuit=None,
            shots=receipt.observation.shots,
            seed=receipt.runtime.seed,
            width=sum(len(register.bits) for register in receipt.classical_layout),
        )

    def admit_batch(self, natives):
        pass

    def launch(self, natives, *, submission_id, run):
        self.data.launches.append((submission_id, tuple((n.shots, n.seed) for n in natives)))
        if self.data.lose_ack:
            raise ConnectionError("lost execution acknowledgement")
        return JobLocator(provider=self.kind, job_id=submission_id)

    def reconcile(self, *args, **kwargs):
        return None

    def refresh(self, locator, natives, *, run):
        self.data.refreshes.append(locator.job_id)
        if self.data.fail_fetch:
            raise ConnectionError("supplied download interruption")
        if self.data.states.get(locator.job_id) != "COMPLETED":
            return BackendRefresh("acknowledged")
        results = tuple(
            (
                str(i),
                SimpleNamespace(
                    raw_output={
                        "counts": {bits: count for bits, count in {
                            "0" * native.width: native.shots * 3 // 4,
                            "0" * (native.width - 1) + "1": native.shots - native.shots * 3 // 4,
                        }.items() if count}
                    },
                    metadata={"native_job_id": locator.job_id},
                ),
            )
            for i, native in enumerate(natives)
        )
        return BackendRefresh("completed", results=results)

    def cancel(self, locator, *, run):
        self.data.cancellations.append(locator.job_id)


@pytest.fixture
def transport(monkeypatch):
    data = SimpleNamespace(
        backend=QueuedBackend(),
        states={},
        launches=[],
        refreshes=[],
        preparations=[],
        fail_fetch=False,
        lose_ack=False,
        cancellations=[],
        compile_states={},
        compile_calls=[],
        compile_refreshes=[],
        native_by_snapshot={},
    )
    monkeypatch.setattr(QueuedBackend, "data", data)
    return data


def selected(family, *, steps=2, dimension=1):
    if family == "rwpe":
        return nwqlib.plan(
            Eigenproblem(A=[[0.4, 0.0], [0.0, -0.7]]),
            method=RWPE(
                initial_state=[1.0, 0.0],
                max_steps=steps,
                tau=0.2,
            ),
            shots=1,
            seed=7,
        )
    method = Lanczos(
        initial_state=[1.0, 0.0],
        krylov_dimension=dimension,
        sampling=SensitivitySampling(total_shots=40 if dimension == 1 else 101, pilot_fraction=0.25),
    )
    return nwqlib.plan(Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]]), method=method, seed=41)


def complete_jobs(run, transport):
    for event in run.trace.events:
        transport.states[event.submission] = "COMPLETED"


def forbidden(*args, **kwargs):
    raise AssertionError("resume replayed original preparation or numerical setup")


def finish(run, transport):
    for _ in range(8):
        if run.result is not None:
            return run.result
        complete_jobs(run, transport)
        run.resume()
    pytest.fail("tiny controller did not finish within its selected acquisitions")


def test_lanczos_floor_uses_returned_pilot_population_without_refunding_exposure(
    tmp_path, monkeypatch, transport
):
    import numpy as np
    from nwqlib.algorithms.lanczos import workflow

    original_refresh = QueuedBackend.refresh

    def partial_pilot(self, locator, natives, *, run):
        response = original_refresh(self, locator, natives, run=run)
        if response.status == "completed" and run.checkpoint_state["stage"] == 0:
            for _, result in response.results:
                result.raw_output["counts"] = {
                    key: count // 2 for key, count in result.raw_output["counts"].items()
                }
        return response

    monkeypatch.setattr(QueuedBackend, "refresh", partial_pilot)
    monkeypatch.setattr(workflow, "_sensitivity_weights", lambda *args: (
        np.array([100., 0.]), {"policy": "discriminating allocation witness"}))
    chosen = selected("lanczos", dimension=2)
    with Run(chosen, backend=transport.backend, directory=tmp_path / "partial-pilot") as run:
        result = finish(run, transport)
        pilot = [chunk for chunk in run.observations.chunks
                 if next(b.value for b in chunk.bindings if b.parameter == "stage") == 0]
        main = [chunk for chunk in run.observations.chunks if chunk not in pilot]
        returned = [chunk.returned_shots for chunk in pilot]
        checkpoint = run.checkpoint_state
        assert returned == [6, 5]
        assert checkpoint["allocation_evidence"]["pilot_floor"] == returned
        assert checkpoint["main_allocations"][1:] == [5]
        assert sum(event.shots for event in run.trace.events) == 101
        assert result.contribution_ids == tuple(chunk.content_id for chunk in main)


@pytest.mark.parametrize("family", ["rwpe", "lanczos"])
def test_pending_reopen_keeps_original_rng_queries_and_once_only_analysis(
    tmp_path, monkeypatch, transport, family
):
    plan = selected(family)
    path = tmp_path / family
    run = Run(plan, backend=transport.backend, directory=path)
    run.resume()
    assert run.result is None and len(transport.launches) == len(transport.preparations) == 1
    original_rng, original_state = run.rng.snapshot(), run.checkpoint_state
    with monkeypatch.context() as patch:
        patch.setattr(numerical, "nominal_eigensystem", forbidden)
        patch.setattr(numerical, "rwpe_update", forbidden)
        run.resume()
    assert run.rng.snapshot() == original_rng and run.checkpoint_state == original_state
    transport.fail_fetch = True
    with pytest.raises(ConnectionError, match="download interruption"):
        run.resume()
    assert run.checkpoint_state == original_state and len(transport.launches) == 1
    run.close()
    transport.fail_fetch = False
    monkeypatch.setattr(type(plan.method), "plan", forbidden)
    with nwqlib.load_run(path, backend=transport.backend) as resumed:
        assert resumed.rng.snapshot() == original_rng
        result = finish(resumed, transport)
        assert len(resumed.trace.events) == 2 and resumed.trace.jobs == 2
        assert sum(event.shots for event in resumed.trace.events) == (2 if family == "rwpe" else 40)
        assert len(set(receipt.runtime.seed for receipt in resumed.prepared_artifacts)) == 2
        before = (list(transport.launches), list(transport.preparations), list(transport.refreshes))
        assert resumed.resume() is resumed and resumed.result is result
        assert before == (transport.launches, transport.preparations, transport.refreshes)
        assert len(resumed.observations.chunks) == 2


@pytest.mark.parametrize("family", ["rwpe", "lanczos"])
def test_lost_submission_ack_keeps_uncertainty_without_replay(tmp_path, transport, family):
    transport.lose_ack = True
    path = tmp_path / family
    with Run(selected(family), backend=transport.backend, directory=path) as run:
        with pytest.raises(ConnectionError, match="lost execution acknowledgement"):
            run.resume()
        assert run.exposure["uncertain"]["circuits"] == 1
        checkpoint = run.checkpoint_state
    with nwqlib.load_run(path, backend=transport.backend) as run:
        run.resume()
        assert run.checkpoint_state == checkpoint and run.result is None
        assert run.trace.events[0].status == "uncertain"
        assert len(transport.launches) == len(transport.preparations) == 1 and not transport.refreshes


def remote_compile(monkeypatch, transport, *, crash_after_ack=False):
    native_prepare = QueuedBackend.prepare

    def local(self, circuit, **kwargs):
        snapshot = kwargs["snapshot"]
        transport.native_by_snapshot[snapshot] = native_prepare(self, circuit, **kwargs)
        return SimpleNamespace(model_dump=lambda **kw: dict(snapshot=snapshot))

    def upload(self, data, **kwargs):
        saved = data.model_dump()
        kwargs["acknowledge"](saved["snapshot"])
        return json.dumps(saved)

    def compile(self, reference, **kwargs):
        snapshot = json.loads(reference)["snapshot"]
        transport.compile_calls.append(snapshot)
        locator = JobLocator(provider=self.kind, job_id=snapshot)
        kwargs["acknowledge"](locator)
        if crash_after_ack:
            raise ConnectionError("crash after compile ACK")
        return locator

    def refresh(self, locator, reference, load_data, **kwargs):
        transport.compile_refreshes.append(locator.job_id)
        if transport.compile_states.get(locator.job_id) != "COMPLETED":
            return BackendRefresh("acknowledged")
        return transport.native_by_snapshot[locator.job_id]

    monkeypatch.setattr(QueuedBackend, "prepare_local", local, raising=False)
    monkeypatch.setattr(QueuedBackend, "upload", upload, raising=False)
    monkeypatch.setattr(QueuedBackend, "start_compile", compile, raising=False)
    monkeypatch.setattr(QueuedBackend, "refresh_compile", refresh, raising=False)


@pytest.mark.parametrize("family", ["rwpe", "lanczos"])
def test_compile_ack_crash_reopens_exact_preparation_and_checkpoint(tmp_path, monkeypatch, transport, family):
    remote_compile(monkeypatch, transport, crash_after_ack=True)
    path = tmp_path / family
    with Run(selected(family, steps=1), backend=transport.backend, directory=path) as run:
        with pytest.raises(ConnectionError, match="compile ACK"):
            run.resume()
        checkpoint = run.checkpoint_state
        sequence = (
            checkpoint["pending"]["checkpoint_sequence"]
            if family == "rwpe"
            else checkpoint["pending"]["sequence"]
        )
        preparation = run.preparation_for_checkpoint(sequence)
        assert preparation is not None and run.trace.preparations == 1 and not run.trace.events
    with nwqlib.load_run(path, backend=transport.backend) as run:
        run.resume()
        assert run.checkpoint_state == checkpoint and run.preparation_for_checkpoint(sequence) == preparation
        transport.compile_states[transport.compile_calls[0]] = "COMPLETED"
        run.resume()
        assert len(transport.compile_calls) == len(transport.preparations) == len(transport.launches) == 1
        assert run.trace.events[0].checkpoint_sequence == sequence


def test_cancellation_reaches_original_job_before_new_admission(tmp_path, transport):
    with Run(selected("lanczos"), backend=transport.backend, directory=tmp_path / "cancel") as run:
        run.resume()
        job = run.trace.submissions[0].locator.job_id
        checkpoint = run.checkpoint_state
        run.cancel()
        run.resume()
        run.resume()
        assert transport.cancellations == [job]
        assert len(transport.launches) == len(transport.preparations) == 1
        assert run.checkpoint_state == checkpoint and run.exposure["reserved"]["jobs"] == 1


def test_common_mixed_compile_frontier_refreshes_each_original_once(tmp_path, monkeypatch, transport):
    from collections import Counter
    from nwqlib._prepared_execution import prepare_experiment, submit_detached
    from nwqlib._remote_preparation import refresh_preparation
    from nwqlib.execution import PendingPreparation
    from nwqlib.algorithms.expectation import ExpectationMethod
    from test_expectation_current import problem

    remote_compile(monkeypatch, transport)
    plan = nwqlib.plan(
        problem((("Z", 1.0), ("X", 1.0), ("Y", 1.0))), method=ExpectationMethod(), shots=4, seed=7
    )
    with Run(plan, backend=transport.backend, directory=tmp_path / "mixed") as run:
        pending = tuple(prepare_experiment(plan.resolve(item.name), run=run) for item in plan.experiments)
        identities = list(transport.compile_calls)
        assert len(identities) == 3 and not transport.launches
        transport.compile_states[identities[0]] = "COMPLETED"
        transport.compile_refreshes.clear()
        refreshed = tuple(refresh_preparation(item.preparation_id, run=run) for item in pending)
        assert Counter(transport.compile_refreshes) == Counter(identities)
        assert not isinstance(refreshed[0], PendingPreparation) and all(
            isinstance(item, PendingPreparation) for item in refreshed[1:]
        )
        submit_detached((refreshed[0],), run=run)
        assert len(transport.preparations) == 3 and len(transport.launches) == 1


@pytest.mark.parametrize("family", ["fixed", "rwpe", "adapt"])
def test_acknowledged_host_preparation_reopens_original_binding_without_new_preparation(
    tmp_path, monkeypatch, family
):
    """Interrupt immediately after durable host preparation, then forbid replanning or preparing
    it again.
    """
    from nwqlib.algorithms.gcim import ADAPT
    from nwqlib.blocks.kernels import BoundKernel
    from nwqlib.operators import ingest_pauli
    import nwqlib._prepared_execution as execution

    problem = Eigenproblem(A=[[0.4, 0.0], [0.0, -0.7]])
    method = (
        ADAPT(initial_state=[1.0, 1.0], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
        if family == "adapt"
        else RWPE(initial_state=[1.0, 0.0], max_steps=1, tau=0.2)
        if family == "rwpe"
        else QCELS(initial_state=[1.0, 0.0], num_times=3, max_time=0.3, grid_size=32, tau=0.2)
    )
    plan = nwqlib.plan(problem, method=method, execution="classical", seed=7)
    path = tmp_path / family
    write = Run._write

    def acknowledge(run, records=(), **kwargs):
        encoded = write(run, records, **kwargs)
        if any(
            kind == "prepared" and getattr(value, "execution", None) == "host_kernel"
            for kind, _, value in records
        ):
            raise RuntimeError("host preparation acknowledged before invocation")
        return encoded

    with monkeypatch.context() as patch:
        patch.setattr(Run, "_write", acknowledge)
        with Run(plan, directory=path) as run:
            with pytest.raises(RuntimeError, match="acknowledged before invocation"):
                run.resume()
            assert run.trace.host_invocations == 0 and not run.trace.events
    monkeypatch.setattr(type(method), "plan", forbidden)
    monkeypatch.setattr(execution, "_prepare_host", forbidden)
    invoke = BoundKernel._invoke
    calls = []
    with nwqlib.load_run(path, backend=None) as run:
        (receipt,) = run.prepared_artifacts

        def one(native):
            calls.append(native.record.content_id)
            result = invoke(native)
            run.cancel()
            return result

        monkeypatch.setattr(BoundKernel, "_invoke", one)
        run.resume()
        assert run.trace.preparations == run.trace.host_invocations == 1 and run.trace.jobs == 0
        assert calls == [receipt.selected_kernel_id]
        (chunk,) = run.observations.chunks
        assert (
            chunk.prepared_id == receipt.content_id and chunk.selected_kernel_id == receipt.selected_kernel_id
        )
        run.resume()
        assert calls == [receipt.selected_kernel_id]
