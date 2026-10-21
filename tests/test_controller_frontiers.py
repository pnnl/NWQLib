"""Original adaptive queries through one injected durable backend; no simulation."""

import json
from types import SimpleNamespace
from typing import ClassVar, Literal

import pytest
import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.lanczos import Lanczos, SensitivitySampling
from nwqlib.algorithms.qpe import RWPE, numerical
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
