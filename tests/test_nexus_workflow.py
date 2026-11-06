"""Public archived Nexus workflow integration with real SDK objects, no service."""

from collections import Counter
from types import SimpleNamespace as NS
from unittest.mock import create_autospec
from uuid import UUID

import pytest

qnx = pytest.importorskip("qnexus")
pytest.importorskip("pytket.extensions.qiskit")

from pytket.backends.backendresult import BackendResult  # noqa: E402
from pytket.utils.outcomearray import OutcomeArray  # noqa: E402
from qnexus.models.annotations import Annotations  # noqa: E402
from qnexus.models.job_status import JobStatus, JobStatusEnum  # noqa: E402
from qnexus.models.references import (  # noqa: E402
    CircuitRef, CompilationResultRef, CompileJobRef, ExecuteJobRef,
    ExecutionResultRef, IncompleteJobItemRef,
)

import nwqlib  # noqa: E402
from nwqlib.algorithms.expectation import ExpectationMethod  # noqa: E402
from nwqlib.backends.nexus import NexusBackend  # noqa: E402
from nwqlib._prepared_execution import Run, prepare_experiment, refresh_submissions, submit_detached  # noqa: E402
from nwqlib._remote_preparation import refresh_preparation  # noqa: E402
from nwqlib._run_archive import load as load_run  # noqa: E402
from test_expectation_current import problem  # noqa: E402
from test_nexus_sdk import sdk_connection as sdk_connection  # noqa: E402


def choice_for_nexus(*, two_items=False):
    terms = (("Z", 1.), ("X", 1.)) if two_items else (("I", 1.), ("Z", 1.))
    return nwqlib.plan(problem(terms), method=ExpectationMethod(), shots=4, seed=7)


@pytest.fixture
def nexus_service(sdk_connection, monkeypatch):
    """Use real Nexus reference types with an offline service that separately counts upload,
    compile, execute and download.
    """
    connection, project = sdk_connection
    service = NS(backend=connection, calls=Counter(), jobs={}, programs={}, compile_inputs={},
        compile_outputs={}, execution_inputs={}, result_programs={}, compile_status=JobStatusEnum.RUNNING,
        execution_status=JobStatusEnum.RUNNING, compile_lost_ack=False, cancel_error=None,
        failed_result=None, result_error=ConnectionError("second result read interrupted"))
    original_local = NexusBackend.prepare_local

    def local(self, *args, **kwargs):
        service.calls["local_preparation"] += 1
        return original_local(self, *args, **kwargs)

    def upload(circuit, project, name):
        service.calls["upload"] += 1
        ref = CircuitRef(id=UUID(int=10+service.calls["upload"]), project=project,
                         annotations=Annotations(name=name))
        service.programs[str(ref.id)] = (ref, circuit)
        return ref

    def compile_request(**kwargs):
        service.calls["compile"] += 1
        identity = UUID(int=100+service.calls["compile"])
        job = CompileJobRef(id=identity, project=project, annotations=Annotations(name=kwargs["name"],
            description=kwargs["description"]), backend_config_store=kwargs["backend_config"],
            last_status=JobStatusEnum.RUNNING, last_message="")
        source, = kwargs["programs"]
        compiled = CircuitRef(id=UUID(int=200+service.calls["compile"]), project=project,
                              annotations=Annotations(name="compiled"))
        service.jobs[str(job.id)] = job
        service.compile_inputs[str(job.id)] = source
        service.compile_outputs[str(job.id)] = compiled
        service.programs[str(compiled.id)] = (compiled, service.programs[str(source.id)][1])
        if service.compile_lost_ack:
            raise ConnectionError("compile accepted but acknowledgement lost")
        return job

    def execute_request(**kwargs):
        service.calls["execute"] += 1
        job = ExecuteJobRef(id=UUID(int=300+service.calls["execute"]), project=project,
            annotations=Annotations(name=kwargs["name"], description=kwargs["description"]),
            backend_config_store=kwargs["backend_config"], last_status=JobStatusEnum.RUNNING, last_message="")
        service.jobs[str(job.id)] = job
        service.execution_inputs[str(job.id)] = tuple(kwargs["programs"])
        assert kwargs["n_shots"] == [4]*len(kwargs["programs"])
        return job

    def get_job(*, id):
        service.calls["get_job"] += 1
        return service.jobs[id]

    def find_jobs(**kwargs):
        service.calls["find_jobs"] += 1
        assert kwargs["page_size"] == 2
        return iter(job for job in service.jobs.values() if job.annotations.name in kwargs["name_exact"])

    def status(job):
        service.calls["status"] += 1
        return JobStatus(service.compile_status if isinstance(job, CompileJobRef) else service.execution_status)

    def results(job, allow_incomplete=False):
        service.calls["result_inventory"] += 1
        assert allow_incomplete
        if isinstance(job, CompileJobRef):
            ref = CompilationResultRef(id=job.id, project=project, annotations=Annotations(name="compilation"),
                                       last_status_detail=JobStatus(service.compile_status))
            return [ref]
        refs = []
        for index, program in enumerate(service.execution_inputs[str(job.id)]):
            if service.execution_status != JobStatusEnum.COMPLETED:
                refs.append(IncompleteJobItemRef(project=project, annotations=Annotations(name="pending"),
                    program_id=program.id, last_status=service.execution_status, last_message="", job_type="execute"))
            else:
                ref = ExecutionResultRef(id=UUID(int=400+100*service.calls["execute"]+index), project=project,
                    annotations=Annotations(name="counts"), job_item_integer_id=40+index,
                    last_status_detail=JobStatus(JobStatusEnum.COMPLETED))
                service.result_programs[str(ref.id)] = program
                refs.append(ref)
        return refs

    def input_program(ref):
        service.calls[("input", str(ref.id))] += 1
        return service.result_programs[str(ref.id)]

    def download(ref):
        service.calls[("download", str(ref.id))] += 1
        if str(ref.id) == service.failed_result:
            raise service.result_error
        program = service.result_programs[str(ref.id)]
        _, circuit = service.programs[str(program.id)]
        return BackendResult(c_bits=circuit.bits, counts=Counter({
            OutcomeArray.from_readouts([[0]]): 3, OutcomeArray.from_readouts([[1]]): 1}))

    def cancel(job):
        service.calls["cancel"] += 1
        if service.cancel_error is not None:
            raise service.cancel_error

    monkeypatch.setattr(NexusBackend, "prepare_local", local)
    for owner, name, function in ((qnx.circuits, "upload", upload), (qnx, "start_compile_job", compile_request),
        (qnx, "start_execute_job", execute_request), (qnx.jobs, "get", get_job), (qnx.jobs, "get_all", find_jobs),
        (qnx.jobs, "status", status), (qnx.jobs, "results", results), (qnx.jobs, "cancel", cancel)):
        monkeypatch.setattr(owner, name, create_autospec(getattr(owner, name), side_effect=function))
    monkeypatch.setattr(CompilationResultRef, "get_input", lambda ref: service.compile_inputs[str(ref.id)])
    monkeypatch.setattr(CompilationResultRef, "get_output", lambda ref: service.compile_outputs[str(ref.id)])
    monkeypatch.setattr(ExecutionResultRef, "get_input", input_program)
    monkeypatch.setattr(ExecutionResultRef, "download_result", download)
    return service


def test_public_nexus_archive_resumes_original_compile_and_execution(tmp_path, monkeypatch, nexus_service):
    """Reopen at compile and execution frontiers, then require exactly one occurrence of each
    external operation.
    """
    from nwqlib._run_journal import LocalJournal
    service = nexus_service
    path = tmp_path / "original"
    with Run(choice_for_nexus(), backend=service.backend, directory=path) as run:
        run.resume()
        pending, = run._state["remote_preparations"].values()
        assert pending.status == "compile_pending"
        compile_id = pending.locator.job_id
        assert run.trace.preparations == 1 and run.trace.jobs == 0
    monkeypatch.setattr(ExpectationMethod, "plan", lambda *a, **k: pytest.fail("resume replanned"))
    with load_run(path, backend=service.backend) as run:
        with monkeypatch.context() as guard:
            guard.setattr(LocalJournal, "read_payload", lambda *a, **k: pytest.fail("queued compile loaded payload"))
            run.resume()
        pending, = run._state["remote_preparations"].values()
        assert pending.locator.job_id == compile_id and run.trace.preparations == 1
    service.compile_status = JobStatusEnum.COMPLETED
    with load_run(path, backend=service.backend) as run:
        run.resume()
        assert run.result is None and run.trace.jobs == 1
        assert run.prepared_artifacts[0].native_operations is None
        prepared_id = run.prepared_artifacts[0].content_id
        job = run.trace.submissions[0].locator
    service.execution_status = JobStatusEnum.COMPLETED
    with load_run(path, backend=service.backend) as run:
        result = run.wait(timeout=1, poll_interval=0)
        assert result.value == 1.5
        assert len(result.data.observations.chunks) == 1
        assert result.data.observations.chunks[0].returned_shots == 4
        assert result.data.trace.submissions[0].locator == job
        assert result.data.receipts[0].content_id == prepared_id
    before = service.calls.copy()
    with load_run(path, backend=service.backend) as run:
        assert run.wait().content_id == result.content_id
    assert service.calls == before
    assert service.calls["local_preparation"] == service.calls["upload"] == service.calls["compile"] == service.calls["execute"] == 1


def test_nexus_saves_the_synthesis_of_its_local_lowering_with_the_local_data(tmp_path, monkeypatch, nexus_service):
    """A remote preparation counts the exact synthesis of its dense unitary when its local data are saved.

    The work is saved in the transaction of the local data, before any provider call, and stays on the
    charge when the compilation completes. A reopened Run reports the same total as the live one.
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import state_input
    from test_dense_synthesis import _dense_synthesis_work

    service = nexus_service
    state = QuantumCircuit(2)
    state.append(UnitaryGate(random_unitary(4, seed=71).data), [0, 1])
    chosen = nwqlib.plan(Expectation(state=state_input(state), observable=ingest_pauli((("ZZ", 1.),), num_qubits=2)),
                         method=ExpectationMethod(), shots=4, seed=7)
    need = _dense_synthesis_work(2)
    saved, write = [], Run._write

    def recorded_write(self, records, **kwargs):
        work = sum(row.synthesis_work for kind, _, row in records if kind == "preparation")
        if work:
            saved.append((work, sorted({kind for kind, _, _ in records})))
        return write(self, records, **kwargs)

    monkeypatch.setattr(Run, "_write", recorded_write)
    path = tmp_path / "run"
    with Run(chosen, backend=service.backend, directory=path) as run:
        run.resume()
        pending, = run._state["remote_preparations"].values()
        assert pending.status == "compile_pending" and run.trace.synthesis_work == need
        assert saved[0] == (need, ["payload", "preparation", "remote_preparation"])
    service.compile_status = JobStatusEnum.COMPLETED
    with load_run(path, backend=service.backend) as run:
        assert run.trace.synthesis_work == need
        run.resume()
        assert run.prepared_artifacts and run.trace.synthesis_work == need
        live = run.trace.synthesis_work
    with load_run(path, backend=service.backend) as run:
        assert run.trace.synthesis_work == live == need


def test_a_budget_equal_to_the_plan_admits_a_row_whose_compilation_is_pending(tmp_path, nexus_service):
    """Two settings under max_total_circuits=2, reopened while the first setting's compilation is pending.

    That row is counted once and has no record yet. It needs no second preparation, so one preparation remains
    and the budget of two finishes the Plan, with one upload per setting.
    """
    from nwqlib.execution import ExecutionLimits

    service = nexus_service
    path = tmp_path / "run"
    with Run(choice_for_nexus(two_items=True), backend=service.backend, directory=path, progress=False,
             limits=ExecutionLimits(max_total_circuits=2)) as run:
        run.resume()
        pending, = run._state["remote_preparations"].values()
        assert pending.status == "compile_pending" and run.trace.preparations == 1
    service.compile_status = service.execution_status = JobStatusEnum.COMPLETED
    with load_run(path, backend=service.backend, progress=False) as run:
        result = run.wait(timeout=5, poll_interval=0)
    assert result.data.trace.preparations == 2 and service.calls["upload"] == 2


def test_nexus_later_read_failure_keeps_prior_item_and_association(tmp_path, nexus_service):
    service = nexus_service
    service.compile_status = service.execution_status = JobStatusEnum.COMPLETED
    path = tmp_path / "items"
    with Run(choice_for_nexus(two_items=True), backend=service.backend, directory=path) as run:
        handles = []
        for experiment in run.plan.experiments:
            pending = prepare_experiment(run.plan.resolve(experiment.name), run=run)
            handles.append(refresh_preparation(pending.preparation_id, run=run))
        submission = submit_detached(tuple(handles), run=run)
        service.failed_result = str(UUID(int=501))
        cause = OSError("underlying transport interruption")
        service.result_error.__cause__ = cause
        with pytest.raises(ConnectionError) as caught:
            refresh_submissions(run=run)
        assert caught.value is service.result_error and caught.value.__cause__ is cause
        first = run.completed_observation(submission.items[0].attempt)
        assert first.returned_shots == 4
        assert run.trace.submissions[0].items[0].provider_result_id == str(UUID(int=500))
        assert service.calls[("download", str(UUID(int=500)))] == 1
        assert not run.trace.submissions[0].results_consumed
    service.failed_result = None
    with load_run(path, backend=service.backend) as run:
        refresh_submissions(run=run)
        assert len(run._state["by_attempt"]) == 2 and run.trace.jobs == 1
        assert service.calls[("download", str(UUID(int=500)))] == 1
        assert service.calls[("download", str(UUID(int=501)))] == 2
        assert sum(event.shots for event in run.trace.events) == 8
    assert service.calls["execute"] == 1


def test_a_refused_compile_cancellation_is_sent_again_and_each_pass_reads_the_job(tmp_path, monkeypatch, nexus_service):
    """A refused request returns to unset, and the compile job is read in the same pass.

    The saved provider status and the error both stay. When the read fails as well, the request's error is
    raised, and the read's error stays on it and in the saved text. A delivered request is not repeated.
    """
    service = nexus_service
    service.cancel_error = refused = RuntimeError("provider unavailable")
    unread = ConnectionError("status read failed")

    def unreadable(self, locator, input_ref_json, load_data, *, run):
        raise unread

    with Run(choice_for_nexus(), backend=service.backend, directory=tmp_path / "run") as run:
        run.resume()
        (pending,) = run._state["remote_preparations"].values()
        job = pending.locator
        with pytest.raises(RuntimeError) as raised:
            run.cancel(reason="stop")
        assert raised.value is refused
        (pending,) = run._state["remote_preparations"].values()
        assert (pending.status, pending.provider_status, pending.cancel_requested, pending.locator) == (
            "compile_pending", "RUNNING", None, job)
        assert "provider unavailable" in pending.failure
        assert (service.calls["cancel"], service.calls["status"]) == (1, 1)
        with monkeypatch.context() as patched:
            patched.setattr(NexusBackend, "refresh_compile", unreadable)
            with pytest.raises(RuntimeError) as raised:
                run.cancel(reason="stop")
        assert raised.value is refused and raised.value.__context__ is unread
        assert any("ConnectionError: status read failed" in note for note in raised.value.__notes__)
        (pending,) = run._state["remote_preparations"].values()
        assert (pending.status, pending.locator) == ("compile_pending", job)
        assert "provider unavailable" in pending.failure and "status read failed" in pending.failure
        assert service.calls["cancel"] == 2
        service.cancel_error = None
        run.cancel(reason="stop")
        run.cancel(reason="stop")
        (pending,) = run._state["remote_preparations"].values()
        assert (pending.cancel_requested, pending.failure) == ("stop", None)
        assert service.calls["cancel"] == 3
    assert service.calls["upload"] == service.calls["compile"] == 1


@pytest.mark.parametrize(("ended", "status", "failure"), [
    (JobStatusEnum.COMPLETED, "completed", None),
    (JobStatusEnum.ERROR, "failed", "Nexus job ERROR"),
    (JobStatusEnum.CANCELLED, "cancelled", "Nexus job CANCELLED"),
])
def test_a_refused_cancellation_of_a_compile_job_that_ended_keeps_what_the_job_reached(tmp_path, nexus_service, ended, status, failure):
    """The provider refuses to cancel a compile job that already ended, and the error is raised once.

    The one read saves where the job ended before the error is raised, so a finished compilation keeps its
    preparation record, a provider's failure keeps its own text, and nothing is left pending, compiled or
    uploaded again, after reopening too.
    """
    service = nexus_service
    path = tmp_path / "run"
    with Run(choice_for_nexus(), backend=service.backend, directory=path) as run:
        run.resume()
        service.compile_status = ended
        service.cancel_error = refused = RuntimeError("job already finished")
        with pytest.raises(RuntimeError) as raised:
            run.cancel(reason="stop")
        assert raised.value is refused
        run.cancel(reason="stop")
        (pending,) = run._state["remote_preparations"].values()
        assert (pending.status, pending.failure) == (status, failure)
        assert len(run.prepared_artifacts) == int(status == "completed")
    with load_run(path, backend=service.backend) as run:
        (pending,) = run._state["remote_preparations"].values()
        assert (pending.status, pending.failure) == (status, failure)
        assert len(run.prepared_artifacts) == int(status == "completed")
        if status == "completed":
            with pytest.raises(RuntimeError, match="run cancelled"):
                run.wait(timeout=0)
        else:
            with pytest.raises(nwqlib.execution.RunFailed) as stopped:
                run.wait(timeout=0)
            assert (stopped.value.stage, stopped.value.status, stopped.value.failure) == ("prepare", status, failure)
    assert service.calls["cancel"] == service.calls["upload"] == service.calls["compile"] == 1
    assert service.calls["local_preparation"] == 1 and service.calls["execute"] == 0


def test_result_allowances_are_admitted_before_any_remote_request(tmp_path, nexus_service):
    """The download and decode allowances of a result follow from the requested shots, which no result exceeds.

    So they are checked when the circuit is prepared. A one-bit circuit of 10000 shots downloads up to
    10000 * (1 + 8) = 90000 bytes, and under max_data_bytes = 84000 the Run refuses during its local
    preparation, before any upload, compile or execute request.
    """
    from nwqlib.execution import ExecutionLimits
    service = nexus_service
    chosen = nwqlib.plan(problem((("I", 1.), ("Z", 1.))), method=ExpectationMethod(), shots=10000, seed=7)
    with Run(chosen, backend=service.backend, directory=tmp_path / "run", progress=False,
             limits=ExecutionLimits(max_data_bytes=84000)) as run:
        with pytest.raises(ValueError, match="90000 bytes exceeds this Run's max_data_bytes=84000"):
            run.resume()
    assert service.calls["upload"] == service.calls["compile"] == service.calls["execute"] == 0


def test_an_over_limit_pooled_query_is_refused_before_any_preparation_or_submission(tmp_path, nexus_service):
    """A sampled SPE query pools shots times its multiplicity into one H2 program. With seed 17 and
    1000 shots the third query requests 21000 shots, outside the inclusive per-program range
    [1, 10000]: the whole-plan admission refuses it before any local preparation, upload, compile or
    execution, so the two earlier legal queries spend nothing. At 100 shots every pooled query is
    inside the range and the same Plan is prepared.
    """
    import numpy as np
    from nwqlib import Eigenproblem
    from nwqlib._prepared_execution import prepare_static
    from nwqlib.algorithms.qpe import SPE
    service = nexus_service
    problem = Eigenproblem(A=np.diag([.25, -.5]))
    method = SPE(initial_state=[0, 1], overlap_lower_bound=1.)
    over = nwqlib.plan(problem, method=method, shots=1000, seed=17)
    with Run(over, backend=service.backend, directory=tmp_path / "over") as run:
        with pytest.raises(ValueError, match=r"requests 21000 shots, outside the inclusive per-program range \[1, 10000\]"):
            prepare_static(over, run=run)
    assert sum(service.calls.values()) == 0
    inside = nwqlib.plan(problem, method=method, shots=100, seed=17)
    with Run(inside, backend=service.backend, directory=tmp_path / "inside") as run:
        prepare_static(inside, run=run, settings="all")
    assert service.calls["local_preparation"] == len(inside.experiments) and service.calls["execute"] == 0
