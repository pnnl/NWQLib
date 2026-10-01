"""Offline qualification with real Nexus and pytket public objects, no service."""

from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from math import pi
from types import SimpleNamespace as NS
from unittest.mock import create_autospec
from uuid import UUID

import numpy as np
import pytest

qnx = pytest.importorskip("qnexus")
pytest.importorskip("pytket.extensions.qiskit")

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister  # noqa: E402
from pytket import Circuit  # noqa: E402
from pytket.backends.backendresult import BackendResult  # noqa: E402
from pytket.circuit import Bit, OpType  # noqa: E402
from pytket.utils.outcomearray import OutcomeArray  # noqa: E402
from qnexus.models.annotations import Annotations  # noqa: E402
from qnexus.models.job_status import JobStatus, JobStatusEnum  # noqa: E402
from qnexus.models.references import (  # noqa: E402
    CircuitRef, CompilationResultRef, CompileJobRef, ExecuteJobRef,
    ExecutionResultRef, IncompleteJobItemRef, ProjectRef,
)

from nwqlib.backends.nexus import NexusBackend  # noqa: E402
from nwqlib.core.planning import ObservationSpec  # noqa: E402
from nwqlib.execution import SubmissionItem  # noqa: E402
from test_nexus_backend import IO  # noqa: E402


@pytest.fixture
def sdk_connection(monkeypatch):
    import httpx
    # Public-boundary test double: any unmocked outbound SDK operation is a failure.
    monkeypatch.setattr(httpx.Client, "send", lambda *a, **k: pytest.fail("unexpected live HTTP request"))
    monkeypatch.setattr(qnx.client.utils, "is_managed_token_environment", lambda: False)
    monkeypatch.setattr(qnx.client, "get_nexus_client",
        lambda: NS(auth=NS(cookies=("myqos_oat",))))
    project = ProjectRef(id=UUID(int=1), annotations=Annotations(name="offline-project"),
                         contents_modified=datetime(2026, 9, 10, tzinfo=timezone.utc))
    connection = NexusBackend(project=str(project.id), device="H2-1", target_region="us",
        max_input_bytes=100_000, max_cost_hqc=12.5)
    monkeypatch.setattr(qnx.projects, "get", create_autospec(qnx.projects.get, return_value=project))
    return connection, project


def test_installed_sdk_is_the_offline_qualified_release():
    # docs/nexus.md and the Nexus Offline Qualification workflow record this release.
    assert version("qnexus") == "0.49.0"


def local_data(connection):
    circuit = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, "z"), ClassicalRegister(2, "a"))
    circuit.global_phase = pi / 2
    circuit.x(1)
    circuit.cx(1, 0)
    circuit.rz(pi / 2, 1)
    circuit.measure(1, 0)
    circuit.measure(0, 2)
    data = connection.prepare_local(circuit, observation=ObservationSpec(kind="counts", shots=3),
        runtime=NS(seed=7), position=len(circuit.data), source_definitions=(),
        run=IO(), snapshot="offline-snapshot")
    return data


def test_real_converter_payload_bit_order_and_reference_round_trip(sdk_connection):
    connection, project = sdk_connection
    data = local_data(connection)
    restored_data = connection.restore_preparation(data.model_dump_json().encode(), run=IO())
    assert restored_data == data
    native = Circuit.from_dict(json.loads(data.circuit_json))
    # Rz(theta) = exp(-i*theta/2) U(0,0,theta); lowering pi/2 Rz
    # subtracts pi/4 from the original pi/2 global phase.
    assert float(native.phase) == 0.25  # pytket phase is in half-turns
    commands = native.get_commands()
    cx, = [command for command in commands if command.op.type == OpType.CX]
    assert [qubit.index[0] for qubit in cx.qubits] == [1, 0]
    phase_gate, = [command for command in commands
                  if command.op.type == OpType.U3 and float(command.op.params[0]) == 0]
    assert tuple(float(p) for p in phase_gate.op.params) == (0., 0., 0.5)
    assert phase_gate.qubits[0].index == [1]
    measured = {(command.qubits[0].index[0], command.bits[0].reg_name, command.bits[0].index[0])
                for command in commands if command.op.type == OpType.Measure}
    assert measured == {(1, "z", 0), (0, "a", 1)}
    assert data.bits == (("z", 0), ("a", 0), ("a", 1))
    ref = CircuitRef(id=UUID(int=2), project=project, annotations=Annotations(name="offline-circuit"))
    restored_ref = connection._program(ref.model_dump_json(), IO())
    assert restored_ref == ref


def test_real_reference_lifecycle_and_full_result_decoding(sdk_connection, monkeypatch):
    connection, project = sdk_connection
    data = local_data(connection)
    uploaded = CircuitRef(id=UUID(int=2), project=project, annotations=Annotations(name="upload"))
    compiled = CircuitRef(id=UUID(int=3), project=project, annotations=Annotations(name="compiled"))
    compile_job = None
    execute_job = None
    events = []

    def upload(circuit, project, name):
        assert isinstance(circuit, Circuit) and float(circuit.phase) == 0.25
        events.append("upload")
        return uploaded

    def compile_request(**kwargs):
        nonlocal compile_job
        assert kwargs["programs"] == [uploaded]
        compile_job = CompileJobRef(id=UUID(int=4), annotations=Annotations(name=kwargs["name"],
            description=kwargs["description"]), project=project, backend_config_store=kwargs["backend_config"],
            last_status=JobStatusEnum.RUNNING, last_message="")
        events.append("compile")
        return compile_job

    def execute_request(**kwargs):
        nonlocal execute_job
        assert kwargs["programs"] == [compiled]
        assert kwargs["n_shots"] == [3] and kwargs["max_cost"] == [12.5]
        assert kwargs["target_region"] == "us"
        execute_job = ExecuteJobRef(id=UUID(int=5), annotations=Annotations(name=kwargs["name"],
            description=kwargs["description"]), project=project, backend_config_store=kwargs["backend_config"],
            last_status=JobStatusEnum.COMPLETED, last_message="")
        events.append("execute")
        return execute_job

    monkeypatch.setattr(qnx.circuits, "upload", create_autospec(qnx.circuits.upload, side_effect=upload))
    monkeypatch.setattr(qnx, "start_compile_job", create_autospec(qnx.start_compile_job, side_effect=compile_request))
    monkeypatch.setattr(qnx, "start_execute_job", create_autospec(qnx.start_execute_job, side_effect=execute_request))
    input_json = connection.upload(data, preparation_id="prep", run=IO(),
                                   acknowledge=lambda identity: events.append(("upload-ack", identity)))
    compile_locator = connection.start_compile(input_json, preparation_id="prep", run=IO(),
        acknowledge=lambda locator: events.append(("compile-ack", locator.job_id)))
    monkeypatch.setattr(qnx.jobs, "get", create_autospec(qnx.jobs.get,
        side_effect=lambda *, id: compile_job if id == str(compile_job.id) else execute_job))
    state = JobStatus(JobStatusEnum.RUNNING)
    monkeypatch.setattr(qnx.jobs, "status", create_autospec(qnx.jobs.status, side_effect=lambda job: state))
    assert connection.refresh_compile(compile_locator, input_json,
        lambda: pytest.fail("queued compile read local payload"), run=IO()).status == "acknowledged"
    compilation = CompilationResultRef(id=UUID(int=6), project=project,
        annotations=Annotations(name="compiled-result"), last_status_detail=JobStatus(JobStatusEnum.COMPLETED))
    monkeypatch.setattr(CompilationResultRef, "get_input", lambda self: uploaded)
    monkeypatch.setattr(CompilationResultRef, "get_output", lambda self: compiled)
    monkeypatch.setattr(qnx.jobs, "results", create_autospec(qnx.jobs.results, return_value=[compilation]))
    state = JobStatus(JobStatusEnum.COMPLETED)
    preparation = connection.refresh_compile(compile_locator, input_json, lambda: data, run=IO())
    assert preparation.operations is None and preparation.native.circuit is None
    assert preparation.target.version == version("qnexus")
    record = NS(provider_options_json=preparation.provider_options_json, snapshot=data.snapshot,
        observation=ObservationSpec(kind="counts", shots=3), classical_layout=data.classical_layout,
        population="unconditional")
    native = connection.restore_native(record, run=IO())
    run = IO()
    locator = connection.launch((native,), submission_id="execute", run=run)
    result_ref = ExecutionResultRef(id=UUID(int=7), project=project, annotations=Annotations(name="counts"),
        last_status_detail=JobStatus(JobStatusEnum.COMPLETED), job_item_integer_id=42)
    # The integer item ID is intentionally not ordinal zero. Association must
    # come from the actual input program, never its result-list position or ID.
    monkeypatch.setattr(ExecutionResultRef, "get_input", lambda self: compiled)
    result = BackendResult(c_bits=[Bit("a", 1), Bit("z", 0), Bit("a", 0)],
        shots=OutcomeArray.from_readouts(np.array([[0, 1, 0], [1, 1, 0], [0, 1, 0]], dtype=np.uint8)))
    monkeypatch.setattr(ExecutionResultRef, "download_result", lambda self: result)
    qnx.jobs.results.return_value = [result_ref]
    original = SubmissionItem(item=0, result_key="0", attempt="attempt", prepared_id="sha256:"+"1"*64)
    run._state["submissions"]["execute"] = NS(locator=locator, items=(original,))
    update = connection.refresh(locator, (native,), run=run)
    assert update.status == "completed"
    assert update.results[0][1].raw_output["counts"] == {"001": 2, "101": 1}
    assert update.associations == (original.revise(provider_result_id=str(result_ref.id)),)
    assert events == ["upload", ("upload-ack", str(uploaded.id)), "compile",
                      ("compile-ack", str(compile_job.id)), "execute"]
    # Simulate reopening using only the durable submission item and completed
    # acquisition. No process-local result cache or circuit payload is available.
    run._state = dict(submissions={"execute": NS(locator=locator, items=update.associations)},
                      by_attempt={"attempt": object()}, backend_context={})
    monkeypatch.setattr(ExecutionResultRef, "get_input", lambda self: pytest.fail("redownload after reopen"))
    monkeypatch.setattr(ExecutionResultRef, "download_result", lambda self: pytest.fail("copy after reopen"))
    assert connection.refresh(locator, (native,), run=run).results == ()
    assert run._state["backend_context"] == {}


def test_real_counts_only_backend_result_and_incomplete_program_models(sdk_connection):
    connection, project = sdk_connection
    data = local_data(connection)
    program = CircuitRef(id=UUID(int=2), project=project, annotations=Annotations(name="counts-only"))
    options = dict(compiled_program=program.model_dump(mode="json"), bits=data.bits)
    record = NS(provider_options_json=json.dumps(options), snapshot=data.snapshot,
        observation=ObservationSpec(kind="counts", shots=3), classical_layout=data.classical_layout,
        population="unconditional")
    native = connection.restore_native(record, run=IO())
    counts = Counter({OutcomeArray.from_readouts([[0, 1, 0]]): 2,
                      OutcomeArray.from_readouts([[1, 1, 0]]): 1})
    result = BackendResult(c_bits=[Bit("a", 1), Bit("z", 0), Bit("a", 0)], counts=counts)
    assert connection._decode(result, native, "job", IO()).raw_output["counts"] == {"001": 2, "101": 1}
    pending = IncompleteJobItemRef(project=project, annotations=Annotations(name="pending"),
        job_type="execute", last_status=JobStatusEnum.RUNNING, last_message="", program_id=program.id)
    assert pending.program_type == "circuit" and pending.program_id == program.id
    assert pending.df().loc[0, "last_status"] == JobStatusEnum.RUNNING
