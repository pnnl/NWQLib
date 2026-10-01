"""Injected public Nexus boundaries: lifecycle and raw joint-count relations.

These tests do not qualify an installed qnexus/pytket SDK or a remote service.
"""

from collections import Counter
from dataclasses import dataclass
import json
from types import ModuleType, SimpleNamespace as NS

import pytest

from nwqlib._prepared_execution import Run
from nwqlib.backends.nexus import NexusBackend, NexusPreparationData, _PreparedNexus
from nwqlib.core.planning import ObservationSpec
from nwqlib.execution import ExecutionLimits, JobLocator, RegisterMap, SubmissionItem


class IO:
    limits = ExecutionLimits(max_data_bytes=1_000_000)

    def __init__(self):
        self.calls = []
        self.synthesis_work = 0
        self._state = dict(submissions={}, by_attempt={}, backend_context={}, synthesis_cache={},
                           synthesis_preparation="preparation")

    def check_data(self, size):
        if size > self.limits.max_data_bytes:
            raise ValueError("data exceeds max_data_bytes")
        self.calls.append(size)

    def _charge_synthesis(self, work, operation):
        # The Run's cumulative check, without its journal.
        if self.synthesis_work + work > self.limits.max_synthesis_work:
            raise ValueError(f"{operation} exceeds max_synthesis_work")
        self.synthesis_work += work

    # The Run's synthesis cache during a preparation, charged through the check above.
    _exact_dense_unitaries = Run._exact_dense_unitaries


class Ref:
    type = "CircuitRef"

    def __init__(self, identity, project="project", name="input"):
        self.id = identity
        self.project = NS(id=project)
        self.annotations = NS(name=name)

    def model_dump(self, mode="json"):
        return dict(id=self.id, project=dict(id=self.project.id), type=self.type)

    @classmethod
    def model_validate_json(cls, text):
        data = json.loads(text)
        return cls(data["id"], data["project"]["id"])


class Config:
    def __init__(self, **values):
        self.values = values

    def __eq__(self, other):
        return isinstance(other, Config) and self.values == other.values

    def model_dump(self, mode="json"):
        return self.values


@dataclass(frozen=True)
class Bit:
    reg_name: str
    position: int

    @property
    def index(self):
        return [self.position]


class Result:
    def __init__(self, bits, counts):
        self.bits = bits
        self.counts = Counter(counts)

    def get_bitlist(self):
        return self.bits

    def get_counts(self, cbits):
        indices = [self.bits.index(bit) for bit in cbits]
        return Counter({tuple(row[i] for i in indices): count for row, count in self.counts.items()})


@pytest.fixture
def backend(monkeypatch):
    import sys
    refs = ModuleType("qnexus.models.references")
    refs.CircuitRef = Ref
    monkeypatch.setitem(sys.modules, "qnexus.models.references", refs)
    tk = ModuleType("pytket")
    tkc = ModuleType("pytket.circuit")
    tkc.Bit = Bit
    result = ModuleType("pytket.backends.backendresult")
    result.BackendResult = Result
    for key, module in (("pytket", tk), ("pytket.circuit", tkc), ("pytket.backends.backendresult", result)):
        monkeypatch.setitem(sys.modules, key, module)
    connection = NexusBackend(project="project", device="H2-1", max_input_bytes=100_000,
        max_cost_hqc=12.5)
    sdk = NS(models=NS(QuantinuumConfig=Config), projects=NS(get=lambda **kw: NS(id=kw["id"])))
    sdk.client = NS(utils=NS(is_managed_token_environment=lambda: False),
        get_nexus_client=lambda: NS(auth=NS(cookies=("myqos_oat",))))
    monkeypatch.setattr(NexusBackend, "_sdk", lambda self: sdk)
    return connection, sdk


def native(identity, shots=7):
    return _PreparedNexus(json.dumps(Ref(identity).model_dump()), identity, "snapshot-"+identity, shots,
        (("z", 0), ("a", 0), ("a", 1)),
        (RegisterMap(name="z", bits=(0,)), RegisterMap(name="a", bits=(1, 2))),
        dict(readout_population="unconditional"))


def make_run():
    return IO()


def job(backend, kind, natives=(), *, identity="job", preparation_id="prep", input_ref=None, status="RUNNING"):
    io = IO()
    programs = [json.loads(n.program_json)["id"] for n in natives] if input_ref is None else [input_ref]
    shots = [n.shots for n in natives] if input_ref is None else []
    return NS(id=identity, project=NS(id="project"), job_type=kind,
        backend_config=backend._config(),
        annotations=NS(name=backend._name(kind, preparation_id), description=backend._association(programs, shots, io)),
        status=status)


def execute_ref(identity, counts, *, state="COMPLETED", program=None):
    return NS(id="result-"+identity, project=NS(id="project"), type="ExecutionResultRef", result_type="PYTKET",
        get_input=lambda: Ref(identity) if program is None else program,
        download_result=lambda: Result([Bit("a", 1), Bit("z", 0), Bit("a", 0)], counts),
        last_status_detail=NS(status=state))


def test_launch_preserves_unequal_shots_and_credit_units_without_wait(backend):
    connection, sdk = backend
    natives = (native("one", 7), native("two", 11))
    called = []
    sdk.start_execute_job = lambda **kw: called.append(kw) or NS(id="accepted")
    locator = connection.launch(natives, submission_id="submission", run=make_run())
    assert locator == JobLocator(provider="nexus", job_id="accepted", project="project", region="us")
    request, = called
    assert [ref.id for ref in request["programs"]] == ["one", "two"]
    assert request["n_shots"] == [7, 11]
    assert request["max_cost"] == [12.5, 12.5]  # per-program HQC, never USD or batch total
    assert request["target_region"] == "us"  # no ambient qnexus target-region override
    assert request["name"] == "nwqlib:execute:submission"
    assert json.loads(request["description"])["shots"] == [7, 11]


def test_lost_ack_reconciles_uniquely_without_resubmission(backend):
    connection, sdk = backend
    natives = (native("one"),)
    calls = []
    error = ConnectionError("accepted-create/lost-ack")

    def launch(**kw):
        calls.append(kw)
        raise error

    sdk.start_execute_job = launch
    with pytest.raises(ConnectionError) as caught:
        connection.launch(natives, submission_id="submission", run=make_run())
    assert caught.value is error
    accepted = job(connection, "execute", natives, preparation_id="submission")
    yielded = []

    def matches(**kw):
        assert kw["page_size"] == 2 and kw["name_exact"] == ["nwqlib:execute:submission"]
        yielded.append(kw)
        return iter([accepted])

    sdk.jobs = NS(get_all=matches)
    locator = connection.reconcile(NS(submission_id="submission"), natives, run=make_run())
    assert locator.job_id == "job" and len(calls) == 1
    sdk.jobs.get_all = lambda **kw: iter([])
    assert connection.reconcile(NS(submission_id="submission"), natives, run=make_run()) is None
    sdk.jobs.get_all = lambda **kw: iter([accepted, accepted, NS()])
    with pytest.raises(ValueError, match="ambiguous"):
        connection.reconcile(NS(submission_id="submission"), natives, run=make_run())
    sdk.jobs.get_all = lambda **kw: iter([accepted])
    with pytest.raises(ValueError, match="association"):
        connection.reconcile(NS(submission_id="submission"), (native("one", 6),), run=make_run())
    assert len(calls) == 1


def test_joint_register_counts_and_result_order(backend):
    connection, sdk = backend
    natives = (native("one", 7), native("two", 11))
    accepted = job(connection, "execute", natives, status="COMPLETED")
    # Provider storage order is [a1,z0,a0], different from both lexicographic
    # register order and original global [z0,a0,a1]. Joint outcomes must survive.
    refs = [execute_ref("two", {(1, 0, 1): 11}), execute_ref("one", {(0, 1, 0): 2, (1, 1, 0): 5})]
    sdk.jobs = NS(get=lambda **kw: accepted, status=lambda ref: NS(status=accepted.status),
                  results=lambda ref, **kw: refs)
    update = connection.refresh(connection._locator(accepted), natives, run=make_run())
    assert update.status == "completed"
    assert [key for key, result in update.results] == ["1", "0"]
    assert update.results[0][1].raw_output["counts"] == {"110": 11}
    assert update.results[1][1].raw_output["counts"] == {"001": 2, "101": 5}
    assert connection.refresh(connection._locator(accepted), natives, run=make_run()) == update


def test_partial_completion_excludes_growing_shot_sets_and_preserves_known_locator(backend):
    connection, sdk = backend
    natives = (native("one"), native("two"))
    accepted = job(connection, "execute", natives)
    refs = [execute_ref("one", {(0, 1, 0): 7}), execute_ref("two", {(0, 1, 0): 2}, state="RUNNING")]
    refs[1].get_input = lambda: pytest.fail("growing result must not be downloaded for association")
    sdk.jobs = NS(get=lambda **kw: accepted, status=lambda ref: NS(status="RUNNING"),
                  results=lambda ref, **kw: refs)
    locator = connection._locator(accepted)
    update = connection.refresh(locator, natives, run=make_run())
    assert update.status == "acknowledged" and [key for key, _ in update.results] == ["0"]
    failure = OSError("download interrupted")

    def broken():
        raise failure

    refs[0].download_result = broken
    with pytest.raises(OSError) as caught:
        connection.refresh(locator, natives, run=make_run())
    assert caught.value is failure and locator.job_id == "job"
    refs[0] = execute_ref("one", {(0, 1, 0): 7})
    refs[1] = execute_ref("two", {(1, 0, 1): 7})
    sdk.jobs.status = lambda ref: NS(status="COMPLETED")
    assert len(connection.refresh(locator, natives, run=make_run()).results) == 2


def test_foreign_program_registers_and_invalid_counts_are_rejected(backend):
    connection, sdk = backend
    natives = (native("one"),)
    accepted = job(connection, "execute", natives)
    refs = [execute_ref("foreign", {(0, 1, 0): 7})]
    sdk.jobs = NS(get=lambda **kw: accepted, status=lambda ref: NS(status="COMPLETED"), results=lambda *a, **k: refs)
    with pytest.raises(ValueError, match="unknown or repeated"):
        connection.refresh(connection._locator(accepted), natives, run=make_run())
    with pytest.raises(ValueError, match="classical registers"):
        connection._decode(Result([Bit("x", 0)], {(0,): 7}), natives[0], "job", IO())
    for counts in ({(0, 1, 0): 8}, {(0, 1, 0): -1}, {(0, 1, 0): 1.2}, {(0, 2, 0): 7}):
        with pytest.raises(ValueError, match="raw counts|shot population"):
            connection._decode(Result([Bit("a", 1), Bit("z", 0), Bit("a", 0)], counts), natives[0], "job", IO())
    with pytest.raises(ValueError, match="distinct compiled"):
        connection.admit_batch((native("one"), native("one")))
    programs = tuple(native(str(index)) for index in range(301))
    connection.admit_batch(programs[:300])
    with pytest.raises(ValueError, match="300 programs"):
        connection.admit_batch(programs)


def test_compile_pending_resume_and_original_association(backend, monkeypatch):
    connection, sdk = backend
    data = NexusPreparationData(snapshot="snapshot", shots=7, circuit_json='{"commands": []}',
        bits=(("z", 0),), quantum_layout=(RegisterMap(name="q", bits=(0,)),),
        classical_layout=(RegisterMap(name="z", bits=(0,)),))
    input_json = json.dumps(Ref("input").model_dump())
    accepted = job(connection, "compile", input_ref="input")
    calls = []
    sdk.start_compile_job = lambda **kw: calls.append(kw) or NS(id=accepted.id)
    sdk.jobs = NS(get=lambda **kw: accepted, status=lambda ref: NS(status=accepted.status),
        results=lambda *a, **kw: pytest.fail("pending compile must not ask for output"))
    locator = connection.start_compile(input_json, preparation_id="prep", run=IO(), acknowledge=lambda locator: None)
    assert connection.refresh_compile(locator, input_json,
        lambda: pytest.fail("queued compile must not load the archived local circuit"), run=IO()).status == "acknowledged"
    monkeypatch.setattr("nwqlib.backends.nexus.version", lambda package: "fixture")
    monkeypatch.setattr("nwqlib.backends.nexus.environment", lambda packages: ())
    accepted.status = "COMPLETED"
    output = NS(type="CompilationResultRef", project=NS(id="project"),
                get_input=lambda: Ref("input"), get_output=lambda: Ref("compiled"))
    sdk.jobs.results = lambda *a, **kw: [output]
    complete = connection.refresh_compile(locator, input_json, lambda: data, run=IO())
    assert complete.operations is None and complete.native_basis == ()
    assert complete.probability_window_exclusions is None
    assert json.loads(complete.provider_options_json)["compiled_program"]["id"] == "compiled"
    assert len(calls) == 1
    record = NS(provider_options_json=complete.provider_options_json, snapshot=data.snapshot,
        observation=ObservationSpec(kind="counts", shots=7), classical_layout=data.classical_layout,
        population="unconditional")
    restored = connection.restore_native(record, run=IO())
    assert json.loads(restored.program_json)["id"] == "compiled" and len(calls) == 1
    output.get_input = lambda: Ref("different")
    with pytest.raises(ValueError, match="different input"):
        connection.refresh_compile(locator, input_json, lambda: data, run=IO())
    # A compile level other than the default 1 marks the receipt with the
    # exclusion "optimization_level".
    output.get_input = lambda: Ref("input")
    leveled = connection.revise(optimization_level=2)
    compiled = job(leveled, "compile", input_ref="input", status="COMPLETED")
    sdk.jobs.get, sdk.jobs.status = lambda **kw: compiled, lambda ref: NS(status="COMPLETED")
    marked = leveled.refresh_compile(locator, input_json, lambda: data, run=IO())
    assert marked.probability_window_exclusions == ("optimization_level",)


def test_compile_correlation_and_cancellation_do_not_create_jobs(backend):
    connection, sdk = backend
    accepted = job(connection, "compile", input_ref="input")
    cancelled = []
    sdk.jobs = NS(get_all=lambda **kw: iter([accepted]), get=lambda **kw: accepted,
                  cancel=lambda ref: cancelled.append(ref.id))
    ref = json.dumps(Ref("input").model_dump())
    locator = connection.reconcile_preparation("prep", "compile", input_ref_json=ref, run=IO())
    assert locator.job_id == "job"
    connection.cancel_preparation(locator, run=IO())
    assert cancelled == ["job"]
    with pytest.raises(ValueError, match="association"):
        connection.reconcile_preparation("prep", "compile",
            input_ref_json=json.dumps(Ref("different").model_dump()), run=IO())


def test_helios_and_noncounts_fail_before_sdk_work():
    kwargs = dict(project="project", max_input_bytes=100)
    with pytest.raises(ValueError, match="Helios"):
        NexusBackend(device="Helios-1", **kwargs)
    connection = NexusBackend(device="H2-1", **kwargs)
    with pytest.raises(ValueError, match="counts only"):
        connection.target_for(NS(kind="probabilities"))


def test_upload_lost_ack_and_resume_from_public_ref(backend, monkeypatch):
    import sys
    connection, sdk = backend
    reconstructed, posted = [], []
    sys.modules["pytket"].Circuit = NS(from_dict=lambda data: reconstructed.append(data) or data)
    data = NexusPreparationData(snapshot="s", shots=2, circuit_json='{"phase": "0.5", "commands": []}',
        bits=(("c", 0),), quantum_layout=(RegisterMap(name="q", bits=(0,)),),
        classical_layout=(RegisterMap(name="c", bits=(0,)),))
    failure = ConnectionError("upload accepted, response lost")

    def upload(**kw):
        posted.append(kw)
        raise failure

    sdk.circuits = NS(upload=upload, get_all=lambda **kw: iter([Ref("uploaded", name="nwqlib:upload:prep")]))
    with pytest.raises(ConnectionError) as caught:
        connection.upload(data, preparation_id="prep", run=IO(), acknowledge=lambda identity: None)
    assert caught.value is failure
    saved = connection.reconcile_preparation("prep", "upload", run=IO())
    assert json.loads(saved)["id"] == "uploaded"
    sdk.start_compile_job = lambda **kw: NS(id="compile-job")
    locator = connection.start_compile(saved, preparation_id="prep", run=IO(), acknowledge=lambda locator: None)
    assert locator.job_id == "compile-job" and len(posted) == 1
    assert reconstructed == [{"phase": "0.5", "commands": []}]


def test_local_conversion_preserves_named_bit_mapping_and_sdk_payload(backend, monkeypatch):
    import sys
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister
    connection, sdk = backend
    converted = []
    converter = ModuleType("pytket.extensions.qiskit")
    payload = {"phase": "0.5", "commands": [{"op": {"type": "H"}, "args": [["q", [0]]]}]}

    def convert(circuit):
        converted.append(circuit)
        return NS(bits=[Bit("a", 0), Bit("a", 1), Bit("z", 0)], to_dict=lambda: payload)

    converter.qiskit_to_tk = convert
    monkeypatch.setitem(sys.modules, "pytket.extensions.qiskit", converter)
    circuit = QuantumCircuit(QuantumRegister(1, "q"), ClassicalRegister(1, "z"), ClassicalRegister(2, "a"))
    circuit.h(0)
    args = dict(observation=ObservationSpec(kind="counts", shots=3), runtime=NS(seed=11), position=1,
                source_definitions=(), run=IO(), snapshot="snapshot")
    data = connection.prepare_local(circuit, **args)
    assert data.bits == (("z", 0), ("a", 0), ("a", 1))
    assert data.classical_layout == (RegisterMap(name="z", bits=(0,)), RegisterMap(name="a", bits=(1, 2)))
    assert json.loads(data.circuit_json) == payload and len(converted) == 1
    assert converted[0].clbits == circuit.clbits
    assert converted[0].data[0].operation.name == "u"
    restored = connection.restore_preparation(data.model_dump_json().encode(), run=IO())
    assert restored == data
    converter.qiskit_to_tk = lambda circuit: NS(bits=[Bit("c", 0)], to_dict=lambda: pytest.fail("foreign inventory"))
    with pytest.raises(ValueError, match="bit inventory"):
        connection.prepare_local(circuit, **args)


def test_upload_ack_precedes_serialization_failure_and_recovers_by_id(backend, monkeypatch):
    import sys
    connection, sdk = backend
    sys.modules["pytket"].Circuit = NS(from_dict=lambda data: data)
    data = NexusPreparationData(snapshot="s", shots=2, circuit_json='{}', bits=(), quantum_layout=(), classical_layout=())
    trace = []
    sdk.circuits = NS(upload=lambda **kw: trace.append("upload") or Ref("accepted"),
                      get=lambda **kw: trace.append(("get", kw["id"])) or Ref(kw["id"]))
    original = NexusBackend._encode

    def fail_after_upload(self, value, run, **kwargs):
        trace.append("serialize")
        raise ValueError("output data limit exhausted")

    monkeypatch.setattr(NexusBackend, "_encode", fail_after_upload)
    with pytest.raises(ValueError, match="data limit"):
        connection.upload(data, preparation_id="prep", run=IO(),
                          acknowledge=lambda identity: trace.append(("ack", identity)))
    assert trace == ["upload", ("ack", "accepted"), "serialize"]
    monkeypatch.setattr(NexusBackend, "_encode", original)
    assert json.loads(connection.restore_upload("accepted", run=IO()))["id"] == "accepted"
    assert trace[-1] == ("get", "accepted") and trace.count("upload") == 1


def test_completed_sibling_readback_reuses_association_not_raw_counts(backend):
    connection, sdk = backend
    natives = (native("one"), native("two"))
    accepted = job(connection, "execute", natives, preparation_id="submission")
    first = execute_ref("one", {(0, 1, 0): 7})
    refs = [first, NS(type="IncompleteJobItemRef", project=NS(id="project"), program_type="circuit", program_id="two")]
    sdk.jobs = NS(get=lambda **kw: accepted, status=lambda ref: NS(status="RUNNING"), results=lambda *a, **kw: refs)
    run = make_run()
    locator = connection._locator(accepted)
    run._state["submissions"]["submission"] = NS(locator=locator, items=(
        SubmissionItem(item=0, attempt="attempt-one", prepared_id="sha256:"+"0"*64),
        SubmissionItem(item=1, attempt="attempt-two", prepared_id="sha256:"+"1"*64, result_key="1")))
    update = connection.refresh(locator, natives, run=run)
    assert [key for key, _ in update.results] == ["0"]
    assert update.associations[0].provider_result_id == "result-one"
    run._state["submissions"]["submission"].items = (
        update.associations[0], run._state["submissions"]["submission"].items[1])
    run._state["by_attempt"]["attempt-one"] = object()  # common owner's durable publication
    first.get_input = lambda: pytest.fail("completed sibling downloaded twice")
    first.download_result = lambda: pytest.fail("completed sibling copied twice")
    assert connection.refresh(locator, natives, run=run).results == ()
    assert run._state["backend_context"] == {}  # no shadow result-association cache


@pytest.mark.parametrize('entry', [
    'upload', 'restore_upload', 'compile', 'launch', 'refresh', 'refresh_compile',
    'reconcile_upload', 'reconcile_execute', 'cancel', 'cancel_compile',
])
def test_absent_local_auth_rejects_every_first_remote_request(backend, monkeypatch, entry):
    import sys
    connection, sdk = backend
    run = IO()
    n = native('program')
    locator = JobLocator(provider='nexus', job_id='job', project='project', region='us')
    calls = []
    def forbidden(*args, **kwargs):
        pytest.fail('request before local authentication admission')
    sdk.client.get_nexus_client = lambda: NS(auth=NS(cookies=()))
    sdk.auth = NS(is_logged_in=forbidden, get_token_expiry=forbidden)
    sdk.projects.get = forbidden
    sdk.circuits = NS(get=forbidden, upload=forbidden, get_all=forbidden)
    sdk.jobs = NS(get=forbidden, status=forbidden, results=forbidden, cancel=forbidden, get_all=forbidden)
    sdk.start_compile_job = sdk.start_execute_job = forbidden
    monkeypatch.setattr(sys.modules['pytket'], 'Circuit', NS(from_dict=forbidden), raising=False)
    operations = {
        'upload': lambda: connection.upload(NS(circuit_json='{}'), preparation_id='prep', run=run, acknowledge=calls.append),
        'restore_upload': lambda: connection.restore_upload('program', run=run),
        'compile': lambda: connection.start_compile(n.program_json, preparation_id='prep', run=run, acknowledge=calls.append),
        'launch': lambda: connection.launch((n,), submission_id='submission', run=run),
        'refresh': lambda: connection.refresh(locator, (n,), run=run),
        'refresh_compile': lambda: connection.refresh_compile(locator, n.program_json, forbidden, run=run),
        'reconcile_upload': lambda: connection.reconcile_preparation('prep', 'upload', run=run),
        'reconcile_execute': lambda: connection.reconcile(NS(submission_id='submission'), (n,), run=run),
        'cancel': lambda: connection.cancel(locator, run=run),
        'cancel_compile': lambda: connection.cancel_preparation(locator, run=run),
    }
    with pytest.raises(ValueError, match='Nexus local authentication is unavailable'):
        operations[entry]()
    assert not calls


@pytest.mark.parametrize('names,managed,admitted', [
    ((), False, False), (('unrelated',), False, False),
    (('myqos_id',), False, True), (('myqos_oat',), False, True), ((), True, True),
])
def test_local_auth_presence_preserves_memory_and_managed_state_without_token_reads(
    backend, names, managed, admitted
):
    connection, sdk = backend
    calls = []
    class CookieNames:
        def __iter__(self):
            return iter(names)
        def get(self, *args, **kwargs):
            pytest.fail('token value read')
    client = NS(auth=NS(cookies=CookieNames()))
    def existing_client():
        calls.append('client')
        return client
    sdk.client.get_nexus_client = existing_client  # No reload argument is accepted.
    sdk.client.utils.is_managed_token_environment = lambda: managed
    def project(**kwargs):
        calls.append('request')
        return NS(id=kwargs['id'])
    sdk.projects.get = project
    if admitted:
        assert connection._project().id == 'project'
        assert calls == (['request'] if managed else ['client', 'request'])
    else:
        with pytest.raises(ValueError, match='local authentication'):
            connection._project()
        assert calls == ['client']


def test_authenticated_sdk_failure_is_not_relabelled_as_missing_auth(backend):
    connection, sdk = backend
    error = RuntimeError('SDK rejected expired authentication')
    def fail(**kwargs):
        raise error
    sdk.projects.get = fail
    with pytest.raises(RuntimeError) as caught:
        connection._project()
    assert caught.value is error


def test_local_conversion_lowers_dense_unitaries_exactly(backend, monkeypatch):
    # Qiskit's own u/cx lowering of this near-identity unitary errs by about
    # 1e-6 per entry. The expected matrix is SciPy's exponential.
    import sys
    import numpy as np
    from scipy.linalg import expm
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    connection, sdk = backend
    converted = []
    converter = ModuleType("pytket.extensions.qiskit")

    def convert(circuit):
        converted.append(circuit)
        return NS(bits=[Bit("z", 0)], to_dict=lambda: {"phase": "0", "commands": []})

    converter.qiskit_to_tk = convert
    monkeypatch.setitem(sys.modules, "pytket.extensions.qiskit", converter)
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary = expm(-1e-6j * (x + x.conj().T) / 2)
    circuit = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(1, "z"))
    circuit.append(UnitaryGate(unitary), [0, 1])
    connection.prepare_local(circuit, observation=ObservationSpec(kind="counts", shots=3), runtime=NS(seed=11),
                             position=1, source_definitions=(), run=IO(), snapshot="dense")
    assert np.abs(Operator(converted[0]).data - unitary).max() <= 1e-13
