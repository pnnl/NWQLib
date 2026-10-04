"""Public IonQ conversion with injected HTTP; no provider or simulation calls."""

import json
import nwqlib
from nwqlib._run_archive import load as load_run
from nwqlib.algorithms.expectation import ExpectationMethod
from test_expectation_current import problem
from types import SimpleNamespace
import pytest
from requests.exceptions import ConnectionError
from nwqlib.backends.ionq import IonQBackend
from nwqlib.core.planning import ObservationSpec, RuntimeOptions
from nwqlib._prepared_execution import (
    Run,
    prepare_experiment as prepare,
    refresh_submissions,
    submit_experiment as submit,
    submit_detached,
)


@pytest.fixture
def api(monkeypatch):
    """Supply offline job, child and artifact responses with independent failure controls and a
    request counter.
    """
    import requests

    exchanges = []
    state = SimpleNamespace(
        status="completed",
        histogram={"0": 3, "1": 1},
        fail_read=False,
        lose_ack=False,
        artifact_format="ionq.result.histogram.json.v1",
        children=None,
        variants=None,
        debiasing=False,
        posts=0,
        body=None,
        fail_child=None,
        fail_artifact=None,
        failure=None,
        child_status={"child-a": "completed", "child-b": "completed"},
        child_overrides={},
    )

    class Response:
        status_code = 200

        def __init__(self, payload):
            self.raw = self
            self.payload = json.dumps(payload).encode()

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def read(self, amt, *, decode_content):
            assert decode_content is True  # requests leaves raw content-encoded bodies undecoded.
            part, self.payload = self.payload[:amt], self.payload[amt:]
            return part

    def request(method, url, **kwargs):
        """Route synthetic responses by the original parent/child identity and expose changed
        artifact or lost-ack cases.
        """
        assert url.startswith("https://api.ionq.co/v0.4/")
        assert kwargs["timeout"] == 30.0 and kwargs["stream"] and (not kwargs["allow_redirects"])
        exchanges.append((method, url, kwargs))
        if method == "POST":
            state.posts += 1
            state.body = json.loads(kwargs["data"])
            if state.lose_ack:
                raise requests.ConnectionError("accepted request lost acknowledgement")
            return Response({"id": "original-ionq-job", "status": "submitted"})
        if method == "PUT":
            state.status = "canceled"
            return Response({"id": "original-ionq-job"})
        if state.fail_read:
            raise requests.ConnectionError("original IonQ retrieval interruption")
        batch = state.body is not None and state.body["type"] == "ionq.multi-circuit.v1"
        if batch:
            child_id = url.split("/jobs/")[1].split("/")[0]
            if child_id == state.fail_child:
                raise state.failure or requests.ConnectionError("original child retrieval interruption")
            if "/artifacts/" in url:
                if child_id == state.fail_artifact:
                    raise state.failure
                return Response({"0": 3, "1": 1} if child_id == "child-a" else {"0": 1, "1": 3})
            if child_id == "original-ionq-job":
                return Response(
                    dict(
                        id=child_id,
                        backend="qpu.forte-1",
                        type="ionq.multi-circuit.v1",
                        status=state.status,
                        shots=4,
                        child_job_ids=state.children or ["child-b", "child-a"],
                        metadata=state.body["metadata"],
                    )
                )
            index = {"child-a": 0, "child-b": 1}[child_id]
            child = dict(
                id=child_id,
                backend="qpu.forte-1",
                type="ionq.circuit.v1",
                parent_job_id="original-ionq-job",
                name=state.body["input"]["circuits"][index]["name"],
                status=state.child_status[child_id],
                shots=4,
                settings={"error_mitigation": {"debiasing": False}},
                results={state.artifact_format: {"id": "raw-" + child_id, "format": state.artifact_format}},
            )
            child.update(state.child_overrides.get(child_id, {}))
            return Response(child)
        if "/artifacts/" in url:
            return Response(state.histogram)
        return Response(
            dict(
                id="original-ionq-job",
                backend="qpu.forte-1",
                type="ionq.circuit.v1",
                status=state.status,
                shots=4,
                child_job_ids=state.children,
                settings={"error_mitigation": {"debiasing": state.debiasing}},
                output={"error_mitigation": {"debiasing": {"variants": state.variants}}},
                results={state.artifact_format: {"id": "raw-artifact", "format": state.artifact_format}},
            )
        )

    monkeypatch.setenv("NWQLIB_IONQ_API_KEY", "injected-test-key")
    monkeypatch.setattr(requests, "request", request)
    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    return (state, backend, exchanges)


def test_public_counts_reach_finite_pauli_analysis_and_repeat_restore(tmp_path, api):
    state, backend, exchanges = api
    method, selected = selection()
    path = tmp_path / "ionq.sqlite"
    with Run(selected, backend=backend, directory=path) as run:
        with pytest.warns(UserWarning, match="offline"):
            handle = handle_for(selected, backend, run)
        receipt = submit_detached((handle,), run=run)
        state.fail_read = True
        with pytest.raises(ConnectionError, match="retrieval interruption"):
            refresh_submissions(run=run)
        assert run.trace.submissions[0].locator == receipt.locator and run.trace.jobs == 1
    state.fail_read = False
    with load_run(path, backend=backend) as run:
        (chunk,) = refresh_submissions(run=run)
        histogram = chunk.histogram()
        assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 3, 1: 1}
        assert chunk.returned_shots == 4 and chunk.job == "original-ionq-job"
        assert run.collect(chunk) and (not run.collect(chunk))
        result = method.analyze(selected, run.data, settings={})
        assert result.value == 1.5
    reads = len(exchanges)
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run)
        assert len(exchanges) == reads and state.posts == 1 and (run.trace.jobs == 1)


def test_one_post_lost_ack_stays_uncertain_and_never_resubmits(tmp_path, api, monkeypatch):
    state, backend, exchanges = api
    _, selected = selection()
    state.lose_ack = True
    path = tmp_path / "lost.sqlite"
    with Run(selected, backend=backend, directory=path) as run:
        with pytest.raises(ConnectionError, match="lost acknowledgement"):
            run.resume()
        assert run.trace.submissions[0].status == "uncertain"
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run)
        assert run.trace.jobs == 1 and run.trace.events[0].shots == 4
        assert run.trace.submissions[0].locator is None and state.posts == 1 and (len(exchanges) == 1)
        from nwqlib.execution import RunFailed

        exposure = run.exposure
        monkeypatch.setattr("nwqlib._prepared_execution.sleep",
                            lambda *args: pytest.fail("unrecoverable IonQ outcome entered polling"))
        with pytest.raises(RunFailed) as caught:
            run.wait(timeout=0)
        assert caught.value.stage == "recovery" and caught.value.status == "uncertain"
        assert run.exposure == exposure
        assert state.posts == 1 and len(exchanges) == 1


def test_sync_pending_then_same_job_completes_and_cancel_preserves_exposure(api, tmp_path):
    state, backend, _ = api
    _, selected = selection()
    with Run(selected, backend=backend, directory=tmp_path / "pending") as run:
        handle = handle_for(selected, backend, run)
        state.status = "started"
        pending = submit(handle, run=run)
        assert pending.status == "acknowledged"
        assert run.trace.submissions[0].locator.job_id == "original-ionq-job"
        assert run.cancel(reason="explicit test cancellation")
        assert not refresh_submissions(run=run)
        assert state.posts == 1 and run.trace.jobs == 1 and (run.trace.events[0].shots == 4)
        assert run.trace.submissions[0].status == "cancelled"


@pytest.mark.parametrize("gateset", ["qis", "native"])
def test_public_converter_preserves_units_and_nontrivial_measured_wires(api, gateset, tmp_path):
    """Nontrivial measurement maps expose endianness errors while QIS/native gate parameters keep
    their original units.
    """
    from qiskit import QuantumCircuit
    from qiskit_ionq import GPIGate, ZZGate

    state, original, _ = api
    backend = original.revise(gateset=gateset)
    qc = QuantumCircuit(3, 3)
    if gateset == "native":
        qc.append(GPIGate(0.125), [2])
        qc.append(ZZGate(0.2), [0, 2])
    else:
        qc.rx(0.125, 2)
    qc.measure(2, 0)
    qc.measure(0, 2)
    _, selected = selection()
    with Run(selected, backend=backend, directory=tmp_path / "converter") as run:
        prep = backend.prepare(
            qc,
            observation=ObservationSpec(kind="counts", shots=4),
            runtime=RuntimeOptions(seed=7),
            position=len(qc.data),
            source_definitions=(),
            run=run,
            snapshot="mapping",
        )
        gates = prep.native.input["circuit"]
        if gateset == "native":
            assert gates[0]["phase"] == 0.125 and gates[1]["angle"] == 0.2
        else:
            assert gates[0]["rotation"] == 0.125
        assert prep.native.measurement_map == (2, None, 0)
        result = backend._decode(
            {"4": 3, "1": 1},
            artifact_format="ionq.result.histogram.json.v1",
            native=prep.native,
            shots=4,
            job_id="mapped",
            run=run,
        )
        assert dict(zip(*(array.tolist() for array in result.raw_output["counts"]))) == {0b001: 3, 0b100: 1}
        result2 = backend._decode(
            {"histogram": {"registers": {"output_all": {"100": 3, "001": 1}}}},
            artifact_format="ionq.result.histogram.json.v2",
            native=prep.native,
            shots=4,
            job_id="mapped",
            run=run,
        )
        assert all((new == old).all() for new, old in zip(result2.raw_output["counts"], result.raw_output["counts"], strict=True))


@pytest.mark.parametrize(
    "change,match",
    [
        ({"artifact_format": "ionq.result.probabilities.json.v2"}, "no raw histogram"),
        ({"histogram": {"0": 0.75, "1": 0.25}}, "nonnegative integers"),
        ({"histogram": {"0": 5}}, "population"),
        ({"children": ["child-other"]}, "child jobs"),
        ({"variants": [{"variant_id": "variant-other", "shots": 4}]}, "merged variants"),
    ],
)
def test_foreign_populations_and_probabilities_never_publish_counts(api, change, match, tmp_path):
    state, backend, _ = api
    _, selected = selection()
    for key, value in change.items():
        setattr(state, key, value)
    with Run(selected, backend=backend, directory=tmp_path / "refused") as run:
        handle = handle_for(selected, backend, run)
        submit(handle, run=run)
        with pytest.raises(ValueError, match=match):
            refresh_submissions(run=run)
        assert run.trace.jobs == 1 and (not run._state["by_attempt"])


def test_unsupported_populations_reject_before_submission_and_sdk_error_propagates(api):
    from qiskit import QuantumCircuit

    state, backend, _ = api
    for connection, match in (
        (backend.revise(debiasing=True), "variant acquisitions"),
        (backend.revise(device="simulator"), "ignores shots"),
    ):
        with pytest.raises(ValueError, match=match):
            connection.target_for(ObservationSpec(kind="counts", shots=4))
    with pytest.raises(ValueError, match="5000"):
        backend.admit_batch(())
    qc = QuantumCircuit(1, 1)
    qc.reset(0)
    qc.measure(0, 0)
    from qiskit_ionq.exceptions import IonQGateError

    with pytest.raises(IonQGateError):
        backend._convert(qc)
    assert state.posts == 0


def test_batch_reordered_children_partial_failure_and_restore_preserve_populations(tmp_path, api):
    """A running parent costs one GET per poll and no child read. After it completes, an
    interrupted child read keeps the reordered association already validated, and both
    children are then collected from the one parent submission.
    """
    state, backend, exchanges = api
    _, selected = selection()
    path = tmp_path / "children.sqlite"
    state.status = "started"
    state.child_status["child-a"] = "started"
    with Run(selected, backend=backend, directory=path) as run:
        handle = handle_for(selected, backend, run)
        receipt = submit_detached((handle, handle), run=run)
        assert state.body["settings"]["error_mitigation"]["debiasing"] is False
        assert [entry["name"] for entry in state.body["input"]["circuits"]] == [
            f"nwqlib:{receipt.submission_id}:0",
            f"nwqlib:{receipt.submission_id}:1",
        ]
        reads = len(exchanges)
        assert not refresh_submissions(run=run)
        assert [url.rsplit("/", 1)[1] for _, url, _ in exchanges[reads:]] == ["original-ionq-job"]
        assert [item.attempt for item in run.trace.submissions[0].items] == [item.attempt for item in receipt.items]
        assert run.trace.jobs == 1 and sum((event.shots for event in run.trace.events)) == 8
    state.status = "completed"
    state.child_status["child-a"] = "completed"
    state.fail_child = "child-a"
    with load_run(path, backend=backend) as run:
        with pytest.raises(ConnectionError, match="child retrieval interruption"):
            refresh_submissions(run=run)
        items = run.trace.submissions[0].items
        assert [item.child_job for item in items] == [None, "child-b"]
        assert not run._state["by_attempt"] and run.trace.jobs == 1
    state.fail_child = None
    with load_run(path, backend=backend) as run:
        first, second = refresh_submissions(run=run)
        assert first.chunk == "1" and first.job == "child-b"
        histogram = first.histogram()
        assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 1, 1: 3}
        assert second.chunk == "0" and second.job == "child-a" and (second.attempt != first.attempt)
        histogram = second.histogram()
        assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 3, 1: 1}
        assert run.collect(first) and run.collect(second)
        assert [item.child_job for item in run.trace.submissions[0].items] == ["child-a", "child-b"]
        assert state.posts == 1 and sum((event.returned_shots for event in run.trace.events)) == 8
        assert run.trace.submissions[0].status == "completed"
    reads = len(exchanges)
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run) and len(exchanges) == reads
        assert len(run._state["by_attempt"]) == 2 and run.trace.jobs == 1
    artifact_reads = [url for method, url, _ in exchanges if "/artifacts/" in url]
    assert len(artifact_reads) == 2


@pytest.mark.parametrize("defect", ["foreign_parent", "duplicate_name", "duplicate_child", "changed_child"])
def test_batch_child_association_errors_precede_artifact_download(tmp_path, api, defect):
    state, backend, exchanges = api
    _, selected = selection()
    with Run(selected, backend=backend, directory=tmp_path / "bad.sqlite") as run:
        handle = handle_for(selected, backend, run)
        receipt = submit_detached((handle, handle), run=run)
        if defect == "foreign_parent":
            state.child_overrides["child-a"] = {"parent_job_id": "foreign-parent"}
        elif defect == "duplicate_name":
            state.child_overrides["child-a"] = {"name": f"nwqlib:{receipt.submission_id}:1"}
        elif defect == "duplicate_child":
            state.children = ["child-b", "child-b"]
        else:
            state.child_overrides["child-a"] = {"id": "foreign-child"}
        with pytest.raises(ValueError, match="IonQ child|child job inventory"):
            refresh_submissions(run=run)
        assert not any(("/artifacts/" in url for _, url, _ in exchanges))
        assert not run._state["by_attempt"] and state.posts == 1


@pytest.mark.parametrize("stage", ["status", "artifact", "decode"])
def test_later_child_error_preserves_completed_results_and_original_error(tmp_path, monkeypatch, api, stage):
    """Inject status, artifact or decoder failure and ensure completed child data and the
    original error cause survive.
    """
    state, backend, exchanges = api
    _, selected = selection()
    path = tmp_path / "partial-error.sqlite"
    cause = OSError("original transport or decoder cause")
    failure = (
        ConnectionError("later child interruption")
        if stage != "decode"
        else ValueError("later decoder error")
    )
    failure.__cause__ = cause
    state.failure = failure
    state.fail_child = "child-a" if stage == "status" else None
    state.fail_artifact = "child-a" if stage == "artifact" else None
    decoding_fails = stage == "decode"
    original_decode = IonQBackend._decode

    def decode(self, payload, **kwargs):
        if decoding_fails and kwargs["job_id"] == "child-a":
            raise failure
        return original_decode(self, payload, **kwargs)

    monkeypatch.setattr(IonQBackend, "_decode", decode)
    with Run(selected, backend=backend, directory=path) as run:
        handle = handle_for(selected, backend, run)
        receipt = submit_detached((handle, handle), run=run)
        with pytest.raises(type(failure)) as caught:
            refresh_submissions(run=run)
        assert caught.value is failure and caught.value.__cause__ is cause
        saved = run.trace.submissions[0]
        assert saved.items[1].child_job == "child-b"
        assert saved.items[1].provider_result_id == "raw-child-b"
        assert not saved.results_consumed and saved.refresh_failure
        if stage != "status":
            assert saved.items[0].child_job == "child-a"
            assert saved.items[0].provider_result_id == "raw-child-a"
            completed = run.completed_observation(receipt.items[1].attempt)
            assert completed.job == "child-b" and completed.returned_shots == 4
            histogram = completed.histogram()
            assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 1, 1: 3}
            assert run.collect(completed)
        else:
            assert not run._state["by_attempt"]
        assert run.trace.jobs == 1 and len(run.prepared_artifacts) == 1
        assert sum((event.shots for event in run.trace.events)) == 8
    state.fail_child = state.fail_artifact = None
    decoding_fails = False
    before = len(exchanges)
    with load_run(path, backend=backend) as run:
        if stage == "artifact":
            state.child_overrides["child-a"] = {
                "results": {
                    state.artifact_format: {"id": "replacement-artifact", "format": state.artifact_format}
                }
            }
            artifact_calls = sum(("/artifacts/" in url for _, url, _ in exchanges))
            with pytest.raises(ValueError, match="artifact changed"):
                refresh_submissions(run=run)
            assert sum(("/artifacts/" in url for _, url, _ in exchanges)) == artifact_calls
            assert run.trace.submissions[0].items[0].provider_result_id == "raw-child-a"
            state.child_overrides.clear()
        remaining = refresh_submissions(run=run)
        assert len(remaining) == (2 if stage == "status" else 1)
        for chunk in remaining:
            assert run.collect(chunk)
        if stage != "status":
            assert remaining[0].job == "child-a" and remaining[0].chunk == "0"
            assert not any(("/jobs/child-b" in url for _, url, _ in exchanges[before:]))
        assert state.posts == 1 and run.trace.jobs == 1 and (len(run.prepared_artifacts) == 1)
        assert sum((event.returned_shots for event in run.trace.events)) == 8
    before = len(exchanges)
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run) and len(exchanges) == before
        assert len(run._state["by_attempt"]) == 2 and run.trace.jobs == 1


def selection():
    method = ExpectationMethod()
    return method, nwqlib.plan(problem(), method=method, shots=4, seed=7)


def handle_for(selected, backend, run):
    return prepare(selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=7))


def test_restore_admits_before_json_and_keeps_original_measurement_map(monkeypatch):
    from dataclasses import replace
    from nwqlib.backends import ionq
    from test_nexus_backend import IO
    from nwqlib.execution import ExecutionLimits

    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    record = SimpleNamespace(
        provider_options_json=json.dumps(dict(measurement_map=[2, None, 0], num_qubits=3)),
        observation=ObservationSpec(kind="counts", shots=4),
        classical_layout=(),
        population="unknown",
    )
    run = IO()
    restored = backend.restore_native(record, run=run)
    assert restored.measurement_map == (2, None, 0) and restored.circuit is restored.input is None
    result = backend._decode(
        {"4": 3, "1": 1},
        artifact_format="ionq.result.histogram.json.v1",
        native=restored,
        shots=4,
        job_id="mapped",
        run=run,
    )
    assert dict(zip(*(array.tolist() for array in result.raw_output["counts"]))) == {0b001: 3, 0b100: 1}
    # An identity map over the first two of three qubits drops the unmeasured q2 of every state.
    prefix = replace(restored, measurement_map=(0, 1))
    result = backend._decode({"5": 3, "1": 1}, artifact_format="ionq.result.histogram.json.v1", native=prefix,
                             shots=4, job_id="prefix", run=run)
    assert dict(zip(*(array.tolist() for array in result.raw_output["counts"]))) == {0b01: 4}
    # A circuit wider than 64 qubits, which prepare admits, keeps the mapping route with Python integers.
    wide = replace(restored, num_qubits=65, measurement_map=(64, 0))
    result = backend._decode({str((1 << 64) | 1): 2, "1": 1, "2": 1}, artifact_format="ionq.result.histogram.json.v1",
                             native=wide, shots=4, job_id="wide", run=run)
    assert result.raw_output["counts"] == {"11": 2, "10": 1, "00": 1}
    # Classical position 64 takes qubit 1 on both artifact formats.
    # Both routes publish positive counts after mapping the provider entries.
    tall = replace(restored, measurement_map=(2, None, 0) + (None,) * 61 + (1,), shots=12)
    expected = {"0" * 62 + "100": 2, "0" * 64 + "1": 3, "1" + "0" * 61 + "101": 7}
    decimal = {"0": 0, "1": 2, "4": 3, "7": 7}
    for artifact, histogram in (("ionq.result.histogram.json.v1", decimal),
                                ("ionq.result.histogram.json.v2",
                                 {"histogram": {"registers": {"output_all": {format(int(key), "03b"): count
                                                                             for key, count in decimal.items()}}}})):
        run.calls.clear()
        result = backend._decode(histogram, artifact_format=artifact, native=tall, shots=12, job_id="tall", run=run)
        assert result.raw_output["counts"] == expected and run.calls == [4 * (65 + 16)]
    run.limits = ExecutionLimits(max_data_bytes=10)
    monkeypatch.setattr(ionq.json, "loads", lambda *a, **k: pytest.fail("JSON parsed before admission"))
    with pytest.raises(ValueError, match="max_data_bytes"):
        backend.restore_native(record, run=run)


@pytest.mark.parametrize("extra_classical_bits", [0, 62], ids=["pair", "wide"])
def test_zero_count_provider_entries_do_not_exceed_readout_cardinality(
    tmp_path, api, extra_classical_bits
):
    """Positive histogram support is at most its shots on either decode route."""
    import warnings
    from nwqlib.ir import ClassicalValue
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import ingest_occupation

    state, backend, _ = api
    state.histogram = {"0": 1, "1": 1, "2": 1, "3": 1, "4": 0}
    selected = nwqlib.plan(
        Expectation(state=ingest_occupation("000", num_qubits=3),
                    observable=ingest_pauli((("ZZZ", 1.0),), num_qubits=3)),
        method=ExpectationMethod(), shots=4, seed=7,
    )
    if extra_classical_bits:
        blocks = selected.blocks
        program = selected.construction.program
        program = program.revise(classical=(*program.classical,
            ClassicalValue(name="padding", dtype="bits", width=extra_classical_bits)))
        selected = selected.revise(
            construction=selected.construction.revise(program=program))._bind(blocks=blocks)
    with Run(selected, backend=backend, directory=tmp_path / "zero-counts") as run:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            handle = handle_for(selected, backend, run)
        assert handle._items == 4
        submit_detached((handle,), run=run)
        chunk, = refresh_submissions(run=run)
        histogram = chunk.histogram()
        assert histogram.width == 3 + extra_classical_bits
        assert dict(zip(histogram.index_list(), histogram.weights.tolist())) == {
            0: 1, 1: 1, 2: 1, 3: 1,
        }
        assert chunk.returned_shots == 4
        assert run.trace.submissions[0].results_consumed and state.posts == 1


@pytest.mark.parametrize("payload", [[], "x", None, {}, {"histogram": []}, {"histogram": {}},
                                     {"histogram": {"registers": []}}])
def test_decode_refuses_a_malformed_v2_payload_as_provider_data(payload):
    """A v2 result without a histogram registers mapping is refused with ValueError, like other malformed histograms."""
    from dataclasses import replace
    from test_nexus_backend import IO

    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    native = replace(_array_route_native(), num_qubits=1, measurement_map=(0,))
    with pytest.raises(ValueError, match="IonQ v2 result requires a histogram registers mapping"):
        backend._decode(payload, artifact_format="ionq.result.histogram.json.v2", native=native, shots=1,
                        job_id="malformed", run=IO())


def _array_route_native():
    from nwqlib.backends.ionq import _PreparedIonQ

    return _PreparedIonQ(circuit=None, input=None, shots=100_000, measurement_map=(), classical_layout=(),
                         num_qubits=0, metadata={"observation": ObservationSpec(kind="counts", shots=4)})


@pytest.mark.parametrize(
    "qubits,measurement_map,keys,array_bytes",
    [
        (20, (0,), 100_000, 5_700_016),
        (20, (3, 7), 100_000, 5_700_032),
        (20, tuple(range(8)), 100_000, 5_702_048),
        (20, (19, 2, 5, 11), 100_000, 5_700_128),
        (4, tuple(range(4)), 16, 1_040),
        (12, tuple(range(12)), 4096, 266_240),
        # Distinct spellings of four native states: at most 2**q, not 2**c, distinct mapped indices.
        (2, (0, 1, 0, 1, None, None, None, None), ("0", "1", "2", "3", "00", "01"), 374),
        # Fewer provider entries than 2**min(q, c): at most m, not 2**64, distinct mapped indices.
        (64, tuple(range(64)), 16, 1_040),
    ],
)
def test_decode_admits_the_live_array_bound_of_the_uint64_route(qubits, measurement_map, keys, array_bytes):
    """Up to 64 bits the decoder checks the represented count keys and then 57*m + 8*min(m, 2**min(q, c)) bytes."""
    from dataclasses import replace
    from test_nexus_backend import IO
    from nwqlib.execution import ExecutionLimits

    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    native = replace(_array_route_native(), num_qubits=qubits, measurement_map=measurement_map)
    histogram = {key: 1 for key in keys} if isinstance(keys, tuple) else {str(state): 1 for state in range(keys)}
    outcomes = len(histogram)
    run = IO()
    run.limits = ExecutionLimits(max_data_bytes=max(array_bytes, outcomes * (len(measurement_map) + 16)))
    result = backend._decode(histogram, artifact_format="ionq.result.histogram.json.v1",
                             native=native, shots=outcomes, job_id="array", run=run)
    assert run.calls == [outcomes * (len(measurement_map) + 16), array_bytes]
    assert int(result.raw_output["counts"][1].sum()) == outcomes


def test_decode_refuses_below_the_array_bound_before_the_state_array(monkeypatch):
    from dataclasses import replace
    import numpy as np
    from test_nexus_backend import IO
    from nwqlib.execution import ExecutionLimits

    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=65536)
    native = replace(_array_route_native(), num_qubits=4, measurement_map=(0, 1, 2, 3))
    histogram = {str(state): 1 for state in range(16)}
    run = IO()
    run.limits = ExecutionLimits(max_data_bytes=1_039)
    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", lambda *a, **k: pytest.fail("state array allocated before admission"))
        with pytest.raises(ValueError, match="max_data_bytes"):
            backend._decode(histogram, artifact_format="ionq.result.histogram.json.v1", native=native, shots=16,
                            job_id="cap", run=run)
    run.limits = ExecutionLimits(max_data_bytes=1_040)
    result = backend._decode(histogram, artifact_format="ionq.result.histogram.json.v1", native=native, shots=16,
                             job_id="cap", run=run)
    assert result.raw_output["counts"][0].tolist() == list(range(16))


@pytest.mark.parametrize("outcomes,qubits,width", [(0, 1, 1), (-1, 1, 1), (1, -1, 1), (1, 65, 1),
                                                  (1, 1, -1), (1, 1, 65)])
def test_decode_byte_bound_refuses_arguments_outside_its_domain(outcomes, qubits, width):
    from nwqlib.backends.ionq import ionq_decode_bytes

    with pytest.raises(ValueError):
        ionq_decode_bytes(outcomes, qubits, width)


@pytest.mark.parametrize("excess", [0, 1, 200_000])
def test_response_reader_holds_at_most_one_byte_past_the_cap(monkeypatch, excess):
    """An oversized body is rejected after at most max_response_bytes + 1 bytes are read.

    A fixed 64 KiB read size lets the reader take up to 64 KiB - 1 bytes past
    the cap before it rejects the body, and the 200,000-byte excess detects
    that. A body of exactly the cap still parses, and a body one byte over it
    is rejected.
    """
    import requests

    cap = 100_000
    body = json.dumps({"id": "x" * (cap + excess - 10)}).encode()
    assert len(body) == cap + excess
    delivered = []

    class Response:
        """Serves ``body`` through both the streaming iterator and the raw reader, counting bytes."""

        status_code = 200

        def __init__(self):
            self.raw = self
            self.offset = 0

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def raise_for_status(self):
            pass

        def read(self, amt, *, decode_content):
            assert decode_content is True
            part = body[self.offset:self.offset + amt]
            self.offset += len(part)
            delivered.append(len(part))
            return part

        def iter_content(self, chunk_size):
            while part := self.read(chunk_size, decode_content=True):
                yield part

    monkeypatch.setenv("NWQLIB_IONQ_API_KEY", "injected-test-key")
    monkeypatch.setattr(requests, "request", lambda method, url, **kwargs: Response())
    backend = IonQBackend(device="qpu.forte-1", max_input_bytes=65536, max_response_bytes=cap)
    run = SimpleNamespace(check_data=lambda size: None)
    if excess == 0:
        assert backend._request("GET", "jobs/cap", run=run) == json.loads(body)
        assert sum(delivered) == cap
    else:
        with pytest.raises(ValueError, match="exceeds max_response_bytes"):
            backend._request("GET", "jobs/cap", run=run)
        assert sum(delivered) == cap + 1


@pytest.mark.parametrize("level", [0, 1, 2])
def test_qis_lowering_keeps_dense_unitaries_exact(api, tmp_path, monkeypatch, level):
    # Qiskit's own Rx/Ry/Rz/CX lowering of this near-identity unitary errs by
    # about 1e-6 per entry. The expected matrix is SciPy's exponential. Levels
    # 0 and 1 receive the exact synthesis; level 2 resynthesizes two-qubit
    # blocks and receives the logical circuit. The selected level reaches the
    # transpiler: two Rz rotations stay two at level 0 and merge into one above
    # it. A level other than the default 0 marks the receipt with the exclusion
    # "optimization_level".
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    state, original, _ = api
    backend = original.revise(gateset="qis", optimization_level=level)
    lowered = []
    convert = IonQBackend._convert
    monkeypatch.setattr(IonQBackend, "_convert", lambda self, circuit: lowered.append(circuit) or convert(self, circuit))
    synthesized, synthesize = [], Run._exact_dense_unitaries
    monkeypatch.setattr(Run, "_exact_dense_unitaries", lambda self, circuit: synthesized.append(circuit) or synthesize(self, circuit))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary = expm(-1e-6j * (x + x.conj().T) / 2)
    qc = QuantumCircuit(2, 2)
    qc.append(UnitaryGate(unitary), [0, 1])
    qc.measure([0, 1], [0, 1])
    _, selected = selection()
    with Run(selected, backend=backend, directory=tmp_path / "dense") as run:
        prep = backend.prepare(qc, observation=ObservationSpec(kind="counts", shots=4),
                               runtime=RuntimeOptions(seed=7), position=len(qc.data),
                               source_definitions=(), run=run, snapshot="dense")
        merge = QuantumCircuit(1, 1)
        merge.rz(0.1, 0)
        merge.rz(0.2, 0)
        merge.measure(0, 0)
        backend.prepare(merge, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=7),
                        position=len(merge.data), source_definitions=(), run=run, snapshot="merge")
    assert lowered[1].count_ops().get("rz", 0) == (2 if level == 0 else 1)
    assert len(synthesized) == (2 if level < 2 else 0)
    body = lowered[0].remove_final_measurements(inplace=False)
    if level < 2:
        assert np.abs(Operator(body).data - unitary).max() <= 1e-13
        assert prep.transformation.startswith("exact dense-unitary synthesis")
    else:
        assert prep.transformation.startswith("public QIS lowering")
    assert prep.probability_window_exclusions == (None if level == 0 else ("optimization_level",))
    if level:
        with pytest.raises(ValueError, match="optimization_level must be 0"):
            original.revise(gateset="native", optimization_level=level).prepare(
                qc, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=7),
                position=len(qc.data), source_definitions=(), run=run, snapshot="native")


def test_qis_receipt_records_the_final_layout_of_a_relabeled_lowering(api, tmp_path):
    # Without a coupling map, level 2 elides the two SWAPs and relabels the
    # wires after them. The receipt records Qiskit's final layout [1, 2, 0]
    # instead of the identity; the measurement map follows the relabeling.
    from qiskit import QuantumCircuit

    _, original, _ = api
    backend = original.revise(gateset="qis", optimization_level=2)
    qc = QuantumCircuit(3, 3)
    qc.h(0)
    qc.x(1)
    qc.swap(0, 1)
    qc.swap(1, 2)
    qc.measure([0, 1, 2], [0, 1, 2])
    _, selected = selection()
    with Run(selected, backend=backend, directory=tmp_path / "relabeled") as run:
        prep = backend.prepare(qc, observation=ObservationSpec(kind="counts", shots=4),
                               runtime=RuntimeOptions(seed=7), position=len(qc.data),
                               source_definitions=(), run=run, snapshot="relabeled")
    assert prep.logical_to_native == (1, 2, 0)
