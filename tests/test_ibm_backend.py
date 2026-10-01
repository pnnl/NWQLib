"""Real Runtime SDK objects with an injected API boundary; no quantum jobs."""

import nwqlib
from uuid import uuid4
from nwqlib._run_archive import load as load_run
from nwqlib.algorithms.expectation import ExpectationMethod
from test_expectation_current import problem

import json
from types import SimpleNamespace
import numpy as np
import pytest
from nwqlib.backends import IBMRuntimeBackend
from nwqlib.core.planning import ObservationSpec, RuntimeOptions
from nwqlib._prepared_execution import (
    Run,
    prepare_experiment as prepare,
    refresh_submissions,
    submit_detached,
)


def test_credentials_resolve_at_explicit_service_call_without_entering_records(monkeypatch, tmp_path):
    import qiskit_ibm_runtime

    calls = []
    monkeypatch.setattr(
        qiskit_ibm_runtime, "IBMQuantumComputeService", lambda **kwargs: calls.append(kwargs) or object()
    )
    monkeypatch.delenv("NWQLIB_IBM_RUNTIME_TOKEN", raising=False)
    backend = IBMRuntimeBackend(device="explicit-device", instance="explicit-instance", max_input_bytes=65536)
    assert not calls
    with pytest.raises(ValueError, match="credential environment variable"):
        backend._new_service()
    assert not calls
    for token in ("supplied-first-token", "supplied-renewed-token"):
        monkeypatch.setenv("NWQLIB_IBM_RUNTIME_TOKEN", token)
        backend._new_service()
        assert calls[-1] == dict(channel="ibm_quantum_platform", token=token, instance="explicit-instance")
        assert token not in backend.model_dump_json()
    backend.revise(account_name="explicit-saved-account")._new_service()
    assert calls[-1] == dict(name="explicit-saved-account", instance="explicit-instance")


@pytest.fixture
def runtime_api(monkeypatch):
    """Use real SDK containers with a synthetic service and reversed PUB order to test
    acquisition association offline.
    """
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, PubResult, SamplerPubResult
    from qiskit_ibm_runtime import IBMBackend, IBMQuantumComputeService, RuntimeEncoder
    from qiskit_ibm_runtime.fake_provider import FakeManilaV2

    instance = "crn:supplied-instance"
    fake = FakeManilaV2()
    configuration = fake.configuration()
    configuration.backend_name = "ibm_supplied"
    requests = []

    class API:
        _instance = instance
        status = "QUEUED"
        pubs = ()
        tags = ()
        reads = 0
        fail_status = False
        primitive = "sampler"
        estimate = 2.25
        standard_error = 0.125
        result_metadata = {"resilience": {"zne_mitigation": False}}
        container_mode = "dedicated"
        cancel_calls = 0
        fail_cancel = False
        transform_result = None

        def backend_status(self, name):
            return dict(
                backend_name=name, backend_version="1", operational=True, pending_jobs=0, status_msg="active"
            )

        def program_run(self, **kwargs):
            requests.append(kwargs)
            self.pubs = kwargs["params"]["pubs"]
            self.primitive = kwargs["program_id"]
            self.tags = kwargs["job_tags"]
            return {"id": "original-ibm-job", "backend": "ibm_supplied"}

        def job_get(self, job_id, **kwargs):
            if self.fail_status:
                raise ConnectionError("supplied temporary status failure")
            return dict(
                id=job_id,
                backend="ibm_supplied",
                program={"id": self.primitive},
                state={"status": self.status},
                tags=self.tags,
            )

        def session_details(self, session_id):
            return dict(id=session_id, backend_name="ibm_supplied", mode=self.container_mode, state="open")

        def job_cancel(self, job_id):
            assert job_id == "original-ibm-job"
            self.cancel_calls += 1
            if self.fail_cancel:
                raise ConnectionError("supplied interrupted cancellation")

        def job_results(self, job_id):
            self.reads += 1
            result = []
            for index, pub in enumerate(self.pubs):
                if self.primitive == "estimator":
                    result.append(
                        PubResult(
                            DataBin(evs=np.asarray(self.estimate), stds=np.asarray(self.standard_error)),
                            metadata={
                                "circuit_metadata": pub.circuit.metadata,
                                "target_precision": pub.precision,
                                "shots": 4096,
                                "num_randomizations": 32,
                            },
                        )
                    )
                    continue
                bits = ["0", "0", "0", "1"] if index == 0 else ["0", "1", "1", "1"]
                result.append(
                    SamplerPubResult(
                        DataBin(readout=BitArray.from_samples(bits)),
                        metadata={"circuit_metadata": pub.circuit.metadata, "shots": 4},
                    )
                )
            output = PrimitiveResult(result[::-1], metadata=self.result_metadata)
            if self.transform_result is not None:
                output = self.transform_result(output)
            return json.dumps(output, cls=RuntimeEncoder)

    api = API()
    service = object.__new__(IBMQuantumComputeService)
    service._channel = "ibm_quantum_platform"
    service._api_clients = {instance: api}
    service._active_api_client = api
    service._check_instance_usage = lambda: None
    backend = IBMBackend(configuration, service=service, api_client=api, instance=instance)
    backend._target, backend._properties = (fake.target, fake.properties())
    service.backend = lambda name, **kwargs: backend
    service._create_backend_obj = lambda name, **kwargs: backend
    monkeypatch.setattr(IBMRuntimeBackend, "_new_service", lambda self: service)
    connection = IBMRuntimeBackend(
        device=backend.name, instance=instance, max_input_bytes=65536, initial_layout=(2,)
    )
    return SimpleNamespace(
        connection=connection, api=api, service=service, backend=backend, requests=requests
    )


@pytest.mark.parametrize("primitive", ["sampler", "estimator"])
@pytest.mark.parametrize("failure_kind", ["decoder", "invalid"])
def test_later_pub_failure_preserves_prior_result_and_skips_its_decode_on_resume(
    tmp_path, monkeypatch, runtime_api, primitive, failure_kind
):
    """Fail a later PUB after one completes, then poison repeat decoding and preserve the
    original exception cause.
    """
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, PubResult, SamplerPubResult

    if primitive == "sampler":
        selected = counts_selection()
    else:
        selected = estimate_selection()
    api, backend = (runtime_api.api, runtime_api.connection)
    cause, failure = (OSError("original PUB decoder cause"), ConnectionError("original late PUB error"))
    failure.__cause__ = cause
    failing, consumed = (True, False)
    decoded_indices = []

    def transform(result):
        pubs = []
        for pub in result:
            index = pub.metadata["circuit_metadata"]["nwqlib_pub"]
            if primitive == "estimator":
                error = -0.125 if failing and failure_kind == "invalid" and (index == 0) else 0.125
                pub = PubResult(
                    DataBin(evs=np.asarray(0.75 if index == 0 else 1.25), stds=np.asarray(error)),
                    metadata=pub.metadata,
                )
            elif failing and failure_kind == "invalid" and (index == 0):
                pub = SamplerPubResult(
                    DataBin(readout=BitArray.from_samples(["0"] * 5)), metadata=pub.metadata
                )
            pubs.append(pub)
        return PrimitiveResult(pubs, metadata=result.metadata)

    api.transform_result = transform

    def visit(pub):
        index = pub.metadata["circuit_metadata"]["nwqlib_pub"]
        assert not (consumed and index == 1), "completed PUB was decoded again"
        decoded_indices.append(index)
        if failing and failure_kind == "decoder" and (index == 0):
            raise failure

    if primitive == "sampler":
        join = SamplerPubResult.join_data

        def join_data(self, *args, **kwargs):
            visit(self)
            return join(self, *args, **kwargs)

        monkeypatch.setattr(SamplerPubResult, "join_data", join_data)
    else:
        estimate = IBMRuntimeBackend._estimate_result

        def estimate_result(self, pub, *args, **kwargs):
            visit(pub)
            return estimate(self, pub, *args, **kwargs)

        monkeypatch.setattr(IBMRuntimeBackend, "_estimate_result", estimate_result)
    path = tmp_path / "partial-pubs.sqlite"
    with Run(selected, backend=backend, directory=path) as run:
        handles = tuple(
            (
                prepare(
                    selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=seed)
                )
                for seed in (7, 8)
            )
        )
        receipt = submit_detached(handles, run=run)
        api.status = "COMPLETED"
        with pytest.raises(ConnectionError if failure_kind == "decoder" else ValueError) as caught:
            refresh_submissions(run=run)
        if failure_kind == "decoder":
            assert caught.value is failure and caught.value.__cause__ is cause
        else:
            assert (
                "shot cap" in str(caught.value)
                if primitive == "sampler"
                else "finite and nonnegative" in str(caught.value)
            )
        prior = run.completed_observation(receipt.items[1].attempt)
        assert prior.job == receipt.locator.job_id and prior.chunk == "1"
        if primitive == "sampler":
            histogram = prior.histogram()
            assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 1, 1: 3}
            assert prior.returned_shots == 4
        else:
            assert prior.values[0].value == 1.25 and prior.values[0].standard_error == 0.125
            assert prior.returned_shots is None
        assert run.collect(prior)
        assert not run.trace.submissions[0].results_consumed
        assert receipt.locator.job_id in run._state["backend_context"]["jobs"]
        assert run.trace.preparations == 2 and run.trace.jobs == 1 and (len(runtime_api.requests) == 1)
        assert api.reads == 1
    failing, consumed = (False, True)
    visited = len(decoded_indices)
    with load_run(path, backend=backend) as run:
        if primitive == "sampler" and failure_kind == "decoder":
            for defect in ("missing", "duplicate", "swapped_snapshot"):

                def corrupt(result):
                    pubs = list(result)
                    if defect == "missing":
                        return PrimitiveResult(pubs[:1], metadata=result.metadata)
                    first, second = pubs
                    owner = dict(first.metadata["circuit_metadata"])
                    owner["nwqlib_pub" if defect == "duplicate" else "nwqlib_snapshot"] = (
                        second.metadata["circuit_metadata"]["nwqlib_pub"]
                        if defect == "duplicate"
                        else second.metadata["circuit_metadata"]["nwqlib_snapshot"]
                    )
                    pubs[0] = SamplerPubResult(
                        first.data, metadata={**first.metadata, "circuit_metadata": owner}
                    )
                    return PrimitiveResult(pubs, metadata=result.metadata)

                api.transform_result = corrupt
                with pytest.raises(ValueError, match="PUB"):
                    refresh_submissions(run=run)
                assert run.completed_observation(receipt.items[1].attempt) == prior
            api.transform_result = transform
        reads = api.reads
        (remaining,) = refresh_submissions(run=run)
        assert remaining.chunk == "0" and remaining.job == receipt.locator.job_id
        assert run.collect(remaining) and (not run.collect(prior))
        assert decoded_indices[visited:] == [0]
        assert api.reads == reads + 1
        assert receipt.locator.job_id not in run._state["backend_context"]["jobs"]
        assert run.trace.preparations == 2 and run.trace.jobs == 1 and (len(runtime_api.requests) == 1)
        if primitive == "sampler":
            assert sum((event.returned_shots for event in run.trace.events)) == 8
        else:
            assert remaining.values[0].value == 0.75 and remaining.values[0].standard_error == 0.125
            assert run.exposure["completed"]["provider_managed_sampling"] == 2
    reads = api.reads
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run) and api.reads == reads
        assert len(run._state["by_attempt"]) == 2 and run.trace.jobs == 1


def test_experimental_options_cannot_override_admitted_readout_or_uncertainty(runtime_api, tmp_path):
    override = {"execution": {"meas_type": "kerneled"}}
    connection = runtime_api.connection.revise(options_json=json.dumps({"experimental": override}))
    run = Run(counts_selection(), backend=connection, directory=tmp_path / str(uuid4()))
    with pytest.raises(ValueError, match="experimental"):
        connection._options(run, "counts")
    run.close()


def test_stored_options_json_is_the_utf8_text_that_admission_measured(runtime_api, tmp_path):
    # A character outside the Basic Multilingual Plane is 4 UTF-8 bytes but a
    # 12-byte surrogate-pair escape in ASCII JSON. The stored identity must be
    # the UTF-8 text whose size the journal encoder admitted.
    tag = "\U0001d11e"
    connection = runtime_api.connection.revise(
        options_json=json.dumps({"environment": {"job_tags": [tag]}}))
    run = Run(counts_selection(), backend=connection, directory=tmp_path / str(uuid4()))
    _, stored = connection._options(run, "counts")
    run.close()
    assert tag in stored and "\\ud834" not in stored
    assert json.loads(stored)["environment"]["job_tags"] == [tag]


@pytest.mark.parametrize("mode", ["batch", "session"])
@pytest.mark.parametrize("lost_cancel_ack", [False, True])
def test_explicit_existing_container_does_not_create_session_and_cancel_is_not_refunded(
    tmp_path, monkeypatch, runtime_api, mode, lost_cancel_ack
):
    """Reuse the supplied container and preserve charged work whether cancellation succeeds or
    loses acknowledgement.
    """
    selection = counts_selection()
    api = runtime_api.api
    api.container_mode = "batch" if mode == "batch" else "dedicated"
    api.fail_cancel = lost_cancel_ack
    api.status = "RUNNING"
    connection = runtime_api.connection.revise(mode=mode, container_id="original-container")
    path = tmp_path / "container.sqlite"
    with Run(selection, backend=connection, directory=path) as run:
        handle = prepare(
            selection.resolve(selection.experiments[0].name), run=run, runtime=RuntimeOptions(seed=7)
        )
        original = submit_detached((handle,), run=run)
        (request,) = runtime_api.requests
        assert request["session_id"] == "original-container" and request["start_session"] is False
        if lost_cancel_ack:
            with pytest.raises(ConnectionError, match="interrupted cancellation"):
                run.cancel(reason="user requested cancellation")
        else:
            run.cancel(reason="user requested cancellation")
        (cancelled,) = run.trace.submissions
        assert (
            cancelled.cancel_requested == "user requested cancellation"
            and cancelled.locator == original.locator
        )
        if lost_cancel_ack:
            assert cancelled.status == "acknowledged" and (not cancelled.results_consumed)
            assert "interrupted cancellation" in cancelled.refresh_failure
        else:
            assert cancelled.status == "acknowledged" and (not cancelled.results_consumed)
            assert cancelled.provider_status == "RUNNING"
        assert api.cancel_calls == 1 and api.reads == 0 and (run.trace.jobs == 1)
        assert sum((event.shots for event in run.trace.events)) == 4
    monkeypatch.setattr(
        IBMRuntimeBackend, "launch", lambda *a, **k: pytest.fail("cancelled job was resubmitted")
    )
    with load_run(path, backend=connection) as run:
        api.status = "CANCELLED"
        assert not refresh_submissions(run=run) and api.cancel_calls == 1
        assert run.trace.jobs == 1 and run.trace.submissions[0].locator == original.locator
        assert run.trace.submissions[0].status == "cancelled" and run.trace.submissions[0].results_consumed


@pytest.mark.parametrize("submission_response", ["lost_ack", "rejected_503"])
def test_actual_http_transport_preserves_one_accepted_job_and_reconciles_original_tag(
    tmp_path, monkeypatch, runtime_api, submission_response
):
    """A loopback HTTP server distinguishes an accepted lost acknowledgement from a rejected
    request eligible for retry.
    """
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from urllib.parse import parse_qs, urlsplit
    from qiskit_ibm_runtime import RuntimeDecoder
    from qiskit.primitives.containers.sampler_pub import SamplerPub
    from qiskit_ibm_runtime.api.clients.runtime import RuntimeClient
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError

    api = runtime_api.api
    exchanges, accepted, post_bodies = ([], [], [])
    fail_get = True
    status_data = dict(
        id="original-ibm-job",
        backend="ibm_supplied",
        program={"id": "sampler"},
        state={"status": "COMPLETED"},
    )
    make_results = api.job_results

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, code, data):
            payload = data.encode() if isinstance(data, str) else json.dumps(data).encode()
            self.send_response(code)
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self):
            wire = self.rfile.read(int(self.headers["Content-Length"]))
            post_bodies.append(wire)
            body = json.loads(wire, cls=RuntimeDecoder)
            exchanges.append(("POST", self.path, body))
            assert self.path == "/jobs"
            if submission_response == "rejected_503" and len(exchanges) == 1:
                self.respond(503, {"error": {"message": "not accepted", "code": "supplied"}})
                return
            accepted.append(body)
            api.pubs = tuple((SamplerPub.coerce(pub) for pub in body["params"]["pubs"]))
            api.tags = body["tags"]
            status_data["tags"] = api.tags
            if submission_response == "lost_ack":
                self.close_connection = True
                return
            self.respond(200, {"id": "original-ibm-job", "backend": "ibm_supplied"})

        def do_GET(self):
            nonlocal fail_get
            path, query = (urlsplit(self.path).path, parse_qs(urlsplit(self.path).query))
            exchanges.append(("GET", path, query))
            if fail_get:
                fail_get = False
                self.respond(503, {"error": {"message": "temporary read failure", "code": "supplied"}})
            elif path == "/jobs":
                assert query["tags"] == [api.tags[-1]] and query["limit"] == ["2"]
                self.respond(200, {"jobs": [status_data], "count": 1})
            elif path.endswith("/results"):
                self.respond(200, make_results("original-ibm-job"))
            else:
                assert path == "/jobs/original-ibm-job"
                self.respond(200, status_data)

    server = HTTPServer(("127.0.0.1", 0), Handler)
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        params = SimpleNamespace(
            get_runtime_api_base_url=lambda: f"http://127.0.0.1:{server.server_port}",
            get_auth_handler=lambda: None,
            connection_parameters=lambda: {},
            instance=runtime_api.connection.instance,
        )
        transport = RuntimeClient(params)
        for name in ("program_run", "jobs_get", "job_get", "job_results"):
            monkeypatch.setattr(api, name, getattr(transport, name), raising=False)
        selection = counts_selection()
        path = tmp_path / "http.sqlite"
        with Run(selection, backend=runtime_api.connection, directory=path) as run:
            handle = prepare(
                selection.resolve(selection.experiments[0].name), run=run, runtime=RuntimeOptions(seed=7)
            )
            if submission_response == "lost_ack":
                with pytest.raises(IBMRuntimeError):
                    submit_detached((handle,), run=run)
                assert run.trace.submissions[0].locator is None
            else:
                submit_detached((handle,), run=run)
            original = run.trace.submissions[0]
            assert len(accepted) == 1 and original.items[0].prepared_id == handle.record.content_id
        with load_run(path, backend=runtime_api.connection) as run:
            chunks = refresh_submissions(run=run)
            assert len(chunks) == 1, run.trace.submissions
            assert chunks[0].job == "original-ibm-job" and chunks[0].prepared_id == handle.record.content_id
            assert run.trace.submissions[0].submission_id == original.submission_id
            assert run.trace.jobs == 1 and len(accepted) == 1
            assert not refresh_submissions(run=run)
        posts = [body for method, _, body in exchanges if method == "POST"]
        assert len(posts) == (1 if submission_response == "lost_ack" else 2)
        assert all((body == post_bodies[0] for body in post_bodies))
        assert len([1 for method, _, _ in exchanges if method == "GET"]) >= 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.mark.parametrize("reuse", [False, True])
def test_actual_sdk_sampler_pub_mapping_and_original_job_resume(tmp_path, monkeypatch, runtime_api, reuse):
    """Reordered PUBs and a reused prepared handle must preserve distinct job-item populations
    after reopening.
    """
    from nwqlib._run_journal import LocalJournal

    connection, api = (runtime_api.connection, runtime_api.api)
    selection = counts_selection()
    path = tmp_path / "ibm.sqlite"
    with Run(selection, backend=connection, directory=path) as run:
        handles = tuple(
            (
                prepare(
                    selection.resolve(selection.experiments[0].name),
                    run=run,
                    runtime=RuntimeOptions(seed=seed),
                )
                for seed in ((7,) if reuse else (7, 8))
            )
        )
        if reuse:
            handles = handles * 2
        assert all((handle.record.logical_to_native == (2,) for handle in handles))
        submitted = submit_detached(handles, run=run)
        assert submitted.locator.job_id == "original-ibm-job" and run.trace.jobs == 1
        assert len(runtime_api.requests) == 1 and len(runtime_api.requests[0]["params"]["pubs"]) == 2
        assert "nwqlib_pub" not in handles[0]._native.circuit.metadata
        assert "nwqlib:" + submitted.submission_id in api.tags
        assert not refresh_submissions(run=run) and api.reads == 0
    plan = selection

    def forbidden(*args, **kwargs):
        raise AssertionError("existing IBM job triggered native preparation, QPY load or new launch")

    monkeypatch.setattr(IBMRuntimeBackend, "prepare", forbidden)
    monkeypatch.setattr(IBMRuntimeBackend, "launch", forbidden)
    monkeypatch.setattr(LocalJournal, "read_payload", forbidden)
    with load_run(path, backend=connection) as run:
        api.fail_status = True
        with pytest.raises(ConnectionError, match="temporary status failure"):
            refresh_submissions(run=run)
        assert run.trace.submissions[0].locator == submitted.locator
        assert run.trace.submissions[0].refresh_failure is not None
        api.fail_status, api.status = (False, "COMPLETED")
        incoming = refresh_submissions(run=run)
        assert len(incoming) == 2, run.trace.submissions
        mapped = {
            chunk.chunk: (chunk.prepared_id, chunk.histogram().width,
                          dict(zip(chunk.histogram().index_list(), chunk.histogram().weights.tolist())))
            for chunk in incoming
        }
        assert mapped["0"] == (handles[0].record.content_id, 1, {0: 3, 1: 1})
        assert mapped["1"] == (handles[1].record.content_id, 1, {0: 1, 1: 3})
        for chunk in incoming:
            run.collect(chunk)
        assert run.trace.jobs == 1 and sum((event.shots for event in run.trace.events)) == 8
        assert not refresh_submissions(run=run) and api.reads == 1 and (len(runtime_api.requests) == 1)
        from nwqlib.algorithms.expectation import ExpectationAnalysis

        result = run.plan.method.analyze(run.plan, run.data, settings={})
        population = result.statistics.populations[0].population
        assert population.zeros == population.ones == 4 and len(population.source_ids) == 2
        assert result.statistics.populations[0].raw_mean == 0
        assert result.statistics.raw_value == result.value == 1
        assert population.preparation_ids == tuple(
            ((chunk.prepared_id,) for chunk in run.observations.chunks)
        )
        restored = ExpectationAnalysis.model_validate_json(result.model_dump_json())
        restored.validate_plan(plan)


def test_sampler_joint_registers_preserve_shot_alignment_and_logical_bit_order(runtime_api, tmp_path):
    """Interleaved classical registers require shot-aligned joint outcomes, which independent
    marginals cannot reconstruct.
    """
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

    selection = counts_selection()
    connection = runtime_api.connection.revise(initial_layout=(2, 0))
    run = Run(selection, backend=connection, directory=tmp_path / str(uuid4()))
    alpha, beta = (ClassicalRegister(2, "alpha"), ClassicalRegister(1, "beta"))
    circuit = QuantumCircuit(QuantumRegister(2, "q"))
    circuit.add_bits([alpha[1], beta[0], alpha[0]])
    circuit.add_register(alpha, beta)
    circuit.h(0)
    circuit.cx(0, 1)
    circuit.x(1)
    circuit.measure(0, alpha[0])
    circuit.measure(1, alpha[1])
    circuit.measure(0, beta[0])
    prepared = connection.prepare(
        circuit,
        observation=ObservationSpec(kind="counts", shots=4),
        runtime=RuntimeOptions(seed=7),
        position=len(circuit.data),
        source_definitions=(),
        run=run,
        snapshot="joint-pub",
    )
    result = SamplerPubResult(
        DataBin(
            alpha=BitArray.from_samples(["01", "10", "01", "10"]),
            beta=BitArray.from_samples(["1", "0", "1", "0"]),
        ),
        metadata={"circuit_metadata": {"nwqlib_snapshot": "joint-pub", "nwqlib_pub": 0}},
    )
    decoded = tuple(
        connection._decode(PrimitiveResult([result]), (prepared.native,), job_id="supplied-joint", run=run)
    )
    indices, counts = decoded[0][1].raw_output["counts"]
    assert (indices.dtype, counts.dtype) == ("uint64", "int64")
    assert dict(zip(indices.tolist(), counts.tolist())) == {0b001: 2, 0b110: 2}
    marginals = SamplerPubResult(
        DataBin(
            alpha=SimpleNamespace(get_counts=lambda: {"01": 2, "10": 2}),
            beta=SimpleNamespace(get_counts=lambda: {"1": 2, "0": 2}),
        ),
        metadata=result.metadata,
    )
    with pytest.raises(ValueError, match="scalar parameter coordinate"):
        tuple(
            connection._decode(
                PrimitiveResult([marginals]), (prepared.native,), job_id="supplied-joint", run=run
            )
        )


@pytest.mark.parametrize(
    ("widths", "shots", "charge"),
    [
        pytest.param((1,), 8192, 278528, id="pair-packing-one-bit"),
        pytest.param((4,), 16, 832, id="pair-unique"),
        pytest.param((4, 5), 4096, 147456, id="pair-packing-two-registers"),
        pytest.param((64,), 5, 680, id="join-concatenate-aligned"),
        pytest.param((3, 7, 2), 100000, 3900096, id="join-concatenate-with-padding"),
        pytest.param((33,), 5, 511, id="join-padded-repack"),
        pytest.param((65,), 16, 2576, id="wide-join-padded-repack"),
        pytest.param((32, 40), 16, 2448, id="wide-join-concatenate-registers"),
    ],
)
def test_sampler_decode_admits_the_join_and_pair_array_bound(
    runtime_api, tmp_path, monkeypatch, widths, shots, charge
):
    """Admit the live-array law derived in sampler_decode_bytes before joining."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.primitives.containers import BitArray, DataBin, PrimitiveResult, SamplerPubResult

    connection = runtime_api.connection
    run = Run(counts_selection(), backend=connection, directory=tmp_path / str(uuid4()))
    registers = tuple(ClassicalRegister(width, f"c{index}") for index, width in enumerate(widths))
    circuit = QuantumCircuit(QuantumRegister(1, "q"), *registers)
    circuit.h(0)
    for register in registers:
        for bit in register:
            circuit.measure(0, bit)
    prepared = connection.prepare(
        circuit, observation=ObservationSpec(kind="counts", shots=shots),
        runtime=RuntimeOptions(seed=7), position=len(circuit.data), source_definitions=(),
        run=run, snapshot="charged-pub",
    )
    rng = np.random.default_rng(3)
    result = SamplerPubResult(
        DataBin(**{
            register.name: BitArray.from_bool_array(
                rng.integers(0, 2, (shots, len(register))).astype(bool))
            for register in registers
        }),
        metadata={"circuit_metadata": {"nwqlib_snapshot": "charged-pub", "nwqlib_pub": 0}},
    )
    charges = []
    check = type(run).check_data
    monkeypatch.setattr(type(run), "check_data",
                        lambda self, value: (charges.append(value), check(self, value))[1])
    join_data = SamplerPubResult.join_data

    def join_after_admission(pub, names=None):
        assert charges == [charge]
        return join_data(pub, names)

    monkeypatch.setattr(SamplerPubResult, "join_data", join_after_admission)
    try:
        decoded = tuple(connection._decode(
            PrimitiveResult([result]), (prepared.native,), job_id="charged", run=run))
        assert charges == [charge]
        raw = decoded[0][1].raw_output["counts"]
        counts = raw[1] if isinstance(raw, tuple) else raw.values()
        assert sum(int(count) for count in counts) == shots
    finally:
        run.close()


def test_estimator_decoder_keeps_missing_uncertainty_and_rejects_invalid_or_foreign_data(
    runtime_api, tmp_path
):
    """Missing uncertainty is legal, but negative uncertainty, vector-shaped estimates and
    foreign coordinates must fail.
    """
    from qiskit.primitives.containers import DataBin, PrimitiveResult, PubResult
    from nwqlib.core.planning import ObservableEstimateSpec

    selection = counts_selection()
    spec = ObservationSpec(
        kind="estimated_observable",
        estimate=ObservableEstimateSpec(
            observable_id=selection.problem.observable.manifest.content_id,
            labels=("I", "Z"),
            coefficients=(1.0, 1.0),
            precision=0.1,
        ),
    )
    native = SimpleNamespace(
        snapshot="supplied-estimate",
        metadata=dict(observation=spec, requested_options_json='{"resilience_level":0}'),
    )
    run = Run(selection, backend=runtime_api.connection, directory=tmp_path / str(uuid4()))
    metadata = dict(
        circuit_metadata={"nwqlib_pub": 0, "nwqlib_snapshot": native.snapshot}, target_precision=0.1
    )

    def decode(data, meta=metadata):
        ((_, result),) = runtime_api.connection._decode(
            PrimitiveResult([PubResult(data, metadata=meta)]), (native,), job_id="supplied-job", run=run
        )
        return result

    absent = decode(DataBin(evs=np.asarray(2.25))).raw_output["estimates"][0]
    assert absent.value == 2.25 and absent.standard_error is None and absent.uncertainty_unavailable
    for error in (-1e-20, float("inf")):
        with pytest.raises(ValueError, match="finite and nonnegative"):
            decode(DataBin(evs=np.asarray(2.25), stds=np.asarray(error)))
    with pytest.raises(ValueError, match="scalar observable"):
        decode(DataBin(evs=np.asarray([2.25]), stds=np.asarray([0.125])))
    with pytest.raises(ValueError, match="PUB coordinate"):
        decode(DataBin(evs=np.asarray(2.25)), {**metadata, "circuit_metadata": {"nwqlib_pub": 1}})
    with pytest.raises(ValueError, match="target precision"):
        decode(DataBin(evs=np.asarray(2.25)), {**metadata, "target_precision": 0.2})
    assert run.trace.jobs == 0 and run.observations.chunks == ()


@pytest.mark.parametrize(
    "value", [float.fromhex("0x1.fffffffffffffp+1023"), float.fromhex("0x0.0000000000001p-1022")]
)
def test_repeated_provider_point_preserves_constant_finite_extremes(value, tmp_path):
    from nwqlib.algorithms.expectation import _provider_mean

    assert _provider_mean((SimpleNamespace(value=value),) * 3) == value


def counts_selection():
    return nwqlib.plan(problem(), method=ExpectationMethod(), shots=4, seed=7)


def estimate_selection(coefficient=1.0):
    return nwqlib.plan(
        problem((("I", 1.0), ("Z", coefficient))), method=ExpectationMethod(estimate_precision=0.1), seed=7
    )


@pytest.mark.parametrize("zne", [False, True])
@pytest.mark.parametrize("quadratic", [False, True])
def test_estimator_weighted_observable_layout_uncertainty_and_saved_resume(
    tmp_path, monkeypatch, runtime_api, zne, quadratic
):
    """Map the weighted observable to physical qubits and distinguish ZNE fit uncertainty from
    sampling standard error.
    """
    coefficient = 1.0 if zne else 1e-12
    plan = estimate_selection(coefficient)
    scale = 4 if quadratic else 1
    if quadratic:
        p = nwqlib.Expectation(state=[2.0, 0.0], observable=plan.problem.observable)
        plan = nwqlib.plan(
            p,
            method=ExpectationMethod(estimate_precision=0.1),
            output=nwqlib.QuadraticForm(observable=p.observable),
            seed=7,
        )
    spec = plan.experiments[0].observation
    assert spec.estimate.labels == ("I", "Z") and spec.estimate.coefficients == (1.0, coefficient)
    api = runtime_api.api
    api.estimate = -0.25 if zne else 1.0
    api.standard_error = 0.125 if zne else coefficient / 64
    api.result_metadata = {"resilience": {"zne_mitigation": zne}}
    backend = runtime_api.connection.revise(options_json='{"resilience_level":2}' if zne else "{}")
    path = tmp_path / "estimate"
    with Run(plan, backend=backend, directory=path) as run:
        if not zne:

            def interrupted(*args, **kwargs):
                raise RuntimeError("before submission intent")

            with monkeypatch.context() as patch:
                patch.setattr(Run, "_begin_submission", interrupted)
                with pytest.raises(RuntimeError, match="before submission intent"):
                    run.resume()
            assert run.trace.preparations == 1 and run.trace.jobs == 0
        else:
            run.resume()
    if not zne:
        with load_run(path, backend=backend) as run, monkeypatch.context() as patch:
            patch.setattr(
                IBMRuntimeBackend,
                "prepare",
                lambda *a, **k: pytest.fail("recompiled original preparation"),
            )
            run.resume()
            assert run.trace.preparations == run.trace.jobs == 1
    with load_run(path, backend=backend) as run:
        (pub,) = runtime_api.requests[0]["params"]["pubs"]
        assert pub.precision == 0.1
        assert pub.observables.tolist() == {"IIIII": 1.0, "IIZII": coefficient}
        assert runtime_api.requests[0]["params"]["resilience_level"] == (2 if zne else 0)
        original = run.trace.submissions[0].locator

    def forbidden(*args, **kwargs):
        pytest.fail("pending Estimator restored a circuit, replanned or resubmitted")

    for name in ("prepare", "launch", "_observable"):
        monkeypatch.setattr(IBMRuntimeBackend, name, forbidden)
    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    api.status = "COMPLETED"
    with load_run(path, backend=backend) as run:
        result = run.wait(timeout=1, poll_interval=0)
        assert result.value == scale * api.estimate and result.statistics is None
        (estimate,) = result.estimates
        assert estimate.value == api.estimate
        (view,) = result.provider_output_estimates
        assert view["source_id"] == estimate.content_id and view["value"] == result.value
        if zne:
            assert (
                estimate.standard_error is None
                and "fit uncertainty" in estimate.uncertainty_unavailable
            )
            assert (
                view["standard_error"] is None
                and "fit uncertainty" in view["uncertainty_unavailable"]
            )
        else:
            assert (
                estimate.standard_error == api.standard_error
                and estimate.uncertainty_unavailable is None
            )
            assert view["standard_error"] == scale * api.standard_error
        assert (
            json.loads(estimate.provider_metadata_json)["reported_data"]["stds"]
            == api.standard_error
        )
        assert result.data.trace.submissions[0].locator == original
        assert result.data.observations.chunks[0].returned_shots is None
        assert run.exposure["completed"]["provider_managed_sampling"] == 1
        assert run.wait() is result
        saved = result.save(tmp_path / "saved-result")
        assert nwqlib.load_result(saved).content_id == result.content_id
    assert api.reads == 1 and len(runtime_api.requests) == 1


def test_public_cloud_run_reopens_original_job_without_planning(tmp_path, monkeypatch, runtime_api):
    from nwqlib.execution import ExecutionLimits

    # Select 64 measured shots of I+Z on |+>. The synthetic sampler returns three
    # zeros in four shots, so the reconstructed value is 1 + (3 - 1)/4 = 1.5.
    selected = nwqlib.plan(
        problem(state=[1.0, 1.0]), method=ExpectationMethod(), shots=64, seed=7
    )
    directory = tmp_path / "cloud"
    limits = ExecutionLimits(max_total_circuits=1, max_total_shots=64, max_data_bytes=8_000_000)
    prepared = nwqlib.prepare(
        selected, backend=runtime_api.connection, directory=directory, limits=limits
    )
    with nwqlib.submit(prepared) as run:
        assert run.result is None and run.trace.jobs == 1
        original = run.plan.content_id
    monkeypatch.setattr(ExpectationMethod, "plan", lambda *a, **k: pytest.fail("resume replanned"))
    runtime_api.api.status = "COMPLETED"
    # The second reopening finds the completed result without another provider read.
    for _ in range(2):
        with nwqlib.load_run(directory, backend=runtime_api.connection) as run:
            run.resume()
            assert run.plan.content_id == original and run.result.value == 1.5
            assert run.trace.jobs == 1 and runtime_api.api.reads == 1
    assert len(runtime_api.requests) == 1


@pytest.mark.parametrize("level", [0, 1, 2])
def test_isa_preparation_uses_exact_dense_synthesis_below_level_two(runtime_api, tmp_path, level):
    # Qiskit's own synthesis of this near-identity unitary errs by about 7e-7
    # per entry. Levels 0 and 1 receive the exact synthesis. Level 2
    # resynthesizes two-qubit blocks itself and receives the logical circuit
    # unchanged. The expected operator is SciPy's exponential on the two
    # logical qubits, identity on the three ancillas.
    # A level other than the default 1 marks the receipt with the exclusion
    # "optimization_level".
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    connection = runtime_api.connection.revise(optimization_level=level, initial_layout=(0, 1))
    run = Run(counts_selection(), backend=connection, directory=tmp_path / str(uuid4()))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary = expm(-1e-6j * (x + x.conj().T) / 2)
    circuit = QuantumCircuit(2, 2)
    circuit.append(UnitaryGate(unitary), [0, 1])
    circuit.measure([0, 1], [0, 1])
    prepared = connection.prepare(circuit, observation=ObservationSpec(kind="counts", shots=4),
                                  runtime=RuntimeOptions(seed=7), position=len(circuit.data),
                                  source_definitions=(), run=run, snapshot="dense")
    body = prepared.native.circuit.remove_final_measurements(inplace=False)
    error = np.abs(Operator.from_circuit(body).data - np.kron(np.eye(8), unitary)).max()
    if level < 2:
        assert error <= 1e-13
        assert prepared.transformation.startswith("exact dense-unitary synthesis")
    else:
        assert prepared.transformation.startswith("IBM ISA circuit")
    assert prepared.probability_window_exclusions == (None if level == 1 else ("optimization_level",))
