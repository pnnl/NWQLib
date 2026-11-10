"""Real Runtime SDK objects with an injected API boundary; no quantum jobs."""

import nwqlib
from uuid import uuid4
from nwqlib._run_archive import load as load_run
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.algorithms.lchs import PTSC, SampledPairs, Trotter, TrotterErrorBudgeted
from test_expectation_current import problem

import json
from types import SimpleNamespace
import numpy as np
import pytest
from conftest import executor_items, executor_program, executor_result_json, offline_ibm_service
from nwqlib.backends import IBMRuntimeBackend
from nwqlib.core.planning import ObservationSpec, RuntimeOptions
from nwqlib._prepared_execution import (
    Run,
    prepare_experiment as prepare,
    refresh_submissions,
    submit_detached,
)


def template_row_outcome(values, width, register):
    """The outcome that the synthetic service returns in every shot of one binding row.

    It is a function of the row's parameter values only, so an experiment's
    saved counts show which row of values produced them.
    """
    from hashlib import sha256

    digest = sha256(np.asarray(values, dtype=np.float64).tobytes() + bytes([register])).digest()
    return int.from_bytes(digest[:8], "little") % (1 << width)


def counts_outcomes(index, item, register_index, register, shape, shots):
    """The outcome of each shot that the synthetic service of ``runtime_api`` returns for one Sampler register.

    Each row of parameter values repeats ``template_row_outcome`` of its
    values. Without parameter values, item 0 returns the outcomes 0, 0, 0, 1
    and every other item 0, 1, 1, 1, repeated to the requested shots.
    """
    values = getattr(item, "circuit_arguments", None)
    if values is not None and values.ndim == 2 and values.shape[0]:
        return np.array([[template_row_outcome(row, register.size, register_index)] for row in values])
    return np.resize([0, 0, 0, 1] if index == 0 else [0, 1, 1, 1], shots)


@pytest.fixture
def runtime_api(monkeypatch):
    """Use the real SDK primitives and decoders with a synthetic executor service that returns the PUBs in
    reverse order, to test acquisition association offline.

    Sampler items return ``counts_outcomes``. Estimator items return 0 in every
    measured bit, so each Pauli term measures +1 and the estimate is the sum
    of the observable's coefficients with zero spread. ``api.estimate_outcomes``
    replaces that rule, and ``api.transform_result(data, passthrough)`` edits
    the item arrays and passthrough data of a result before it is sent.
    ``api.item_metadata`` gives each item's metadata fields. Every result has
    one timing span, as a result from the service does.
    ``api.params`` holds the last job request's executor parameters.
    """
    instance = "crn:supplied-instance"
    requests = []

    class API:
        _instance = instance
        status = "QUEUED"
        params = None
        program_id = "executor"
        tags = ()
        reads = 0
        fail_status = False
        container_mode = "dedicated"
        cancel_calls = 0
        fail_cancel = False
        transform_result = None
        item_metadata = None
        estimate_outcomes = staticmethod(lambda *args: 0)

        def backend_status(self, name):
            return dict(
                backend_name=name, backend_version="1", operational=True, pending_jobs=0, status_msg="active"
            )

        def program_run(self, **kwargs):
            requests.append(kwargs)
            self.params = kwargs["params"]
            self.program_id = kwargs["program_id"]
            self.tags = kwargs["job_tags"]
            return {"id": "original-ibm-job", "backend": "ibm_supplied"}

        def job_get(self, job_id, **kwargs):
            if self.fail_status:
                raise ConnectionError("supplied temporary status failure")
            return dict(
                id=job_id,
                backend="ibm_supplied",
                program={"id": self.program_id},
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
            program = executor_program(self.params)
            estimator = program._semantic_role == "estimator_v2"
            data = executor_items(program, self.estimate_outcomes if estimator else counts_outcomes)
            # Reverse the PUBs: a Sampler result lists its items in reverse
            # with their circuit metadata, and an Estimator result its
            # qiskit-mitigation tasks, which name their items.
            passthrough = dict(program.passthrough_data)
            processor = passthrough["post_processor"] = dict(passthrough["post_processor"])
            processor["circuits_metadata"] = processor["circuits_metadata"][::-1]
            if estimator:
                passthrough["qiskit_mitigation"] = passthrough["qiskit_mitigation"][::-1]
            else:
                data = data[::-1]
            if self.transform_result is not None:
                data, passthrough = self.transform_result(data, passthrough)
            span = {"start": "2026-11-10T00:00:00", "stop": "2026-11-10T00:00:01",
                    "parts": [{"idx_item": 0, "size": 1, "permutation": [], "element_range": [0, 1, 1]}]}
            return executor_result_json(program, data, passthrough=passthrough, item_metadata=self.item_metadata,
                                        chunk_timing=[span])

    api = API()
    service, backend = offline_ibm_service(api, instance)
    monkeypatch.setattr(IBMRuntimeBackend, "_new_service", lambda self: service)
    connection = IBMRuntimeBackend(
        device=backend.name, instance=instance, initial_layout=(2,)
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

    An invalid Sampler PUB returns five shots of the four requested. An invalid Estimator PUB
    returns no shots, so its estimate and stds are NaN. The Estimator's PUB 0 measures 0 and PUB 1
    measures 1 in every shot, so for I + Z they estimate 2 and 0 with zero spread.
    """
    from qiskit.primitives.containers import SamplerPubResult

    if primitive == "sampler":
        selected = counts_selection()
    else:
        selected = estimate_selection()
    api, backend = (runtime_api.api, runtime_api.connection)
    cause, failure = (OSError("original PUB decoder cause"), ConnectionError("original late PUB error"))
    failure.__cause__ = cause
    failing, consumed = (True, False)
    decoded_indices = []
    api.estimate_outcomes = lambda index, item, register_index, register, *args: (1 << register.size) - 1 if index else 0

    def transform(data, passthrough):
        if failing and failure_kind == "invalid":
            # An Estimator item's index is its PUB's. Sampler items follow their circuit metadata.
            position = 0 if primitive == "estimator" else next(
                position for position, metadata in enumerate(passthrough["post_processor"]["circuits_metadata"])
                if metadata["nwqlib_pub"] == 0)
            shots = 0 if primitive == "estimator" else 5
            data[position] = {name: np.zeros((*array.shape[:-2], shots, array.shape[-1]), bool)
                              for name, array in data[position].items()}
        return data, passthrough

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
            assert prior.values[0].value == 0.0 and prior.values[0].standard_error == 0.0
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

                def corrupt(data, passthrough):
                    metadata = passthrough["post_processor"]["circuits_metadata"]
                    if defect == "missing":
                        passthrough["post_processor"]["circuits_metadata"] = metadata[:1]
                        return data[:1], passthrough
                    first, second = metadata
                    key = "nwqlib_pub" if defect == "duplicate" else "nwqlib_snapshot"
                    metadata[0] = {**first, key: second[key]}
                    return data, passthrough

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
            assert remaining.values[0].value == 2.0 and remaining.values[0].standard_error == 0.0
            assert run.exposure["completed"]["provider_managed_sampling"] == 2
    reads = api.reads
    with load_run(path, backend=backend) as run:
        assert not refresh_submissions(run=run) and api.reads == reads
        assert len(run._state["by_attempt"]) == 2 and run.trace.jobs == 1


def test_explicit_existing_container_does_not_create_session_and_cancel_is_not_refunded(
    tmp_path, monkeypatch, runtime_api
):
    """Reuse the supplied session and preserve charged work after cancellation succeeds."""
    selection = counts_selection()
    api = runtime_api.api
    api.container_mode = "dedicated"
    api.fail_cancel = False
    api.status = "RUNNING"
    connection = runtime_api.connection.revise(mode="session", container_id="original-container")
    path = tmp_path / "container.sqlite"
    with Run(selection, backend=connection, directory=path) as run:
        handle = prepare(
            selection.resolve(selection.experiments[0].name), run=run, runtime=RuntimeOptions(seed=7)
        )
        original = submit_detached((handle,), run=run)
        (request,) = runtime_api.requests
        assert request["session_id"] == "original-container" and request["start_session"] is False
        run.cancel(reason="user requested cancellation")
        (cancelled,) = run.trace.submissions
        assert (
            cancelled.cancel_requested == "user requested cancellation"
            and cancelled.locator == original.locator
        )
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


def test_actual_http_transport_preserves_one_accepted_job_and_reconciles_original_tag(
    tmp_path, monkeypatch, runtime_api
):
    """A loopback HTTP server accepts the submission and loses its acknowledgement."""
    from http.server import BaseHTTPRequestHandler, HTTPServer
    from threading import Thread
    from urllib.parse import parse_qs, urlsplit
    from qiskit_ibm_runtime import RuntimeDecoder
    from qiskit_ibm_runtime.api.client import RuntimeClient
    from qiskit_ibm_runtime.exceptions import IBMRuntimeError

    api = runtime_api.api
    exchanges, accepted, post_bodies = ([], [], [])
    fail_get = True
    status_data = dict(
        id="original-ibm-job",
        backend="ibm_supplied",
        program={"id": "executor"},
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
            assert self.path == "/jobs" and body["program_id"] == "executor"
            accepted.append(body)
            api.params = body["params"]
            api.tags = body["tags"]
            status_data["tags"] = api.tags
            self.close_connection = True

        def do_GET(self):
            nonlocal fail_get
            path, query = (urlsplit(self.path).path, parse_qs(urlsplit(self.path).query))
            exchanges.append(("GET", path, query))
            if fail_get:
                fail_get = False
                self.respond(503, {"error": {"message": "temporary read failure", "code": "supplied"}})
            elif path == "/jobs":
                assert query["tags"] == [api.tags[-1]] and query["limit"] == ["2"] and query["program"] == ["executor"]
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
            with pytest.raises(IBMRuntimeError):
                submit_detached((handle,), run=run)
            assert run.trace.submissions[0].locator is None
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
        assert len(posts) == 1
        assert all((body == post_bodies[0] for body in post_bodies))
        assert len([1 for method, _, _ in exchanges if method == "GET"]) >= 3
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def test_actual_sdk_sampler_pub_mapping_and_original_job_resume(tmp_path, monkeypatch, runtime_api):
    """Reordered PUBs must preserve distinct job-item populations after reopening."""
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
                for seed in (7, 8)
            )
        )
        assert all((handle.record.logical_to_native == (2,) for handle in handles))
        submitted = submit_detached(handles, run=run)
        assert submitted.locator.job_id == "original-ibm-job" and run.trace.jobs == 1
        assert len(runtime_api.requests) == 1 and len(executor_program(runtime_api.requests[0]["params"]).items) == 2
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
        pytest.param((4, 5), 4096, 241664, id="pair-packing-two-registers"),
        pytest.param((65,), 16, 2848, id="wide-join-padded-repack"),
    ],
)
def test_sampler_decode_admits_the_join_and_pair_array_bound(
    runtime_api, tmp_path, monkeypatch, widths, shots, charge
):
    """Check the byte estimate S*(9*Q + w + 32) of sampler_decode_bytes once, before joining."""
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


def counts_selection():
    return nwqlib.plan(problem(), method=ExpectationMethod(), shots=4, seed=7)


def estimate_selection(coefficient=1.0, state=None):
    return nwqlib.plan(
        problem((("I", 1.0), ("Z", coefficient)), state=state), method=ExpectationMethod(estimate_precision=0.1),
        seed=7
    )


@pytest.mark.parametrize(("options", "reason", "spread"), [
    pytest.param("{}", None, False, id="level-0"),
    pytest.param('{"resilience_level":2}', 2, False, id="level-2"),
    pytest.param('{"resilience_level":2,"resilience":{"zne_mitigation":true}}', 1, False, id="zne-on"),
    pytest.param('{"resilience_level":1,"resilience":{"zne_mitigation":false}}', None, True, id="zne-off-spread"),
])
def test_estimator_weighted_observable_layout_uncertainty_and_saved_resume(
    tmp_path, monkeypatch, runtime_api, options, reason, spread
):
    """Map the weighted observable to physical qubits, decode the client-side Estimator's values from
    the returned bits through the real SDK, and save them with the provider metadata.

    The full observable, the coefficient 1e-12 included, is handed to the SDK. Without spread the
    synthetic service measures 0 in every shot, so each Pauli term that the provider keeps measures +1
    and the estimate is the sum of the kept coefficients with zero spread: 2 for I + Z, and 1 or
    1 + 1e-12 for I + 1e-12 Z, as the provider's coercion may remove the small term. That observable
    is estimated on |+>, whose circuit has a gate: qiskit-mitigation 0.2.0 refuses a circuit without
    instructions when its observable keeps only the identity. With spread it returns random outcomes
    from a fixed seed, and the estimate and stds are the values that the SDK decodes from them. The
    standard error is the provider's stds when ZNE is known to be off, here at resilience level 0 and
    with zne_mitigation false. With spread the SDK's stds and ensemble_standard_error differ, so the
    standard error is shown to be the stds. With zne_mitigation true it is unavailable with
    _ESTIMATE_REASONS[1]. Resilience level 2 alone does not state in the request whether ZNE ran, so it
    is unavailable with _ESTIMATE_REASONS[2]. The raw stds stays in the provider metadata either way.
    """
    from qiskit_ibm_runtime import EstimatorV2
    from nwqlib.backends.ibm_runtime import _ESTIMATE_REASONS

    level0 = options == "{}"
    coefficient = 1e-12 if level0 else 1.0
    plan = estimate_selection(coefficient, state=[1.0, 1.0] if level0 else None)
    spec = plan.experiments[0].observation
    assert spec.estimate.labels == ("I", "Z") and spec.estimate.coefficients == (1.0, coefficient)
    api = runtime_api.api
    if spread:
        rng = np.random.default_rng(3)
        api.estimate_outcomes = lambda index, item, register_index, register, shape, shots: rng.integers(
            0, 1 << register.size, size=(*shape, shots), dtype=np.uint64)
    sent, run_estimator = [], EstimatorV2.run
    monkeypatch.setattr(EstimatorV2, "run", lambda self, pubs, **kwargs: (
        sent.append(tuple(pubs)), run_estimator(self, pubs, **kwargs))[1])
    # The data fields of the PUB that the SDK decodes, before NWQLib reads them.
    decoded, estimate_result = [], IBMRuntimeBackend._estimate_result
    monkeypatch.setattr(IBMRuntimeBackend, "_estimate_result", lambda self, pub, *args, **kwargs: (
        decoded.append({name: float(pub.data[name]) for name in ("evs", "stds", "ensemble_standard_error")
                        if name in pub.data}), estimate_result(self, pub, *args, **kwargs))[1])
    level = json.loads(options).get("resilience_level", 0)
    backend = runtime_api.connection.revise(options_json=options)
    path = tmp_path / "estimate"
    with Run(plan, backend=backend, directory=path) as run:
        if level0:

            def interrupted(*args, **kwargs):
                raise RuntimeError("before submission intent")

            with monkeypatch.context() as patch:
                patch.setattr(Run, "_begin_submission", interrupted)
                with pytest.raises(RuntimeError, match="before submission intent"):
                    run.resume()
            assert run.trace.preparations == 1 and run.trace.jobs == 0
        else:
            run.resume()
    if level0:
        with load_run(path, backend=backend) as run, monkeypatch.context() as patch:
            patch.setattr(
                IBMRuntimeBackend,
                "prepare",
                lambda *a, **k: pytest.fail("recompiled original preparation"),
            )
            run.resume()
            assert run.trace.preparations == run.trace.jobs == 1
    with load_run(path, backend=backend) as run:
        ((pub,),) = sent
        assert pub[3] == 0.1
        assert pub[1].tolist() == {"IIIII": 1.0, "IIZII": coefficient}
        processor = executor_program(runtime_api.requests[0]["params"]).passthrough_data["post_processor"]
        assert processor["precision"] == 0.1 and processor["options"]["resilience_level"] == level
        original = run.trace.submissions[0].locator

    def forbidden(*args, **kwargs):
        pytest.fail("pending Estimator restored a circuit, replanned or resubmitted")

    for name in ("prepare", "launch", "_observable"):
        monkeypatch.setattr(IBMRuntimeBackend, name, forbidden)
    monkeypatch.setattr(ExpectationMethod, "plan", forbidden)
    api.status = "COMPLETED"
    with load_run(path, backend=backend) as run:
        result = run.wait(timeout=1, poll_interval=0)
        (sdk,) = decoded
        expected = (sdk["evs"],) if spread else (1.0, 1.0 + coefficient) if level0 else (2.0,)
        assert result.value in expected and result.statistics is None
        if not spread:
            assert sdk["stds"] == 0.0
        (estimate,) = result.estimates
        assert estimate.value == result.value
        (view,) = result.provider_output_estimates
        assert view["source_id"] == estimate.content_id and view["value"] == result.value
        if reason is None:
            assert estimate.standard_error == sdk["stds"] and estimate.uncertainty_unavailable is None
            assert view["standard_error"] == sdk["stds"]
        else:
            assert estimate.standard_error is None and estimate.uncertainty_unavailable == _ESTIMATE_REASONS[reason]
            assert view["standard_error"] is None and view["uncertainty_unavailable"] == _ESTIMATE_REASONS[reason]
        if spread:
            assert estimate.standard_error != sdk["ensemble_standard_error"]
        provider = json.loads(estimate.provider_metadata_json)
        assert provider["reported_data"]["stds"] == sdk["stds"]
        assert provider["primitive"]["options"]["resilience_level"] == level
        assert provider["pub"]["circuit_metadata"]["nwqlib_pub"] == 0
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


def sampler_options_of(request):
    """The three circuit-changing options with which the client-side Sampler built the job of ``request``.

    The SDK applies them on this computer and records its finalized options
    in the job's passthrough data.
    """
    options = executor_program(request["params"]).passthrough_data["post_processor"]["options"]
    return {"dynamical_decoupling": {"enable": options["dynamical_decoupling"]["enable"]},
            "twirling": {name: options["twirling"][name] for name in ("enable_gates", "enable_measure")}}


def cx_path_depth(circuit):
    """Largest number of CX gates on one dependency path, and the CX count.

    Every operation, barriers and measurements included, orders the qubits and
    classical bits it touches. CX is the only two-qubit gate of FakeManilaV2.
    """
    depth, gates = {}, 0
    for item in circuit.data:
        wires = (*item.qubits, *item.clbits)
        weight = int(item.operation.name == "cx")
        level = max((depth.get(wire, 0) for wire in wires), default=0) + weight
        depth.update(dict.fromkeys(wires, level))
        gates += weight
    return max(depth.values(), default=0), gates


def test_isa_native_depth_is_saved_with_its_preparation_and_decides_a_requested_limit(
    runtime_api, tmp_path, monkeypatch
):
    """The recorded depth is that of the ISA circuit that executes, qualified only for counts
    whose effective options, saved and sent, keep the provider's dynamical decoupling and
    twirling off. It survives reopening without
    compiling again, and a limit refuses before the submission is recorded when the depth is
    missing or unqualified, or when the Run is reopened where another version of a recorded
    package is installed or one cannot be read. Without a limit, nothing is looked up. A
    reopened Run sends the saved effective options, which the installed SDK checks before the
    submission is recorded, and retrieves the submitted job without these checks.
    """
    import importlib.metadata
    from qiskit import QuantumCircuit
    import qiskit_ibm_runtime.options_models as sdk_options
    from nwqlib._prepared_execution import PreparedHandle, restore_prepared
    from nwqlib.backends.inspection import NativeDepthCapRefusal, check_native_depth_cap
    from nwqlib.execution import NativeDepth
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation

    # The client-side Sampler takes False as the default of the three
    # transformation options. An unset one must be sent and saved as that
    # explicit False, and an explicit True must stay.
    frozen = {"dynamical_decoupling": {"enable": False}, "twirling": {"enable_gates": False, "enable_measure": False}}
    twirled_options = {**frozen, "twirling": {"enable_gates": True, "enable_measure": False}}
    split = {"num_randomizations": 2, "shots_per_randomization": 2}
    installed = importlib.metadata.version("qiskit-ibm-runtime")

    # Three CX on the linear chain 0-1-2-3, two of them parallel.
    state = QuantumCircuit(4)
    state.h(0)
    state.cx(0, 1)
    state.h(2)
    state.cx(2, 3)
    state.cx(1, 2)
    plan = nwqlib.plan(Expectation(state=state, observable=ingest_pauli((("ZZZZ", 1.0),), num_qubits=4)),
                       method=ExpectationMethod(), shots=4, seed=7)
    point = plan.resolve(plan.experiments[0].name)
    connection = runtime_api.connection.revise(initial_layout=(0, 1, 2, 3))
    path = tmp_path / "depth"
    with Run(plan, backend=connection, directory=path) as run:
        handle = prepare(point, run=run)
        circuit = handle.inspect_circuit()
        assert {item.operation.name for item in circuit.data} <= {*runtime_api.backend.target.operation_names, "barrier"}
        depth, gates = cx_path_depth(circuit)
        assert gates > depth >= 1
        recorded = handle.record.native_depth
        assert recorded.width == circuit.num_qubits
        assert connection.options_json == "{}" and json.loads(handle.record.provider_options_json) == frozen
        check_native_depth_cap((handle,), None)
        assert recorded.qualified
        assert (recorded.native_two_qubit_depth, recorded.native_two_qubit_gates) == (depth, gates)
        check_native_depth_cap((handle,), depth)
        with pytest.raises(NativeDepthCapRefusal) as refused:
            check_native_depth_cap((handle,), depth - 1)
        assert (refused.value.experiment, refused.value.prepared_id) == (point.experiment, handle.record.content_id)
        assert (refused.value.depth, refused.value.cap, refused.value.reason) == (depth, depth - 1, None)
        # The same circuit with an unqualified depth: no limit admits an unknown depth.
        unknown = NativeDepth(qualified=False, reason="supplied unqualified depth", width=circuit.num_qubits)
        rebuilt = PreparedHandle._make(handle.record.revise(native_depth=unknown), handle.realization,
                                       handle._native, handle._items, handle._setting, handle._bindings)
        with pytest.raises(NativeDepthCapRefusal) as refused:
            check_native_depth_cap((rebuilt,), depth)
        assert (refused.value.depth, refused.value.reason) == (None, "supplied unqualified depth")
        assert run.trace.jobs == 0
    with monkeypatch.context() as patch, load_run(path, backend=connection) as run:
        patch.setattr(IBMRuntimeBackend, "prepare", lambda *a, **k: pytest.fail("reopening compiled again"))
        restored = restore_prepared(handle.record.content_id, run=run, for_submission=True)
        assert restored.record.native_depth == recorded
        assert restored.record.provider_options_json == handle.record.provider_options_json
        check_native_depth_cap((restored,), depth)
        with pytest.raises(NativeDepthCapRefusal):
            check_native_depth_cap((restored,), depth - 1)

    # The saved Run reopened where another qiskit-ibm-runtime, here called
    # 99.0.0, is installed, whose Sampler turns gate twirling on unless the
    # request sets it. Only the installed-version lookup and the SDK's options
    # class are replaced.
    lookup, looked_up, replaced = importlib.metadata.version, [], {"qiskit-ibm-runtime": "99.0.0"}

    def reopened_version(name):
        looked_up.append(name)
        value = replaced[name] if name in replaced else lookup(name)
        if isinstance(value, Exception):
            raise value
        return value

    sampler_options = sdk_options.SamplerOptions

    def twirling_by_default(**values):
        options = sampler_options(**values)
        if options.twirling.enable_gates is None:
            options.twirling.enable_gates = True
        return options

    def rejecting(**values):
        raise ValueError("this SDK does not accept the saved Sampler options")

    def all_zero(data, passthrough):
        """Every classical bit 0 in each shot of each item."""
        return [{name: np.zeros_like(array) for name, array in item.items()} for item in data], passthrough

    sent = len(runtime_api.requests)
    with monkeypatch.context() as patch, load_run(path, backend=connection) as run:
        patch.setattr(IBMRuntimeBackend, "prepare", lambda *a, **k: pytest.fail("reopening compiled again"))
        patch.setattr(importlib.metadata, "version", reopened_version)
        patch.setattr(sdk_options, "SamplerOptions", twirling_by_default)
        restored = restore_prepared(handle.record.content_id, run=run, for_submission=True)
        looked_up.clear()
        check_native_depth_cap((restored,), None)
        assert looked_up == []
        with monkeypatch.context() as limited:
            limited.setattr(ExpectationMethod, "before_submit",
                            lambda self, plan, prepared, *, run: check_native_depth_cap(prepared, depth))
            with pytest.raises(NativeDepthCapRefusal) as refused:
                submit_detached((restored,), run=run)
        assert refused.value.depth is None and refused.value.native_depth == recorded
        assert f"qiskit-ibm-runtime {installed}" in refused.value.reason
        assert "qiskit-ibm-runtime 99.0.0 is installed" in refused.value.reason
        # With qiskit-ibm-runtime as recorded, an installed Qiskit whose
        # version cannot be read refuses on its own.
        replaced.update({"qiskit-ibm-runtime": installed, "qiskit": importlib.metadata.PackageNotFoundError("qiskit")})
        with pytest.raises(NativeDepthCapRefusal) as unreadable:
            check_native_depth_cap((restored,), depth)
        assert unreadable.value.depth is None
        assert "installed version of qiskit cannot be read" in unreadable.value.reason
        replaced.update({"qiskit-ibm-runtime": "99.0.0"})
        del replaced["qiskit"]
        assert not run.trace.submissions and len(runtime_api.requests) == sent
        # The installed SDK checks the saved options before the submission is recorded.
        patch.setattr(sdk_options, "SamplerOptions", rejecting)
        with pytest.raises(ValueError, match="does not accept the saved"):
            submit_detached((restored,), run=run)
        assert not run.trace.submissions and len(runtime_api.requests) == sent
        # Without a limit the job carries the saved effective options, although
        # this SDK would turn gate twirling on for the configuration's "{}".
        patch.setattr(sdk_options, "SamplerOptions", twirling_by_default)
        submitted = submit_detached((restored,), run=run)
        assert len(runtime_api.requests) == sent + 1 and sampler_options_of(runtime_api.requests[-1]) == frozen
    # The job submitted there is retrieved there.
    runtime_api.api.status, runtime_api.api.transform_result = "COMPLETED", all_zero
    with monkeypatch.context() as patch, load_run(path, backend=connection) as run:
        patch.setattr(importlib.metadata, "version", reopened_version)
        patch.setattr(sdk_options, "SamplerOptions", twirling_by_default)
        (chunk,) = refresh_submissions(run=run)
        assert (chunk.job, chunk.prepared_id) == (submitted.locator.job_id, handle.record.content_id)
        assert dict(zip(chunk.histogram().index_list(), chunk.histogram().weights.tolist())) == {0: 4}
        assert len(runtime_api.requests) == sent + 1
    # Gate twirling requested: it is sent as requested, and the depth of the
    # same ISA circuit is unqualified, so a limit refuses it before the
    # submission is recorded while execution without a limit proceeds.
    twirled = connection.revise(options_json=json.dumps({"twirling": {"enable_gates": True, **split}}))
    sent = len(runtime_api.requests)
    with Run(plan, backend=twirled, directory=tmp_path / "twirled") as run:
        handle = prepare(point, run=run)
        unqualified = handle.record.native_depth
        assert (unqualified.qualified, unqualified.width) == (False, runtime_api.backend.num_qubits)
        assert "twirling.enable_gates=True" in unqualified.reason
        saved = json.loads(handle.record.provider_options_json)
        assert saved == {**twirled_options, "twirling": {**twirled_options["twirling"], **split}}
        with monkeypatch.context() as patch:
            patch.setattr(ExpectationMethod, "before_submit",
                          lambda self, plan, prepared, *, run: check_native_depth_cap(prepared, depth))
            with pytest.raises(NativeDepthCapRefusal) as refused:
                submit_detached((handle,), run=run)
        assert (refused.value.depth, refused.value.reason) == (None, unqualified.reason)
        assert not run.trace.submissions and len(runtime_api.requests) == sent
        submit_detached((handle,), run=run)
        assert len(runtime_api.requests) == sent + 1 and sampler_options_of(runtime_api.requests[-1]) == twirled_options
    # A provider estimate, at the default resilience level 0, is unqualified.
    estimate = estimate_selection()
    with Run(estimate, backend=runtime_api.connection, directory=tmp_path / "estimate") as run:
        handle = prepare(estimate.resolve(estimate.experiments[0].name), run=run)
        unqualified = handle.record.native_depth
        assert (unqualified.qualified, unqualified.width) == (False, runtime_api.backend.num_qubits)
        assert "Estimator" in unqualified.reason
        assert json.loads(handle.record.provider_options_json) == {"resilience_level": 0}
        check_native_depth_cap((handle,), None)
        with pytest.raises(NativeDepthCapRefusal):
            check_native_depth_cap((handle,), depth)
    with Run(plan) as run:
        aer = prepare(point, run=run)
        assert aer.record.native_depth is None
        check_native_depth_cap((aer,), None)
        with pytest.raises(NativeDepthCapRefusal) as refused:
            check_native_depth_cap((aer,), depth)
        assert refused.value.native_depth is refused.value.depth is None and refused.value.reason


def test_isa_preparation_uses_exact_dense_synthesis_below_level_two(runtime_api, tmp_path):
    # Qiskit's own synthesis of this near-identity unitary errs by about 7e-7
    # per entry. The default level 1 receives the exact synthesis. The
    # expected operator is SciPy's exponential on the two logical qubits,
    # identity on the three ancillas.
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    connection = runtime_api.connection.revise(optimization_level=1, initial_layout=(0, 1))
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
    assert error <= 1e-13
    assert prepared.transformation.startswith("exact dense-unitary synthesis")
    assert prepared.probability_window_exclusions is None


def fractional_device(runtime_api):
    """Make the synthetic service open the SDK's fake fractional backend (rx and rzz in its target).

    Returns the device and the list of keyword arguments with which it was opened.
    """
    from qiskit_ibm_runtime import IBMBackend
    from qiskit_ibm_runtime.fake_provider import FakeFractionalBackend

    fake = FakeFractionalBackend()
    configuration = fake.configuration()
    configuration.backend_name = "ibm_supplied"
    device = IBMBackend(configuration, service=runtime_api.service, api_client=runtime_api.api,
                        instance=runtime_api.connection.instance)
    device._target, device._properties = (fake.target, fake.properties())
    opened = []
    runtime_api.service.backend = lambda name, **kwargs: (opened.append(kwargs), device)[1]
    runtime_api.service._create_backend_obj = lambda name, **kwargs: device
    return device, opened


def sampled_lchs_plan(time=.7, control_copies=1, **options):
    """Eight sampled-pair LCHS rows of two shots on one system qubit, with a source, so rows mix
    initial and source inputs and carry branch phases."""
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs import LCHS
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.records import QuadraticForm

    linear = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                            initial_state=[.6, .8j], source=[.3, -.4j], time=time)
    return nwqlib.plan(linear, method=LCHS(estimator=SampledPairs(repeats=2, control_copies=control_copies),
                                           ham_evo=Trotter(steps=2), **options),
                       output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45)], num_qubits=1)),
                       shots=16, seed=11)


def row_key(realization):
    """The static point key ``pairs[row=i]`` of a sampled-pair row's Realization."""
    return "pairs[row=%d]" % next(int(item.value) for item in realization.bindings if item.parameter == "row")


def logical_row_operator(plan, realization, width):
    """The unitary of the point ``realization``'s own lowered logical circuit, identity on ``width`` minus its qubits."""
    from qiskit.quantum_info import Operator
    from nwqlib.blocks.lowering import lower_qiskit

    _, construction = realization._selected_construction(plan)
    selected = {record.content_id for record in construction.selections}
    circuit = lower_qiskit(construction, blocks=tuple(block for block in plan.blocks
                                                      if block.record.content_id in selected)).circuit
    circuit.remove_final_measurements()
    return np.kron(np.eye(2 ** (width - circuit.num_qubits)), Operator(circuit).data)


@pytest.mark.parametrize("options", [
    dict(ham_evo=Trotter(steps=2), estimator=SampledPairs(
        repeats=2, product_formula="richardson")),
    dict(ham_evo=Trotter(steps=2), estimator=SampledPairs(
        repeats=2, product_formula="mpf")),
    dict(ham_evo=TrotterErrorBudgeted(order=1), estimator=SampledPairs(
        repeats=2, product_formula=PTSC(pf_allowance=1e-2, mu_cap=1.5, taylor_cutoff=8)))],
    ids=["richardson", "mpf", "ptsc"])
def test_product_formula_recipe_row_templates_bind_to_each_row_circuit(options, monkeypatch):
    """Each recipe row's template, bound to that row's values, is the row's own pair circuit.

    The oracle is the row builder `construct_pair`, compared as a unitary without global-phase alignment.
    Recipe rows apply several weighted product-formula words, so their gate sequences differ from plain rows.
    Every experiment belongs to exactly one template, and where a template holds rows with different values,
    binding another row's values gives a different unitary, so a template that ignored its parameters fails.
    Grouping the rows walks no SELECT occurrence; only building each template does.
    """
    from qiskit.quantum_info import Operator
    from nwqlib import LinearDynamics
    from nwqlib.algorithms.lchs import LCHS, sampled_native
    from nwqlib.algorithms.lchs.sampled_plan import construct_pair, point_row
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems.records import QuadraticForm

    linear = LinearDynamics(A=ingest_pauli([("I", .3+.1j), ("X", .25), ("Z", .2j), ("Y", -.15j)], num_qubits=1),
                            initial_state=[.6, .8j], source=[.3, -.4j], time=.7)
    chosen = nwqlib.plan(linear, method=LCHS(**options),
                         output=QuadraticForm(observable=ingest_pauli([("X", .7), ("Z", -.45)], num_qubits=1)),
                         shots=16, seed=11)
    block = next(item for item in chosen.blocks if item.record.signature.name == "lchs_pair")
    points = tuple(chosen._static_points())
    rows_of = {point.key: point_row(point) for point in points}
    walks, building = {"grouping": 0, "template": 0}, [False]
    walk_select, build_template = sampled_native._walk_select, sampled_native.build_pair_template

    def counted_walk(*args, **kwargs):
        walks["template" if building[0] else "grouping"] += 1
        return walk_select(*args, **kwargs)

    def counted_build(*args, **kwargs):
        building[0] = True
        try:
            return build_template(*args, **kwargs)
        finally:
            building[0] = False

    checks, check_table = [], sampled_native._real_pauli_table

    def counted_check(*args, **kwargs):
        checks.append(args[1])
        return check_table(*args, **kwargs)

    monkeypatch.setattr(sampled_native, "_walk_select", counted_walk)
    monkeypatch.setattr(sampled_native, "build_pair_template", counted_build)
    monkeypatch.setattr(sampled_native, "_real_pauli_table", counted_check)
    templates = chosen.method.row_templates(chosen, points)
    # Grouping walks no repeated SELECT, and each template walks it when its circuit is read.
    assert walks == {"grouping": 0, "template": 0}
    assert all(template.circuit.data for template in templates)
    monkeypatch.undo()
    assert walks["grouping"] == 0 and walks["template"] >= len(templates)
    # The Plan's L_c and H_c tables are checked once for all rows, when the first row needs them.
    assert len(checks) <= 2 < len(points)
    assert sorted(key for template in templates for key, _ in template.rows) == sorted(rows_of)
    for template in templates:
        for key, values in template.rows:
            bound = template.circuit.assign_parameters(dict(zip(template.parameters, values)))
            expected = Operator(construct_pair(block, (("row", rows_of[key]),), None)).data
            assert np.max(np.abs(Operator(bound).data - expected)) <= 1e-12
            other = next((v for _, v in template.rows if v != values), None)
            if other is not None:
                wrong = template.circuit.assign_parameters(dict(zip(template.parameters, other)))
                assert np.max(np.abs(Operator(wrong).data - expected)) > 1e-6


@pytest.mark.parametrize("fractional", [False, True], ids=["standard", "fractional"])
def test_sampled_lchs_rows_share_compiled_templates_in_one_job_and_resume_by_row(tmp_path, monkeypatch, runtime_api,
                                                                                  fractional):
    """Rows of one gate sequence share one compiled template, bound per row in one job.

    `prepare` compiles the first row's template, and closing the Run, which `prepared` still
    references, releases that compiled circuit. The reopened Run uses its saved circuits
    without compiling it again and compiles each other template once. On the fractional target
    the structured SELECT's CRZ compiles to RZZ gates with parameterized angles, which each row
    folds into IBM's [0, pi/2]: the rows of one correction class share one circuit built from
    their compiled template, whose QPY is saved once, so one template can give several circuits
    and PUBs. The problem time 2.0 makes the first row's correction differ from another row's of
    its template, so the reopened Run builds that class from the saved compiled template. Each
    saved circuit is read at most once. Every row keeps its preparation record, attempt and two
    shots. Each bound ISA circuit equals the experiment's own lowered row circuit without
    global-phase alignment. The service returns the PUBs in reverse order, and each binding row's
    outcome is a function of that row's values (`template_row_outcome`), so each experiment's
    counts must carry the outcome of its own saved values. PUB 0 is invalid on the first read:
    the other rows are saved, and the reopened Run decodes only PUB 0's rows, without planning,
    drawing rows, compiling, folding, launching or reading a saved circuit, and its Result is
    analyzed again unchanged.
    """
    import gc
    import weakref
    from collections import Counter
    from qiskit import qpy
    from qiskit.primitives.containers import BitArray
    from qiskit.quantum_info import Operator
    from nwqlib._run_journal import LocalJournal
    from nwqlib.algorithms.lchs import sampled_rows

    selected = sampled_lchs_plan(time=2.0 if fractional else .7, lcu_select="structured",
                                 control_copies=2)
    points = tuple(selected._static_points())
    names = tuple(point.key for point in points)
    templates = selected.method.row_templates(selected, points)
    assert 1 < len(templates) < len(names)
    connection, api = runtime_api.connection.revise(initial_layout=None, use_fractional_gates=fractional), runtime_api.api
    if fractional:
        fractional_device(runtime_api)
    compiled, compiled_circuits, loads, reads = [], [], [], []
    prepare_isa, load, read = IBMRuntimeBackend.prepare, qpy.load, LocalJournal.read_payload

    def compile_isa(self, circuit, **options):
        compiled.append(circuit.num_parameters)
        preparation = prepare_isa(self, circuit, **options)
        compiled_circuits.append(weakref.ref(preparation.native.circuit))
        return preparation

    monkeypatch.setattr(IBMRuntimeBackend, "prepare", compile_isa)
    monkeypatch.setattr(qpy, "load", lambda *args, **kwargs: (loads.append(1), load(*args, **kwargs))[1])
    monkeypatch.setattr(LocalJournal, "read_payload", lambda self, reference: (
        reads.append(reference.content_id), read(self, reference))[1])
    directory = tmp_path / "sampled-lchs"
    prepared = nwqlib.prepare(selected, backend=connection, directory=directory)
    (first,) = prepared.run.prepared_artifacts
    prepared.run.close()
    assert len(compiled) == 1 and compiled[0] > 0
    # Closing frees the compiled circuit although `prepared` still references the Run. On the
    # fractional target only the Run's template cache held it, since the first row runs a
    # circuit folded from it.
    gc.collect()
    assert compiled_circuits[0]() is None
    # Fixture premise: on the fractional target the first row runs a folded circuit that needs
    # correction gates, and its compiled template is saved apart from that circuit.
    assert (first.template.correction is not None) == fractional
    if fractional:
        assert set(first.template.correction) - {"0"} and first.template.base not in (None, first.payload)
    later = len(next(t for t in templates if row_key(first.realization) in dict(t.rows)).rows) > 1
    with load_run(directory, backend=connection) as run:
        run.resume()
        assert run.result is None and run.trace.jobs == 1 and len(runtime_api.requests) == 1
        program = executor_program(runtime_api.requests[0]["params"])
        pubs = program.passthrough_data["post_processor"]["circuits_metadata"]
        records = run.prepared_artifacts
        assert len(records) == run.trace.preparations == len(names)
        # One compilation per logical template; one circuit, snapshot, saved QPY and PUB per
        # correction class (the template itself without folding).
        assert len(compiled) == len(templates) and all(compiled)
        assert sum(record.template.compiled for record in records) == len(templates)
        snapshots = Counter(record.snapshot for record in records)
        classes = {(record.template.key, record.template.correction) for record in records}
        assert len(snapshots) == len(classes) == len({record.payload.content_id for record in records}) == len(pubs)
        assert sorted(item.circuit_arguments.shape[0] for item in program.items) == sorted(snapshots.values())
        bases = {record.template.key: record.template.base for record in records}
        assert all(record.template.base == bases[record.template.key] for record in records)
        assert all((base is not None) == fractional for base in bases.values())
        assert not set(bases.values()) & {record.payload for record in records}
        # The reopened Run reads the first row's circuit, and the compiled template from which
        # the other rows of its template are folded, once each.
        assert len(reads) == len(set(reads)) == len(loads)
        assert set(reads) == {first.payload.content_id} | (
            {first.template.base.content_id} if fractional and later else set())
        assert all(record.construction_work_reserved == 0 for record in records if not record.template.compiled)
        assert [event.shots for event in run.trace.events] == [2] * len(names)
        for record in records:
            bound = run._state["handles"][record.content_id].inspect_circuit()
            bound.remove_final_measurements()
            expected = logical_row_operator(selected, record.realization, bound.num_qubits)
            assert np.max(np.abs(Operator.from_circuit(bound).data - expected)) <= 1e-10
        first = {"read": True}

        def invalid_pub_zero(data, passthrough):
            """On the first read, PUB 0 returns five shots in each row, more than the two requested."""
            if first["read"]:
                position = next(position for position, metadata
                                in enumerate(passthrough["post_processor"]["circuits_metadata"])
                                if metadata["nwqlib_pub"] == 0)
                data[position] = {name: np.zeros((*array.shape[:-2], 5, array.shape[-1]), bool)
                                  for name, array in data[position].items()}
            return data, passthrough

        api.status, api.transform_result = "COMPLETED", invalid_pub_zero
        with pytest.raises(ValueError, match="template rows or shot cap"):
            run.resume()
        missing = len(pubs[0]["nwqlib_items"])
        assert sum(event.status == "completed" for event in run.trace.events) == len(names) - missing
        assert run.result is None and api.reads == 1
    first["read"] = False

    def forbidden(*args, **kwargs):
        raise AssertionError("reopened Run planned, drew rows, compiled, folded, launched or read a saved circuit")

    for owner, attribute in ((type(selected.method), "plan"), (sampled_rows, "draw_pair_rows"),
                             (IBMRuntimeBackend, "prepare"), (IBMRuntimeBackend, "fold_template"),
                             (IBMRuntimeBackend, "template_class"), (IBMRuntimeBackend, "launch"),
                             (LocalJournal, "read_payload"), (qpy, "load")):
        monkeypatch.setattr(owner, attribute, forbidden)
    joined = []
    concatenate = BitArray.concatenate_bits
    monkeypatch.setattr(BitArray, "concatenate_bits", staticmethod(lambda arrays: (
        joined.append(1), concatenate(arrays))[1]))
    with load_run(directory, backend=connection) as run:
        run.resume()
        assert run.result is not None and len(joined) == missing and api.reads == 2
        assert run.trace.jobs == 1 and len(runtime_api.requests) == 1
        assert sum(event.shots for event in run.trace.events) == selected.shots
        for chunk in run.observations.chunks:
            record = run.prepared_artifact(chunk.prepared_id)
            positions = {register.name: register.bits for register in record.classical_layout}
            outcome = 0
            for index, register in enumerate(record.native_classical_layout):
                value = template_row_outcome(record.template.values, len(register.bits), index)
                outcome |= sum(((value >> bit) & 1) << position
                               for bit, position in enumerate(positions[register.name]))
            histogram = chunk.histogram()
            assert dict(zip(histogram.index_list(), histogram.weights.tolist())) == {outcome: 2}
        again = run.result.analyze()
        assert (again.model_dump(exclude={"origin", "content_id"})
                == run.result.model_dump(exclude={"origin", "content_id"}))


def test_fractional_gates_open_the_device_fold_bound_rzz_and_check_template_rows(tmp_path, runtime_api):
    """The device is opened with use_fractional_gates=True and the SDK's fake fractional target.
    A bound RZZ(-0.3) is compiled into IBM's [0, pi/2] with the same unitary (no global-phase
    alignment). A parameterized RZZ keeps its parameter through compilation, and each row of
    values is folded into [0, pi/2]. The rows cover both signs, exact 0 and pi/2, both sides of
    pi/2 and of pi, and a 2*pi shift. Each folded circuit, bound to its row, must equal the
    analytic RZZ diagonal times the template's nonzero, parameter-dependent global phase without
    phase alignment, and pass the SDK's ISA check, and one job of all rows passes the SDK's RZZ
    check. pi/4 and 2*pi+pi/4 need the same correction gates, so only the folded global phase
    gives the opposite sign of the second. The compiled template bound to -0.4 without folding is
    refused before a job could be recorded. Options that IBM documents as unsupported with
    fractional gates are refused, and measurement twirling is accepted."""
    from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
    from qiskit.circuit import Parameter
    from qiskit.quantum_info import Operator
    from qiskit_ibm_runtime.utils import validate_isa_circuits

    device, opened = fractional_device(runtime_api)
    connection = runtime_api.connection.revise(initial_layout=None, use_fractional_gates=True)
    run = Run(counts_selection(), backend=connection, directory=tmp_path / str(uuid4()))
    try:
        def prepare(circuit, snapshot):
            return connection.prepare(circuit, observation=ObservationSpec(kind="counts", shots=4),
                runtime=RuntimeOptions(seed=7), position=len(circuit.data), source_definitions=(), run=run,
                snapshot=snapshot)

        circuit = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "readout"))
        circuit.h(0)
        circuit.rzz(-0.3, 0, 1)
        circuit.rx(0.7, 1)
        unitary = Operator(circuit).data
        circuit.measure([0, 1], [0, 1])
        prepared = prepare(circuit, "bound-rzz")
        assert opened and all(kwargs["use_fractional_gates"] is True for kwargs in opened)
        assert "fractional gates enabled" in prepared.target.domain and {"rx", "rzz"} <= set(prepared.native_basis)
        angles = [float(item.operation.params[0]) for item in prepared.native.circuit.data
                  if item.operation.name == "rzz"]
        assert angles and all(0 <= angle <= np.pi / 2 for angle in angles)
        body = prepared.native.circuit.remove_final_measurements(inplace=False)
        expected = np.kron(np.eye(2 ** (body.num_qubits - 2)), unitary)
        assert np.max(np.abs(Operator.from_circuit(body).data - expected)) <= 1e-10
        connection.admit_batch((prepared.native,))

        theta = Parameter("theta")
        template = QuantumCircuit(QuantumRegister(2, "q"), ClassicalRegister(2, "readout"))
        template.rzz(theta, 0, 1)
        # theta then remains in each folded circuit only through this phase.
        template.global_phase = 0.3 + theta / 4
        template.measure([0, 1], [0, 1])
        compiled = prepare(template, "template-rzz").native
        assert compiled.parameters == ("theta",)
        # The crossing displacement 2**-20 is far above binary64 roundoff at these angles.
        step = 2.0 ** -20
        angles = (-np.pi / 4, np.pi / 4, 3 * np.pi / 4, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 2 - step,
                  np.pi / 2 + step, np.pi - step, np.pi + step, 2 * np.pi + np.pi / 4)
        folded = connection.fold_template(compiled, [{"theta": angle} for angle in angles])
        assert folded[1][0] == folded[-1][0]
        classes, rows = {}, []
        for angle, (correction, columns) in zip(angles, folded, strict=True):
            if correction not in classes:
                classes[correction] = connection.template_class(compiled, correction, snapshot=f"class-{correction}",
                                                                run=run)[0]
                validate_isa_circuits([classes[correction].circuit], device.target)
            circuit = classes[correction].circuit
            rows.append(connection.bind_template(classes[correction], tuple(columns[parameter.name]
                                                                           for parameter in circuit.parameters),
                                                 runtime=None)[0])
            body = circuit.assign_parameters({parameter: columns[parameter.name] for parameter in circuit.parameters})
            body.remove_final_measurements()
            # RZZ(theta) = exp(-i theta Z0 Z1 / 2) is diagonal with Z0 Z1 = (1, -1, -1, 1).
            expected = np.exp(1j * (0.3 + angle / 4)) * np.diag(np.exp(-0.5j * angle * np.array([1, -1, -1, 1])))
            difference = Operator.from_circuit(body).data - np.kron(np.eye(2 ** (body.num_qubits - 2)), expected)
            assert np.max(np.abs(difference)) <= 1e-10, angle
        connection.admit_batch(tuple(rows))
        # The SDK's RZZ check still runs before a job is recorded.
        with pytest.raises(ValueError, match=r"RZZ angle outside \[0, pi/2\]"):
            connection.admit_batch((connection.bind_template(compiled, (-0.4,), runtime=None)[0],))
        assert not runtime_api.requests

        connection.revise(options_json='{"twirling": {"enable_measure": true}}')._options(run, "counts")
        with pytest.raises(ValueError, match="gate twirling"):
            connection.revise(options_json='{"twirling": {"enable_gates": true}}')._options(run, "counts")
        for options, reason in (('{"resilience_level": 2}', "gate twirling"),
                                ('{"resilience": {"pec_mitigation": true}}', "probabilistic error cancellation"),
                                ('{"resilience": {"zne_mitigation": true, "zne": {"amplifier": "pea"}}}', "PEA")):
            with pytest.raises(ValueError, match=reason):
                connection.revise(options_json=options)._options(run, "estimated_observable")
        connection.revise(options_json='{"resilience_level": 1}')._options(run, "estimated_observable")
    finally:
        run.close()


def test_interrupted_template_preparation_never_compiles_a_template_twice(tmp_path, monkeypatch, runtime_api):
    """An interruption between a compilation's charge and its record, or before a later row's record, costs no recompile.

    The sampled LCHS Plan of `sampled_lchs_plan` at time 2.0 runs on the fractional target with the structured
    SELECT, so rows are folded into correction classes, and row 7 shares the first row's template but needs a
    class first built after reopening (checked below). First, the journal commit that would record the first
    compilation fails after the compilation was charged: the reopened Run has the charge without a record and
    refuses with PreparationNotRebuilt, compiling nothing and submitting nothing. With the explicit request, that
    one row, whose template no record holds, compiles once more, each other template compiles once, every row runs
    in one job and the count grows by one. Second, a Run that holds no charge without a record gets the request in
    every call below, which changes nothing there. Prepare records the
    first row, and the commit that would prepare another row of its template fails: that row keeps no charge
    or record. The next reopened Run loads the saved compiled template, builds row 7's class again without
    compiling and compiles each other logical template once; its job is then refused at admission, so every row
    is prepared and none is submitted. The last reopened Run restores every row from its own saved class circuit,
    with no compilation, class build or template load, and sends one job. Each restored circuit, bound to its
    record's values, equals its experiment's own lowered row circuit without global-phase alignment, so a row
    restored with another class circuit of its template, whose RZZ fold correction differs, fails there.
    """
    from qiskit.quantum_info import Operator
    from nwqlib._prepared_execution import PreparationNotRebuilt
    from nwqlib._run_journal import LocalJournal
    from nwqlib.algorithms.protocol import _DeferredRowTemplate

    calls = {"prepare": [], "template_class": [], "restore_template": []}
    for attribute in calls:
        original = getattr(IBMRuntimeBackend, attribute)

        def counted(self, *args, _original=original, _calls=calls[attribute], **kwargs):
            _calls.append(args[1] if _original.__name__ == "template_class" else 1)
            return _original(self, *args, **kwargs)
        monkeypatch.setattr(IBMRuntimeBackend, attribute, counted)
    commit = LocalJournal.commit

    def fail_commit(condition):
        armed = [True]

        def failing(self, records=(), **kwargs):
            if armed[0] and any(kind == "prepared" and condition(value) for kind, _, value in records):
                armed[0] = False
                raise OSError("injected interruption before the preparation commit")
            return commit(self, records, **kwargs)
        monkeypatch.setattr(LocalJournal, "commit", failing)
        return armed

    fractional_device(runtime_api)
    connection = runtime_api.connection.revise(initial_layout=None, use_fractional_gates=True)
    selected = sampled_lchs_plan(time=2.0, lcu_select="structured")
    points = tuple(selected._static_points())
    names = tuple(point.key for point in points)
    templates = selected.method.row_templates(selected, points)

    armed = fail_commit(lambda record: True)
    with pytest.raises(OSError, match="injected interruption"):
        nwqlib.prepare(selected, backend=connection, directory=tmp_path / "first")
    assert not armed[0] and len(calls["prepare"]) == 1
    with load_run(tmp_path / "first", backend=connection) as run:
        assert run.prepared_artifacts == () and run.trace.preparations == 1
        with pytest.raises(PreparationNotRebuilt):
            run.resume()
    assert len(calls["prepare"]) == 1 and not runtime_api.requests

    monkeypatch.setattr(LocalJournal, "commit", commit)
    calls["prepare"].clear()
    with load_run(tmp_path / "first", backend=connection) as run:
        run.resume(rebuild_preparations=True)
        assert len(calls["prepare"]) == len(templates) and len(runtime_api.requests) == 1
        assert run.trace.preparations == len(names) + 1 and run.trace.jobs == 1
        for record in run.prepared_artifacts:
            bound = run._state["handles"][record.content_id].inspect_circuit()
            bound.remove_final_measurements()
            expected = logical_row_operator(selected, record.realization, bound.num_qubits)
            assert np.max(np.abs(Operator.from_circuit(bound).data - expected)) <= 1e-10
    runtime_api.requests.clear()
    for counted_calls in calls.values():
        counted_calls.clear()
    prepared = nwqlib.prepare(selected, backend=connection, directory=tmp_path / "later")
    (first,) = prepared.run.prepared_artifacts
    prepared.run.close()
    assert "pairs[row=7]" in dict(next(t for t in templates if row_key(first.realization) in dict(t.rows)).rows)
    # From here each template builds when asked, so the rows of a compiled template, folded here, must not ask.
    ordinary, read = type(selected.method).row_templates, []

    def deferred(self, plan, requested):
        return tuple(_DeferredRowTemplate(key=t.key, block_id=t.block_id, parameters=t.parameters, rows=t.rows,
                                          build=lambda t=t: read.append(t.key) or t.circuit)
                     for t in ordinary(self, plan, requested))

    monkeypatch.setattr(type(selected.method), "row_templates", deferred)
    armed = fail_commit(lambda record: record.template.key == first.template.key)
    with load_run(tmp_path / "later", backend=connection) as run:
        with pytest.raises(OSError, match="injected interruption"):
            run.resume(rebuild_preparations=True)
    assert not armed[0] and not runtime_api.requests and not read
    built = [correction for correction in calls["template_class"] if correction != first.template.correction]
    admit = IBMRuntimeBackend.admit_batch

    def refuse(self, natives):
        raise ValueError("injected admission refusal")
    monkeypatch.setattr(IBMRuntimeBackend, "admit_batch", refuse)
    with load_run(tmp_path / "later", backend=connection) as run:
        item = run._state["workflow_items"].get("pairs[row=7]")
        assert item is None or (item.preparation_id is None and item.prepared_id is None)
        with pytest.raises(ValueError, match="injected admission refusal"):
            run.resume(rebuild_preparations=True)
        assert len(calls["prepare"]) == len(templates) and len(calls["restore_template"]) == 2
        assert sorted(read) == sorted(t.key for t in templates if t.key != first.template.key)
        later = next(record for record in run.prepared_artifacts if row_key(record.realization) == "pairs[row=7]")
        assert later.template.correction in built and not later.template.compiled
        assert calls["template_class"].count(later.template.correction) == 2
        assert len(run.prepared_artifacts) == len(names) and not run.trace.submissions and not runtime_api.requests
    monkeypatch.setattr(IBMRuntimeBackend, "admit_batch", admit)
    for counted_calls in calls.values():
        counted_calls.clear()
    with load_run(tmp_path / "later", backend=connection) as run:
        run.resume(rebuild_preparations=True)
        assert not any(calls.values())
        assert run.trace.jobs == 1 and len(runtime_api.requests) == 1
        for record in run.prepared_artifacts:
            bound = run._state["handles"][record.content_id].inspect_circuit()
            bound.remove_final_measurements()
            expected = logical_row_operator(selected, record.realization, bound.num_qubits)
            assert np.max(np.abs(Operator.from_circuit(bound).data - expected)) <= 1e-10


def three_setting_counts():
    """A Plan with three settings, each one job on IBM: ZZZ, XXX and YYY need three measurement bases.

    The three-qubit state compiles to enough gates that a QPY copy dwarfs what a readout reserves.
    """
    from nwqlib.operators import ingest_pauli
    from nwqlib.problems import Expectation

    return nwqlib.plan(Expectation(state=np.arange(1., 9.),
                                   observable=ingest_pauli((("ZZZ", 1.), ("XXX", .5), ("YYY", .25)), num_qubits=3)),
                       method=ExpectationMethod(), shots=8, seed=7)


def test_a_data_limit_inside_a_circuit_copy_leaves_a_run_that_an_explicit_rebuild_finishes(tmp_path, ibm_runtime_spy):
    """One job has run when max_data_bytes stops the second circuit's QPY copy, which leaves it charged without a record.

    The limit comes from an uninterrupted Run's own counts, half way through that copy. No job is sent again and each
    committed charge adds one preparation, a failed one included. A rebuild under the same limit fails again, so a new
    preparation has no way around the limit and a second failure needs a second request. Raising the limit leads past
    it. The synthetic counts make the values equal only as a check of the associations, and the seeds are the
    uninterrupted Run's.
    """
    from nwqlib.execution import ExecutionLimits
    from nwqlib._prepared_execution import PreparationNotRebuilt

    chosen, backend, requests = three_setting_counts(), ibm_runtime_spy.connection, ibm_runtime_spy.api.requests
    counted, holder = [], {}

    def observe(stage, done, total):
        if stage == "prepare" and done is None and holder:
            counted.append(holder["run"].trace.data_bytes)

    prepared = nwqlib.prepare(chosen, backend=backend, directory=tmp_path / "whole", progress=observe)
    holder["run"] = prepared.run
    with prepared.run as run:
        nwqlib.submit(prepared)
        value = run.wait(timeout=30, poll_interval=0).value
        records = run.prepared_artifacts
        seeds = {record.realization_id: record.runtime.seed for record in records}
    limit = counted[0] + records[1].payload.bytes // 2
    requests.clear()
    prepared = nwqlib.prepare(chosen, backend=backend, directory=tmp_path / "run", progress=False,
                              limits=ExecutionLimits(max_data_bytes=limit))
    with prepared.run as run, pytest.raises(ValueError, match="max_data_bytes"):
        nwqlib.submit(prepared)
        run.wait(timeout=30, poll_interval=0)
    assert len(requests) == 1
    with load_run(tmp_path / "run", backend=backend) as run:
        assert run.trace.preparations == 2 and len(run.prepared_artifacts) == 1
        with pytest.raises(PreparationNotRebuilt):
            run.resume()
        with pytest.raises(ValueError, match="max_data_bytes"):
            run.resume(rebuild_preparations=True)
        assert run.trace.preparations == 3 and len(requests) == 1
        with pytest.raises(PreparationNotRebuilt):
            run.resume()
        run.extend_limits(max_data_bytes=None)
        run.resume(rebuild_preparations=True)
        assert run.trace.preparations == 4 and len(run.prepared_artifacts) == 2
        result = run.wait(timeout=30, poll_interval=0)
        assert result.value == value and run.trace.preparations == 5 and run.trace.jobs == 3 and len(requests) == 3
        assert {record.realization_id: record.runtime.seed for record in run.prepared_artifacts} == seeds


def circuit_and_qpy_size():
    """A circuit and the length of its QPY copy dumped to a plain stream, the oracle for the size b of the copy."""
    from io import BytesIO
    from qiskit import QuantumCircuit, qpy

    circuit = QuantumCircuit(2)
    for angle in np.linspace(.1, 3., 30):
        circuit.rz(angle, 0)
        circuit.cx(0, 1)
    stream = BytesIO()
    qpy.dump(circuit, stream)
    return circuit, len(stream.getvalue())


def test_the_qpy_copy_is_compared_with_the_stored_data_left_under_ample_memory(tmp_path, ibm_runtime_spy):
    """A copy of b bytes needs b bytes of max_data_bytes, whatever the memory allows, before the buffer takes them.

    The two limits stand b/2 below and 2b above what the new Run has counted, so the few bytes that the limit's own
    digits add to its saved row cannot decide the outcome.
    """
    from nwqlib._limits import ResourceLimitError
    from nwqlib.execution import ExecutionLimits

    circuit, size = circuit_and_qpy_size()
    backend = ibm_runtime_spy.connection

    def copy(extra_bytes, name, counted=0):
        with Run(counts_selection(), backend=backend, directory=tmp_path / name, progress=False,
                 limits=ExecutionLimits(max_data_bytes=counted + extra_bytes)) as run:
            left = run._available_data_bytes()
            return backend._qpy(circuit, run), counted + extra_bytes - left

    _, counted = copy(1 << 30, "sizing")
    assert len(copy(2 * size, "fits", counted)[0]) == size
    with pytest.raises(ValueError, match="max_data_bytes") as refused:
        copy(size // 2, "half", counted)
    assert not isinstance(refused.value, ResourceLimitError)


@pytest.mark.parametrize("source", ("max_bytes", "physical memory"))
def test_the_qpy_copy_is_compared_with_the_memory_limit_as_it_grows_without_a_data_limit(tmp_path, monkeypatch, ibm_runtime_spy,
                                                                                          source):
    """A QPY copy of b bytes is held twice, so 2b is the size that the memory limit admits, with max_data_bytes=None.

    The oracle for b is the length of the same circuit dumped to a plain stream. A limit of 2b admits the copy and
    2b - 1 refuses it. A limit of b refuses with at most half the copy in the buffer, so the comparison comes before
    the buffer takes the bytes.
    """
    import nwqlib._limits as limits
    from nwqlib._limits import ResourceLimitError
    from nwqlib.backends._qpy_buffer import QpyBuffer

    circuit, size = circuit_and_qpy_size()
    offsets, write = [], QpyBuffer.write

    def writes(self, data):
        written = write(self, data)
        offsets.append(self.tell())
        return written

    monkeypatch.setattr(QpyBuffer, "write", writes)

    def copy(limit, name):
        offsets.clear()
        if source == "physical memory":
            monkeypatch.setattr(limits, "physical_memory_bytes", lambda: limit)
        method = ExpectationMethod(max_bytes=limit if source == "max_bytes" else None)
        with Run(nwqlib.plan(problem(), method=method, shots=4, seed=7), backend=ibm_runtime_spy.connection,
                 directory=tmp_path / name, progress=False) as run:
            return ibm_runtime_spy.connection._qpy(circuit, run)

    assert len(copy(2 * size, "fits")) == size and max(offsets) == size
    with pytest.raises(ResourceLimitError, match="as QPY"):
        copy(2 * size - 1, "one less")
    with pytest.raises(ResourceLimitError, match="as QPY"):
        copy(size, "half")
    assert max(offsets, default=0) <= size // 2


@pytest.mark.parametrize("mode", ("batch", "session"))
def test_an_existing_container_opens_the_device_with_the_prepared_fractional_gates(tmp_path, runtime_api, mode):
    """Eight sampled LCHS rows prepared with fractional gates run in an existing Batch or Session container.

    The device is opened with ``use_fractional_gates=True`` for preparation and again for the
    container, so the SDK checks the circuits against the target they were compiled for and one job
    is acknowledged. A device opened without fractional gates lacks ``rzz`` and ``rx``.
    """
    from copy import copy
    from qiskit_ibm_runtime.utils.backend_converter import convert_to_target

    enabled, _ = fractional_device(runtime_api)
    disabled = copy(enabled)
    disabled._target = convert_to_target(enabled.configuration(), enabled.properties(), include_fractional_gates=False)
    assert "rzz" not in disabled.target.operation_names
    opened = []

    def device(name, *, use_fractional_gates=False, **kwargs):
        opened.append(use_fractional_gates)
        return enabled if use_fractional_gates else disabled

    runtime_api.service.backend = device
    runtime_api.api.container_mode = "batch" if mode == "batch" else "dedicated"
    connection = runtime_api.connection.revise(mode=mode, container_id="original-container", initial_layout=None,
                                               use_fractional_gates=True)
    with Run(sampled_lchs_plan(), backend=connection, directory=tmp_path / "run", progress=False) as run:
        run.resume()
        assert opened == [True, True] and len(runtime_api.requests) == 1
        assert [submission.status for submission in run.trace.submissions] == ["acknowledged"]


@pytest.mark.parametrize("reopen", (False, True))
def test_template_rows_refused_by_the_data_budget_keep_their_seeds_for_a_retry(tmp_path, monkeypatch, runtime_api,
                                                                               reopen):
    """The rows of eight sampled LCHS points on IBM are refused by max_data_bytes before any is saved.

    Raising the limit with ``run.extend_limits``, on the same Run or after reopening it, gives every
    row the seed of a Run whose budget was ample from the start. A stand-in stops each resume before
    template compilation, so nothing is prepared or sent.
    """
    import nwqlib._prepared_execution as execution
    from nwqlib.execution import ExecutionLimits

    class BeforeCompilation(Exception):
        pass

    monkeypatch.setattr(execution, "_bind_template", lambda *args, **kwargs: (_ for _ in ()).throw(BeforeCompilation()))
    connection = runtime_api.connection.revise(initial_layout=None)
    selected = sampled_lchs_plan()
    with Run(selected, backend=connection, directory=tmp_path / "size", progress=False) as sizing:
        created = sum(sizing._state[key] for key in ("data_bytes", "array_bytes", "payload_bytes", "external_bytes"))

    def seeds(folder, budget):
        run = Run(selected, backend=connection, directory=folder, progress=False,
                  limits=ExecutionLimits(max_data_bytes=budget))
        try:
            if budget < 2_000_000:
                with pytest.raises(ValueError, match="max_data_bytes"):
                    run.resume()
                assert not run._state["workflow_items"] and run.trace.preparations == 0
                if reopen:
                    run.close()
                    run = load_run(folder, backend=connection, progress=False)
                run.extend_limits(max_data_bytes=2_000_000)
            with pytest.raises(BeforeCompilation):
                run.resume()
            return {key: item.runtime.seed for key, item in run._state["workflow_items"].items()}
        finally:
            run.close()

    ample = seeds(tmp_path / "ample", 2_000_000)
    assert len(ample) == 8 and seeds(tmp_path / "retry", created + 1536) == ample
    assert not runtime_api.requests


def test_an_estimate_reserves_its_known_fields_before_submission(tmp_path, runtime_api):
    """A provider estimate's identity, value, uncertainty fields and options text are known before submission.

    The Run reserves them with the declaration. The estimate saves the effective options text a second time, so
    a longer request raises the reservation by exactly the JSON size of the added text. The submission is
    written with that reservation, so a limit 64 bytes below the prepared Run's bytes plus the reservation is
    refused before the request, while a limit 64 bytes above a submitted Run's bytes plus the reservation
    submits. The margins cover the few bytes by which saved timing fields vary between Runs.
    """
    from nwqlib.execution import ExecutionLimits
    runtime_api.api.status = "QUEUED"
    chosen, large = estimate_selection(), ExecutionLimits(max_data_bytes=4_000_000)

    def reservation(options, directory):
        backend = runtime_api.connection.revise(options_json=options)
        with Run(chosen, backend=backend, directory=tmp_path / directory, limits=large, progress=False) as run:
            run.resume()
            return run.trace.data_bytes, run.trace.data_bytes_reserved, backend
    short_text, long_text = '{"resilience_level":0}', '{"max_execution_time":300,"resilience_level":0}'
    stored, reserved, backend = reservation("{}", "default")
    _, longer, _ = reservation('{"max_execution_time": 300}', "longer")
    assert longer - reserved == len(json.dumps(long_text)) - len(json.dumps(short_text))
    with nwqlib.prepare(chosen, backend=backend, directory=tmp_path / "prepared", limits=large,
                        progress=False).run as run:
        prepared = run.trace.data_bytes
    assert len(runtime_api.requests) == 2
    with Run(chosen, backend=backend, directory=tmp_path / "short", progress=False,
             limits=ExecutionLimits(max_data_bytes=prepared + reserved - 64)) as run:
        with pytest.raises(ValueError, match=f"max_data_bytes={prepared + reserved - 64} cannot admit"):
            run.resume()
    assert len(runtime_api.requests) == 2
    with Run(chosen, backend=backend, directory=tmp_path / "enough", progress=False,
             limits=ExecutionLimits(max_data_bytes=stored + reserved + 64)) as run:
        run.resume()
    assert len(runtime_api.requests) == 3


def test_large_provider_metadata_of_a_paid_estimate_is_saved_after_one_fetch(tmp_path, runtime_api):
    """The provider metadata of an admitted estimate is saved at its actual size, with no limit on its encoding.

    73,728 characters of scheduler timing in the PUB's item metadata exceed max_data_bytes = 65,536. The
    first wait saves the estimate 2, the sum of the coefficients of I + Z for outcomes that are all 0, with all
    of its metadata and then refuses the derived result row, naming extend_limits. After the limit is raised
    the result is read from the saved estimate, so the job is submitted and fetched once.
    """
    from nwqlib.execution import ExecutionLimits
    api = runtime_api.api
    api.status = "QUEUED"
    backend = runtime_api.connection.revise(options_json="{}")
    directory = tmp_path / "estimate"
    with Run(estimate_selection(), backend=backend, directory=directory,
             limits=ExecutionLimits(max_data_bytes=65536), progress=False) as run:
        run.resume()
        locator = run.trace.submissions[0].locator
    api.status, api.item_metadata = "COMPLETED", [{"scheduler_timing": {"timing": "x" * 73728, "circuit_duration": 1}}]
    with load_run(directory, backend=backend) as run:
        with pytest.raises(ValueError, match="extend_limits"):
            run.wait(timeout=1, poll_interval=0)
        estimate, = run.observations.chunks[0].values
        assert estimate.value == 2.0
        timing = json.loads(estimate.provider_metadata_json)["pub"]["compilation"]["scheduler_timing"]["timing"]
        assert timing == "x" * 73728
    with load_run(directory, backend=backend) as run:
        run.extend_limits(max_data_bytes=4_000_000)
        assert run.wait(timeout=1, poll_interval=0).value == 2.0
        assert run.trace.submissions[0].locator == locator
    assert len(runtime_api.requests) == api.reads == 1


def test_sampler_decode_arrays_are_admitted_before_submission(tmp_path, runtime_api):
    """The requested shots bound the Sampler decode, so preparation can refuse before IBM receives a job."""
    from nwqlib.backends.ibm_runtime import sampler_decode_bytes
    from nwqlib.execution import ExecutionLimits
    decode_bytes = sampler_decode_bytes((1,), 30000)
    chosen = nwqlib.plan(problem((("I", 1.), ("Z", 1.))), method=ExpectationMethod(), shots=30000, seed=7)
    with Run(chosen, backend=runtime_api.connection, directory=tmp_path / "run",
             limits=ExecutionLimits(max_data_bytes=200000), progress=False) as run:
        with pytest.raises(ValueError, match=rf"{decode_bytes} bytes exceeds this Run's max_data_bytes=200000"):
            run.resume()
    assert not runtime_api.requests


@pytest.mark.parametrize(("kind", "options", "reason"), [
    pytest.param("counts", '{"twirling":{"enable_measure":true}}', "twirling.num_randomizations",
                 id="automatic-twirling-split"),
    pytest.param("counts", '{"twirling":{"enable_gates":true,"num_randomizations":3,"shots_per_randomization":2}}',
                 "product is the 4 shots", id="twirling-split-of-other-shots"),
    pytest.param("counts", '{"twirling":{"enable_measure":true,"num_randomizations":1,"shots_per_randomization":2}}',
                 "product is the 4 shots", id="twirling-split-of-fewer-shots"),
    pytest.param("counts", '{"environment":{"private":true}}', "only once", id="private-counts"),
    pytest.param("estimate", '{"environment":{"private":true}}', "only once", id="private-estimate"),
])
def test_a_private_job_or_a_twirling_split_of_other_shots_is_refused_before_the_submission_intent(
        tmp_path, runtime_api, kind, options, reason):
    """IBM returns a private job's result only once, so a Run could not read it again, and a twirled Sampler
    without a split equal to the requested shots can return another number of shots. Both requests are refused
    during preparation: the Run records no submission and no circuit, shot or provider-managed exposure, and no
    request is sent. Each refusal names its cause."""
    from nwqlib.execution import ExecutionLimits
    chosen = counts_selection() if kind == "counts" else estimate_selection()
    backend = runtime_api.connection.revise(options_json=options)
    with Run(chosen, backend=backend, directory=tmp_path / "run", progress=False,
             limits=ExecutionLimits(max_data_bytes=4_000_000)) as run:
        with pytest.raises(ValueError, match=reason):
            run.resume()
        assert not run.trace.submissions
        assert all(not any(counts.values()) for counts in run.exposure.values())
    assert not runtime_api.requests


def test_twirled_counts_with_an_explicit_split_return_the_requested_shots(tmp_path, runtime_api):
    """Measurement twirling with num_randomizations * shots_per_randomization equal to the requested 4 shots is sent
    through the real SDK, whose decoder undoes the measurement flips that the synthetic service applied. The counts
    are the service's outcomes 0, 0, 0, 1 for the one PUB."""
    connection = runtime_api.connection.revise(
        options_json='{"twirling":{"enable_measure":true,"num_randomizations":1,"shots_per_randomization":4}}')
    with Run(counts_selection(), backend=connection, directory=tmp_path / "run", progress=False) as run:
        run.resume()
        runtime_api.api.status = "COMPLETED"
        (chunk,) = refresh_submissions(run=run)
    assert executor_program(runtime_api.requests[0]["params"]).passthrough_data["post_processor"]["twirling"]
    histogram = chunk.histogram()
    assert chunk.returned_shots == 4 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 3, 1: 1}


def test_each_point_of_an_estimator_batch_is_checked_before_the_first_preparation(tmp_path, monkeypatch,
                                                                                    runtime_api):
    """A Method that specializes each point's estimate has every point checked against the target first.

    The second point's coefficients (0, 0) make a zero observable, which IBM's Estimator refuses, so
    ``prepare`` refuses with that point's key before any preparation or request.
    """
    import nwqlib._prepared_execution as execution
    from nwqlib.core.planning import ReadoutDetails
    from test_selected_runtime import AxisToy, axis_plan

    class PointEstimate(AxisToy):
        def specialize_experiment(self, plan, experiment, bindings):
            estimate = experiment.readout.estimate.revise(coefficients=(0.0, 1.0 - bindings["row"]))
            return experiment.revise(readout=experiment.readout.revise(estimate=estimate))

    chosen = axis_plan(2, method=PointEstimate())
    program = chosen.construction.program
    definitions = []
    for definition in program.definitions:
        if definition.id == "axis":
            definition = definition.revise(node=definition.node.revise(observation_kind="estimated_observable",
                                                                       repetitions=1))
        elif definition.id == "body":
            definition = definition.revise(node=definition.node.revise(children=("allocate", "release")))
        elif definition.id == "measure":
            continue
        definitions.append(definition)
    estimate = estimate_selection().experiments[0].observation.estimate
    chosen = chosen.revise(
        construction=chosen.construction.revise(program=program.revise(classical=(), definitions=tuple(definitions))),
        experiments=(chosen.experiments[0].revise(readout=ReadoutDetails(estimate=estimate)),))
    prepared = []
    original = execution._prepare_static_item
    monkeypatch.setattr(execution, "_prepare_static_item",
                        lambda *args, **kwargs: prepared.append(1) or original(*args, **kwargs))
    with pytest.raises(ValueError, match=r"Experiment 'series\[row=1\]' requests readout kind "
                                         r"'estimated_observable': IBM Estimator does not accept a zero observable"):
        nwqlib.prepare(chosen, backend=runtime_api.connection.revise(initial_layout=None),
                       directory=tmp_path / "run", progress=False)
    assert prepared == [] and not runtime_api.requests


@pytest.mark.parametrize("fractional", [False, True], ids=["compiled-template", "folded-classes"])
def test_template_experiments_record_the_depth_of_their_bound_class_circuit(runtime_api, tmp_path, monkeypatch,
                                                                             fractional):
    """Each template experiment records the qualified depth of its class circuit, scanned once per class.

    Every experiment of a class runs the same gates on the same qubits with its own values, so the record equals
    the depth of each experiment's bound circuit. Folding RZZ angles adds only one-qubit gates, so the classes
    share their two-qubit depth. Gate twirling, which IBM allows without fractional gates, leaves the depth
    unqualified with the option as the reason.
    """
    from nwqlib.backends import inspection

    target =fractional_device(runtime_api)[0].target if fractional else runtime_api.backend.target
    connection = runtime_api.connection.revise(initial_layout=None, use_fractional_gates=fractional)
    selected = sampled_lchs_plan(time=2.0, lcu_select="structured")
    scans, scan = [], inspection.inspect_native_depth
    monkeypatch.setattr(inspection, "inspect_native_depth",
                        lambda circuit, *args, **kwargs: (scans.append(circuit.num_parameters),
                                                          scan(circuit, *args, **kwargs))[1])
    monkeypatch.setattr(IBMRuntimeBackend, "admit_batch", lambda self, natives: (_ for _ in ()).throw(
        ValueError("prepare every row without submitting")))

    def prepared_rows(backend, directory):
        """Prepare every row of the Plan, refusing the job at admission, and return the open Run."""
        nwqlib.prepare(selected, backend=backend, directory=directory).run.close()
        run = load_run(directory, backend=backend)
        with pytest.raises(ValueError, match="without submitting"):
            run.resume()
        assert not run.trace.submissions and not runtime_api.requests
        return run

    run = prepared_rows(connection, tmp_path / "depth")
    classes = {}
    try:
        for record in run.prepared_artifacts:
            circuit = run._state["handles"][record.content_id]._native.circuit
            values = dict(zip(record.template.parameters, record.template.values, strict=True))
            bound = circuit.assign_parameters({parameter: values[parameter.name] for parameter in circuit.parameters})
            assert record.native_depth.qualified and record.native_depth == scan(bound, target)
            classes.setdefault((record.template.key, record.template.correction), set()).add(record.native_depth)
    finally:
        run.close()
    assert all(len(depths) == 1 for depths in classes.values())
    # One scan per class, of its bound circuit, and none of an unbound template.
    assert scans == [0] * len(classes) and len(run.prepared_artifacts) > len(classes)
    scans.clear()
    with load_run(tmp_path / "depth", backend=connection) as reopened:
        assert [record.native_depth for record in reopened.prepared_artifacts] == [
            record.native_depth for record in run.prepared_artifacts]
    assert not scans
    if fractional:
        by_key = {}
        for (key, _), (depth,) in classes.items():
            by_key.setdefault(key, set()).add(depth.native_two_qubit_depth)
        assert len(classes) > len(by_key) and all(len(depths) == 1 for depths in by_key.values())
    if not fractional:
        # IBM refuses gate twirling with fractional gates, so this check uses the compiled template.
        twirled = '{"twirling":{"enable_gates":true,"num_randomizations":1,"shots_per_randomization":2}}'
        run = prepared_rows(connection.revise(options_json=twirled), tmp_path / "twirled")
        try:
            reasons = {record.native_depth.reason for record in run.prepared_artifacts}
        finally:
            run.close()
        assert len(reasons) == 1 and "twirling.enable_gates=True" in reasons.pop()
