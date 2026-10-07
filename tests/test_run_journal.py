"""Atomic current Run transitions, original pending jobs and bounded SQLite data."""

import builtins
import os
from pathlib import Path

import pytest

from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment
from nwqlib._run_archive import load
from nwqlib._run_journal import LocalJournal
from nwqlib.execution import ExecutionLimits, PayloadRef
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend
from test_saved_evidence import supplied_qhd as supplied_qhd


def test_binary_payload_roundtrip_reuses_storage_and_preserves_plain_saved_bytes(tmp_path):
    data = b"an immutable native input"
    ref = PayloadRef(format="test/bytes", key="original-input", bytes=len(data))
    path = tmp_path / "bytes.sqlite"
    store = LocalJournal(path, 4096, create=True)
    store.commit((("payload", ref.content_id, ref),), payloads=((ref, data),))
    with pytest.raises(ValueError, match="duplicate binary"):
        store.commit(payloads=((ref, data), (ref, data)))
    before = store.usage.copy()
    store.commit((("payload", ref.content_id, ref),), payloads=((ref, data),))
    assert store.usage == before and store.read_payload(ref) == data
    store.close()
    store = LocalJournal(path, 4096, create=False)
    try:
        assert store.read_payload(ref) == data
        statements = []
        store.connection.set_trace_callback(statements.append)
        with pytest.raises(ValueError, match="missing or reordered"):
            store.read_payload(ref.revise(bytes=len(data) + 1))
        assert not any("SELECT payload FROM binary" in query for query in statements)
    finally:
        store.close()


def test_interrupted_binary_write_publishes_neither_reference_nor_partial_bytes(tmp_path, monkeypatch):
    import nwqlib._run_journal as writer

    data = b"partly written native input"
    ref = PayloadRef(format="test/bytes", key="interrupted-input", bytes=len(data))
    path = tmp_path / "binary.sqlite"
    store = LocalJournal(path, 4096, create=True)
    original = writer._write_bytes

    def interrupted(*args):
        original(*args)
        raise OSError("interrupted binary write")

    monkeypatch.setattr(writer, "_write_bytes", interrupted)
    with pytest.raises(OSError, match="interrupted binary"):
        store.commit((("payload", ref.content_id, ref),), payloads=((ref, data),))
    assert not list(store.rows("payload")) and store.usage["data_bytes"] == 0
    store.close()
    store = LocalJournal(path, 4096, create=False)
    assert store.connection.execute("SELECT count(*) FROM binary").fetchone()[0] == 0
    store.close()


def test_whole_delta_admission_precedes_json_and_atomic_replacement(tmp_path, monkeypatch):
    import nwqlib._run_journal as storage

    store = LocalJournal(tmp_path / "delta.sqlite", 128, create=True)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(
                storage.json, "dumps", lambda *a, **k: pytest.fail("unadmitted delta reached JSON output")
            )
            with pytest.raises(ValueError, match="byte limit"):
                # Each 66-byte JSON string fits alone, but their 132-byte
                # transaction exceeds the 128-byte cap.
                store.commit((("row", "a", "x" * 64), ("row", "b", "y" * 64)))
        assert list(store.rows("row")) == []
        store.commit((("row", "a", "x"), ("row", "b", "y")))
        before = store.usage["data_bytes"]
        store.commit((("row", "a", "z"),))
        assert store.usage["data_bytes"] == before
        assert [value for _, value, _ in store.rows("row")] == ["z", "y"]
        with pytest.raises(ValueError, match="duplicate row"):
            store.commit((("row", "a", 1), ("row", "a", 2)))
        store.commit((("cache", "old", "remove"),))
        old_bytes = store.usage["data_bytes"]
        store.commit((("cache", "new", "keep"),), deleted=(("cache", "old"),))
        assert list(store.rows("cache")) == [("new", "keep", False)]
        assert store.usage["data_bytes"] == old_bytes - 2
        for deleted in ((("cache", "new"),), (("chunk", "published"),)):
            with pytest.raises(ValueError, match="conflict|only cache"):
                store.commit((("cache", "new", "bad"),), deleted=deleted)
        assert list(store.rows("cache")) == [("new", "keep", False)]
        before = store.usage["data_bytes"]
        with pytest.raises(ValueError, match="byte limit"):
            store.commit((("cache", "new", "x"*128),), deleted=(("cache", "absent"),))
        assert store.usage["data_bytes"] == before
        assert list(store.rows("cache")) == [("new", "keep", False)]
    finally:
        store.close()


def test_run_encodes_each_changed_row_once_and_rolls_back_delete(tmp_path, monkeypatch):
    import nwqlib._run_journal as storage
    import sqlite3

    with Run(selected(), backend=InjectedBackend(), directory=tmp_path / "once") as run:
        calls = []
        original = storage._encode_admitted

        def encoded(value, limit):
            calls.append(value)
            return original(value, limit)

        monkeypatch.setattr(storage, "_encode_admitted", encoded)
        run._write((("cache", "one", {"scientific": 3}),))
        assert calls == [{"scientific": 3}]
        before = run._state["data_sizes"].copy(), run._state["data_bytes"]
        journal = run._state["journal"]
        journal.connection.execute("CREATE TRIGGER reject_cache_delete BEFORE DELETE ON records "
                                   "WHEN OLD.kind='cache' BEGIN SELECT RAISE(ABORT, 'interrupted delete'); END")
        with pytest.raises(sqlite3.IntegrityError, match="interrupted delete"):
            run._write(deleted=(("cache", "one"),))
        assert (run._state["data_sizes"], run._state["data_bytes"]) == before
        assert list(journal.rows("cache")) == [("one", {"scientific": 3}, False)]
        expected = journal.connection.execute("SELECT sum(length(CAST(payload AS BLOB))) FROM records").fetchone()[0]
        assert journal.usage["data_bytes"] == expected


@pytest.mark.parametrize("durable", [False, True])
def test_actual_collection_order_survives_save_and_duplicate_collection(tmp_path, monkeypatch, durable):
    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    # With durable=False the detached Run uses the default run directory; keep it in tmp_path.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    source = tmp_path / "source"
    with Run(selected(), backend=InjectedBackend(), directory=source if durable else None) as run:
        handles = [prepare_experiment(run.plan.resolve("counts"), run=run) for _ in range(2)]
        chunks = [submit_experiment(handle, run=run) for handle in handles]
        assert run.collect(chunks[1]) and run.collect(chunks[0])
        before = run.observations
        assert not run.collect(chunks[1]) and run.observations == before
        path = run.save(tmp_path / "saved")
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert [c.content_id for c in restored.observations.chunks] == [chunks[1].content_id, chunks[0].content_id]
        assert not restored.collect(chunks[1])


def test_a_chunk_row_keeps_its_readout_only_through_an_equal_receipt(tmp_path, monkeypatch):
    # Reopening restores a chunk's observation from its receipt (the reopened
    # identities above), so a chunk declaring another readout must not be stored.
    import json

    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    with Run(selected(), backend=InjectedBackend(), directory=tmp_path / "row") as run:
        chunk = submit_experiment(prepare_experiment(run.plan.resolve("counts"), run=run), run=run)
        row, = run._state["journal"].connection.execute("SELECT payload FROM records WHERE kind='chunk'").fetchone()
        assert "observation" not in json.loads(row) and json.loads(row)["prepared_id"] == chunk.prepared_id
        changed = chunk.revise(observation=chunk.observation.revise())
        with pytest.raises(ValueError, match="receipt's readout"):
            run._write((("chunk", changed.content_id, changed),))


def test_one_binary_identity_keeps_two_actual_publication_producers(tmp_path):
    import numpy as np
    from nwqlib._prepared_execution import submit_detached
    from nwqlib.artifacts import ArrayOutput
    from nwqlib.core.records import Basis, Source

    with Run(selected(), backend=InjectedBackend(), directory=tmp_path / "two-outputs") as run:
        # One actual reserved acquisition owns both publications and their output allowance.
        prepared = prepare_experiment(run.plan.resolve("counts"), run=run)
        submission = submit_detached((prepared,), run=run)
        item, = submission.items
        acquisition = (run.run_id, item.attempt, submission.locator.job_id, item.result_key)
        values = np.array([1+2j, 3-4j])
        output = ArrayOutput(name="computed vector", kind="vector",
            basis=Basis(identity="test", dimension=2, ordering="increasing index"),
            frame="physical", global_phase="physical")
        producers = tuple(Source(name=name, version="1", domain="two independently declared outputs",
                                 reference="explicit tiny vector publication")
                          for name in ("first acquisition producer", "second acquisition producer"))
        provenances = [dict(plan_id=run.plan.content_id, realization_id=prepared.realization.content_id,
                            construction_id=prepared.record.construction_id, producer_id=producer.content_id,
                            acquisition=acquisition, source=producer) for producer in producers]
        # Outside an acquisition outcome, the same legal publication is rejected
        # before its reservation reaches the journal.
        journal = run._state["journal"]
        rows = journal.connection.execute("SELECT count(*) FROM records").fetchone()
        with pytest.raises(ValueError, match="requires its acquisition outcome"):
            run.artifacts._publish(values, output=output, provenance=provenances[0])
        assert journal.connection.execute("SELECT count(*) FROM records").fetchone() == rows
        assert not run.data.artifacts
        run._start_publications()
        handles = [run.artifacts._publish(values, output=output, provenance=provenance)
                   for provenance in provenances]
        run._flush_publications(())
        assert handles[0].manifest.content_id != handles[1].manifest.content_id
        assert handles[0].manifest.digest == handles[1].manifest.digest
        assert journal.connection.execute("SELECT sum(length(payload)) FROM binary").fetchone()[0] == values.nbytes
        assert len(run.data.artifacts) == 2 and len(list(journal.rows("payload"))) == 1
        # Each manifest keeps its own supplied producer, both in the handle and the store inventory.
        assert [h.manifest.producer_id for h in handles] == [p.content_id for p in producers]
        assert {h.manifest.producer_id for h in run.data.artifacts} == {p.content_id for p in producers}
        assert run._state["array_bytes"] == 2*values.nbytes


def test_record_fields_are_admitted_before_allocating_portable_metadata(monkeypatch):
    from nwqlib._run_journal import encode
    from nwqlib.core.records import Source

    source = Source(name="supplied provenance", version="1", domain="test metadata",
                    reference="x" * 512)
    with monkeypatch.context() as patch:
        patch.setattr(Source, "model_dump", lambda *a, **k: pytest.fail("metadata copied before byte admission"))
        with pytest.raises(ValueError, match="byte limit"):
            encode(source, 128)
    assert Source.model_validate_json(encode(source, 8192)) == source


@pytest.mark.parametrize("kind", ("counts", "probabilities", "pauli_expectation", "host_scalars"))
def test_readout_reservation_covers_commit_statistic_envelope(kind):
    """A 25-wire declaration needs one prototype, with no outcome enumeration."""
    from types import SimpleNamespace
    from nwqlib._prepared_execution import readout_bytes
    from nwqlib._run_journal import _json_bound
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.execution import CountBin, PauliValue, ScalarValue, RegisterMap

    width, shots, allowance = 25, 100_000, 65_536
    labels = ("X"*width,) if kind == "pauli_expectation" else ("a\"\\\n中",)
    extra = dict(shots=shots) if kind == "counts" else (
        dict(qubits=tuple(range(width))) if kind == "probabilities" else dict(labels=labels))
    spec = ObservationSpec(kind=kind, **extra)
    layout = (RegisterMap(name="bits", bits=tuple(range(width))),)
    item, count = {
        "counts": (CountBin(bits="1"*width, count=shots), shots),
        "probabilities": (None, 1 << width),
        "pauli_expectation": (PauliValue(label="X"*width, value=-1e-300), 1),
        "host_scalars": (ScalarValue(label=labels[0], value=-1e300, frame="encoded_branch",
                                     parent_id="sha256:"+"a"*64), 1),
    }[kind]
    handle = SimpleNamespace(record=SimpleNamespace(observation=spec, quantum_layout=layout,
        classical_layout=layout if kind == "counts" else ()), _items=count, _bindings=(),
        _setting="scalar sizing", realization=SimpleNamespace(experiment="size"),
        _native=SimpleNamespace(record=SimpleNamespace(scalars=labels, data_bytes=0, application_bytes=0)))
    reserved = readout_bytes(handle, metadata_bytes=allowance)
    # The actual commit owner supplies the bound per JSON statistic. Multiplication
    # tests large cardinalities while allocating only this one scalar record.
    # Probabilities are a binary array: eight bytes per possible outcome.
    required = 8*count if item is None else count*_json_bound(item, allowance) + max(0, count-1)
    assert reserved >= required + allowance
    assert readout_bytes(handle, metadata_bytes=1) == reserved-allowance+1
    if kind == "host_scalars":
        # Receipt bytes a kernel declares join the fixed reservation exactly.
        handle._native.record.application_bytes = 1234
        assert readout_bytes(handle, metadata_bytes=allowance) == reserved+1234


def test_declared_pauli_population_does_not_spend_variable_completion_allowance():
    """Thousands of declared labels are fixed input size, not provider metadata."""
    from itertools import islice, product
    from types import SimpleNamespace
    from nwqlib._prepared_execution import readout_bytes, _completion_metadata_bound
    from nwqlib._run_journal import _json_bound
    from nwqlib.core.planning import ObservationSpec
    from nwqlib.core.records import Source
    from nwqlib.execution import ObservationChunk, PauliValue, RegisterMap

    labels = tuple("I"*19+"".join(p) for p in islice(product("IXYZ", repeat=6), 3000))
    spec = ObservationSpec(kind="pauli_expectation", labels=labels)
    layout = (RegisterMap(name="q", bits=tuple(range(25))),)
    handle = SimpleNamespace(record=SimpleNamespace(observation=spec, quantum_layout=layout,
        classical_layout=()), _items=len(labels), _bindings=(), _setting="pauli",
        realization=SimpleNamespace(experiment="energy"))
    limit, allowance = 1_000_000, 65_536
    assert _json_bound(spec, limit) > allowance
    reserved = readout_bytes(handle, metadata_bytes=allowance, max_bytes=limit)
    identity = "sha256:"+"0"*64
    chunk = ObservationChunk(run_id="run", plan_id=identity, realization_id=identity,
        prepared_id=identity, experiment="energy", setting="pauli", bindings=(),
        quantum_layout=layout, classical_layout=(), attempt="attempt", job="job", chunk="0",
        observation=spec, population="unconditional", returned_shots=None, trajectories=1,
        values=tuple(PauliValue(label=label, value=0.) for label in labels),
        source=Source(name="injected", version="1", domain="exact scalar records", reference="test"))
    completed, variable = _completion_metadata_bound((("chunk", "0", chunk),), limit)
    assert completed == 1 and variable < allowance
    assert _json_bound(chunk, limit) <= reserved
    small = ObservationChunk(**{**{name: getattr(chunk, name) for name in type(chunk).model_fields},
        "observation": spec.revise(labels=labels[:1]), "values": chunk.values[:1]})
    assert _completion_metadata_bound((("chunk", "0", small),), limit) == (completed, variable)
    enlarged = chunk.revise(source=chunk.source.revise(reference="x"*allowance))
    assert _completion_metadata_bound((("chunk", "0", enlarged),), limit)[1] > allowance


def test_json_bound_covers_real_utf8_encoding_without_ascii_size_inflation():
    import json
    from nwqlib._run_journal import _json_bound, _portable, encode
    from nwqlib.execution import ScalarValue

    value = dict(text="quoted \" \\ \n 中 🙂", empty=(), integers=(0, -999999),
                 flags=(True, False, None), statistic=ScalarValue(label="energy", value=-1e-300))
    encoded = json.dumps(value, default=_portable, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    bound = _json_bound(value, 1000)
    assert len(encoded.encode("utf-8")) <= bound
    assert encode(value, bound) == encoded
    # Fixed-width ASCII identifiers dominate the repeated batch header. Their
    # exact size prevents sixfold phantom growth each time that row changes.
    assert _json_bound("x"*100, 102) == 102


def test_failed_preparation_persistence_stops_before_native_and_recovers_same_run(tmp_path, monkeypatch):
    path = tmp_path / "prepare"
    run = Run(selected(), backend=InjectedBackend(), directory=path)
    before = run.trace

    def failed(*args, **kwargs):
        raise OSError("before preparation")

    monkeypatch.setattr(run._state["journal"], "commit", failed)
    with pytest.raises(OSError, match="before preparation"):
        prepare_experiment(run.plan.resolve("counts"), run=run)
    assert run.trace == before and InjectedBackend.calls == []
    with pytest.raises(ValueError, match="load the durable run"):
        prepare_experiment(run.plan.resolve("counts"), run=run)
    run.close()
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        handle = prepare_experiment(restored.plan.resolve("counts"), run=restored)
        assert handle.record.plan_id == run.plan.content_id and restored.trace.preparations == 1


@pytest.mark.parametrize("window", ["intent", "completion", "after_commit"])
def test_interrupted_transition_preserves_actual_committed_boundary(tmp_path, monkeypatch, window):
    InjectedBackend.supports_synchronous = True
    path = tmp_path / window
    run = Run(selected(), backend=InjectedBackend(), directory=path)
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    store = run._state["journal"]
    commit = store.commit

    def interrupted(records=(), **kwargs):
        wanted = "reserved" if window == "intent" else "completed"
        hit = any(kind == "event" and getattr(value, "status", None) == wanted for kind, _, value in records)
        if hit and window != "after_commit":
            raise OSError("interrupted transition")
        commit(records, **kwargs)
        if hit:
            raise OSError("interrupted transition")

    monkeypatch.setattr(store, "commit", interrupted)
    with pytest.raises(OSError, match="interrupted transition"):
        submit_experiment(handle, run=run)
    run.close()
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        if window == "intent":
            assert not restored.trace.events and not any(
                kind == "launch" for kind, _ in InjectedBackend.calls
            )
        elif window == "completion":
            # The earlier acknowledgement committed: this is an original known
            # provider job with missing local completion, so it remains pending.
            assert restored.exposure["reserved"]["shots"] == 7
            with pytest.raises(ValueError, match="completed"):
                restored.completed_observation(restored.trace.events[0].attempt)
        else:
            (event,) = restored.trace.events
            assert event.status == "completed" and restored.exposure["completed"]["shots"] == 7
            chunk = restored.completed_observation(event.attempt)
            assert restored.collect(chunk) and not restored.collect(chunk)
            assert len(restored.observations.chunks) == 1
        assert not any(kind == "refresh" for kind, _ in InjectedBackend.calls)


def test_lock_alias_and_checkpoint_bound_preserve_original_controller(tmp_path, monkeypatch):
    import nwqlib._run_journal as storage

    path = tmp_path / "run"
    run = Run(
        selected(), backend=InjectedBackend(), directory=path, limits=ExecutionLimits(max_data_bytes=100_000)
    )
    assert run._state["journal"].connection.execute("PRAGMA synchronous").fetchone()[0] == 3
    alias = tmp_path / "alias"
    alias.symlink_to(path, target_is_directory=True)
    for directory in (path, alias):
        with pytest.raises(BlockingIOError):
            load(directory, backend=InjectedBackend(), method=CountsMethod)
    before = run._state["journal"].usage.copy()
    with monkeypatch.context() as patch:
        patch.setattr(storage.json, "dumps", lambda *a, **k: pytest.fail("oversized checkpoint encoded"))
        with pytest.raises(ValueError, match="max_data_bytes=100000"):
            run.checkpoint({"too_large": "x" * 100_000})
    assert run._state["journal"].usage == before
    run.checkpoint({"next": 1})
    run.close()
    assert (path / "run.sqlite.lock").exists()
    with pytest.raises(FileExistsError):
        Run(selected(), backend=InjectedBackend(), directory=path)
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.checkpoint_state == {"next": 1}


def test_method_context_factory_failure_and_lifetime_are_actual_run_owned():
    run = Run(selected())
    failure = RuntimeError("factory failure")

    def failed():
        raise failure

    with pytest.raises(RuntimeError) as caught:
        run._method_context(failed)
    assert caught.value is failure
    calls = []

    def factory():
        calls.append(object())
        return calls[-1]

    context = run._method_context(factory)
    assert run._method_context(factory) is context and len(calls) == 1
    with Run(selected()) as other:
        assert other._method_context(factory) is not context
    run.close()
    assert run._state["method_context"] is None and len(calls) == 2
    with pytest.raises(ValueError, match="closed"):
        run._method_context(factory)


def test_in_memory_checkpoint_needs_no_optional_sqlite_or_fcntl(monkeypatch):
    original = builtins.__import__

    def import_without(name, *args, **kwargs):
        if name.split(".")[0] in {"sqlite3", "fcntl"}:
            raise ImportError("optional module absent")
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without)
    with Run(selected()) as run:
        assert run.checkpoint({"scalar": 2, "rng": [1, 2, 3]}) == 1
        assert run.checkpoint_state == {"scalar": 2, "rng": [1, 2, 3]}


@pytest.mark.skipif(os.name != "posix", reason="actual child-process crash uses fork and POSIX lock")
@pytest.mark.parametrize("completed", [False, True])
def test_process_exit_reopens_original_job_rng_and_once_only_result(tmp_path, monkeypatch, completed):
    path = tmp_path / "process"
    child = os.fork()
    if child == 0:
        try:
            run = Run(selected(), backend=InjectedBackend(), directory=path)
            run.resume()
            if completed:
                InjectedBackend.status = "completed"
                run.resume()
            os._exit(0)
        except BaseException:
            os._exit(1)
    _, status = os.waitpid(child, 0)
    assert os.waitstatus_to_exitcode(status) == 0
    monkeypatch.setattr(CountsMethod, "plan", lambda *a, **k: pytest.fail("reopen replanned"))
    InjectedBackend.status = "completed"
    with load(path, backend=InjectedBackend(), method=CountsMethod) as run:
        rng = run.rng.snapshot()
        result = run.wait(timeout=1, poll_interval=0)
        assert result.counts == 7 and run.trace.jobs == 1 and run.rng.snapshot() == rng
        assert not any(kind in {"prepare", "launch"} for kind, _ in InjectedBackend.calls)
        assert len(run.observations.chunks) == 1
        assert sum(kind == "refresh" for kind, _ in InjectedBackend.calls) == int(not completed)


@pytest.mark.parametrize("failure", [False, True])
def test_array_and_completed_observation_publish_in_one_atomic_transition(
    tmp_path, monkeypatch, supplied_qhd, failure
):
    """Fail after writing array bytes and require rollback of both payload and completed
    observation.
    """
    import numpy as np
    import nwqlib._run_journal as writer
    import nwqlib

    case = supplied_qhd
    path = tmp_path / "arrays"
    original = writer._write_bytes

    def interrupted(*args):
        original(*args)
        raise OSError("array completion interrupted")

    run = Run(case.plan, directory=path)
    if failure:
        stored = []
        flush = Run._flush_publications

        def observed(self, *args, **kwargs):
            stored.append(self.trace.data_bytes)
            return flush(self, *args, **kwargs)

        monkeypatch.setattr(writer, "_write_bytes", interrupted)
        monkeypatch.setattr(Run, "_flush_publications", observed)
        with pytest.raises(OSError, match="array completion interrupted"):
            run.resume()
        assert not run.data.artifacts and run.artifacts.data_bytes == 0
        # The failed outcome leaves the stored byte count at its pre-outcome value.
        assert run.trace.data_bytes == stored[-1]
    else:
        run.resume()
        assert run.result.valid_mass == 0.75 and len(case.calls) == 1
    run.close()
    from nwqlib.backends import AerBackend

    with nwqlib.load_run(path, backend=AerBackend()) as restored:
        if failure:
            assert not restored.observations.chunks and not restored.data.artifacts
            assert restored.exposure["uncertain"]["circuits"] == 1
            assert (
                restored._state["journal"].connection.execute("SELECT count(*) FROM binary").fetchone()[0]
                == 0
            )
        else:
            result = restored.result
            np.testing.assert_array_equal(result.data.artifact(result.artifact).array, case.raw)
            assert not result.data.artifact(result.artifact).array.flags.writeable
            assert len(restored.observations.chunks) == 1 and restored.trace.jobs == 1
        assert len(case.calls) == 1


def test_aer_process_exit_preserves_prefix_and_raises_owned_unavailable_recovery(tmp_path, monkeypatch):
    import subprocess
    import sys
    import nwqlib
    from nwqlib.execution import RunFailed

    path = tmp_path/'interrupted-aer'
    code = '''
import os, sys
from types import SimpleNamespace
import nwqlib
from nwqlib.backends import qiskit_aer
from nwqlib.algorithms.expectation import ExpectationMethod
calls = 0
def execute(prepared, **kwargs):
    global calls
    calls += 1
    if calls == 2:
        os._exit(0)
    return SimpleNamespace(raw_output={"counts": {"0": prepared.shots}},
        metadata={"native_job_id": "first-acquisition"})
qiskit_aer._submit_aer_execution = execute
plan = nwqlib.plan(nwqlib.Expectation(state=[1,1], observable=[[1,1],[1,-1]]), method=ExpectationMethod(), shots=8)
run = nwqlib.prepare(plan, directory=sys.argv[1]).run
run.resume()
raise AssertionError("second acquisition did not interrupt")
'''
    subprocess.run([sys.executable, '-c', code, str(path)], check=True, timeout=30)
    from nwqlib.backends import qiskit_aer
    monkeypatch.setattr(qiskit_aer, '_submit_aer_execution', lambda *a, **k: pytest.fail('native replay'))
    monkeypatch.setattr('nwqlib._prepared_execution.sleep', lambda *a: pytest.fail('impossible pending loop'))
    from nwqlib.backends import AerBackend
    with nwqlib.load_run(path, backend=AerBackend()) as run:
        assert len(run.observations.chunks) == 1
        assert [event.status for event in run.trace.events] == ['completed', 'uncertain']
        original = run.trace
        assert run.exposure['completed']['shots'] == run.exposure['uncertain']['shots'] == 8
        for continuation in (run.resume, run.wait, lambda: run.wait(timeout=0)):
            with pytest.raises(RunFailed) as caught:
                continuation()
            assert (caught.value.stage, caught.value.status) == ('recovery', 'uncertain')
            assert caught.value.attempt == original.events[1].attempt
            assert run.trace.events == original.events
            assert run.observations.chunks[0].returned_shots == 8


def test_manifests_sharing_one_payload_digest_read_each_with_its_own_dtype(tmp_path, monkeypatch):
    """Two restored manifests with equal bytes share one buffer and keep their own dtypes.

    A k = 3 sparse marginal whose uint64 indices [1, 6] and float64 values
    have identical bytes publishes two manifests with one digest. The
    reopened Run's store reads that payload once and gives each manifest a
    view of its own dtype.
    """
    import numpy as np
    from types import SimpleNamespace
    import nwqlib
    from nwqlib.backends import AerBackend
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.core.planning import Experiment, ObservationSpec
    from nwqlib.execution import Run, prepare_experiment, submit_experiment
    from nwqlib.ir import Allocate, Definition, Program, Register, Release, Sequence
    from test_prepared_execution import base_plan

    k = 3
    program = Program(root="body", registers=(Register(name="q", width=k),), definitions=(
        Definition(id="allocate", node=Allocate(wire="q")), Definition(id="release", node=Release(wire="q")),
        Definition(id="body", node=Sequence(children=("allocate", "release")))))
    plan = base_plan().revise(construction=SelectedConstruction(program=program, selections=()), shots=None,
        experiments=(Experiment(name="marginal", setting="all qubits",
                                observation=ObservationSpec(kind="probabilities", qubits=tuple(range(k)))),))._bind()
    indices = np.array([1, 6], dtype=np.uint64)
    marginal = np.zeros(1 << k)
    marginal[indices] = indices.view(np.float64)
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: SimpleNamespace(
        raw_output={"probabilities": marginal.copy()}, metadata={"native_job_id": "supplied"}))
    path = tmp_path / "run"
    with Run(plan, directory=path) as run:
        chunk = submit_experiment(prepare_experiment(plan.resolve("marginal"), run=run), run=run)
        arrays = chunk.values[0]
        assert arrays.encoding == "sparse" and arrays.indices.digest == arrays.probabilities.digest
    with nwqlib.load_run(path, backend=AerBackend()) as reopened:
        histogram = reopened.completed_observation(chunk.attempt).histogram()
        assert histogram.indices().dtype == np.uint64 and histogram.indices().tolist() == [1, 6]
        assert histogram.weights.dtype == np.float64
        np.testing.assert_array_equal(histogram.weights, marginal[indices])
        assert len(reopened.artifacts._hydrated) == 1


@pytest.mark.parametrize("support", ("sparse", "dense"))
def test_probability_arrays_publish_reopen_and_save_bit_for_bit(tmp_path, monkeypatch, support):
    """A k = 12 marginal is stored as binary arrays: sparse when 2*s < D, else dense by position.

    The published chunk, the reopened Run and a saved and loaded RunData read
    the same bytes under the same identity. The chunk row holds manifests and
    summaries only, reopening reads no array until one is requested, and a
    dense chunk's stored-entry count differs from its nonzero count.
    """
    import numpy as np
    from types import SimpleNamespace
    import nwqlib
    from nwqlib._choice_archive import ArchiveFiles
    from nwqlib.backends import AerBackend
    from nwqlib.backends import qiskit_aer as aer
    from nwqlib.blocks import SelectedConstruction
    from nwqlib.core.planning import Experiment, ObservationSpec
    from nwqlib.execution import Run, prepare_experiment, submit_experiment
    from nwqlib.ir import Allocate, Definition, Program, Register, Release, Sequence
    from nwqlib.saved_evidence import load_data, save_data
    from test_prepared_execution import base_plan

    k = 12
    program = Program(root="body", registers=(Register(name="q", width=k),), definitions=(
        Definition(id="allocate", node=Allocate(wire="q")), Definition(id="release", node=Release(wire="q")),
        Definition(id="body", node=Sequence(children=("allocate", "release")))))
    plan = base_plan().revise(construction=SelectedConstruction(program=program, selections=()), shots=None,
        experiments=(Experiment(name="marginal", setting="all qubits",
                                observation=ObservationSpec(kind="probabilities", qubits=tuple(range(k)))),))._bind()
    rng = np.random.default_rng(3)
    marginal = rng.random(1 << k)
    marginal[rng.random(1 << k) < (0.9 if support == "sparse" else 0.25)] = 0.
    marginal /= marginal.sum()
    monkeypatch.setattr(aer, "_submit_aer_execution", lambda native: SimpleNamespace(
        raw_output={"probabilities": marginal.copy()}, metadata={"native_job_id": "supplied"}))
    nonzero = np.flatnonzero(marginal)
    path = tmp_path / "run"
    with Run(plan, directory=path) as run:
        chunk = submit_experiment(prepare_experiment(plan.resolve("marginal"), run=run), run=run)
        run.collect(chunk)
        attempt = chunk.attempt
        arrays = chunk.values[0]
        assert arrays.encoding == support and arrays.nonzero == len(nonzero)
        if support == "dense":
            assert arrays.entries == 1 << k != arrays.nonzero and arrays.indices is None
            np.testing.assert_array_equal(chunk.histogram().weights, marginal)
        else:
            assert arrays.entries == arrays.nonzero and 2 * arrays.nonzero < 1 << k
            assert chunk.histogram().indices().tolist() == nonzero.tolist()
            np.testing.assert_array_equal(chunk.histogram().weights, marginal[nonzero])
        # The row holds manifests and summaries, not one entry per outcome.
        assert run._state["data_sizes"][("chunk", chunk.content_id)] < 8192
        data = run.data
    with nwqlib.load_run(path, backend=AerBackend()) as reopened:
        restored = reopened.completed_observation(attempt)
        assert restored == chunk and restored.content_id == chunk.content_id
        histogram = restored.histogram()
        assert (histogram.width, histogram.entries) == (k, arrays.entries) and not reopened.artifacts._hydrated
        np.testing.assert_array_equal(histogram.weights, chunk.histogram().weights)
        np.testing.assert_array_equal(histogram.packed_indices(), chunk.histogram().packed_indices())
        # A copy of a reopened chunk reads its arrays through the same payload source.
        import copy
        for copied in (copy.copy(restored), copy.deepcopy(restored), restored.model_copy()):
            assert copied == restored
            np.testing.assert_array_equal(copied.histogram().weights, histogram.weights)
            np.testing.assert_array_equal(copied.histogram().packed_indices(), histogram.packed_indices())
    files = ArchiveFiles(tmp_path / "saved", None)
    files.path.mkdir()
    saved = save_data(data, files, plan.method)
    loaded = load_data(saved, files, plan.method)
    (again,) = loaded.observations.chunks
    assert again == chunk and again.content_id == chunk.content_id
    np.testing.assert_array_equal(again.histogram().weights, chunk.histogram().weights)
