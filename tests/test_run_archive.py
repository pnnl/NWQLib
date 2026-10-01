"""Actual scientific cache snapshots and pending Run copies; no replanning."""

import json

import numpy as np
import pytest
import sympy as sp

import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.algorithms.qhd import QHD, QuadraticSchedule
from nwqlib.algorithms.qpe import QCELS
from nwqlib.operators import ingest_pauli
from nwqlib.problems import Eigenproblem, Optimization
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend
from test_saved_evidence import supplied_qhd as supplied_qhd


def classical_plan(family):
    if family == "qhd":
        x = sp.Symbol("x")
        problem = Optimization(objective=(x - sp.Rational(1, 5)) ** 2, variables=(x,), bounds=((-1.0, 1.0),))
        method = QHD(num_grid_points=3, num_steps=1, total_time=0.1, keep_state=True)
    else:
        problem = Eigenproblem(A=[[1.0, 0.0], [0.0, -1.0]])
        method = (
            QCELS(initial_state=[1.0, 0.0], num_times=3, grid_size=32, max_time=0.3, tau=0.2)
            if family == "qpe"
            else ADAPT(initial_state=[1.0, 1.0], pool=(ingest_pauli((("Y", 1j),), num_qubits=1),))
        )
    return nwqlib.plan(problem, method=method, execution="classical", seed=7)


def test_reopened_empty_cache_cannot_delete_a_shared_plan_input(tmp_path):
    from nwqlib._run_archive import _CacheDelta, finish_caches

    path = tmp_path / "input-protection"
    with Run(selected(), backend=InjectedBackend(), directory=path) as run:
        assert not list(run._state["journal"].rows("cache"))
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as run:
        files = run._state["archive_files"]
        name = next(name for name in files._written_files if name.endswith(".npy"))
        source = files.file(name)
        original = source.read_bytes()
        # Exercise the committed reference transition: a disposable cache
        # shares an immutable input, then its last reference is removed.
        addition = _CacheDelta(records=(("cache", "shared-input", {"file": name}),),
                               references=(("shared-input", {name}),))
        run._write(addition.records)
        finish_caches(run, addition, committed=True)
        removal = _CacheDelta(deleted=(("cache", "shared-input"),))
        run._write(deleted=removal.deleted)
        finish_caches(run, removal, committed=True)
        assert source.is_file() and source.read_bytes() == original
    # Restoring the original scientific input still succeeds after collection.
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as run:
        assert run.plan.problem.content_id == selected().problem.content_id


def test_reopen_cleans_uncommitted_cache_before_capacity_and_preserves_owned_inputs(tmp_path):
    from nwqlib.execution import ExecutionLimits

    class CacheMethod(CountsMethod):
        def save_run_context(self, context, files):
            return dict(values=files.write_numpy_json("values.json", context))

        @staticmethod
        def load_run_context(saved, files):
            return files.read_numpy_json(saved["values"])

    path = tmp_path / "interrupted-cache"
    plan = selected().revise(method=CacheMethod())
    with Run(plan, backend=InjectedBackend(), directory=path,
             limits=ExecutionLimits(max_data_bytes=200_000)) as run:
        run._method_context(lambda: dict(values=np.array([3., 4.])))
        run.checkpoint(dict(step=1))
        files = run._state["archive_files"]
        committed = {name: (path / name).read_bytes() for name in files._written_files}
        # A process can exit after a cache file is closed but before SQLite
        # publishes its pointer. Reopening must reclaim that file even when it
        # alone exceeds the data cap, without parsing its incomplete contents.
        files._begin_cache_files()
        orphan = files.write_array("interrupted.npy", np.array([9.]))
        with (path / orphan).open("r+b") as stream:
            stream.truncate(200_001)
        note = path / "notes.txt"
        note.write_text("user-owned file")
        # Another controller must not delete an active owner's provisional file.
        with pytest.raises(BlockingIOError):
            nwqlib.load_run(path, backend=InjectedBackend(), method=CacheMethod)
        assert (path / orphan).is_file()
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CacheMethod) as restored:
        assert not (path / orphan).exists()
        assert note.read_text() == "user-owned file"
        assert all((path / name).read_bytes() == data for name, data in committed.items())
        np.testing.assert_array_equal(restored._state["method_context"]["values"], [3., 4.])
        assert restored.checkpoint_state == dict(step=1)
        def stored():
            return sum(file.stat().st_size for file in path.iterdir()
                       if file.is_file() and file.name not in {"run.sqlite", "run.sqlite.lock", "run.sqlite-journal"})
        assert restored._state["external_bytes"] == stored()
        # Loading charged nothing, and later writes are charged to the Run's own
        # total. A replaced cache adds its new files and refunds the superseded
        # ones, and a cache beyond max_data_bytes is refused without a file.
        restored._state["method_context"]["values"] = np.array([5., 6., 7.])
        restored.checkpoint(dict(step=2))
        assert restored._state["external_bytes"] == stored()
        before = set(path.iterdir())
        restored._state["method_context"]["values"] = np.zeros(25_001)
        with pytest.raises(ValueError, match="max_data_bytes"):
            restored.checkpoint(dict(step=3))
        assert set(path.iterdir()) == before and restored._state["external_bytes"] == stored()
        np.testing.assert_array_equal(restored._state["method_context"]["values"], [5., 6., 7.])


def test_reopen_writes_the_uncertain_submission_and_reclaims_an_oversized_orphan(tmp_path, monkeypatch):
    """A stop after a submission intent commits leaves that intent for reopening to revise.

    Reopening writes the uncertain revision of the submission to the journal
    and keeps its shots in the uncertain exposure. An unpublished cache file
    larger than max_data_bytes is reclaimed, so the Run can still be
    reopened.
    """
    from nwqlib._choice_archive import CACHE_FILE_PREFIX
    from nwqlib._prepared_execution import prepare_experiment, submit_experiment
    from nwqlib._run_archive import load
    from nwqlib.execution import ExecutionLimits

    InjectedBackend.supports_synchronous = True
    path = tmp_path / "recovery"
    run = Run(selected(), backend=InjectedBackend(), directory=path, limits=ExecutionLimits(max_data_bytes=200_000))
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    store = run._state["journal"]
    commit = store.commit
    stopped = []

    def stop_after_intent(records=(), **kwargs):
        # Commit 4 (intent and reserved event) succeeds, and the process stops before any later commit.
        if stopped:
            raise OSError("process stopped after the submission intent")
        commit(records, **kwargs)
        if any(kind == "submission" for kind, _, _ in records):
            stopped.append(True)

    monkeypatch.setattr(store, "commit", stop_after_intent)
    with pytest.raises(OSError, match="process stopped"):
        submit_experiment(handle, run=run)
    run.close()
    orphan = path / (CACHE_FILE_PREFIX + "interrupted.npy")
    orphan.write_bytes(bytes(200_001))
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert not orphan.exists()
        (submission,) = restored.trace.submissions
        (event,) = restored.trace.events
        assert submission.status == event.status == "uncertain"
        saved = [row["status"] for _, row, _ in restored._state["journal"].rows("submission")]
        assert saved == ["uncertain"] and restored.exposure["uncertain"]["shots"] == event.shots


@pytest.mark.parametrize("family", ["qhd", "qpe", "adapt"])
def test_completed_archive_keeps_numerical_caches_arrays_and_result_without_reexecution(
    tmp_path, monkeypatch, family
):
    """Poison planning, execution and analysis after saving to prove reopening uses existing
    numerical caches and arrays.
    """
    plan = classical_plan(family)
    run = Run(plan)
    result = run.wait()
    path = run.save(tmp_path / family)
    native_before = tuple(
        (receipt.content_id, receipt.selected_kernel_id) for receipt in run.prepared_artifacts
    )
    rng = run.rng.snapshot()

    def forbidden(*a, **k):
        raise AssertionError("load replayed selected numerical work")

    monkeypatch.setattr(type(plan.method), "plan", forbidden)
    monkeypatch.setattr(type(plan.method), "execute", forbidden)
    monkeypatch.setattr(type(plan.method), "analyze", forbidden)
    with nwqlib.load_run(path, backend=None) as restored:
        assert restored.wait().content_id == result.content_id
        assert restored.rng.snapshot() == rng
        assert (
            tuple((r.content_id, r.selected_kernel_id) for r in restored.prepared_artifacts) == native_before
        )
        assert restored.trace.events == run.trace.events
        if result.data.artifacts:
            for handle in result.data.artifacts:
                loaded = restored.data.artifact(handle.manifest).array
                np.testing.assert_array_equal(loaded, handle.array)
                assert not loaded.flags.writeable
        with monkeypatch.context() as patch:
            patch.setattr(
                np, "save", lambda *a, **k: pytest.fail("unchanged numerical cache copied after reopen")
            )
            restored.checkpoint({"saved_result": result.content_id})
    run.close()


def test_pending_copy_preserves_original_locator_rng_exposure_and_once_only_collection(tmp_path, monkeypatch):
    original = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "original")
    original.resume()
    locator = original.trace.submissions[0].locator
    rng = original.rng.snapshot()
    path = original.save(tmp_path / "snapshot")
    assert original.trace.jobs == 1 and original.result is None
    original.close()
    monkeypatch.setattr(CountsMethod, "plan", lambda *a, **k: pytest.fail("snapshot load replanned"))
    InjectedBackend.status = "completed"
    with nwqlib.load_run(path, backend=InjectedBackend(), method=CountsMethod) as run:
        assert run.trace.submissions[0].locator == locator and run.rng.snapshot() == rng
        result = run.wait(timeout=1, poll_interval=0)
        assert result.counts == 7 and run.trace.jobs == 1 and len(run.observations.chunks) == 1
        assert not run.collect(run.observations.chunks[0])
    assert sum(name == "launch" for name, _ in InjectedBackend.calls) == 1


def test_failed_archive_does_not_remove_original_run_or_previous_user_directory(tmp_path, monkeypatch):
    from nwqlib._choice_archive import ArchiveFiles

    source = tmp_path / "source"
    run = Run(selected(), backend=InjectedBackend(), directory=source)
    run.resume()
    old = tmp_path / "existing"
    old.mkdir()
    (old / "user-file").write_text("keep")
    with pytest.raises(FileExistsError):
        run.save(old)
    assert (old / "user-file").read_text() == "keep"
    writer = ArchiveFiles.write_json

    def interrupted(files, name, data):
        writer(files, name, data)
        raise OSError("snapshot interruption")

    monkeypatch.setattr(ArchiveFiles, "write_json", interrupted)
    with pytest.raises(OSError, match="snapshot interruption"):
        run.save(tmp_path / "failed")
    assert not (tmp_path / "failed").exists() and (source / "run.sqlite").exists()
    assert run.trace.jobs == 1 and run.result is None
    run.close()


@pytest.mark.parametrize("origin", ["durable", "memory", "copy"])
def test_published_array_has_one_durable_payload_and_survives_close(tmp_path, supplied_qhd, origin):
    import sqlite3
    import subprocess
    import sys
    from nwqlib.backends import AerBackend

    case = supplied_qhd
    source = tmp_path / "source"
    run = Run(case.plan, directory=None if origin == "memory" else source)
    result = run.wait()
    path = run.save(tmp_path / "saved") if origin != "durable" else source
    run.close()
    assert not tuple(path.glob("*.npy"))  # This QHD input has no independent input/cache arrays.
    with sqlite3.connect(path / "run.sqlite") as connection:
        assert connection.execute("SELECT sum(length(payload)) FROM binary").fetchone()[0] == case.raw.nbytes
    with nwqlib.load_run(path, backend=AerBackend()) as restored:
        loaded = restored.result
        handle = loaded.data.artifact(loaded.artifact)
        # Reopening registers the array; it is read on first use, not by a display.
        assert "stored array, not loaded" in loaded._array_text(loaded.artifact)
        np.testing.assert_array_equal(handle.array, case.raw)
        assert "not loaded" not in loaded._array_text(loaded.artifact)
        assert not handle.array.flags.writeable
        assert restored.hydrate(loaded.artifact) is handle
        assert loaded.artifact.acquisition == result.artifact.acquisition
        assert restored.trace.jobs == 1
    np.testing.assert_array_equal(handle.array, case.raw)
    assert len(case.calls) == 1
    if origin == "memory":
        # A fresh process knows neither the injected acquisition nor the old
        # ndarray. It must recover values solely from the saved binary owner.
        child = subprocess.run([sys.executable, "-c", '''
import sys
import numpy as np
from nwqlib import load_run
from nwqlib.backends import AerBackend
with load_run(sys.argv[1], backend=AerBackend()) as run:
    result = run.result
    array = result.data.artifact(result.artifact).array
    expected = np.array([.5j,.5,-.5j,0,-.5,0,0,0])
    np.testing.assert_array_equal(array, expected)
    assert result.artifact.acquisition[0] == run.run_id
    assert not array.flags.writeable and run.trace.jobs == 1
print("fresh-process binary array and original acquisition preserved")
''', str(path)], text=True, capture_output=True)
        assert child.returncode == 0, child.stdout + child.stderr
        assert "original acquisition preserved" in child.stdout


def test_memory_array_save_interruption_preserves_original_payload(tmp_path, supplied_qhd, monkeypatch):
    import nwqlib._sqlite_bytes as storage

    with Run(supplied_qhd.plan) as run:
        result = run.wait()
        original = result.data.artifact(result.artifact).array
        write = storage.write_bytes

        def interrupted(*args, **kwargs):
            write(*args, **kwargs)
            raise OSError("interrupted snapshot binary")

        monkeypatch.setattr(storage, "write_bytes", interrupted)
        path = tmp_path / "failed"
        with pytest.raises(OSError, match="snapshot binary"):
            run.save(path)
        assert not path.exists()
        assert result.data.artifact(result.artifact).array is original
        np.testing.assert_array_equal(original, supplied_qhd.raw)


def test_adapt_collect_before_cache_checkpoint_restores_canonical_order(tmp_path, monkeypatch):
    plan = classical_plan("adapt")
    with Run(plan) as reference:
        expected = reference.wait()
        expected_invocations = reference.trace.host_invocations
    original = Run.collect
    path = tmp_path / "adapt-collect"

    def interrupted(run, chunk):
        fresh = original(run, chunk)
        if fresh:
            raise RuntimeError("collected before method cache update")
        return fresh

    with monkeypatch.context() as patch:
        patch.setattr(Run, "collect", interrupted)
        with Run(plan, directory=path) as run:
            with pytest.raises(RuntimeError, match="before method cache"):
                run.resume()
            assert len(run.observations.chunks) == 1
            first = run.observations.chunks[0].content_id
    with nwqlib.load_run(path, backend=None) as restored:
        context = restored._state["method_context"]
        assert context.contribution_order == [first]
        rows = list(restored._state["journal"].rows("cache"))
        saved = next(row["method_context"] for _, row, _ in rows if row["kind"] == "method")
        assert saved["observations_in_run"] is True
        assert not {"observations", "contribution_order", "contribution_seen"} & saved.keys()
        result = restored.wait()
        assert result.eigenvalue == pytest.approx(expected.eigenvalue, rel=0, abs=2e-13)
        assert restored.trace.host_invocations == expected_invocations
        assert restored.observations.chunks[0].content_id == first
        saved_result = result.save(tmp_path / "standalone-adapt")
    saved_context = json.loads((saved_result / "result.json").read_text())["data"]["method_context"]
    assert saved_context["observations_in_run"] is True
    assert not {"observations", "contribution_order", "contribution_seen"} & saved_context.keys()
    standalone = nwqlib.load_result(saved_result)
    assert standalone.data.method_context.keys() == result.data.method_context.keys()
    assert standalone.data.observations == result.data.observations


def _two_qubit_adapt(execution, max_iterations):
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    h = .8*np.kron(z, np.eye(2)) - .3*np.kron(np.eye(2), z) + .45*np.kron(x, x) + .2*np.kron(y, y)
    pool = tuple(ingest_pauli(((label, 1j),), num_qubits=2) for label in ("YI", "IY", "XY"))
    return nwqlib.plan(Eigenproblem(A=h),
        method=ADAPT(initial_state=[.5]*4, pool=pool, theta=np.pi/4, max_iterations=max_iterations),
        execution=execution, seed=7)


@pytest.mark.parametrize("execution,iterations", [("classical", 3), ("quantum", 2)])
def test_adapt_checkpoints_write_the_pencil_history_and_method_cache_once_per_analysis(
    tmp_path, monkeypatch, execution, iterations
):
    """Every committed checkpoint reproduces the live controller state, although acquisition
    steps and native preparations write neither the pencil, the decision history nor the durable
    method cache.
    """
    import json
    from collections import Counter
    from nwqlib import _run_archive
    plan = _two_qubit_adapt(execution, iterations)
    with Run(plan) as reference:
        expected = reference.wait()
    written, methods = Counter(), []
    checkpoint, method_frontier = Run.checkpoint, _run_archive._method_frontier

    def committed(run, value, *, changed=None):
        sequence = checkpoint(run, value, changed=changed)
        written.update(tuple(value) if changed is None else changed)
        assert run.checkpoint_state == json.loads(json.dumps(value))
        return sequence

    monkeypatch.setattr(Run, "checkpoint", committed)
    monkeypatch.setattr(_run_archive, "_method_frontier", lambda *args: methods.append(1) or method_frontier(*args))
    with Run(plan, directory=tmp_path / "adapt-fields") as run:
        result = run.wait()
        state = run.checkpoint_state
    assert result.eigenvalue == pytest.approx(expected.eigenvalue, rel=0, abs=2e-13)
    assert result.selected == expected.selected
    analyses, recorded = state["analysis_attempts"], len(state["history"])
    assert recorded >= 2 and written["phase"] > 10 * analyses
    assert written["pencil"] == analyses and written["history"] <= 1 + 3 * recorded
    assert len(methods) <= analyses + 2


def test_exact_adapt_resumes_reused_native_readout_without_replaying_acquisitions(tmp_path, monkeypatch):
    from nwqlib.algorithms.gcim import adapt_acquisition as acquisition
    from nwqlib.backends import qiskit_aer as aer

    plan = _two_qubit_adapt("quantum", 2)
    with Run(plan) as reference:
        expected = reference.wait()
        expected_jobs = reference.trace.jobs
        half = expected_jobs // 2
        backend = reference.backend
    submit = acquisition.submit_experiment
    submitted = 0

    def interrupted(*args, **kwargs):
        nonlocal submitted
        if submitted == half:
            raise RuntimeError("stop before next acquisition")
        submitted += 1
        return submit(*args, **kwargs)

    path = tmp_path / "adapt-exact-reuse"
    with Run(plan, directory=path) as run:
        with monkeypatch.context() as patch:
            patch.setattr(acquisition, "submit_experiment", interrupted)
            with pytest.raises(RuntimeError, match="before next acquisition"):
                run.resume()
        prefix = tuple(chunk.content_id for chunk in run.observations.chunks)
        assert len({chunk.attempt for chunk in run.observations.chunks}) == half
    with monkeypatch.context() as patch:
        patch.setattr(aer, "_prepare_aer_execution", lambda *a, **k: pytest.fail("load lowered again"))
        restored = nwqlib.load_run(path, backend=backend)
    with restored:
        result = restored.wait()
        assert result.eigenvalue == pytest.approx(expected.eigenvalue, rel=0, abs=2e-13)
        assert result.selected == expected.selected
        assert result.stop_reason == expected.stop_reason
        assert restored.trace.jobs == expected_jobs
        assert tuple(chunk.content_id for chunk in restored.observations.chunks[:len(prefix)]) == prefix


@pytest.mark.parametrize("kind", ["amplitudes", "probabilities", "pauli_expectation"])
def test_aer_qpy_restore_preserves_readout_position_and_wire_order(kind):
    import io
    from types import SimpleNamespace
    from qiskit import QuantumCircuit, qpy
    from nwqlib.backends import AerBackend, qiskit_aer as aer

    circuit = QuantumCircuit(2, global_phase=.3)
    circuit.x(0)
    circuit.h(1)
    labels = ("IZ", "ZI")
    native = aer._prepare_aer_execution(aer.AER_STATEVECTOR_TARGET, circuit, shots=None, seed=7,
        probability_qubits=(1, 0) if kind == "probabilities" else None,
        pauli_expectation_readout=(1, labels) if kind == "pauli_expectation" else None)
    stream = io.BytesIO()
    qpy.dump(native.circuit, stream)
    stream.seek(0)
    saved, = qpy.load(stream)
    observation = SimpleNamespace(kind=kind, shots=None, qubits=(1, 0), position=1, labels=labels)
    record = SimpleNamespace(observation=observation, runtime=SimpleNamespace(seed=7))
    metadata = dict(native.metadata, simulator_method=native.simulator.options.method,
        simulator_memory_mb=1024, simulator_zero_threshold=native.simulator.options.zero_threshold)
    restored = AerBackend().restore_native_data(record, saved, metadata, run=None)
    output = aer._submit_aer_execution(restored).raw_output
    if kind == "amplitudes":
        np.testing.assert_allclose(output["statevector"], np.exp(.3j)*np.array([0, 1, 0, 1])/np.sqrt(2),
            rtol=0, atol=2e-15)
    elif kind == "probabilities":
        np.testing.assert_allclose(output["probabilities"], [0., 0., .5, .5], rtol=0, atol=2e-15)
    else:
        # ZI is observed before H(1); moving readout to the end would give zero.
        assert output["pauli_expectations"] == pytest.approx({"IZ": -1, "ZI": 1}, rel=0, abs=2e-15)


def test_plan_size_selection_stays_out_of_the_small_run_manifest(tmp_path):
    """A multi-MB Plan record lives in the selection file, so load_run can reopen the Run.

    The QHD reconstruction record lists the blocks of every product-formula
    step. With 80 steps, two variables and six grid points the Plan record is
    about 3 MB. load_run reads run.json under a one MiB cap before the journal
    supplies the data allowance, so run.json only names the selection file.
    Nothing is executed.
    """
    import json

    x, y = sp.symbols("x y", real=True)
    objective = ((2 * x**2 - 1) ** 2 + sp.Rational(3, 5) * x + 2 * (y - sp.Rational(3, 10)) ** 2
                 + sp.Rational(6, 5) * x * y)
    problem = Optimization(objective=objective, variables=(x, y), bounds=((-1.2, 1.2), (-1.2, 1.2)))
    method = QHD(num_grid_points=6, num_steps=80, total_time=10.0, schedule=QuadraticSchedule(gamma=0.3))
    selected = nwqlib.plan(problem, method=method, seed=7)
    path = tmp_path / "qhd"
    with Run(selected, directory=path) as run:
        backend = run.backend
    manifest = json.loads((path / "run.json").read_text())
    assert (path / "run.json").stat().st_size < 4096
    assert (path / manifest["selection"]).stat().st_size > 1024**2
    with nwqlib.load_run(path, backend=backend) as reopened:
        assert reopened.plan.content_id == selected.content_id


def test_oversized_run_manifest_rejects_before_its_selection_is_loaded(tmp_path, monkeypatch):
    """The one MiB cap on run.json is checked before the manifest is parsed.

    Padding keeps the manifest valid JSON, so only the size check can reject it,
    and the patched load_plan shows that no selection was read first.
    """
    import nwqlib._choice_archive as choice_archive
    from test_run_lifecycle import InjectedBackend

    path = tmp_path / "run"
    Run(selected(), backend=InjectedBackend(), directory=path).close()
    manifest = path / "run.json"
    manifest.write_text(manifest.read_text() + " " * 1024**2)

    def unexpected(*args, **kwargs):
        raise AssertionError("the selection was loaded before the manifest size check")

    monkeypatch.setattr(choice_archive, "load_plan", unexpected)
    with pytest.raises(ValueError, match="run header exceeds one MiB"):
        nwqlib.load_run(path, backend=InjectedBackend())
