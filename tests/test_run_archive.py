"""Actual scientific cache snapshots and pending Run copies; no replanning."""


import numpy as np
import pytest
import sympy as sp

import nwqlib
from nwqlib._prepared_execution import Run
from nwqlib.algorithms.gcim import ADAPT
from nwqlib.algorithms.qhd import QHD
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


def test_published_array_has_one_durable_payload_and_survives_close(tmp_path, supplied_qhd):
    import sqlite3
    import subprocess
    import sys
    from nwqlib.backends import AerBackend

    case = supplied_qhd
    run = Run(case.plan, directory=None)
    result = run.wait()
    path = run.save(tmp_path / "saved")
    run.close()
    assert not tuple(path.glob("*.npy"))  # This QHD input has no independent input/cache arrays.
    with sqlite3.connect(path / "run.sqlite") as connection:
        assert connection.execute("SELECT sum(length(payload)) FROM binary").fetchone()[0] == case.raw.nbytes
    with nwqlib.load_run(path, backend=AerBackend()) as restored:
        loaded = restored.result
        handle = loaded.data.artifact(loaded.artifact)
        np.testing.assert_array_equal(handle.array, case.raw)
        assert not handle.array.flags.writeable
        assert loaded.artifact.acquisition == result.artifact.acquisition
        assert restored.trace.jobs == 1
    np.testing.assert_array_equal(handle.array, case.raw)
    assert len(case.calls) == 1
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


def _two_qubit_adapt(execution, max_iterations, shots=None, basis_growth=None, sampling=None):
    x = np.array([[0, 1], [1, 0]])
    y = np.array([[0, -1j], [1j, 0]])
    z = np.diag([1, -1])
    h = .8*np.kron(z, np.eye(2)) - .3*np.kron(np.eye(2), z) + .45*np.kron(x, x) + .2*np.kron(y, y)
    pool = tuple(ingest_pauli(((label, 1j),), num_qubits=2) for label in ("YI", "IY", "XY"))
    return nwqlib.plan(Eigenproblem(A=h),
        method=ADAPT(initial_state=[.5]*4, pool=pool, theta=np.pi/4, max_iterations=max_iterations,
                     basis_growth=basis_growth, sampling=sampling),
        execution=execution, shots=shots, seed=7)


@pytest.mark.parametrize("bounded", [False, True])
def test_exact_adapt_resumes_reused_native_readout_without_replaying_acquisitions(bounded, tmp_path, monkeypatch):
    """An exact ADAPT Run interrupted halfway resumes from disk to the uninterrupted result, without repeating work.

    The reopened Run runs only the remaining jobs and keeps its earlier
    chunks first. With BoundedRitzGrowth(max_chain_length=2) the Run is
    interrupted while it obtains the rows of its second round. The reopened
    Run examines the extensions of only the rounds that the interrupted one
    had not reached, so the interrupted and the reopened Run together
    examine as many rounds as the uninterrupted one, and it ends with the
    same explicit chains, the same examined extensions with their reasons
    and scores, and the same cursor. A saved Result reanalyzes at another
    cutoff without acquisition.
    """
    from nwqlib.algorithms.gcim import BoundedRitzGrowth, adapt_acquisition as acquisition, bounded_growth
    from nwqlib.backends import qiskit_aer as aer

    plan = _two_qubit_adapt("quantum", 3 if bounded else 2,
                            basis_growth=BoundedRitzGrowth(max_chain_length=2) if bounded else None)
    frontiers, frontier = [], bounded_growth.frontier
    monkeypatch.setattr(bounded_growth, "frontier", lambda *a, **k: frontiers.append(1) or frontier(*a, **k))
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
        phase = run.checkpoint_state["phase"]
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
    if bounded:
        # Three rounds examined: 3 uninterrupted, 2 before and 1 after the interruption.
        assert phase == "rows" and len(frontiers) == 6
        rounds = [row.candidates for row in expected.history]
        assert len(result.basis) > 2 and (result.basis, result.frontier_cursor) == (
            expected.basis, expected.frontier_cursor)
        assert [row.candidates for row in result.history] == rounds
        loaded = nwqlib.load_result(result.save(tmp_path / "bounded-result"))
        assert (loaded.basis, loaded.value_basis, loaded.history) == (result.basis, result.value_basis, result.history)
        # The rows-phase gate set is execution state, never saved scientific state.
        assert "pending_items" not in result.data.method_context and "pending_items" not in loaded.data.method_context
        monkeypatch.setattr(aer, "_submit_aer_execution", lambda *a, **k: pytest.fail("reanalysis acquired"))
        again = loaded.analyze(overlap_cutoff=1e-10)
        assert again.eigenvalue == pytest.approx(result.eigenvalue, rel=0, abs=2e-13)
        assert again.value_basis == result.value_basis


def test_grouped_adapt_resume_keeps_its_population_and_exposure(tmp_path, monkeypatch):
    """An interrupted grouped sampled ADAPT run resumes to the counts, moments and exposure of an uninterrupted one.

    The words of H and the pool are real (an even number of Y in H, one Y
    per generator) and the reference is a real vector, so the default
    grouped acquisition measures real parts. The Run stops before half of
    its jobs. Reopened from disk it acquires only the remaining queries,
    with the seeds drawn in the same order, so its earlier chunks stay
    first, every query has the uninterrupted counts, and the jobs, shots,
    selections, group moments and energy are those of the uninterrupted
    run. Contribution identities name each Run's own chunks.

    With PencilSampling(B = 8m, f = 1/4) the same selection ends on the
    final basis of m = b(b+1)G/2 settings, which a pilot and a main stage
    measure again, 2m more jobs and B more shots. Stopped before the fourth
    main-stage submission and reopened from disk, the Run submits no
    selection or pilot query, calls no allocation function and ends with the
    jobs, shots, counts, decision and energy of the uninterrupted one, and
    its Result uses the main-stage chunks alone. A cancellation inside the
    pilot or the main stage ends the Run with the last pencil of the growth,
    and its Result uses the selection chunks alone. With
    room for the selection shots and B - 1 more, the Run is refused when the
    selection ends, before any stage submission, and after extend_limits it
    ends with the energy of the uninterrupted one. An allocation or a
    main-stage analysis that stops is continued only by
    run.resume(reanalyze=True), which measures nothing again.
    """
    from nwqlib.algorithms import PencilSampling
    from nwqlib.algorithms.gcim import adapt_acquisition as acquisition, pencil_workflow
    from nwqlib.execution import ExecutionLimits

    plan = _two_qubit_adapt("quantum", 2, shots=64)
    assert (plan.reconstruction.acquisition.matrix_measurement, plan.reconstruction.acquisition.quadratures) == (
        "grouped", "real")
    with Run(plan) as reference:
        expected = reference.wait()
        jobs, shots = reference.trace.jobs, sum(event.shots for event in reference.trace.events)
        counts = {c.realization_id: c.values for c in reference.observations.chunks}
        backend = reference.backend
    submit, submitted = acquisition.submit_experiment, 0

    def interrupted(*args, **kwargs):
        nonlocal submitted
        if submitted == jobs // 2:
            raise RuntimeError("stop before next acquisition")
        submitted += 1
        return submit(*args, **kwargs)

    path = tmp_path / "adapt-grouped"
    with Run(plan, directory=path) as run:
        with monkeypatch.context() as patch:
            patch.setattr(acquisition, "submit_experiment", interrupted)
            with pytest.raises(RuntimeError, match="before next acquisition"):
                run.resume()
        prefix = tuple(chunk.content_id for chunk in run.observations.chunks)
    with nwqlib.load_run(path, backend=backend) as restored:
        result = restored.wait()
        assert restored.trace.jobs == jobs and sum(event.shots for event in restored.trace.events) == shots
        assert tuple(chunk.content_id for chunk in restored.observations.chunks[:len(prefix)]) == prefix
        assert {c.realization_id: c.values for c in restored.observations.chunks} == counts
    assert (result.selected, result.theta, result.eigenvalue) == (expected.selected, expected.theta, expected.eigenvalue)

    def moments(record):
        return [m.model_dump(exclude={"contribution_id", "content_id"}) for m in record.group_moments]

    assert moments(result) == moments(expected)
    assert len(moments(result)) == 4 * 5 // 2 * len(plan.reconstruction.energy_groups)  # b(b+1)G/2, b = 4

    m = len(moments(result))
    sampled = _two_qubit_adapt("quantum", 2, shots=64, sampling=PencilSampling(total_shots=8 * m, pilot_fraction=.25))
    with Run(sampled) as reference:
        uninterrupted = reference.wait()
        assert (reference.trace.jobs, sum(event.shots for event in reference.trace.events)) == (
            jobs + 2 * m, shots + 8 * m)
        jobs, shots = reference.trace.jobs, sum(event.shots for event in reference.trace.events)
        counts = {c.realization_id: c.values for c in reference.observations.chunks}
    stage_submit, staged = pencil_workflow.submit_experiment, 0

    def stopped(*args, **kwargs):
        nonlocal staged
        if staged == m + 3:
            raise RuntimeError("stop inside the main stage")
        staged += 1
        return stage_submit(*args, **kwargs)

    path = tmp_path / "adapt-sampled"
    with Run(sampled, directory=path) as run:
        with monkeypatch.context() as patch:
            patch.setattr(pencil_workflow, "submit_experiment", stopped)
            with pytest.raises(RuntimeError, match="inside the main stage"):
                run.resume()
        final = run.checkpoint_state["final"]
        prefix = tuple(chunk.content_id for chunk in run.observations.chunks)
    assert (final["stage"], final["index"], len(prefix)) == (1, 3, jobs - m + 3)

    def fail(*args, **kwargs):
        pytest.fail("the reopened Run measured a selection or pilot query or chose its shots again")

    with monkeypatch.context() as patch:
        patch.setattr(acquisition, "submit_experiment", fail)
        for name in ("pilot_decision", "floored_allocation", "uniform_allocation", "pilot_split"):
            patch.setattr(pencil_workflow, name, fail)
        with nwqlib.load_run(path, backend=backend) as restored:
            result = restored.wait()
            assert restored.trace.jobs == jobs and sum(event.shots for event in restored.trace.events) == shots
            assert tuple(chunk.content_id for chunk in restored.observations.chunks[:len(prefix)]) == prefix
            assert {c.realization_id: c.values for c in restored.observations.chunks} == counts
    assert (result.shot_allocation, result.eigenvalue) == (uninterrupted.shot_allocation, uninterrupted.eigenvalue)

    def stage(chunk):
        return {item.parameter: item.value for item in chunk.bindings}["stage"]

    main_ids = {c.content_id for c in result.data.observations.chunks if stage(c) == 1}
    assert set(result.contribution_ids) == main_ids == {moment.contribution_id for moment in result.group_moments}

    # A cancellation after 3 pilot submissions, or after the pilot and 3 main-stage submissions, ends the Run
    # with the last pencil that the growth solved. Its Result uses the selection chunks (stage -1) alone.
    for pilot, main in ((3, 0), (m, 3)):
        done = 0
        with Run(sampled) as run:
            def cancelling(*args, **kwargs):
                nonlocal done
                chunk = stage_submit(*args, **kwargs)
                done += 1
                if done == pilot + main:
                    run.cancel()
                return chunk

            with monkeypatch.context() as patch:
                patch.setattr(pencil_workflow, "submit_experiment", cancelling)
                cancelled = run.wait()
        stages = [stage(c) for c in cancelled.data.observations.chunks]
        assert (cancelled.stop_reason, stages.count(0), stages.count(1)) == ("cancelled", pilot, main)
        growth = {c.content_id for c, s in zip(cancelled.data.observations.chunks, stages) if s == -1}
        assert set(cancelled.contribution_ids) == growth
        assert len(cancelled.group_moments) == m
        assert {moment.contribution_id for moment in cancelled.group_moments} <= growth
        assert cancelled.eigenvalue == cancelled.history[-1].energy

    # The stages need room for all B shots before the first of them.
    with Run(sampled, limits=ExecutionLimits(max_total_shots=shots - 1)) as run:
        with pytest.raises(ValueError, match="max_total_shots"):
            run.resume()
        assert (run.trace.jobs, sum(event.shots for event in run.trace.events)) == (jobs - 2 * m, shots - 8 * m)
        run.extend_limits(max_total_shots=shots)
        assert run.wait().eigenvalue == uninterrupted.eigenvalue

    # An allocation or a main-stage analysis that stops leaves its marker, so a plain resume refuses to
    # continue, and run.resume(reanalyze=True) completes the Run from its saved data with the jobs, shots,
    # decision and energy of the uninterrupted one and one recorded retry, without measuring again.
    for target, name, retry in ((pencil_workflow, "pilot_decision", 0),
                                (acquisition, "_pencil_from_observations", 1)):
        original = getattr(target, name)
        with Run(sampled) as run:
            def injected(*args, **kwargs):
                if run.checkpoint_state["phase"] == "final":
                    raise RuntimeError("stopped in the final phase")
                return original(*args, **kwargs)

            with monkeypatch.context() as patch:
                patch.setattr(target, name, injected)
                with pytest.raises(RuntimeError, match="stopped in the final phase"):
                    run.resume()
            submitted = run.trace.jobs
            with pytest.raises(RuntimeError, match="was interrupted"):
                run.resume()
            assert run.trace.jobs == submitted
            run.resume(reanalyze=True)
            result = run.wait()
            assert (run.trace.jobs, sum(event.shots for event in run.trace.events)) == (jobs, shots)
            assert (result.shot_allocation, result.eigenvalue) == (uninterrupted.shot_allocation,
                                                                  uninterrupted.eigenvalue)
            (entry,) = run.checkpoint_state["analysis_retries"]
            assert entry["stage"] == retry


@pytest.mark.parametrize("kind", ["amplitudes", "probabilities", "pauli_expectation"])
def test_aer_qpy_restore_preserves_readout_position_and_wire_order(kind):
    import io
    from types import SimpleNamespace
    from qiskit import QuantumCircuit, qpy
    from nwqlib.backends import AerBackend, qiskit_aer as aer
    from nwqlib.execution import ExecutionLimits

    circuit = QuantumCircuit(2, global_phase=.3)
    circuit.x(0)
    circuit.h(1)
    labels = ("IZ", "ZI")
    memory_mb = ExecutionLimits().simulator_memory_mb
    native = aer._prepare_aer_execution(aer.AER_STATEVECTOR_TARGET, circuit, shots=None, seed=7,
        simulator_memory_mb=memory_mb, probability_qubits=(1, 0) if kind == "probabilities" else None,
        pauli_expectation_readout=(1, labels) if kind == "pauli_expectation" else None)
    stream = io.BytesIO()
    qpy.dump(native.circuit, stream)
    stream.seek(0)
    saved, = qpy.load(stream)
    observation = SimpleNamespace(kind=kind, shots=None, qubits=(1, 0), position=1, labels=labels)
    record = SimpleNamespace(observation=observation, runtime=SimpleNamespace(seed=7))
    metadata = dict(native.metadata, simulator_method=native.simulator.options.method,
        simulator_memory_mb=memory_mb, simulator_zero_threshold=native.simulator.options.zero_threshold)
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
