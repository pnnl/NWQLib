"""Atomic current Run transitions, original pending jobs and bounded SQLite data."""

import os

import pytest

from nwqlib._prepared_execution import Run
from nwqlib._run_archive import load
from test_run_lifecycle import CountsMethod, InjectedBackend, selected
from test_run_lifecycle import clear_backend as clear_backend
from test_saved_evidence import supplied_qhd as supplied_qhd


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
