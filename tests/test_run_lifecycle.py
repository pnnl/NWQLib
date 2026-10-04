"""Actual one-qubit lifecycle, durable interruption and selected-data storage."""

from types import SimpleNamespace
from typing import ClassVar, Literal

import pytest

from nwqlib.algorithms.protocol import AlgorithmDescriptor, Method
from nwqlib.blocks import SelectedConstruction
from nwqlib.core.analysis import Result
from nwqlib.core.planning import Experiment, ObservationSpec, Plan, RandomStreams
from nwqlib.core.records import Record, Source
from nwqlib.execution import CountsSampling, ExecutionLimits, JobLocator
from nwqlib.ir import Allocate, ClassicalValue, Definition, Measure, Program, Register, Release, Sequence
from nwqlib.problems import Eigenproblem, Samples
from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment


class CountsResult(Result):
    counts: int


class CountsMethod(Method):
    result_type: ClassVar[type] = CountsResult

    @property
    def descriptor(self):
        return AlgorithmDescriptor(method="counts-test", version="1")

    def plan(self, problem, *, output, execution, shots, rng):
        nodes = {"root": Sequence(children=("allocate", "measure", "release")),
                 "allocate": Allocate(wire="q"), "measure": Measure(wire="q", result="c"),
                 "release": Release(wire="q")}
        program = Program(root="root", registers=(Register(name="q", width=1),),
            classical=(ClassicalValue(name="c", dtype="bits", width=1),),
            definitions=tuple(Definition(id=name, node=node) for name, node in nodes.items()))
        return Plan(problem=problem, method=self, output=output, execution=execution, shots=shots,
            randomness=rng.snapshot(), construction=SelectedConstruction(program=program, selections=()),
            experiments=(Experiment(name="counts", setting="counts", observation=ObservationSpec(kind="counts", shots=shots)),))._bind()

    def analyze(self, plan, data, *, settings):
        chunks = data.observations.chunks
        return CountsResult(plan_id=plan.content_id, construction_id=plan.construction.content_id,
            observation_id=data.observations.content_id, contribution_ids=tuple(c.content_id for c in chunks),
            counts=sum(c.returned_shots for c in chunks))

    def save_archive(self, plan, files):
        return dict(plan=plan.to_record(), problem=files.write_problem(plan.problem),
                    output=files.write_output(plan.output))

    @classmethod
    def load_archive(cls, saved, files):
        return files.read_plan(saved["plan"], problem=files.read_problem(saved["problem"]),
            method=cls(), output=files.read_output(saved["output"]))._bind()


def selected(shots=7):
    return CountsMethod().plan(Eigenproblem(A=[[1, 0], [0, -1]]), output=Samples(),
                              execution="quantum", shots=shots, rng=RandomStreams(31))


def test_empty_run_rejects_a_method_loader_that_changes_the_selected_plan(tmp_path, monkeypatch):
    from nwqlib._run_archive import load

    path = tmp_path / "empty"
    with Run(selected(), backend=InjectedBackend(), directory=path):
        pass
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.plan.shots == 7 and restored.trace.jobs == 0
    reader = CountsMethod.load_archive.__func__

    def wrong_shots(cls, saved, files):
        return reader(cls, saved, files).revise(shots=9)._bind()

    monkeypatch.setattr(CountsMethod, "load_archive", classmethod(wrong_shots))
    with pytest.raises(ValueError, match="original selected Plan"):
        load(path, backend=InjectedBackend(), method=CountsMethod)
    assert InjectedBackend.calls == []


class InjectedBackend(Record):
    kind: Literal["injected"] = "injected"
    supports_synchronous: ClassVar[bool] = False
    requires_prepared_payload: ClassVar[bool] = False
    calls: ClassVar[list] = []
    status: ClassVar[str] = "acknowledged"
    fail: ClassVar[bool] = False

    def target_for(self, observation):
        return SimpleNamespace(readouts=("counts",))

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot):
        from nwqlib.backends.connection import NativePreparation, circuit_layout
        self.calls.append(("prepare", runtime.seed))
        source = Source(name="injected", version="1", domain="offline test", reference="test:injected")
        native = SimpleNamespace(circuit=circuit, observation=observation)
        return NativePreparation(native=native, target=source, compiler=source, native_basis=("measure",),
            environment=(), quantum_layout=circuit_layout(circuit, circuit.qregs),
            classical_layout=circuit_layout(circuit, circuit.cregs), logical_to_native=(0,),
            operations=len(circuit.data), population="unconditional", counts_sampling=CountsSampling(kind="fixed_seed", seed=runtime.seed), transformation="actual one-qubit measurement")

    def submit(self, native, *, submission_id, run):
        locator = self.launch((native,), submission_id=submission_id, run=run)
        run._ack_submission(submission_id, locator)
        if self.fail:
            raise RuntimeError("native read failed")
        return self._result(native)

    def admit_batch(self, natives):
        pass

    def launch(self, natives, *, submission_id, run):
        self.calls.append(("launch", submission_id))
        return JobLocator(provider=self.kind, job_id="original-job")

    def restore_native(self, record, payload=None, *, run):
        self.calls.append(("restore", record.snapshot))
        return SimpleNamespace(circuit=None, observation=record.observation)

    def reconcile(self, submission, natives, *, run):
        self.calls.append(("reconcile", submission.submission_id))
        return submission.locator

    def refresh(self, locator, natives, *, run):
        from nwqlib.backends.connection import BackendRefresh
        self.calls.append(("refresh", locator.job_id))
        return BackendRefresh(status=self.status,
            results=() if self.status != "completed" else (("0", self._result(natives[0])),))

    def cancel(self, locator, *, run):
        self.calls.append(("cancel", locator.job_id))

    def _result(self, native):
        return SimpleNamespace(raw_output={"counts": {"0": native.observation.shots}},
                               metadata={"native_job_id": "original-job"})


@pytest.fixture(autouse=True)
def clear_backend():
    InjectedBackend.calls.clear()
    InjectedBackend.status = "acknowledged"
    InjectedBackend.fail = False
    InjectedBackend.supports_synchronous = False


@pytest.mark.parametrize("outcome", ("prepare_failure", "solve_success", "solve_failure", "header_failure"))
def test_convenience_run_ownership_releases_original_lock(tmp_path, monkeypatch, outcome):
    """An unreachable internal Run cannot keep its process's exclusive lock."""
    from pathlib import Path
    import nwqlib
    from nwqlib._run_archive import load

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    path = tmp_path / "prepare"
    original_prepare = InjectedBackend.prepare
    original_write = Run._write

    def fail_prepare(*args, **kwargs):
        raise ConnectionError("preparation interrupted")

    def fail_header(self, records=(), **kwargs):
        if any(kind == "header" for kind, _, _ in records):
            raise ConnectionError("header interrupted")
        return original_write(self, records, **kwargs)

    if outcome == "prepare_failure":
        monkeypatch.setattr(InjectedBackend, "prepare", fail_prepare)
        with pytest.raises(ConnectionError, match="preparation interrupted"):
            nwqlib.prepare(selected(), backend=InjectedBackend(), directory=path)
    elif outcome == "header_failure":
        monkeypatch.setattr(Run, "_write", fail_header)
        with pytest.raises(ConnectionError, match="header interrupted"):
            Run(selected(), backend=InjectedBackend(), directory=path)
    else:
        InjectedBackend.status = "completed"
        if outcome == "solve_failure":
            InjectedBackend.supports_synchronous = True
            InjectedBackend.fail = True
            with pytest.raises(RuntimeError, match="native read failed"):
                nwqlib.solve(selected(), backend=InjectedBackend())
        else:
            result = nwqlib.solve(selected(), backend=InjectedBackend())
            assert result.counts == 7 and result.data.observations.chunks
        path, = (tmp_path / ".nwqlib" / "runs").iterdir()
    monkeypatch.setattr(InjectedBackend, "prepare", original_prepare)
    monkeypatch.setattr(Run, "_write", original_write)
    # Header failure deliberately has no committed header; test its lock owner
    # directly. The other paths must also remain readable as scientific runs.
    if outcome == "header_failure":
        from nwqlib._run_journal import LocalJournal
        journal = LocalJournal(path / "run.sqlite", 100_000_000, create=False)
        journal.close()
    else:
        with load(path, backend=InjectedBackend(), method=CountsMethod) as reopened:
            assert reopened.plan.problem.A.dense_array().tolist() == [[1, 0], [0, -1]]
            if outcome == "solve_success":
                assert reopened.result.counts == result.counts


def test_a_creation_refused_inside_its_manifest_publishes_no_run_json(tmp_path):
    """A Run refused while it writes ``run.json`` leaves no ``run.json``, so no truncated manifest is loaded.

    ``max_data_bytes`` is charged for each part of a file before it is written, so a limit that ends inside
    the manifest refuses the creation there. The limit is set halfway through the manifest of an otherwise
    identical Run, after the selected inputs and ``selection.json``. ``_run_archive._write_manifest`` writes
    the manifest under a temporary name and renames it once complete, so ``load_run`` then reports a Run
    whose creation stopped before its manifest instead of parsing a truncated file.
    """
    import nwqlib

    whole = tmp_path / "whole"
    nwqlib.prepare(selected(), backend=InjectedBackend(), directory=whole).run.close()
    sizes = {path.name: path.stat().st_size for path in whole.iterdir()}
    stored = sum(size for name, size in sizes.items() if not name.startswith("run."))
    folder = tmp_path / "refused"
    with pytest.raises(ValueError, match="max_data_bytes"):
        nwqlib.prepare(selected(), backend=InjectedBackend(), directory=folder,
                       limits=ExecutionLimits(max_data_bytes=stored + sizes["run.json"] // 2))
    assert (folder / "selection.json").stat().st_size == sizes["selection.json"]
    assert not (folder / "run.json").exists()
    with pytest.raises(FileNotFoundError):
        nwqlib.load_run(folder, backend=InjectedBackend())


@pytest.mark.parametrize("status", ("failed", "cancelled", "acknowledged", "completed"))
def test_wait_distinguishes_terminal_backend_from_pending(tmp_path, status):
    """Original remote status reaches wait without another launch or a fake Result."""
    import nwqlib

    InjectedBackend.status = status
    prepared = nwqlib.prepare(selected(), backend=InjectedBackend(), directory=tmp_path / status)
    with prepared.run as run:
        nwqlib.submit(prepared)
        if status == "completed":
            assert run.wait(timeout=0).counts == 7
        elif status == "acknowledged":
            with pytest.raises(TimeoutError, match="pending"):
                run.wait(timeout=0)
        else:
            from nwqlib.execution import RunFailed
            with pytest.raises(RunFailed) as caught:
                run.wait(timeout=0)
            failure = caught.value
            assert (failure.stage, failure.status) == ("execute", status)
            assert failure.locator.job_id == "original-job"
            assert failure.directory == run.directory
            assert failure.exposure["uncertain"]["shots"] == 7
            with pytest.raises(TypeError):
                failure.exposure["uncertain"]["shots"] = 0
            assert run.result is None
        assert sum(call[0] == "launch" for call in InjectedBackend.calls) == 1


def test_failure_updates_physical_exposure_and_reopens_original_job(tmp_path):
    from nwqlib._run_archive import load
    InjectedBackend.supports_synchronous = True
    InjectedBackend.fail = True
    path = tmp_path / "failed"
    run = Run(selected(), backend=InjectedBackend(), directory=path)
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    with pytest.raises(RuntimeError, match="native read failed"):
        submit_experiment(handle, run=run)
    assert run.exposure["uncertain"] == dict(jobs=1, circuits=1, shots=7, provider_managed_sampling=0)
    assert run.exposure["reserved"] == dict(jobs=0, circuits=0, shots=0, provider_managed_sampling=0)
    submission, = run.trace.submissions
    assert submission.locator.job_id == "original-job" and submission.native_invocations == 1
    assert submission.timing.seconds >= 0
    before = run.rng.snapshot()
    run.extend_limits(max_total_shots=run.limits.max_total_shots + 7)
    amendment, = run.limit_amendments
    assert (amendment.circuit_preparations, amendment.circuit_attempts, amendment.raw_shots) == (1, 1, 7)
    assert amendment.pending_output_bytes > 0
    run.close()
    restored = load(path, backend=InjectedBackend(), method=CountsMethod)
    assert restored.exposure["uncertain"] == dict(jobs=1, circuits=1, shots=7, provider_managed_sampling=0)
    assert restored.rng.snapshot() == before
    assert restored.limit_amendments == (amendment,)
    InjectedBackend.status = "completed"
    from nwqlib._prepared_execution import refresh_submissions
    chunk, = refresh_submissions(run=restored)
    assert restored.collect(chunk) and not restored.collect(chunk)
    assert restored.exposure["completed"] == dict(jobs=1, circuits=1, shots=7, provider_managed_sampling=0)
    assert len([c for c in InjectedBackend.calls if c[0] == "launch"]) == 1
    restored.close()


@pytest.mark.parametrize("failure", [None, "read", "foreign_result"])
def test_sync_submission_keeps_acknowledged_locator_for_success_and_failed_read(tmp_path, monkeypatch, failure):
    """A custom synchronous backend acknowledges its job before reading the result. Run keeps
    that locator after success or failure, rejects a result from another job, and a later
    refresh retrieves the original job without another launch.
    """
    from nwqlib._prepared_execution import refresh_submissions
    from nwqlib._run_archive import load
    # Account context exists only in the acknowledged locator, not in the returned job name.
    acknowledged = JobLocator(provider="injected", job_id="original-job", account="original-account")

    def launch(self, natives, *, submission_id, run):
        self.calls.append(("launch", submission_id))
        return acknowledged

    def foreign(self, native):
        return SimpleNamespace(raw_output={"counts": {"0": native.observation.shots}},
                               metadata={"native_job_id": "foreign-job"})

    monkeypatch.setattr(InjectedBackend, "launch", launch)
    InjectedBackend.supports_synchronous = True
    InjectedBackend.fail = failure == "read"
    path = tmp_path / "sync"
    with Run(selected(), backend=InjectedBackend(), directory=path) as run:
        handle = prepare_experiment(run.plan.resolve("counts"), run=run)
        if failure is None:
            run.collect(submit_experiment(handle, run=run))
        else:
            with monkeypatch.context() as patch:
                if failure == "foreign_result":
                    patch.setattr(InjectedBackend, "_result", foreign)
                error, message = ((RuntimeError, "native read failed") if failure == "read"
                                  else (ValueError, "original job locator"))
                with pytest.raises(error, match=message):
                    submit_experiment(handle, run=run)
        original, = run.trace.submissions
        assert original.locator == acknowledged
        assert original.status == ("uncertain" if failure else "completed")
        assert run.trace.jobs == 1
    InjectedBackend.fail = False
    InjectedBackend.status = "completed"
    with load(path, backend=InjectedBackend(), method=CountsMethod) as run:
        if failure == "foreign_result":
            # A refreshed result is joined to its submission item when it is
            # decoded. Reopening does not repeat that join, so this is where a
            # result from another job must be refused.
            with monkeypatch.context() as patch:
                patch.setattr(InjectedBackend, "_result", foreign)
                with pytest.raises(ValueError, match="job/item association"):
                    refresh_submissions(run=run)
            assert not run.observations.chunks
        incoming = refresh_submissions(run=run)
        assert len(incoming) == int(failure is not None)
        submission, = run.trace.submissions
        assert submission.locator == acknowledged and submission.status == "completed"
        for chunk in incoming:
            run.collect(chunk)
        assert len(run.observations.chunks) == 1 and not refresh_submissions(run=run)
    calls = [name for name, _ in InjectedBackend.calls]
    assert calls.count("launch") == 1 and calls.count("refresh") == {None: 0, "read": 1, "foreign_result": 2}[failure]


def test_pending_resume_keeps_seed_caps_and_once_only_collection(tmp_path, monkeypatch):
    from nwqlib._run_archive import load
    from nwqlib.saved_evidence import load_result
    path = tmp_path / "pending"
    run = Run(selected(), backend=InjectedBackend(), directory=path,
              limits=ExecutionLimits(max_total_circuits=2, max_total_shots=7))
    run.plan.method.prepare(run.plan, run=run)
    # A successful unused preparation counts, but does not invent an acquisition.
    prepare_experiment(run.plan.resolve("counts"), run=run)
    run.resume()
    assert run.result is None and run.exposure["reserved"]["shots"] == 7
    before, calls = run.trace, tuple(InjectedBackend.calls)
    rng = run.rng.snapshot()
    run.extend_limits(max_total_circuits=3, max_total_shots=14)
    first, = run.limit_amendments
    assert (first.sequence, first.circuit_preparations, first.circuit_attempts, first.raw_shots) == (1, 2, 1, 7)
    assert first.old == before.limits and first.new == run.limits
    assert first.stored_data_bytes == before.data_bytes
    assert first.pending_output_bytes == before.events[0].data_bytes_reserved > 0
    assert first.recorded_at.utcoffset() is not None
    with pytest.raises(ValueError, match="frozen"):
        first.raw_shots = 0
    assert before.limit_amendments == () and run.trace.data_bytes > before.data_bytes
    assert tuple(InjectedBackend.calls) == calls and run.rng.snapshot() == rng
    run.close()
    restored = load(path, backend=InjectedBackend(), method=CountsMethod)
    assert restored.limits.max_total_shots == 14 and restored.rng.snapshot() == rng
    assert restored.limit_amendments == (first,)
    InjectedBackend.status = "completed"
    result = restored.wait(timeout=1, poll_interval=0)
    assert result.counts == 7 and len(result.data.observations.chunks) == 1
    assert restored.wait() is result and restored.rng.snapshot() == rng
    captured, report, before = result.data.trace, result.report(), restored.trace
    calls = tuple(InjectedBackend.calls)
    restored.extend_limits(max_total_shots=21, max_data_bytes=restored.limits.max_data_bytes + 4096)
    second = restored.limit_amendments[1]
    assert (second.sequence, second.circuit_preparations, second.circuit_attempts, second.raw_shots) == (2, 2, 1, 7)
    assert second.old == first.new and second.new == restored.limits
    assert second.stored_data_bytes == before.data_bytes and second.pending_output_bytes == 0
    assert result.data.trace is captured and result.report() == report
    assert tuple(InjectedBackend.calls) == calls and restored.rng.snapshot() == rng
    saved_result = result.save(tmp_path / "result")
    copy = restored.save(tmp_path / "copy")
    restored.close()

    def forbidden(*args, **kwargs):
        raise AssertionError("cap history restoration replayed scientific work")
    monkeypatch.setattr(CountsMethod, "plan", forbidden)
    monkeypatch.setattr(CountsMethod, "analyze", forbidden)
    for source in (path, copy):
        with load(source, backend=InjectedBackend(), method=CountsMethod) as reopened:
            assert reopened.limit_amendments == (first, second)
            assert reopened.limits == second.new and reopened.rng.snapshot() == rng
            assert reopened.result.data.trace == captured
            assert reopened.result.report() == report
    assert load_result(saved_result, method=CountsMethod).data.trace == captured
    assert [c[0] for c in InjectedBackend.calls].count("prepare") == 2
    assert [c[0] for c in InjectedBackend.calls].count("launch") == 1


def test_in_memory_cap_history_save_keeps_completed_result_prefix(tmp_path, monkeypatch):
    """A completed Result keeps its earlier limit-history prefix after the live Run receives
    another cap increase.
    """
    from nwqlib._run_archive import load
    monkeypatch.setattr(InjectedBackend, "launch", None)
    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    def submit(self, native, *, submission_id, run):
        self.calls.append(("submit", submission_id))
        return self._result(native)
    monkeypatch.setattr(InjectedBackend, "submit", submit)
    run = Run(selected(), backend=InjectedBackend(), limits=ExecutionLimits(max_total_shots=7))
    assert run.directory is None
    run.extend_limits(max_total_shots=14)
    result = run.wait(timeout=1, poll_interval=0)
    captured = result.data.trace
    run.extend_limits(max_total_shots=21)
    first, second = run.limit_amendments
    assert (first.circuit_preparations, first.circuit_attempts, first.raw_shots) == (0, 0, 0)
    assert (second.circuit_preparations, second.circuit_attempts, second.raw_shots) == (1, 1, 7)
    path = run.save(tmp_path / "memory")
    assert run.directory is None
    run.close()
    with load(path, backend=InjectedBackend(), method=CountsMethod) as reopened:
        assert reopened.limit_amendments == (first, second)
        assert reopened.limits == second.new
        assert reopened.result.data.trace == captured
    assert [name for name, _ in InjectedBackend.calls].count("submit") == 1


def test_cap_amendment_is_atomic_and_equal_values_do_not_write(tmp_path, monkeypatch):
    """A database trigger fails the amendment transaction so neither the current cap nor history
    may change.
    """
    import sqlite3
    from nwqlib._run_archive import load
    run = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "atomic",
              limits=ExecutionLimits(max_total_shots=7))
    run.extend_limits(max_total_shots=14)
    before, history, rng = run.trace, run.limit_amendments, run.rng.snapshot()
    store = run.artifacts
    journal = run._state["journal"]
    original_commit = journal.commit
    deltas = []
    def commit(records=(), **kwargs):
        assert run.limits == before.limits and run.limit_amendments == history
        deltas.append(tuple((kind, key) for kind, key, _ in records))
        return original_commit(records, **kwargs)
    monkeypatch.setattr(journal, "commit", commit)
    assert run.extend_limits(max_total_shots=14) is run
    with pytest.raises(ValueError, match="extend_limits"):
        run.extend_limits(max_total_shots=13)
    assert deltas == [] and run.trace == before
    journal.connection.execute("CREATE TRIGGER fail_amendment BEFORE INSERT ON records "
        "WHEN NEW.kind='limit_amendment' BEGIN SELECT RAISE(ABORT,'amendment write failed'); END")
    with pytest.raises(sqlite3.IntegrityError, match="amendment write failed"):
        run.extend_limits(max_total_shots=21, max_data_bytes=before.limits.max_data_bytes + 4096)
    assert deltas == [(('limits', 'current'), ('limit_amendment', '2'))]
    assert run.limits == before.limits and run.limit_amendments == history
    assert run.trace == before and run.rng.snapshot() == rng and InjectedBackend.calls == []
    assert journal.max_bytes == before.limits.max_data_bytes
    assert store._cap == before.limits.max_data_bytes
    assert len(tuple(journal.rows("limit_amendment"))) == 1
    assert tuple(journal.rows("limits"))[0][1]["max_total_shots"] == 14
    journal.connection.execute("DROP TRIGGER fail_amendment")
    run.close()
    with load(tmp_path / "atomic", backend=InjectedBackend(), method=CountsMethod) as reopened:
        assert reopened.limits == before.limits and reopened.limit_amendments == history


def test_cap_history_metadata_is_admitted_before_publication(monkeypatch):
    # A synchronous backend without launch keeps this Run in memory.
    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    monkeypatch.setattr(InjectedBackend, "launch", None)
    run = Run(selected(), backend=InjectedBackend(), limits=ExecutionLimits(max_data_bytes=65536))
    # Fill the actual journal owner with scalar data, without quantum work.
    available = run.limits.max_data_bytes - run.trace.data_bytes
    # Two JSON quote bytes leave exactly one byte, too little for any
    # nonempty amendment record, independently of the admission formula.
    run._write((("test", "fill", "x" * (available - 3)),))
    before = run.trace
    with pytest.raises(ValueError, match="max_data_bytes"):
        run.extend_limits(max_total_shots=2_000_000)
    assert run.trace == before and run.limit_amendments == ()
    run.extend_limits(max_data_bytes=81920)
    amendment, = run.limit_amendments
    assert amendment.stored_data_bytes == before.data_bytes
    assert run.trace.data_bytes > before.data_bytes and InjectedBackend.calls == []
    run.close()


@pytest.mark.parametrize("oversized", (False, True))
def test_completed_counts_fit_reserved_bytes_or_report_excess_metadata(monkeypatch, oversized):
    """One acquisition fits at the cap, while unbounded provider text is explicit."""
    monkeypatch.setattr(InjectedBackend, "supports_synchronous", True)
    monkeypatch.setattr(InjectedBackend, "launch", None)
    allowance = 8192
    with Run(selected(), backend=InjectedBackend(),
             limits=ExecutionLimits(max_completion_metadata_bytes=allowance)) as run:
        handle = prepare_experiment(run.plan.resolve("counts"), run=run)

        def acquired(backend, native, *, submission_id, run):
            # The admission has already funded this acquisition. Leave exactly
            # its reserved bytes and no unreserved completion headroom.
            if not oversized:
                limit = run.trace.data_bytes + sum(run._state["pending_output"].values())
                run._state["limits"] = run.limits.revise(max_data_bytes=limit)
            return SimpleNamespace(raw_output={"counts": {"0": 7}},
                metadata={"native_job_id": "x"*(allowance if oversized else 1)})

        monkeypatch.setattr(InjectedBackend, "submit", acquired)
        if oversized:
            with pytest.raises(ValueError, match="completion metadata"):
                submit_experiment(handle, run=run)
            assert run.trace.events[0].status == "uncertain"
            assert run.exposure["uncertain"]["shots"] == 7
        else:
            chunk = submit_experiment(handle, run=run)
            assert chunk.returned_shots == 7
            assert run.trace.events[0].status == "completed"
            assert run._state["pending_output"] == {}
            assert run.trace.data_bytes <= run.limits.max_data_bytes


def test_saved_unused_aer_preparation_executes_without_new_lowering(tmp_path, monkeypatch):
    from nwqlib._run_archive import load
    from nwqlib.backends import AerBackend
    import nwqlib.blocks.lowering as lowering
    import nwqlib.backends.qiskit_aer as aer
    run = Run(selected(), limits=ExecutionLimits(simulator_memory_mb=32))
    run.plan.method.prepare(run.plan, run=run)
    expected_seed = run.rng.snapshot()
    path = run.save(tmp_path / "aer")
    def forbidden(*args, **kwargs):
        raise AssertionError("saved native preparation was rebuilt")
    monkeypatch.setattr(lowering, "_lower_qiskit", forbidden)
    monkeypatch.setattr(aer, "_prepare_aer_execution", forbidden)
    restored = load(path, backend=AerBackend(), method=CountsMethod)
    assert restored.rng.snapshot() == expected_seed
    # A reopened handle rebuilds its native object from its saved QPY at first use.
    native, = (handle._restore_native() for handle in restored._state["handles"].values())
    assert native.simulator.options.max_memory_mb == 32
    result = restored.wait(timeout=1, poll_interval=0)
    histogram = result.data.observations.chunks[0].histogram()
    assert result.counts == 7 and histogram.width == 1 and histogram.index_list()[0] == 0
    assert restored.trace.preparations == 1 and restored.trace.jobs == 1
    restored.close()


def test_cancellation_calls_original_pending_job_and_does_not_refund(tmp_path):
    run = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "cancel")
    run.resume()
    run.cancel()
    assert ("cancel", "original-job") in InjectedBackend.calls
    assert run.exposure["reserved"] == dict(jobs=1, circuits=1, shots=7, provider_managed_sampling=0)
    with pytest.raises(ValueError, match="cancellation"):
        run.check_capacity(circuits=1, shots=1)
    assert [name for name, _ in InjectedBackend.calls].count("launch") == 1
    run.close()


def test_cancel_without_backend_operation_keeps_original_job_readable(tmp_path):
    class LocalProcessBackend(InjectedBackend):
        # Like local NWQ-Sim, the backend cannot stop an already launched job.
        cancel: ClassVar[None] = None

    run = Run(selected(), backend=LocalProcessBackend(), directory=tmp_path / "local")
    run.resume()
    run.cancel()
    assert run.warnings == ("backend has no cancellation operation; original job remains observable",)
    with pytest.raises(ValueError, match="cancellation"):
        run.check_capacity(circuits=1, shots=1)
    # The original job may still finish; the same Run reads it without a new launch.
    InjectedBackend.status = "completed"
    result = run.wait(timeout=1, poll_interval=0)
    assert result.counts == 7 and run.cancel_requested == "user requested cancellation"
    calls = [name for name, _ in InjectedBackend.calls]
    assert calls.count("launch") == 1 and "cancel" not in calls
    run.close()


def test_cache_checkpoint_reuses_immutable_arrays_and_removes_superseded_files(tmp_path, monkeypatch):
    import numpy as np
    from nwqlib._run_archive import load
    class CacheMethod(CountsMethod):
        def save_run_context(self, context, files):
            return dict(values=files.write_array("values.npy", context["values"]))
        @staticmethod
        def load_run_context(saved, files):
            return dict(values=files.read_array(saved["values"]))
    original = selected()
    plan = original.revise(method=CacheMethod())
    run = Run(plan, backend=InjectedBackend(), directory=tmp_path / "caches")
    context = run._method_context(lambda: dict(values=np.array([1., 2.])))
    run.checkpoint(dict(step=0))
    run.checkpoint(dict(step=1))
    assert len(tuple(run.directory.glob("*values.npy"))) == 1
    context["values"] = np.array([3., 4.])
    run.checkpoint(dict(step=2))
    assert len(tuple(run.directory.glob("*values.npy"))) == 1
    run.close()
    restored = load(tmp_path / "caches", backend=InjectedBackend(), method=CacheMethod)
    np.testing.assert_array_equal(restored._state["method_context"]["values"], [3., 4.])
    assert restored.checkpoint_state == dict(step=2)
    with monkeypatch.context() as patch:
        patch.setattr(np, "save", lambda *a, **k: pytest.fail("unchanged mapped cache was copied"))
        restored.checkpoint(dict(step=3))
    assert len(tuple(restored.directory.glob("*values.npy"))) == 1
    restored._state["method_context"]["values"] = np.array([5., 6.])
    restored.checkpoint(dict(step=4))
    assert len(tuple(restored.directory.glob("*values.npy"))) == 1
    restored.close()


def test_checkpoint_writes_only_changed_fields_and_restores_the_whole_state(tmp_path, monkeypatch):
    """A step that changes one small field neither re-encodes nor rewrites a large one."""
    import json
    from pathlib import Path
    import nwqlib._run_journal as storage
    from nwqlib._run_archive import load
    path = tmp_path / "fields"
    run = Run(selected(), backend=InjectedBackend(), directory=path)
    history = ["x" * 1000] * 50
    assert run.checkpoint(dict(history=history, step=0)) == 1
    encoded, original = [], storage._encode_admitted

    def spy(value, max_bytes):
        encoded.append(original(value, max_bytes))
        return encoded[-1]

    with monkeypatch.context() as patch:
        patch.setattr(storage, "_encode_admitted", spy)
        for step in (1, 2, 3):
            assert run.checkpoint(dict(history=history, step=step), changed=("step",)) == step + 1
    assert len(encoded) == 9 and max(map(len, encoded)) < 1000
    # An added or removed field must be named; a rejected step leaves the commit intact.
    with pytest.raises(ValueError, match="including every added or removed one"):
        run.checkpoint(dict(history=history, step=4, added=True), changed=("step",))
    assert run.checkpoint_state == dict(history=history, step=3) and run.checkpoint_sequence == 4
    assert run.checkpoint(dict(step=5), changed=("step", "history")) == 5
    assert [key for key, _, _ in run._state["journal"].rows("checkpoint_field")] == ["step"]
    assert json.loads(run.data.controller) == dict(sequence=5, state=dict(step=5))
    run.close()
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.checkpoint_state == dict(step=5) and restored.checkpoint_sequence == 5
        assert restored.checkpoint(dict(step=6, added=[1]), changed=("step", "added")) == 6
    with load(path, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.checkpoint_state == dict(step=6, added=[1])
    # Without a directory the detached Run uses the default run directory; keep it in tmp_path.
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    memory = Run(selected(), backend=InjectedBackend())
    memory.checkpoint(dict(history=history, step=0))
    memory.checkpoint(dict(history=history, step=1), changed=("step",))
    copy = memory.save(tmp_path / "memory-copy")
    memory.close()
    with load(copy, backend=InjectedBackend(), method=CountsMethod) as restored:
        assert restored.checkpoint_state == dict(history=history, step=1) and restored.checkpoint_sequence == 2


def test_extended_data_limit_is_read_before_loading_new_data(tmp_path):
    from nwqlib._run_archive import load
    path = tmp_path / "extended"
    run = Run(selected(), backend=InjectedBackend(), directory=path,
              limits=ExecutionLimits(max_data_bytes=100_000))
    run.extend_limits(max_data_bytes=1_000_000)
    run.checkpoint(dict(payload="x" * 110_000))
    run.close()
    restored = load(path, backend=InjectedBackend(), method=CountsMethod)
    assert restored.limits.max_data_bytes == 1_000_000
    assert restored.checkpoint_state["payload"] == "x" * 110_000
    restored.close()


def test_analysis_recovery_requires_explicit_supported_interruption(tmp_path):
    run = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "recovery")
    with pytest.raises(ValueError, match="no interrupted controller"):
        run.resume(reanalyze=True)
    assert not InjectedBackend.calls and not run.trace.events
    run.close()


@pytest.mark.parametrize("synchronous,missing", [(True, None), (True, "submit"), (False, None), (False, "launch")])
def test_backend_admission_requires_its_submission_route(tmp_path, monkeypatch, synchronous, missing):
    """A backend without its route's submission operation is rejected before preparation or intent."""
    InjectedBackend.supports_synchronous = synchronous
    if missing is not None:
        monkeypatch.setattr(InjectedBackend, missing, None)
    directory = tmp_path / "run"
    if missing is None:
        with Run(selected(), backend=InjectedBackend(), directory=directory) as run:
            assert not run.trace.events
    else:
        with pytest.raises(TypeError, match="selected preparation and submission"):
            Run(selected(), backend=InjectedBackend(), directory=directory)
        assert not directory.exists()
    assert InjectedBackend.calls == []


@pytest.mark.parametrize("synchronous", [False, True])
def test_method_admission_precedes_submission_and_is_not_repeated_on_refresh(tmp_path, monkeypatch, synchronous):
    InjectedBackend.supports_synchronous = synchronous
    run = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "admit")
    handle = prepare_experiment(run.plan.resolve("counts"), run=run)
    seen = []
    def reject(self, plan, prepared, *, run):
        seen.append(prepared)
        raise ValueError("selected statistical premise fails")
    with monkeypatch.context() as patch:
        patch.setattr(CountsMethod,"before_submit",reject)
        with pytest.raises(ValueError,match="statistical premise"):
            submit_experiment(handle,run=run)
    assert seen == [(handle,)] and not run.trace.events and run.trace.jobs == 0
    if not synchronous:
        submit_experiment(handle,run=run)
        monkeypatch.setattr(CountsMethod,"before_submit",reject)
        InjectedBackend.status = "completed"
        from nwqlib._prepared_execution import refresh_submissions
        assert len(refresh_submissions(run=run)) == 1
        assert seen == [(handle,)]
    run.close()


@pytest.mark.parametrize("snapshot", ["current", "stale"])
def test_run_publishes_only_a_result_attached_to_its_current_data(tmp_path, monkeypatch, snapshot):
    original = CountsMethod.analyze
    before, returned = [], []

    def analyze(self, plan, data, *, settings):
        # A consistent Result of the pre-acquisition snapshot passes
        # Result._attach, so only the Run's publication check can reject it.
        data = data if snapshot == "current" else before[0]
        returned.append(original(self, plan, data, settings=settings)._attach(plan, data))
        return returned[-1]

    monkeypatch.setattr(CountsMethod, "analyze", analyze)
    InjectedBackend.status = "completed"
    InjectedBackend.supports_synchronous = True
    with Run(selected(), backend=InjectedBackend(), directory=tmp_path / snapshot) as run:
        before.append(run.data)
        if snapshot == "current":
            run.resume()
            assert run.result is returned[0] and run.result.counts == 7
            assert run.result.data.trace.events == run.trace.events
            assert run.result.data.trace.submissions == run.trace.submissions
        else:
            with pytest.raises(ValueError, match="acquisition exposure"):
                run.resume()
            # The stale snapshot has no observations; the run collected one.
            assert returned[0].counts == 0 and len(run.observations.chunks) == 1
            assert run.result is None
        assert len(list(run._state["journal"].rows("result"))) == (snapshot == "current")


def test_repeated_imported_handle_in_one_batch_has_one_receipt_and_two_attempts(tmp_path):
    from nwqlib._prepared_execution import submit_detached
    plan=selected()
    with Run(plan,backend=InjectedBackend(),directory=tmp_path/"source") as source:
        handle=prepare_experiment(plan.resolve("counts"),run=source)
    with Run(plan,backend=InjectedBackend(),directory=tmp_path/"destination") as destination:
        submission=submit_detached((handle,handle),run=destination)
        assert len(destination.prepared_artifacts)==1 and destination.trace.preparations==0
        assert len(submission.items)==2 and len(destination.trace.events)==2
        assert destination.trace.jobs==1 and sum(event.shots for event in destination.trace.events)==14
        assert len({event.attempt for event in destination.trace.events})==2


def test_acquired_counts_identity_follows_provider_job_child_and_item():
    """Different physical job, child job or submitted item coordinates are different count
    populations, while re-reading one acquired item keeps its identity.
    """
    from nwqlib._counts import CountsSources
    from nwqlib.execution import SubmissionItem, SubmissionRecord

    prepared = "sha256:" + "1" * 64
    backend = Source(name="fixture", version="1", domain="test", reference="test:counts identity")

    def identity(job="job-a", child_job=None, item=0, provider="fixture"):
        member = SubmissionItem(attempt="attempt", prepared_id=prepared, item=item, child_job=child_job)
        submission = SubmissionRecord(submission_id="submission", run_id="run", backend=backend,
            items=(member,), status="acknowledged", started="t0",
            locator=JobLocator(provider=provider, job_id=job))
        # No saved preparation receipt: only the acquisition coordinates determine the identity.
        sources = CountsSources(lambda _: None, lambda _: (submission, member))
        return sources.resolve(prepared, "attempt").identity

    original = identity()
    assert identity() == original
    # The same job id at another provider is another physical job.
    assert len({original, identity(job="job-b"), identity(child_job="child-1"), identity(item=1),
                identity(provider="other-provider")}) == 5


@pytest.mark.parametrize("repeat_fails", (False, True))
def test_repeated_submit_keeps_each_explicit_owner(tmp_path, monkeypatch, repeat_fails):
    """A second submit launches new work without surrendering the first Run."""
    from pathlib import Path
    import nwqlib
    from nwqlib._run_archive import load

    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path))
    InjectedBackend.supports_synchronous = True
    prepared = nwqlib.prepare(selected(), backend=InjectedBackend(), directory=tmp_path / "first")
    with prepared.run as first:
        assert nwqlib.submit(prepared) is first
        assert first.result.counts == 7
        InjectedBackend.fail = repeat_fails
        if repeat_fails:
            with pytest.raises(RuntimeError, match="native read failed"):
                nwqlib.submit(prepared)
            second_path, = (tmp_path / ".nwqlib" / "runs").iterdir()
            with load(second_path, backend=InjectedBackend(), method=CountsMethod) as recovered:
                assert recovered.result is None
                assert recovered.trace.submissions[0].locator.job_id == "original-job"
        else:
            with nwqlib.submit(prepared) as second:
                assert second is not first and second.run_id != first.run_id
                assert second.result.counts == 7
                with pytest.raises(BlockingIOError):
                    load(second.directory, backend=InjectedBackend(), method=CountsMethod)
        with pytest.raises(BlockingIOError):
            load(first.directory, backend=InjectedBackend(), method=CountsMethod)
        assert first.result.counts == 7
        assert sum(call[0] == "launch" for call in InjectedBackend.calls) == 2


def test_method_can_publish_available_data_after_another_attempt_fails(tmp_path, monkeypatch):
    """Lifecycle preserves the Method's interpretation of actual partial observations."""
    run = Run(selected(), backend=InjectedBackend(), directory=tmp_path / "partial")
    with run:
        InjectedBackend.supports_synchronous = True
        handle = prepare_experiment(run.plan.resolve("counts"), run=run)
        run.collect(submit_experiment(handle, run=run))
        InjectedBackend.supports_synchronous = False
        submit_experiment(handle, run=run)
        InjectedBackend.status = "failed"

        def available_result(self, plan, *, run):
            return self.analyze(plan, run.data, settings={})

        monkeypatch.setattr(CountsMethod, "execute", available_result)
        result = run.wait(timeout=0)
        assert result.counts == 7 and len(result.data.observations.chunks) == 1
        assert [item.status for item in result.data.trace.submissions] == ["completed", "failed"]
        assert sum(event.shots for event in result.data.trace.events) == 14
        assert sum(call[0] == "launch" for call in InjectedBackend.calls) == 2


@pytest.mark.parametrize('interrupted', [False, True])
def test_host_exception_is_terminal_but_process_interruption_stays_uncertain(tmp_path, monkeypatch, interrupted):
    import nwqlib as nw
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.blocks.kernels import BoundKernel
    from nwqlib.execution import RunFailed

    calls = []
    error = KeyboardInterrupt('process interrupted') if interrupted else ArithmeticError('selected host arithmetic failed')
    def fail(self):
        calls.append(1)
        raise error
    selected = nw.plan(nw.Expectation(state=[1,1], observable=[[1,0],[0,-1]]),
                       method=ExpectationMethod(), execution='classical')
    prepared = nw.prepare(selected, directory=tmp_path/'failed-host')
    with monkeypatch.context() as patch:
        patch.setattr(BoundKernel, '_invoke', fail)
        with pytest.raises(type(error)) as caught:
            nw.submit(prepared)
    assert caught.value is error and calls == [1]
    expected = 'uncertain' if interrupted else 'failed'
    run = prepared.run
    assert run.trace.events[0].status == expected
    work = run.trace.host_work_reserved
    with pytest.raises(RunFailed) as caught:
        run.wait(timeout=0)
    assert caught.value.stage == ('recovery' if interrupted else 'execute')
    assert caught.value.status == expected
    if not interrupted:
        assert caught.value.failure == 'ArithmeticError: selected host arithmetic failed'
        assert caught.value.locator is None
    run.close()
    with nw.load_run(tmp_path/'failed-host', backend=None) as restored:
        assert restored.trace.host_invocations == 1 and restored.trace.host_work_reserved == work
        assert restored.trace.events[0].status == expected
        with pytest.raises(RunFailed):
            restored.wait(timeout=0)
    assert calls == [1]


@pytest.mark.parametrize('interrupted', [False, True])
def test_terminal_host_failure_allows_method_owned_partial_result(tmp_path, monkeypatch, interrupted):
    import nwqlib as nw
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.blocks.kernels import BoundKernel

    selected = nw.plan(nw.Expectation(state=[1,1], observable=[[1,0],[0,-1]]),
                       method=ExpectationMethod(), execution='classical')
    with nw.prepare(selected, directory=tmp_path/'partial').run as run:
        def fail(self):
            raise KeyboardInterrupt('host interrupted') if interrupted else ArithmeticError('host failed')
        with monkeypatch.context() as patch:
            patch.setattr(BoundKernel, '_invoke', fail)
            with pytest.raises(KeyboardInterrupt if interrupted else ArithmeticError):
                run.resume()
        # The Method explicitly permits an unavailable value with no observations.
        # Run.wait must not preempt that legitimate partial-result policy.
        def partial(self, plan, *, run):
            return self.analyze(plan, run.data, settings={})
        monkeypatch.setattr(ExpectationMethod, 'execute', partial)
        result = run.wait(timeout=0)
        assert result.value is None and not result.data.observations.chunks
        assert result.data.trace.host_invocations == 1
        if not interrupted:
            assert 'failed host invocations' in str(result)
        assert result.data.trace.events[0].status == ('uncertain' if interrupted else 'failed')


def test_host_completion_storage_failure_stays_uncertain_after_reopen(tmp_path, monkeypatch):
    import sqlite3
    import nwqlib as nw
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.blocks.kernels import BoundKernel

    choice = nw.plan(nw.Expectation(state=[1,1], observable=[[1,0],[0,-1]]),
                     method=ExpectationMethod(), execution='classical')
    prepared = nw.prepare(choice, directory=tmp_path/'storage-gap')
    run = prepared.run
    journal = run._state['journal']
    journal.connection.execute("CREATE TRIGGER reject_completion BEFORE UPDATE ON records "
        "WHEN NEW.kind='event' AND json_extract(NEW.payload,'$.status')='completed' "
        "BEGIN SELECT RAISE(ABORT, 'completion storage failure'); END")
    calls = []
    invoke = BoundKernel._invoke
    def counted(self):
        calls.append(1)
        return invoke(self)
    monkeypatch.setattr(BoundKernel, '_invoke', counted)
    with pytest.raises(sqlite3.IntegrityError, match='completion storage failure'):
        nw.submit(prepared)
    assert run._state['storage_failed'] and calls == [1]
    assert not run._state['receipts'] and run.trace.events[0].status == 'reserved'
    journal.connection.execute('DROP TRIGGER reject_completion')
    run.close()
    with nw.load_run(tmp_path/'storage-gap', backend=None) as restored:
        assert restored.trace.events[0].status == 'uncertain'
        from nwqlib.execution import RunFailed
        with pytest.raises(RunFailed) as caught:
            restored.wait(timeout=0)
        assert caught.value.stage == 'recovery' and caught.value.status == 'uncertain'
    assert calls == [1]


@pytest.mark.parametrize('refreshable,reconcilable,acknowledged', [
    (False, False, False), (True, False, False), (True, False, True), (True, True, False),
])
def test_uncertain_recovery_depends_on_actual_retrieval_capabilities(
    tmp_path, monkeypatch, refreshable, reconcilable, acknowledged
):
    from nwqlib.execution import RunFailed

    monkeypatch.setattr(InjectedBackend, 'supports_synchronous', True)
    backend = InjectedBackend()
    run = Run(selected(), backend=backend, directory=tmp_path/'retrieval')
    failure = KeyboardInterrupt('original acquisition interrupted')

    def interrupted(self, native, *, submission_id, run):
        if acknowledged:
            run._ack_submission(submission_id, JobLocator(provider=self.kind, job_id='original-job'))
        raise failure

    monkeypatch.setattr(InjectedBackend, 'submit', interrupted)
    with pytest.raises(KeyboardInterrupt) as caught:
        run.resume()
    assert caught.value is failure
    if not refreshable:
        monkeypatch.setattr(InjectedBackend, 'refresh', None)
    if not reconcilable:
        monkeypatch.setattr(InjectedBackend, 'reconcile', None)
    before = run.exposure
    attempt = run.trace.events[0].attempt
    monkeypatch.setattr(InjectedBackend, 'submit', lambda *a, **k: pytest.fail('acquisition replay'))
    monkeypatch.setattr('nwqlib._prepared_execution.sleep', lambda *a: pytest.fail('impossible wait'))
    if refreshable and (acknowledged or reconcilable):
        assert run.resume() is run and run.result is None
        with pytest.raises(TimeoutError):
            run.wait(timeout=0)
    else:
        for continuation in (run.resume, run.wait, lambda: run.wait(timeout=0)):
            with pytest.raises(RunFailed) as caught:
                continuation()
            error = caught.value
            assert (error.stage, error.status, error.attempt) == ('recovery', 'uncertain', attempt)
            assert error.failure == 'KeyboardInterrupt: original acquisition interrupted'
            assert str(tmp_path/'retrieval') in str(error) and attempt in str(error)
            assert dict(error.exposure['uncertain']) == before['uncertain']
    assert run.trace.events[0].status == 'uncertain' and not run.observations.chunks
    assert not any(kind == 'launch' for kind, _ in backend.calls)
    run.close()


def test_readout_refusal_names_selected_target_before_native_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(InjectedBackend, 'target_for',
        lambda self, observation: SimpleNamespace(name='selected-device', readouts=('probabilities',)))
    with Run(selected(), backend=InjectedBackend(), directory=tmp_path/'unsupported') as run:
        with pytest.raises(ValueError) as caught:
            prepare_experiment(run.plan.resolve('counts'), run=run)
        message = str(caught.value)
        assert 'selected-device' in message and "'counts'" in message and 'probabilities' in message
        assert not InjectedBackend.calls and run.trace.preparations == 0


def test_reopened_run_reports_to_the_progress_callback_given_to_load_run(tmp_path):
    """load_run gives the reopened Run the supplied progress callback."""
    import nwqlib as nw
    from nwqlib.algorithms.expectation import ExpectationMethod

    def interrupt(stage, done, total):
        if stage == 'analyze':
            raise KeyboardInterrupt('interrupted before the Result')

    selected = nw.plan(nw.Expectation(state=[1, 1], observable=[[1, 0], [0, -1]]),
                       method=ExpectationMethod(), execution='classical')
    prepared = nw.prepare(selected, directory=tmp_path/'run', progress=interrupt)
    with pytest.raises(KeyboardInterrupt):
        nw.submit(prepared)
    prepared.run.close()
    reports = []
    with nw.load_run(tmp_path/'run', backend=None, progress=lambda *report: reports.append(report)) as run:
        run.wait()
    assert reports[-2:] == [('analyze', 0, 1), ('analyze', 1, 1)]


def test_reopened_synchronous_aer_attempt_releases_only_its_output_reservation(tmp_path, monkeypatch):
    """Reopening an interrupted synchronous Aer publication frees its output bytes, not its exposure.

    ``Run._restore_submissions`` releases the pending output of an
    unacknowledged synchronous Aer submission without a locator, because its
    result existed only in the interrupted process. The job, circuit and shot
    counts stay those of the live process, now uncertain, and a second reopen
    repeats the same accounting. On each reopened Run both ``resume`` and
    ``wait`` raise the recovery error with the original attempt, the saved
    event, submission, directory and exposure, and repeat no backend call; after
    ``cancel`` the failed retrieval is not reported as a failed execution.
    The detached case, which keeps its reservation, is
    ``test_nwqsim_targets.py::test_local_lost_acknowledgement_binds_only_the_original_output_without_relaunch``.
    """
    import nwqlib
    from nwqlib._run_journal import LocalJournal
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import AerBackend
    from nwqlib.execution import RunFailed
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import ingest_occupation

    plan = nwqlib.plan(Expectation(state=ingest_occupation("01", num_qubits=2),
                                   observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2)),
                       method=ExpectationMethod(), shots=8, seed=5)
    original = LocalJournal.commit

    def commit(self, records=(), **kwargs):
        if any(kind == "chunk" for kind, _, _ in records):
            raise OSError("injected publication failure")
        return original(self, records, **kwargs)

    def totals(exposure):
        return {name: sum(row[name] for row in exposure.values()) for name in ("jobs", "circuits", "shots")}

    calls = []
    native = AerBackend.submit
    monkeypatch.setattr(AerBackend, "submit", lambda self, *args, **kwargs: calls.append(1) or native(
        self, *args, **kwargs))
    monkeypatch.setattr(LocalJournal, "commit", commit)
    prepared = nwqlib.prepare(plan, directory=tmp_path / "run", progress=False)
    with prepared.run as run:
        with pytest.raises(OSError, match="injected publication failure"):
            nwqlib.submit(prepared).wait()
        live = totals(run.exposure)
        attempt = run.trace.events[0].attempt
        assert sum(run._state["pending_output"].values()) > 0
    monkeypatch.setattr(LocalJournal, "commit", original)
    assert live == dict(jobs=1, circuits=1, shots=8)
    for _ in range(2):
        with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
            assert run._state["pending_output"] == {}
            assert totals(run.exposure) == live and run.exposure["uncertain"]["shots"] == 8
            (submission,) = run.trace.submissions
            (event,) = run.trace.events
            assert submission.status == "failed" and submission.results_consumed and submission.locator is None
            assert event.status == "uncertain" and event.attempt == attempt
            assert submission.failure == "interrupted synchronous Aer result is unavailable after reopen"
            for continuation in (run.resume, lambda: run.wait(timeout=0)):
                with pytest.raises(RunFailed) as caught:
                    continuation()
                error = caught.value
                assert (error.stage, error.status, error.attempt) == (
                    "recovery", "uncertain", event.attempt,
                )
                assert error.locator is None and error.directory == run.directory
                assert error.failure == submission.failure
                assert error.trace.events == (event,)
                assert error.trace.submissions == (submission,)
                assert dict(error.exposure["uncertain"]) == dict(run.exposure["uncertain"])
                assert run.result is None and not run.observations.chunks
                assert len(calls) == 1
    with nwqlib.load_run(tmp_path / "run", backend=AerBackend(), progress=False) as run:
        run.cancel()
        with pytest.raises(RuntimeError, match="run cancelled"):
            run.wait(timeout=0)
    assert calls == [1]
