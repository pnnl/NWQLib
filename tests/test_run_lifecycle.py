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


@pytest.mark.parametrize("status", ("failed", "acknowledged"))
def test_wait_distinguishes_terminal_backend_from_pending(tmp_path, status):
    """Original remote status reaches wait without another launch or a fake Result."""
    import nwqlib

    InjectedBackend.status = status
    prepared = nwqlib.prepare(selected(), backend=InjectedBackend(), directory=tmp_path / status)
    with prepared.run as run:
        nwqlib.submit(prepared)
        if status == "acknowledged":
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


def test_readout_refusal_names_selected_target_before_native_preparation(tmp_path, monkeypatch):
    monkeypatch.setattr(InjectedBackend, 'target_for',
        lambda self, observation: SimpleNamespace(name='selected-device', readouts=('probabilities',)))
    with Run(selected(), backend=InjectedBackend(), directory=tmp_path/'unsupported') as run:
        with pytest.raises(ValueError) as caught:
            prepare_experiment(run.plan.resolve('counts'), run=run)
        message = str(caught.value)
        assert 'selected-device' in message and "'counts'" in message and 'probabilities' in message
        assert not InjectedBackend.calls and run.trace.preparations == 0
