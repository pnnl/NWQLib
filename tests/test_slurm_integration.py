"""Common Slurm lifecycle with supplied native packets; zero simulations/site calls."""

import nwqlib
from nwqlib._run_archive import load as load_run
from nwqlib.algorithms.expectation import ExpectationMethod
from nwqlib.problems import Expectation
from nwqlib.problems.inputs import ingest_occupation
from nwqlib.operators.inputs import ingest_pauli

import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from nwqlib.backends import NWQSimSlurmBackend, SlurmProfile
from nwqlib.backends.slurm import SlurmLauncher
from nwqlib.core.planning import RuntimeOptions
from nwqlib._prepared_execution import (
    Run,
    prepare_experiment as prepare,
    refresh_submissions,
    submit_detached,
)

REVISION = "efd02262ff9c5def4f2eac1df6416c1ee2023d5b"


@pytest.fixture
def supplied(tmp_path, monkeypatch):
    """Model scheduler responses offline with separate queue, accounting, lost-ack and
    build-description controls.
    """
    from nwqlib.backends import nwqsim

    backend = NWQSimSlurmBackend(
        profile=SlurmProfile(
            cluster="perlmutter",
            account="original-account",
            walltime="00:05:00",
            nodes=1,
            ranks_per_node=1,
            threads=3,
            executable="/shared/original-runner",
            build_identity=REVISION,
            spool=str(tmp_path / "spool"),
        ),
        accounting_since="2026-09-10T00:00:00",
        accounting_until="2026-09-11T00:00:00",
        max_input_bytes=65536,
        max_output_bytes=65536,
        max_buffer_bytes=65536,
        max_response_bytes=4096,
        timeout_seconds=2,
    )
    calls, descriptions = ([], [])

    def describe(argv, **kwargs):
        assert argv == [backend.profile.executable, "--describe"]
        descriptions.append(argv)
        return SimpleNamespace(
            stdout=json.dumps(
                dict(
                    format="nwqlib.nwqsim/3",
                    backends=[state.backend.profile.backend],
                    methods=[state.backend.profile.method],
                    readouts=["counts", "pauli", "probabilities", "amplitudes", "trajectory"]
                    if state.backend.profile.backend == "CPU" and state.backend.profile.method == "SV"
                    else ["counts"],
                    distributed=state.backend.profile.backend == "MPI",
                    gpu_error_check=state.backend.profile.backend in {"NVGPU", "AMDGPU"},
                    nwqsim_revision=state.revision,
                    compiler="injected-compiler",
                )
            )
        )

    def scheduler(argv, *, max_output_bytes):
        calls.append(argv)
        if state.error is not None:
            error, state.error = (state.error, None)
            raise error
        if argv[0] == "sbatch":
            state.sid = Path(argv[-1]).parent.name
            state.input = json.loads((Path(argv[-1]).parent / "input.json").read_bytes())
            if state.before_ack is not None:
                state.before_ack()
            if state.lose_ack:
                raise state.lost_ack
            return "7309;perlmutter\n"
        if argv[0] == "squeue":
            assert "--jobs=7309" in argv and "--clusters=perlmutter" in argv
            return state.queue
        if argv[0] == "sacct":
            if "--format=JobIDRaw,JobName%80" in argv:
                assert f"--name=nwqlib-{state.sid}" in argv
                assert "--accounts=original-account" in argv
                assert "--starttime=2026-09-10T00:00:00" in argv
                assert "--endtime=2026-09-11T00:00:00" in argv
                return f"7309|nwqlib-{state.sid}\n" if state.found else ""
            assert "--jobs=7309" in argv
            return state.accounting
        assert argv == ("scancel", "--clusters=perlmutter", "7309")
        if state.cancel_error is not None:
            raise state.cancel_error
        return ""

    state = SimpleNamespace(
        backend=backend,
        calls=calls,
        descriptions=descriptions,
        revision=REVISION,
        queue="7309|RUNNING\n",
        accounting="",
        found=True,
        lose_ack=False,
        lost_ack=TimeoutError("accepted sbatch; acknowledgement lost"),
        error=None,
        cancel_error=None,
        before_ack=None,
        sid=None,
        input=None,
        journal=tmp_path / "run.sqlite",
    )
    monkeypatch.setattr(nwqsim.subprocess, "run", describe)
    monkeypatch.setattr(
        NWQSimSlurmBackend,
        "_launcher",
        lambda self: SlurmLauncher(
            self.profile, transport=scheduler, max_response_bytes=self.max_response_bytes
        ),
    )
    return state


def prepared(run, selected):
    return prepare(selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=23))


def packet(state, **changes):
    fields = dict(
        format="nwqlib.nwqsim/3",
        input_id=state.input["input_id"],
        status="completed",
        backend=state.backend.profile.backend,
        method=state.backend.profile.method,
        ranks=state.backend.profile.nodes * state.backend.profile.ranks_per_node,
        nwqsim_revision=REVISION,
        native_simulations=1,
        kind="counts",
        shots=4,
        counts={"0": 1, "1": 3},
    )
    fields.update(changes)
    path = Path(state.backend.profile.spool) / state.sid / "result.json"
    path.write_text(json.dumps(fields))
    return path


def test_lost_ack_reopen_binds_original_native_payload_and_publishes_once(supplied):
    """Recover an accepted sbatch by its original name and publish its packet once despite
    unknown scheduler status.
    """
    state, selected = (supplied, selection())
    state.lose_ack = True
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        handle = prepared(run, selected)

        def inspect_intent():
            (submission,) = run.trace.submissions
            assert submission.status == "intent" and submission.submission_id == state.sid
            assert run.trace.jobs == 1 and sum((event.shots for event in run.trace.events)) == 4

        state.before_ack = inspect_intent
        with pytest.raises(TimeoutError) as caught:
            submit_detached((handle,), run=run)
        assert caught.value is state.lost_ack
        record = handle.record
        assert state.input["seed"] == 23 and state.input["shots"] == 4
        assert state.input["backend"] == "CPU" and state.input["method"] == "SV"
        assert state.input["input_id"] == record.snapshot
        assert len({record.content_id, record.snapshot, state.sid, "7309"}) == 4
    state.queue, state.accounting = ("", "")
    with load_run(state.journal, backend=state.backend) as run:
        assert not refresh_submissions(run=run)
        (original,) = run.trace.submissions
        assert original.locator.job_id == "7309" and original.locator.instance == state.sid
        assert original.locator.account == "original-account" and original.provider_status == "unknown"
        assert original.native_simulations is None and (not original.results_consumed)
        packet(state)
        (chunk,) = refresh_submissions(run=run)
        assert chunk.job == "7309" and chunk.prepared_id == record.content_id
        assert chunk.observation == record.observation and chunk.returned_shots == 4
        histogram = chunk.histogram()
        assert histogram.width == 1 and dict(zip(histogram.index_list(), histogram.weights.tolist())) == {0: 1, 1: 3}
        assert run.collect(chunk)
        (finished,) = run.trace.submissions
        assert finished.provider_status == "unknown" and finished.native_simulations == 1
        assert not refresh_submissions(run=run)
    previous_calls = tuple(state.calls)
    for _ in range(2):
        with load_run(state.journal, backend=state.backend) as run:
            assert not refresh_submissions(run=run)
            assert len(run.observations.chunks) == 1 and (not run.collect(chunk))
            assert run.trace.jobs == 1 and sum((event.shots for event in run.trace.events)) == 4
            assert run.prepared_artifact(record.content_id) == record
    assert tuple(state.calls) == previous_calls
    assert sum((call[0] == "sbatch" for call in state.calls)) == 1 and len(state.descriptions) == 1


def test_known_job_missing_terminal_output_and_transport_error_remain_refreshable(supplied):
    state, selected = (supplied, selection())
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        original = submit_detached((prepared(run, selected),), run=run)
    state.queue, state.accounting = ("", "7309|COMPLETED|0:0\n")
    with load_run(state.journal, backend=state.backend) as run:
        assert not refresh_submissions(run=run)
        (status,) = run.trace.submissions
        assert status.status == "uncertain" and status.native_simulations is None
        assert "raw result is not visible" in status.failure and (not status.results_consumed)
        # COMPLETED before the file is visible on a shared file system is not a failure.
        assert not status.terminal_without_result
        failure = ConnectionError("original scheduler error")
        state.error = failure
        with pytest.raises(ConnectionError) as caught:
            refresh_submissions(run=run)
        assert caught.value is failure and run.trace.submissions[0].locator == original.locator
        packet(state)
        (chunk,) = refresh_submissions(run=run)
        assert chunk.job == "7309" and run.trace.submissions[0].native_simulations == 1
    assert not any(("--format=JobIDRaw,JobName%80" in call for call in state.calls))
    assert sum((call[0] == "sbatch" for call in state.calls)) == 1


@pytest.mark.parametrize("state_name, code", [("BOOT_FAIL", "0:0"), ("DEADLINE", "0:0"), ("FAILED", "137:0"), ("NODE_FAIL", "0:0"),
                                              ("OUT_OF_MEMORY", "0:125"), ("PREEMPTED", "0:0"), ("TIMEOUT", "0:15")])
def test_a_job_that_ended_without_output_stops_waiting_and_a_late_file_is_still_read(supplied, state_name, code):
    """Offline scheduler responses for the job states that the Slurm documentation lists as ending a job.

    A pending job carries no failure. Once the job has ended without a result file, waiting stops with the
    outcome uncertain and counted, in this process and after reopening, and nothing is submitted again. A
    file that does arrive for a clean exit is still read. No live site was involved.
    """
    from nwqlib.execution import RunFailed

    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue = "7309|PENDING\n"
        refresh_submissions(run=run)
        (pending,) = run.trace.submissions
        assert (pending.status, pending.failure, pending.terminal_without_result) == ("acknowledged", None, False)
        state.queue, state.accounting = "", f"7309|{state_name}|{code}\n"
        with pytest.raises(RunFailed) as stopped:
            run.wait(timeout=1, poll_interval=.01)
        assert (stopped.value.stage, stopped.value.status) == ("recovery", "uncertain")
        assert stopped.value.locator == pending.locator and stopped.value.attempt == run.trace.events[0].attempt
        assert run.exposure["uncertain"]["shots"] == 4
    with load_run(state.journal, backend=state.backend) as run:
        (saved,) = run.trace.submissions
        assert saved.terminal_without_result and saved.status == "uncertain"
        with pytest.raises(RunFailed):
            run.wait(timeout=1, poll_interval=.01)
        if code == "0:0":
            packet(state)
            (chunk,) = refresh_submissions(run=run)
            assert chunk.job == "7309"
            assert not run.trace.submissions[0].terminal_without_result
    assert sum(call[0] == "sbatch" for call in state.calls) == 1


def test_a_completed_job_whose_file_is_not_visible_yet_keeps_waiting(supplied):
    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue, state.accounting = "", "7309|COMPLETED|0:0\n"
        with pytest.raises(TimeoutError):
            run.wait(timeout=0)
        assert not run.trace.submissions[0].terminal_without_result


def test_a_job_the_scheduler_cancelled_without_output_is_a_confirmed_cancellation(supplied):
    from nwqlib.execution import RunFailed

    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue, state.accounting = "", "7309|CANCELLED by 1000|0:0\n"
        with pytest.raises(RunFailed) as stopped:
            run.wait(timeout=1, poll_interval=.01)
        assert (stopped.value.stage, stopped.value.status) == ("execute", "cancelled")
        assert not run.trace.submissions[0].terminal_without_result


@pytest.mark.parametrize("code", ["7:0"])
def test_nonzero_exit_or_signal_blocks_native_success_and_preserves_population(supplied, code):
    state, selected = (supplied, selection())
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        submit_detached((prepared(run, selected),), run=run)
        packet(state)
        state.queue, state.accounting = ("", f"7309|FAILED|{code}\n")
        assert not refresh_submissions(run=run)
        (status,) = run.trace.submissions
        assert status.status == "failed" and status.native_simulations == 1
        assert f"ExitCode={code}" in status.failure and f"ExitCode={code}" in status.provider_status
        assert (
            not run.observations.chunks
            and run.trace.jobs == 1
            and (sum((event.shots for event in run.trace.events)) == 4)
        )


@pytest.mark.parametrize(
    "changed",
    [
        {"input_id": "foreign-input"},
    ],
)
def test_foreign_or_malformed_completed_packet_cannot_publish_or_poison_retry(supplied, changed):
    state, selected = (supplied, selection())
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        original = submit_detached((prepared(run, selected),), run=run)
        packet(state, **changed)
        with pytest.raises(ValueError, match="prepared input|build or evolution population"):
            refresh_submissions(run=run)
        (status,) = run.trace.submissions
        assert status.locator == original.locator and status.native_simulations is None
        assert not status.results_consumed and (not run.observations.chunks)
        packet(state)
        (chunk,) = refresh_submissions(run=run)
        assert chunk.job == "7309" and run.trace.submissions[0].native_simulations == 1


def test_a_delivered_cancellation_request_is_not_sent_again_after_a_failed_read_and_reopening(supplied):
    state, selected = (supplied, selection())
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        submit_detached((prepared(run, selected),), run=run)
        run.cancel(reason="requested stop")
        failure = ConnectionError("lost cancellation acknowledgement")
        state.error = failure
        with pytest.raises(ConnectionError) as caught:
            refresh_submissions(run=run)
        assert caught.value is failure
    with load_run(state.journal, backend=state.backend) as run:
        assert not refresh_submissions(run=run)
        assert run.trace.submissions[0].cancel_requested == "requested stop"
        assert run.trace.jobs == 1 and sum((event.shots for event in run.trace.events)) == 4
        state.queue, state.accounting = ("", "7309|CANCELLED|0:15\n")
        packet(state, status="failed", error="native invocation cancelled", native_simulations=0)
        assert not refresh_submissions(run=run)
        (status,) = run.trace.submissions
        assert status.status == "cancelled" and status.native_simulations == 0
        assert "ExitCode=0:15" in status.failure and (not run.observations.chunks)
    assert sum((call[0] == "scancel" for call in state.calls)) == 1


def test_an_interrupted_cancellation_request_stays_recorded_and_is_not_sent_again(supplied):
    """Ctrl-C during scancel may have reached the scheduler, so the request stays recorded and the job is not read in that pass."""
    state, selected = (supplied, selection())
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        submit_detached((prepared(run, selected),), run=run)
        state.error = KeyboardInterrupt()
        with pytest.raises(KeyboardInterrupt):
            run.cancel(reason="stop")
        assert state.calls[-1][0] == "scancel"
        assert run.trace.submissions[0].cancel_requested == "stop"
    with load_run(state.journal, backend=state.backend) as run:
        assert not refresh_submissions(run=run)
        assert run.trace.submissions[0].cancel_requested == "stop"
    assert sum(call[0] == "scancel" for call in state.calls) == 1


def test_a_refused_cancellation_does_not_keep_a_finished_job_from_being_collected(supplied):
    """The job ended with its result file visible, and no refresh has read that yet. scancel refuses a finished job.

    Offline scheduler responses. The one request goes first, its error is raised after the status and the
    result are saved, and nothing is left to send again, so the result stays reachable, after reopening too.
    """
    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    state.cancel_error = RuntimeError("scancel: error: Invalid job id specified")
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue, state.accounting = "", "7309|COMPLETED|0:0\n"
        packet(state)
        with pytest.raises(RuntimeError) as refused:
            run.cancel(reason="stop")
        assert refused.value is state.cancel_error
        (saved,) = run.trace.submissions
        assert saved.status == "completed" and run.exposure["completed"]["shots"] == 4
        assert "Invalid job id" in saved.refresh_failure
        run.cancel(reason="stop")
    with load_run(state.journal, backend=state.backend) as run:
        result = run.wait(timeout=1, poll_interval=.01)
        assert len(result.data.observations.chunks) == 1
        assert result.data.trace.submissions[0].locator.job_id == "7309"
    assert sum(call[0] == "scancel" for call in state.calls) == sum(call[0] == "sbatch" for call in state.calls) == 1


def test_a_persistently_refused_cancellation_still_reads_the_job_until_its_file_arrives(supplied):
    """scancel keeps refusing. Each pass sends the request, reads the job, keeps what it read and raises the error.

    Offline scheduler responses. COMPLETED without a visible file keeps its documented meaning, that the
    file can still appear, and no timeout or hidden state stops the job from being read after reopening.
    """
    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    state.cancel_error = RuntimeError("scancel: slurmctld is unreachable")
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue = "7309|PENDING\n"
        with pytest.raises(RuntimeError):
            run.cancel(reason="stop")
        (saved,) = run.trace.submissions
        assert (saved.status, saved.provider_status, saved.cancel_requested) == ("acknowledged", "PENDING", None)
        assert "unreachable" in saved.refresh_failure
        state.queue, state.accounting = "", "7309|COMPLETED|0:0\n"
        with pytest.raises(RuntimeError):
            refresh_submissions(run=run)
        (saved,) = run.trace.submissions
        assert (saved.status, saved.terminal_without_result) == ("uncertain", False)
        assert saved.provider_status.startswith("COMPLETED") and "unreachable" in saved.refresh_failure
        assert run.exposure["completed"]["shots"] == 0
    packet(state)
    with load_run(state.journal, backend=state.backend) as run:
        with pytest.raises(RuntimeError):
            run.resume()
        (saved,) = run.trace.submissions
        assert saved.status == "completed" and run.exposure["completed"]["shots"] == 4
        assert len(run.wait(timeout=1, poll_interval=.01).data.observations.chunks) == 1
    assert sum(call[0] == "scancel" for call in state.calls) == 3
    assert sum(call[0] == "sbatch" for call in state.calls) == 1


def test_a_job_that_ended_without_output_takes_no_request_and_reports_its_recovery_failure(supplied):
    """A saved end without a result needs no request, and a refused one must not hide the recovery failure from wait.

    Offline scheduler responses. The one request goes out while the job is not yet known to have ended.
    A file that arrives later is still read after reopening, again with no request.
    """
    from nwqlib.execution import RunFailed

    state, selected = (supplied, selection())
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal)
    state.cancel_error = RuntimeError("scancel: error: Invalid job id specified")
    with chosen.run as run:
        nwqlib.submit(chosen)
        state.queue, state.accounting = "", "7309|DEADLINE|0:0\n"
        with pytest.raises(RuntimeError):
            run.cancel(reason="stop")
        assert run.trace.submissions[0].terminal_without_result
        for _ in range(2):
            with pytest.raises(RunFailed) as stopped:
                run.wait(timeout=1, poll_interval=.01)
            assert (stopped.value.stage, stopped.value.status) == ("recovery", "uncertain")
        assert run.exposure["uncertain"]["shots"] == 4
    packet(state)
    with load_run(state.journal, backend=state.backend) as run:
        result = run.wait(timeout=1, poll_interval=.01)
        assert len(result.data.observations.chunks) == 1
        assert not run.trace.submissions[0].terminal_without_result
    assert sum(call[0] == "scancel" for call in state.calls) == sum(call[0] == "sbatch" for call in state.calls) == 1


def test_a_request_beyond_the_data_limit_is_refused_before_its_submission(supplied):
    """The runner may write max_output_bytes = 2 MiB, which is over max_data_bytes = 1 MiB, so nothing may reach the scheduler.

    Offline scheduler responses. The refusal leaves no submission, event or uncertain shot, and the
    named limit leads past it.
    """
    from nwqlib.execution import ExecutionLimits

    state, selected = (supplied, selection())
    state.backend = state.backend.model_copy(update=dict(max_output_bytes=1 << 21))
    chosen = nwqlib.prepare(selected, backend=state.backend, directory=state.journal,
                            limits=ExecutionLimits(max_data_bytes=1 << 20))
    with chosen.run as run:
        with pytest.raises(ValueError, match=f"{1 << 21} bytes exceeds this Run's max_data_bytes={1 << 20}"):
            nwqlib.submit(chosen)
        assert not run.trace.submissions and not run.trace.events
        assert run.exposure["uncertain"]["shots"] == 0
        assert not any(call[0] == "sbatch" for call in state.calls)
        run.extend_limits(max_data_bytes=1 << 30)
        run.resume()
        (submission,) = run.trace.submissions
        assert submission.status == "acknowledged" and submission.locator.job_id == "7309"
    assert sum(call[0] == "sbatch" for call in state.calls) == 1


@pytest.mark.parametrize(("revision", "message"), [
    # The known unqualified source must fail on its own revision check.
    ("b35763d846e6512ed817d3f88ac8ce79a7e82a7e", "undefined behavior"),
])
def test_build_identity_and_known_unqualified_source_fail_before_lowering(supplied, monkeypatch, revision, message):
    import qiskit

    state, selected = (supplied, selection())
    state.revision = revision

    def forbidden(*args, **kwargs):
        raise AssertionError("lowering entered before build admission")

    monkeypatch.setattr(qiskit, "transpile", forbidden)
    with Run(
        selected, backend=state.backend, directory=Path(state.backend.profile.spool).parent / "extra-run"
    ) as run:
        with pytest.raises(ValueError, match=message):
            prepared(run, selected)
        assert run.trace.jobs == 0 and (not state.calls)


@pytest.mark.parametrize(
    "backend,method",
    [("MPI", "SV"), ("NVGPU", "SV")],
)
def test_selected_slurm_target_allocation_and_packet_survive_resume(supplied, backend, method):
    state, selected = (supplied, selection(3))
    gpu = backend in {"NVGPU", "AMDGPU"}
    profile = state.backend.profile.revise(
        backend=backend,
        method=method,
        ranks_per_node=2 if backend == "MPI" else 1,
        gpus_per_task=1 if gpu else None,
        gpu_bind="closest" if gpu else None,
    )
    state.backend = state.backend.revise(profile=profile)
    with Run(selected, backend=state.backend, directory=state.journal) as run:
        handle = prepared(run, selected)
        submit_detached((handle,), run=run)
        assert state.input["backend"] == backend and state.input["method"] == method
        assert state.input["ranks"] == profile.ranks_per_node
        script = (Path(profile.spool) / state.sid / "job.sh").read_text()
        assert f"--ntasks={profile.ranks_per_node}" in script
        assert ("--gpus-per-task=1" in script) is gpu
    packet(state, counts={"000": 1, "001": 3})  # grouped readout keeps all three classical bits
    state.queue, state.accounting = ("", "7309|COMPLETED|0:0\n")
    with load_run(state.journal, backend=state.backend) as run:
        (chunk,) = refresh_submissions(run=run)
        assert chunk.prepared_id == handle.record.content_id and chunk.job == "7309"
        assert chunk.returned_shots == 4 and run.trace.submissions[0].native_simulations == 1
        assert sum((event.shots for event in run.trace.events)) == 4
        assert run.collect(chunk)
    with load_run(state.journal, backend=state.backend) as run:
        assert not refresh_submissions(run=run)
    assert sum((call[0] == "sbatch" for call in state.calls)) == 1


def selection(q=1):
    problem = Expectation(
        state=ingest_occupation("0" * q, num_qubits=q),
        observable=ingest_pauli((("I" * q, 1.0), ("I" * (q - 1) + "Z", 1.0)), num_qubits=q),
    )
    return nwqlib.plan(problem, method=ExpectationMethod(), shots=4, seed=23)
