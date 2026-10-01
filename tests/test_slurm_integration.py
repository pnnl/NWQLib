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


@pytest.mark.parametrize("code", ["7:0", "0:9"])
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
        {"nwqsim_revision": "foreign-build"},
        {"backend": "NVGPU"},
        {"method": "DM"},
        {"native_simulations": 2},
        {"native_simulations": True},
        {"native_simulations": 1.0},
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


def test_cancel_original_job_is_sent_once_across_interrupted_request(supplied):
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


@pytest.mark.parametrize(("revision", "message"), [
    ("foreign-build", "build identity"),
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


def test_cpu_single_rank_configuration_roundtrip_and_profile_admission(supplied):
    backend = supplied.backend
    assert NWQSimSlurmBackend.model_validate_json(backend.model_dump_json()) == backend
    with pytest.raises(ValueError, match="only the MPI backend"):
        backend.revise(profile=backend.profile.revise(nodes=2))
    with pytest.raises(ValueError, match="cannot consume a GPU"):
        backend.revise(profile=backend.profile.revise(gpus_per_task=1, gpu_bind="closest"))
    with pytest.raises(ValueError, match="time window"):
        backend.revise(accounting_until=backend.accounting_since)


def test_corrected_build_keeps_finite_u_phase_domain(supplied):
    import sys
    from qiskit import QuantumCircuit
    from nwqlib.core.planning import ObservationSpec

    circuit = QuantumCircuit(1, 1)
    circuit.u(0, sys.float_info.max, sys.float_info.max, 0)
    circuit.measure(0, 0)
    with Run(
        selection(),
        backend=supplied.backend,
        directory=Path(supplied.backend.profile.spool).parent / "extra-run",
    ) as run:
        result = supplied.backend.prepare(
            circuit,
            observation=ObservationSpec(kind="counts", shots=4),
            runtime=RuntimeOptions(seed=23),
            position=len(circuit.data),
            source_definitions=(),
            run=run,
            snapshot="finite-u-preservation",
        )
    request = json.loads(result.payload)
    assert request["gates"][0]["params"] == [0, sys.float_info.max, sys.float_info.max]
    assert result.target.version == REVISION and (not supplied.calls)


@pytest.mark.parametrize(
    "backend,method",
    [("CPU", "DM"), ("MPI", "SV"), ("NVGPU", "SV"), ("NVGPU", "DM"), ("AMDGPU", "SV"), ("AMDGPU", "DM")],
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


def test_selected_optimization_level_reaches_the_native_preparation(supplied):
    # Level 1 merges the two Rz rotations and marks the receipt with the
    # exclusion "optimization_level", as the NWQ-Sim adapter does.
    from qiskit import QuantumCircuit
    from nwqlib.core.planning import ObservationSpec

    backend = supplied.backend.revise(optimization_level=1)
    circuit = QuantumCircuit(1, 1)
    circuit.rz(0.1, 0)
    circuit.rz(0.2, 0)
    circuit.measure(0, 0)
    with Run(selection(), backend=backend, directory=Path(backend.profile.spool).parent / "level-run") as run:
        result = backend.prepare(circuit, observation=ObservationSpec(kind="counts", shots=4),
                                 runtime=RuntimeOptions(seed=23), position=len(circuit.data),
                                 source_definitions=(), run=run, snapshot="level")
    assert len(json.loads(result.payload)["gates"]) == 1
    assert "optimization_level" in result.probability_window_exclusions


@pytest.mark.parametrize("response_bytes", [1, 5, 4096])
def test_trajectory_admission_prices_each_point_at_a_ten_digit_job_id(supplied, response_bytes):
    """A Slurm trajectory reserves ten job-ID digits per point before sbatch, for any response cap.

    Without a forecast the early condition is ``32*K + 168 <= M`` for K Pauli
    points: ``NWQSimSlurmBackend.native_job_id_length`` returns 10, the
    largest decimal length of a positive uint32 job ID that
    ``SlurmLauncher._locator`` admits, so each point's job and empty values
    envelope costs at most ``22 + 10`` bytes, and the fresh outcome
    transaction grows by at most 168 bytes
    (``_prepared_execution.detached_completion_growth``). The bound does not
    depend on ``max_response_bytes``. One byte below it the Run refuses before
    any scheduler call. At it the submission reaches sbatch when the cap
    covers the 22-byte acknowledgement floor, and otherwise launch refuses
    before sbatch.
    """
    from test_observation_schedule import pauli, three_call_plan, trajectory

    from nwqlib.execution import ExecutionLimits

    plan = three_call_plan(trajectory(pauli("a", 1, "ZZ", "XY"), pauli("b", 2, "XX"), pauli("end", None, "ZI")),
                           state=(1., 2j, -3., 4.))
    required = 32 * 3 + 168
    backend = supplied.backend.revise(max_response_bytes=response_bytes)
    spool = Path(backend.profile.spool)

    def launch(allowance):
        with Run(plan, backend=backend, directory=spool.parent / f"run-{allowance}",
                 limits=ExecutionLimits(max_completion_metadata_bytes=allowance)) as run:
            handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
            submit_detached((handle,), run=run)

    with pytest.raises(ValueError, match=f"need at least {required} bytes, more than "
                                         f"max_completion_metadata_bytes={required - 1}, before native work"):
        launch(required - 1)
    assert supplied.calls == []
    if response_bytes < 22:
        with pytest.raises(ValueError, match="requires max_response_bytes >= 22"):
            launch(required)
        assert supplied.calls == []
    else:
        launch(required)
        assert [call[0] for call in supplied.calls] == ["sbatch"]


def test_configuration_precheck_refuses_before_the_submission_intent(supplied):
    """A response cap below the acknowledgement floor is refused before the intent commits.

    ``SlurmLauncher.admit`` runs from ``NWQSimSlurmBackend.admit_batch``, which
    ``submit_detached`` calls before ``Run._begin_submission``. The refused Run
    therefore holds no submission, event, exposure or output reservation, so a
    refresh has no job to search for, and a Run with the corrected
    configuration submits the same Plan.
    """
    state, selected = (supplied, selection())
    backend = state.backend.revise(max_response_bytes=21)
    with Run(selected, backend=backend, directory=state.journal) as run:
        handle = prepared(run, selected)
        with pytest.raises(ValueError, match="requires max_response_bytes >= 22"):
            submit_detached((handle,), run=run)
        assert run.trace.submissions == () and run.trace.events == () and run.trace.jobs == 0
        assert all(not any(counts.values()) for counts in run.exposure.values())
        assert run._state["pending_output"] == {} and run.trace.data_bytes_reserved == 0
        assert not refresh_submissions(run=run)
    assert state.calls == [] and not (Path(backend.profile.spool)).exists()
    with Run(selected, backend=backend.revise(max_response_bytes=4096),
             directory=state.journal.parent / "corrected.sqlite") as run:
        submit_detached((prepared(run, selected),), run=run)
        assert run.trace.jobs == 1
    assert [call[0] for call in state.calls] == ["sbatch"]


@pytest.mark.parametrize("response_bytes, reply", [
    (1, "4294967295;perlmutter\n"), (21, "4294967295;perlmutter\n"),
    (22, "4294967295;perlmutter\n"), (22, "4294967295;perlmutter\r\n")])
def test_launch_refuses_a_response_cap_below_the_longest_acknowledgement(supplied, response_bytes, reply):
    """Launch needs ``12 + len(cluster)`` response bytes before any write (``SlurmLauncher.launch``).

    The longest canonical acknowledgement of a supported job ID from
    ``perlmutter`` is ``4294967295;perlmutter\\n``, 22 bytes. A smaller cap is
    refused with no spool directory and no scheduler call, so no job is
    submitted that the launcher could not then track. At 22 bytes the job
    is tracked. The floor assumes one LF, so a CRLF reply of 23 bytes exceeds
    the 22-byte cap after the one sbatch call (``SlurmLauncher._command``),
    and the spool stays in place for reconciliation.
    """
    from test_nexus_backend import IO

    calls = []

    def scheduler(argv, *, max_output_bytes):
        calls.append(argv)
        return reply

    profile = supplied.backend.profile
    launcher = SlurmLauncher(profile, transport=scheduler, max_response_bytes=response_bytes)
    sid = "7f6e5d4c-3b2a-4190-8a7b-6c5d4e3f2a1b"
    if response_bytes < 22:
        with pytest.raises(ValueError, match=f"requires max_response_bytes >= 22 .*got {response_bytes}"):
            launcher.launch(b"{}", submission_id=sid, max_input_bytes=100, run=IO())
        assert calls == [] and not Path(profile.spool).exists()
    elif reply.endswith("\r\n"):
        with pytest.raises(ValueError, match="exceeded its admitted text response"):
            launcher.launch(b"{}", submission_id=sid, max_input_bytes=100, run=IO())
        assert len(calls) == 1 and (Path(profile.spool) / sid / "job.sh").exists()
    else:
        locator = launcher.launch(b"{}", submission_id=sid, max_input_bytes=100, run=IO())
        assert locator.job_id == "4294967295" and len(calls) == 1
