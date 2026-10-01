"""Offline scheduler protocol checks; no scheduler or simulation is invoked."""

from uuid import uuid4

import pytest
from test_nexus_backend import IO

from nwqlib.backends.slurm import SlurmLauncher, SlurmProfile


def profile(tmp_path, **changes):
    return SlurmProfile(
        cluster="perlmutter",
        account="allocation",
        qos="regular",
        walltime="00:05:00",
        nodes=1,
        ranks_per_node=1,
        threads=2,
        executable="/shared/a runner; literal",
        spool=str(tmp_path),
        build_identity="user-verified revision",
        **changes,
    )


class Transport:
    def __init__(self, *responses):
        self.responses = iter(responses)
        self.calls = []

    def __call__(self, argv, *, max_output_bytes):
        self.calls.append(argv)
        response = next(self.responses)
        if isinstance(response, Exception):
            raise response
        return response


run = IO()


def test_lost_acknowledgement_reconciles_without_second_launch(tmp_path):
    sid = str(uuid4())
    error = TimeoutError("accepted but acknowledgement lost")
    transport = Transport(error, f"73|nwqlib-{sid}\n", "73|RUNNING\n", "")
    launcher = SlurmLauncher(profile(tmp_path), transport=transport, max_response_bytes=1024)
    with pytest.raises(TimeoutError) as caught:
        launcher.launch(b"original payload", submission_id=sid, max_input_bytes=100, run=run)
    assert caught.value is error
    with pytest.raises(FileExistsError):
        launcher.launch(b"original payload", submission_id=sid, max_input_bytes=100, run=run)
    locator = launcher.reconcile(sid, since="2026-09-10T10:00:00", until="2026-09-10T11:00:00", run=run)
    assert (locator.job_id, locator.instance, locator.cluster) == ("73", sid, "perlmutter")
    assert launcher.refresh(locator, run=run).state == "RUNNING"
    launcher.cancel(locator, run=run)
    assert [call[0] for call in transport.calls] == ["sbatch", "sacct", "squeue", "scancel"]
    assert transport.calls[-1] == ("scancel", "--clusters=perlmutter", "73")
    assert (tmp_path / sid / "input.json").read_bytes() == b"original payload"


@pytest.mark.parametrize(
    "accounting,state,failure",
    [
        ("", "unknown", "accounting may be delayed"),
        ("73|COMPLETED|0:0\n", "COMPLETED", "raw result is not visible"),
        ("73|FAILED|1:0\n", "FAILED", "nonzero scheduler exit"),
        ("73_1|COMPLETED|0:0\n73_2|RUNNING|0:0\n", "uncertain", "partial/array"),
    ],
)
def test_scheduler_state_does_not_fabricate_publication(tmp_path, accounting, state, failure):
    launcher = SlurmLauncher(profile(tmp_path), transport=Transport("", accounting), max_response_bytes=1024)
    locator = launcher._locator("73", str(uuid4()))
    status = launcher.refresh(locator, run=run)
    assert status.state == state and failure in status.failure
    assert status.result_path is None


def test_same_job_refresh_preserves_raw_result_and_original_error(tmp_path):
    sid = str(uuid4())
    (tmp_path / sid).mkdir()
    raw = tmp_path / sid / "result.json"
    raw.write_text('{"status":"failed"}')
    error = ConnectionError("scheduler unavailable")
    transport = Transport(error, "", "73|COMPLETED|0:0\n", "", "73|COMPLETED|0:0\n")
    launcher = SlurmLauncher(profile(tmp_path), transport=transport, max_response_bytes=1024)
    locator = launcher._locator("73", sid)
    with pytest.raises(ConnectionError) as caught:
        launcher.refresh(locator, run=run)
    assert caught.value is error
    first = launcher.refresh(locator, run=run)
    assert launcher.refresh(locator, run=run) == first
    assert first.result_path == raw  # Raw failure packet is not decoded or published here.
    assert all("--jobs=73" in call for call in transport.calls)
    assert raw.read_text() == '{"status":"failed"}'


def test_script_quotes_paths_and_rejects_directive_injection(tmp_path):
    import shlex

    sid = str(uuid4())
    transport = Transport("73\n", "")
    launcher = SlurmLauncher(profile(tmp_path, modules=("compiler/reviewed",)), transport=transport,
                             max_response_bytes=1024)
    launcher.launch(b"{}", submission_id=sid, max_input_bytes=100, run=run)
    assert launcher.reconcile(sid, since="2026-09-10T10:00:00", until="2026-09-10T11:00:00", run=run) is None
    directory = tmp_path / sid
    assert transport.calls[0][-1] == str(directory / "job.sh")
    # Parse the submitted script as the shell does. Comment lines other than
    # #SBATCH directives are dropped.
    lines = [words for line in (directory / "job.sh").read_text().splitlines()
             if (words := shlex.split(line, comments=not line.startswith("#SBATCH")))]
    directives = [words[1] for words in lines if words[0] == "#SBATCH"]
    (name,) = [argument.removeprefix("--name=") for argument in transport.calls[1] if argument.startswith("--name=")]
    assert "--job-name=" + name in directives
    for option in ("--account=allocation", "--clusters=perlmutter", "--qos=regular", "--time=00:05:00",
                   "--nodes=1", "--ntasks-per-node=1", "--cpus-per-task=2", "--no-requeue"):
        assert option in directives
    assert ["module", "load", "compiler/reviewed"] in lines and ["export", "OMP_NUM_THREADS=2"] in lines
    (command,) = [words for words in lines if words[0] == "exec"]
    assert command[:2] == ["exec", "srun"]
    assert {"--nodes=1", "--ntasks=1", "--cpus-per-task=2", "--cpu-bind=cores"} <= set(command[2:-4])
    # The quoted executable path stays one argument, followed by input, result and input cap.
    assert command[-4:] == ["/shared/a runner; literal", str(directory / "input.json"),
                            str(directory / "result.json"), "100"]
    with pytest.raises(ValueError):
        profile(tmp_path, partition="regular\n#SBATCH --array=1-20")
    with pytest.raises(ValueError):
        profile(tmp_path, gpu_bind="closest")


def test_multiple_ranks_cannot_launch_independent_simulations(tmp_path):
    selected = profile(tmp_path).revise(nodes=2, ranks_per_node=4)
    assert "--ntasks=8" in selected.script(str(uuid4()), max_input_bytes=100)
    transport = Transport()
    launcher = SlurmLauncher(selected, transport=transport, max_response_bytes=1024)
    with pytest.raises(ValueError, match="only the MPI backend"):
        launcher.launch(b"{}", submission_id=str(uuid4()), max_input_bytes=100, run=run)
    assert transport.calls == [] and list(tmp_path.iterdir()) == []


def test_ambiguous_accounting_never_selects_first_job(tmp_path):
    sid = str(uuid4())
    transport = Transport(f"73|nwqlib-{sid}\n74|nwqlib-{sid}\n")
    launcher = SlurmLauncher(profile(tmp_path), transport=transport, max_response_bytes=1024)
    assert launcher.reconcile(sid, since="2026-09-10T10:00:00", until="2026-09-10T11:00:00", run=run) is None
    with pytest.raises(ValueError):
        launcher.cancel(launcher._locator("73", sid).revise(cluster="other"), run=run)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("ack", ["73;other\n", "73_2;perlmutter\n", "73;perlmutter;unexpected\n"])
def test_unusable_ack_keeps_intent_without_retry(tmp_path, ack):
    sid = str(uuid4())
    transport = Transport(ack)
    launcher = SlurmLauncher(profile(tmp_path), transport=transport, max_response_bytes=1024)
    with pytest.raises(ValueError):
        launcher.launch(b"{}", submission_id=sid, max_input_bytes=100, run=run)
    assert (tmp_path / sid / "job.sh").is_file()
    assert len(transport.calls) == 1


def test_reservation_failure_prevents_scheduler_contact(tmp_path):
    transport = Transport()
    launcher = SlurmLauncher(profile(tmp_path), transport=transport, max_response_bytes=1024)

    class Exhausted:
        def check_data(self, size):
            raise ValueError("I/O allowance exhausted")

    with pytest.raises(ValueError, match="allowance"):
        launcher.refresh(launcher._locator("73", str(uuid4())), run=Exhausted())
    assert transport.calls == []


def test_local_transport_separates_streams_and_preserves_cli_error(monkeypatch):
    import subprocess
    from nwqlib.backends.slurm import SlurmTransport

    monkeypatch.setenv("SBATCH_ARRAY_INX", "1-20")
    transport = SlurmTransport(timeout_seconds=2)
    assert (
        transport(
            ("/bin/sh", "-c", 'test -z "$SBATCH_ARRAY_INX"; printf 73; printf warning >&2'),
            max_output_bytes=64,
        )
        == "73"
    )
    with pytest.raises(subprocess.CalledProcessError) as caught:
        transport(("/bin/sh", "-c", "printf out; printf failure >&2; exit 7"), max_output_bytes=64)
    assert (caught.value.returncode, caught.value.output, caught.value.stderr) == (7, b"out", b"failure")


def test_local_transport_combined_cap_and_deadline():
    import subprocess
    from nwqlib.backends.slurm import SlurmTransport

    transport = SlurmTransport(timeout_seconds=2)
    with pytest.raises(OverflowError, match="combined byte cap"):
        transport(("/bin/sh", "-c", "printf 12345; printf 67890 >&2"), max_output_bytes=8)
    with pytest.raises(subprocess.TimeoutExpired):
        SlurmTransport(timeout_seconds=0.05)(("/bin/sh", "-c", "exec sleep 3"), max_output_bytes=64)


def test_missing_scheduler_command_preserves_original_cause_and_names_cli(monkeypatch):
    import subprocess
    from nwqlib.backends.slurm import SlurmTransport

    error = FileNotFoundError(2, 'missing executable', 'sbatch')
    def missing(*args, **kwargs):
        raise error
    monkeypatch.setattr(subprocess, 'Popen', missing)
    with pytest.raises(FileNotFoundError) as caught:
        SlurmTransport(timeout_seconds=1)(('sbatch', '--parsable'), max_output_bytes=64)
    assert caught.value is error and caught.value.filename == 'sbatch'
    assert "Slurm command 'sbatch'" in caught.value.__notes__[0]


def test_success_storage_bounds_price_each_transaction_by_the_response_size():
    """The per-transaction bounds of ``slurm_success_storage_bounds`` for C-byte scheduler responses.

    With ``L`` the JSON size of a locator with a one-digit job ID,
    acknowledgement grows by ``L + 71`` to ``L + C + 70``, completed status
    by ``3 + 13`` to ``2 + 6*C + 11 + 20``, and the detached outcome by
    ``(161, 168)``, or ``(92, 99)`` with an already revised event
    (``_prepared_execution.detached_completion_growth``). C = 1 and C = 5.
    """
    from nwqlib._run_journal import _json_bound
    from nwqlib.backends.slurm import slurm_success_storage_bounds
    from nwqlib.execution import JobLocator

    locator = _json_bound(JobLocator(provider="slurm", job_id="7", cluster="perlmutter", account="allocation",
                                     instance=str(uuid4())), 1 << 40)
    for revised, outcomes in ((False, (161, 168)), (True, (92, 99))):
        assert slurm_success_storage_bounds(cluster="perlmutter", account="allocation", response_bytes=1,
                                            max_bytes=1 << 40, event_already_revised=revised) == {
            "ack": (locator + 71, locator + 71),
            "status": (16, 39),
            "outcomes": outcomes,
            "intent_to_consumed": (locator + 71 + 16 + outcomes[0], locator + 71 + 39 + outcomes[1]),
        }
    # Each further response byte adds one job-ID digit and six status bytes.
    wider = slurm_success_storage_bounds(cluster="perlmutter", account="allocation", response_bytes=5,
                                         max_bytes=1 << 40)
    assert wider["ack"] == (locator + 71, locator + 75) and wider["status"] == (16, 63)
    with pytest.raises(ValueError, match="response_bytes must be a positive integer"):
        slurm_success_storage_bounds(cluster="perlmutter", account="allocation", response_bytes=0, max_bytes=1 << 40)


@pytest.mark.parametrize("job_id", ["12345678901", "9" * 4096, "4294967296", "12345_67", "123+0"])
def test_locator_admits_only_a_single_positive_uint32_job_id(tmp_path, job_id):
    """A Slurm job ID is a positive uint32 of at most ten decimal digits, or it is refused.

    ``NWQSimSlurmBackend.native_job_id_length`` prices trajectory job text at
    that ten-digit bound, so a longer ID would exceed the admitted envelope,
    and a larger one lies outside Slurm's uint32 job-ID domain. An array
    task or heterogeneous component names one part of a job collection. The
    largest ten-digit uint32 and the smallest ten-digit ID stay accepted.
    """
    sid = str(uuid4())
    launcher = SlurmLauncher(profile(tmp_path), transport=Transport(), max_response_bytes=1024)
    with pytest.raises(ValueError, match=r"single job ID in 1\.\.4294967295"):
        launcher._locator(job_id, sid)
    for edge in ("4294967295", "1000000000"):
        assert launcher._locator(edge, sid).job_id == edge
