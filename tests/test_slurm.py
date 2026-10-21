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
