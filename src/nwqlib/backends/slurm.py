"""Selected native NWQ-Sim target and bounded Slurm scheduler control.

The backend reuses native preparation/decoding and common journal publication.
The launcher owns only scheduler/file control, with injectable bounded transport.
"""

from dataclasses import dataclass
from datetime import datetime
from math import isfinite
import os
from pathlib import Path
import re
import selectors
import shlex
import subprocess
from time import monotonic
from typing import Annotated, ClassVar, Literal
from uuid import UUID

from pydantic import Field, model_validator

from nwqlib.core.records import PositiveInt, Real, Record, Text
from nwqlib.execution import JobLocator
from nwqlib.operators.access import Count


class SlurmTransport:
    """One local CLI invocation with a combined output cap and finite deadline.

    Killing a timed-out sbatch client does not cancel a possibly accepted job.
    The caller keeps the original submission intent and reconciles it.
    """

    def __init__(self, *, timeout_seconds):
        """Set the deadline, in seconds, that applies to each CLI call."""
        if (isinstance(timeout_seconds, bool) or not isinstance(timeout_seconds, (int, float))
                or not isfinite(timeout_seconds) or timeout_seconds <= 0):
            raise ValueError("Slurm CLI deadline must be finite and positive")
        self.timeout_seconds = timeout_seconds

    def __call__(self, argv, *, max_output_bytes):
        """Run one scheduler CLI command and return its stdout text.

        Inherited ``SBATCH_*`` variables would override the rendered script's
        ``#SBATCH`` directives, and ``SRUN_*`` variables would add srun options
        that the script leaves unset, so both are removed. Both pipes
        are drained together under one byte cap and one deadline, so a noisy or
        hung client cannot block or grow without bound. On any failure only
        this local client is killed and reaped.
        """
        if type(max_output_bytes) is not int or max_output_bytes <= 0:
            raise ValueError("Slurm CLI output cap must be positive")
        environment = {key: value for key, value in os.environ.items()
                       if not key.startswith(("SBATCH_", "SRUN_"))}
        buffers = {"stdout": bytearray(), "stderr": bytearray()}
        deadline = monotonic() + self.timeout_seconds
        try:
            process = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                       stderr=subprocess.PIPE, env=environment)
        except FileNotFoundError as error:
            error.add_note(f"Slurm command {argv[0]!r} is unavailable; configure its executable path or run where the scheduler CLI is installed.")
            raise
        try:
            with selectors.DefaultSelector() as selector:
                selector.register(process.stdout, selectors.EVENT_READ, "stdout")
                selector.register(process.stderr, selectors.EVENT_READ, "stderr")
                total = 0
                while selector.get_map():
                    remaining = deadline - monotonic()
                    if remaining <= 0:
                        raise subprocess.TimeoutExpired(argv, self.timeout_seconds,
                            output=bytes(buffers["stdout"]), stderr=bytes(buffers["stderr"]))
                    for key, _ in selector.select(remaining):
                        # Read at most 64 KiB (an untuned pipe read size), and
                        # never more than one byte past the cap, which is
                        # enough to detect an overflow.
                        chunk = os.read(key.fileobj.fileno(), min(65536, max_output_bytes - total + 1))
                        if not chunk:
                            selector.unregister(key.fileobj)
                            continue
                        total += len(chunk)
                        if total > max_output_bytes:
                            raise OverflowError("Slurm CLI stdout/stderr exceeded the combined byte cap")
                        buffers[key.data].extend(chunk)
                process.wait(timeout=max(0, deadline - monotonic()))
            stdout, stderr = bytes(buffers["stdout"]), bytes(buffers["stderr"])
            if process.returncode:
                raise subprocess.CalledProcessError(process.returncode, argv, output=stdout, stderr=stderr)
            return stdout.decode("utf-8")
        except BaseException as error:
            if isinstance(error, subprocess.TimeoutExpired):
                error.output = bytes(buffers["stdout"])
                error.stderr = bytes(buffers["stderr"])
            # Reap only this local CLI, never issue scheduler cancel.
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
            process.stderr.close()


def _token(value):
    """Admit one literal option token, so a profile value cannot inject another directive."""
    if not re.fullmatch(r"[A-Za-z0-9_./,:+=@%-]+", value):
        raise ValueError("Slurm option must be one literal token without control characters")
    return value


def _submission(value):
    """Admit only a canonical UUID, since it names the spool directory and the job."""
    if str(UUID(value)) != value:
        raise ValueError("Slurm submission requires its original canonical UUID")
    return value


class SlurmProfile(Record):
    """Slurm site settings for an NWQ-Sim runner: allocation, runner, binding and shared pending-job directory.

    Build it with keyword arguments, for example
    `SlurmProfile(cluster=..., account=..., walltime="00:05:00", nodes=1, ranks_per_node=1, threads=1, executable=..., build_identity=..., spool=...)`,
    and pass it as `profile=` to [`NWQSimSlurmBackend`][nwqlib.backends.slurm.NWQSimSlurmBackend].
    `cluster`, `account`, `walltime`, `nodes`, `ranks_per_node`, `threads`,
    `executable`, `build_identity` and `spool` are required. Node, rank and thread
    counts describe the requested resources, not measured use. Every option value
    must be one literal token, so that it cannot end its `#SBATCH` line and add
    another directive. The [Slurm guide](../slurm.md) gives Perlmutter and
    Frontier profiles.

    Attributes:
        cluster: Required. One cluster name, letters, digits, `_` and `-` only.
            `"all"` and comma lists are rejected, because job lookups name exactly
            one cluster.
        account: Required. Slurm account to charge.
        partition: Default `None`. Slurm partition.
        qos: Default `None`. Slurm quality of service.
        constraint: Default `None`. Slurm node constraint, such as `"gpu"`.
        walltime: Required. Positive time limit in Slurm's
            `[days-]hours:minutes:seconds` form.
        nodes: Required. Positive. Number of nodes.
        ranks_per_node: Required. Positive. MPI ranks per node.
        threads: Required. Positive. Threads per rank.
        gpus_per_task: Default `None`. Positive. GPUs per task, given together
            with `gpu_bind`. The `"NVGPU"` and `"AMDGPU"` backends require
            exactly 1, and the `"CPU"` and `"MPI"` backends require `None`.
        gpu_bind: Default `None`. Slurm GPU binding, given together with
            `gpus_per_task`.
        cpu_bind: Default `"cores"`. Slurm CPU binding.
        modules: Default `()`. Environment modules to load before the run.
        executable: Required. Absolute path of the runner on the compute nodes.
        build_identity: Required. Source commit of the runner's NWQ-Sim build. It
            records which build ran and does not show that a GPU or distributed
            build works.
        spool: Required. Absolute path of a directory shared with the compute
            nodes.
        backend: Default `"CPU"`. NWQ-Sim backend, with the values and
            restrictions of [`NWQSimBackend`][nwqlib.backends.nwqsim.NWQSimBackend].
        method: Default `"SV"`. `"SV"` or `"DM"`, as in `NWQSimBackend`.

    Raises:
        ValueError: If the `backend` and `method` combination is not supported,
            `cluster` is not one name, an option value is not one token,
            `walltime` is malformed or zero, `gpus_per_task` and `gpu_bind` are
            not given together, or a path is not absolute.
    """

    cluster: Text
    account: Text
    partition: Text | None = None
    qos: Text | None = None
    constraint: Text | None = None
    walltime: Text
    nodes: PositiveInt
    ranks_per_node: PositiveInt
    threads: PositiveInt
    gpus_per_task: PositiveInt | None = None
    gpu_bind: Text | None = None
    cpu_bind: Text = "cores"
    modules: tuple[Text, ...] = ()
    executable: Text
    build_identity: Text
    spool: Text
    backend: Literal["CPU", "MPI", "NVGPU", "AMDGPU", "NVGPU_MPI"] = "CPU"
    method: Literal["SV", "DM"] = "SV"

    @model_validator(mode="after")
    def _validate(self):
        """Admit only values that render into unambiguous batch directives.

        - The backend/method pair must be a route that ``nwqsim._route``
          admits. ``admit_execution`` checks the rank count separately.
        - The cluster is one explicit name, because the locator and every
          ``--clusters`` query name exactly one cluster. Slurm reads ``all``
          or a comma list as several clusters.
        - Every option value is one literal token (``_token``), so it cannot
          end its ``#SBATCH`` line and inject another directive.
        - The walltime has Slurm's ``[days-]hours:minutes:seconds`` form and is
          positive.
        - A GPU count and a GPU binding are given together or not at all.
        - The runner and spool are absolute paths without control characters,
          because they are written into the script and resolved on compute
          nodes whose working directory differs.
        """
        from nwqlib.backends.nwqsim import _route
        _route(self.backend, self.method, 1)
        if not re.fullmatch(r"[A-Za-z0-9_-]+", self.cluster) or self.cluster == "all":
            raise ValueError("profile requires one explicit cluster, not a cluster list")
        for value in (self.cluster, self.account, self.partition, self.qos, self.constraint,
                      self.cpu_bind, self.gpu_bind, *self.modules):
            if value is not None:
                _token(value)
        if not re.fullmatch(r"(?:[0-9]+-)?[0-9]+:[0-5][0-9]:[0-5][0-9]", self.walltime):
            raise ValueError("walltime requires [days-]hours:minutes:seconds")
        if not any(int(part) for part in re.split("[-:]", self.walltime)):
            raise ValueError("walltime must be finite and positive")
        if (self.gpus_per_task is None) != (self.gpu_bind is None):
            raise ValueError("GPU allocation requires explicit GPU binding and vice versa")
        for path in (self.executable, self.spool):
            if not Path(path).is_absolute() or any(ord(char) < 32 for char in path):
                raise ValueError("runner and shared spool require absolute paths without control characters")
        return self

    def admit_execution(self):
        """Check the total rank count and GPU request against the selected native route.

        GPU routes are single-process with exactly one GPU per task. CPU and MPI
        routes must not hold a GPU allocation they cannot use.
        """
        from nwqlib.backends.nwqsim import _route
        _route(self.backend, self.method, self.nodes * self.ranks_per_node)
        if self.backend in {"NVGPU", "AMDGPU"}:
            if self.gpus_per_task != 1:
                raise ValueError("single-process GPU execution requires exactly one GPU per task")
        elif self.gpus_per_task is not None:
            raise ValueError("CPU/MPI execution cannot consume a GPU allocation")

    def script(self, submission_id, *, max_input_bytes):
        """Return the Slurm batch script for one submission, without running it.

        Use it to inspect what a submission would send to `sbatch`. Rendering a
        script does not test the site. The job name `nwqlib-<submission UUID>` is
        what a lost submission acknowledgement is searched for. `--no-requeue` keeps
        Slurm from running the same request a second time after a node failure or
        preemption, and `srun --kill-on-bad-exit` stops all ranks when one fails.
        Batch stdout and stderr go to `/dev/null`, because only the runner's result
        file is read.

        Args:
            submission_id (str): Submission UUID that names the job and its
                subdirectory of `spool`.
            max_input_bytes (int): Positive limit in bytes on the request file, passed
                to the runner.

        Returns:
            script (str): The script text, ending with a newline.

        Raises:
            ValueError: If `max_input_bytes` is not a positive integer.
            ValueError: If `submission_id` is not a canonical UUID.
        """
        directory = Path(self.spool) / _submission(submission_id)
        if type(max_input_bytes) is not int or max_input_bytes <= 0:
            raise ValueError("max_input_bytes must be a positive integer")
        options = dict(account=self.account, clusters=self.cluster, partition=self.partition,
                       qos=self.qos, constraint=self.constraint, time=self.walltime,
                       nodes=self.nodes, **{"ntasks-per-node": self.ranks_per_node,
                       "cpus-per-task": self.threads, "gpus-per-task": self.gpus_per_task})
        lines = ["#!/bin/bash"] + [f"#SBATCH --{key}={value}" for key, value in options.items() if value is not None]
        lines += [f"#SBATCH --job-name=nwqlib-{submission_id}", "#SBATCH --output=/dev/null",
                  "#SBATCH --error=/dev/null", "#SBATCH --no-requeue", "set -euo pipefail",
                  f"cd -- {shlex.quote(str(directory))}"]
        if self.modules:
            lines.append("module load " + shlex.join(self.modules))
        lines += [f"export OMP_NUM_THREADS={self.threads}",
                  "# Only the runner's designated publisher writes result.json; no rank writes a journal."]
        command = ["srun", "--kill-on-bad-exit=1", f"--nodes={self.nodes}",
                   f"--ntasks={self.nodes * self.ranks_per_node}",
                   f"--ntasks-per-node={self.ranks_per_node}", f"--cpus-per-task={self.threads}",
                   f"--cpu-bind={self.cpu_bind}"]
        if self.gpu_bind is not None:
            command += [f"--gpus-per-task={self.gpus_per_task}", f"--gpu-bind={self.gpu_bind}"]
        command += [self.executable, str(directory / "input.json"), str(directory / "result.json"), str(max_input_bytes)]
        return "\n".join(lines + ["exec " + shlex.join(command), ""])


@dataclass(frozen=True)
class SlurmStatus:
    """One scheduler observation. result_path alone makes no scientific claim.

    Attributes:
        state: Scheduler state observed for the original job.
        exit_code: Scheduler-reported process exit code, or None before it is known.
        result_path: Original expected output path, or None if no result path is available.
        failure: Recorded scheduler/output failure context. None when absent.
    """

    state: str
    exit_code: str | None
    result_path: Path | None
    failure: str | None = None


class SlurmLauncher:
    """One submission or refresh per explicit call, with bounded CLI transport.

    ``locator.instance`` holds the NWQLib submission UUID, and
    ``locator.cluster`` and ``locator.account`` route scheduler commands. The
    serialized profile names the original spool. Each script and input write
    and each scheduler response is admitted through ``run``.
    ``max_response_bytes`` caps the combined stdout and stderr of each command.
    An injected transport must enforce that cap during capture, must not retry,
    and must remove inherited ``SBATCH_*`` variables, which otherwise override
    the script's directives, including allocation and array options. The
    default ``SlurmTransport`` does all three and also removes ``SRUN_*``
    variables.
    """

    def __init__(self, profile, *, max_response_bytes, timeout_seconds=None, transport=None):
        """Bind one profile and a response cap. ``transport`` replaces the CLI, for example in tests."""
        if not isinstance(profile, SlurmProfile):
            raise TypeError("profile must be a SlurmProfile")
        if type(max_response_bytes) is not int or max_response_bytes <= 0:
            raise ValueError("max_response_bytes must be positive")
        self.profile = profile
        self.transport = SlurmTransport(timeout_seconds=timeout_seconds) if transport is None else transport
        self.max_response_bytes = max_response_bytes

    def _command(self, argv, run):
        """Admit the response allowance through the Run, then run one CLI command."""
        run.check_data(self.max_response_bytes)
        output = self.transport(tuple(argv), max_output_bytes=self.max_response_bytes)
        if not isinstance(output, str) or len(output.encode()) > self.max_response_bytes:
            raise ValueError("Slurm transport exceeded its admitted text response")
        return output

    def _locator(self, job_id, submission_id):
        """Build the locator of one plain Slurm job with a positive uint32 ID.

        The canonical decimal ID has at most 10 ASCII digits and is at most
        4294967295. Array tasks and heterogeneous components identify parts of
        a job collection, while the runner protocol expects exactly one job.
        """
        if not re.fullmatch(r"[1-9][0-9]{0,9}", job_id) or int(job_id) > 0xFFFFFFFF:
            raise ValueError("Slurm launcher requires a single job ID in 1..4294967295")
        return JobLocator(provider="slurm", job_id=job_id, cluster=self.profile.cluster,
                          account=self.profile.account, instance=_submission(submission_id))

    def _directory(self, locator):
        """Spool directory of the original job, after checking that the locator belongs to this profile.

        The directory is named by the submission UUID that the locator keeps in
        ``instance``. The provider, cluster, account, job id and UUID form are
        checked before that path is built, so a foreign locator reaches no file
        or scheduler command.
        """
        if (locator.provider != "slurm" or locator.cluster != self.profile.cluster
                or locator.account != self.profile.account):
            raise ValueError("Slurm locator differs from the original cluster/account")
        self._locator(locator.job_id, locator.instance)
        return Path(self.profile.spool) / locator.instance

    def admit(self, payload, *, max_input_bytes):
        """Run every launch check that depends only on this profile and the payload.

        These checks read no Run and write nothing, so the common owner runs
        them before it commits the submission intent (through
        ``NWQSimSlurmBackend.admit_batch``), and a refusal leaves no submission,
        event, exposure or output reservation. ``launch`` repeats them for a
        direct caller.

        The response floor requires max_response_bytes to cover ten job-ID
        digits, a separator, the profile's ASCII cluster name and one trailing
        newline. This floor admits the canonical acknowledgement
        of any supported job ID. It does not bound later status or accounting rows.

        ``_locator`` admits one positive uint32 job ID, with at most ten decimal
        digits, and ``SlurmProfile`` admits a single ASCII cluster name. Slurm's
        parsable output (https://slurm.schedmd.com/sbatch.html#OPT_parsable) is
        the job ID and, when present, the cluster separated by ``;``. For the
        canonical Unix acknowledgement with at most one trailing LF, the maximum
        encoded length is B_ack = 10 + 1 + |cluster|_ASCII + 1 = 12 + |cluster|_ASCII.
        This is the smallest uniform allowance for every admitted ID and the
        known cluster in that reply form, not the actual length of every
        acknowledgement. The cap applies before ``.strip()``, so the newline
        counts. The floor assumes the canonical sbatch line with no other
        stdout or stderr bytes. A transport that introduces CRLF needs another byte or must
        normalize it within its own declared protocol. For ``perlmutter`` the
        floor is 22 bytes, reached by ``4294967295;perlmutter\\n``.

        Then the rank and GPU request, the build identity of a non-CPU runner,
        the script rendering and the payload's input byte allowance are checked.
        """
        minimum = 12 + len(self.profile.cluster.encode("ascii"))
        if self.max_response_bytes < minimum:
            raise ValueError(
                f"Slurm launch requires max_response_bytes >= {minimum} for the sbatch "
                f"acknowledgement from cluster {self.profile.cluster!r}, "
                f"got {self.max_response_bytes}. Raise max_response_bytes before launch"
            )
        self.profile.admit_execution()
        if self.profile.backend != "CPU":
            # Low-level callers bypass common preparation. Verify the real
            # selected binary here before allocating cooperating ranks or a GPU.
            from nwqlib.backends.nwqsim import _describe
            if _describe(self.profile.executable, self.profile.backend, self.profile.method)["nwqsim_revision"] != self.profile.build_identity:
                raise ValueError("Slurm profile build identity differs from the actual selected runner")
        # The submission UUID only names the spool directory and the job, so
        # rendering with the nil UUID checks the same script inputs.
        self.profile.script(str(UUID(int=0)), max_input_bytes=max_input_bytes)
        if not isinstance(payload, bytes) or len(payload) > max_input_bytes:
            raise ValueError("prepared runner payload exceeds its input byte allowance")

    def launch(self, payload, *, submission_id, max_input_bytes, run):
        """Run ``admit``, then write the spool files once and call ``sbatch --parsable`` once.

        The caller has already persisted the intent. If the acknowledgement is
        lost, the files stay in place and ``reconcile`` searches for the job
        name. A repeated launch with the same UUID fails at directory creation,
        before any scheduler call.
        """
        self.admit(payload, max_input_bytes=max_input_bytes)
        return self._submit(payload, submission_id=submission_id, max_input_bytes=max_input_bytes, run=run)

    def _submit(self, payload, *, submission_id, max_input_bytes, run):
        """Write the spool files and call sbatch once for a payload that ``admit`` accepted."""
        script = self.profile.script(submission_id, max_input_bytes=max_input_bytes).encode()
        # One payload write plus one read per native rank. The script is written
        # once. Scheduler internals are outside this explicit file-work inventory.
        ranks = self.profile.nodes * self.profile.ranks_per_node
        payload_bytes = (1 + ranks) * len(payload) + len(script)
        run.check_data(payload_bytes)
        directory = Path(self.profile.spool) / submission_id
        # An unresolved accepted request keeps its directory. Never resubmit it.
        directory.mkdir(parents=True, exist_ok=False)
        (directory / "input.json").write_bytes(payload)
        (directory / "job.sh").write_bytes(script)
        response = self._command(["sbatch", "--parsable", f"--chdir={directory}", str(directory / "job.sh")], run).strip()
        parts = response.split(";")
        if len(parts) > 2 or (len(parts) == 2 and parts[1] != self.profile.cluster):
            raise ValueError("sbatch acknowledgement differs from the requested cluster; reconcile the original intent")
        return self._locator(parts[0], submission_id)

    def refresh(self, locator, *, run):
        """Observe the original job with one ``squeue`` call, and ``sacct`` only when it left the queue.

        Scheduler state, exit code and result-file visibility are reported
        separately. None of them alone establishes scientific completion,
        which the native result decoder decides.
        """
        directory = self._directory(locator)
        route = [f"--clusters={locator.cluster}", f"--jobs={locator.job_id}"]
        active = self._command(["squeue", "--noheader", "--format=%i|%T", *route], run)
        rows = [line.split("|") for line in active.splitlines() if line.strip()]
        result = directory / "result.json"
        result = result if result.is_file() else None
        if rows:
            if len(rows) != 1 or len(rows[0]) != 2 or rows[0][0] != locator.job_id:
                return SlurmStatus("uncertain", None, result, "partial/array or mismatched queue response")
            return SlurmStatus(rows[0][1], None, result)
        accounting = self._command(["sacct", "--noheader", "--parsable2", "--allocations",
                                    "--format=JobIDRaw,State,ExitCode", *route], run)
        rows = [line.split("|") for line in accounting.splitlines() if line.strip()]
        if not rows:
            return SlurmStatus("unknown", None, result, "job absent from queue; accounting may be delayed")
        if len(rows) != 1 or len(rows[0]) != 3 or rows[0][0] != locator.job_id:
            return SlurmStatus("uncertain", None, result, "partial/array or mismatched accounting response")
        _, state, code = rows[0]
        if not re.fullmatch(r"[0-9]+:[0-9]+", code):
            raise ValueError("invalid Slurm exit-code response")
        failure = "nonzero scheduler exit" if code != "0:0" else None
        if state == "COMPLETED" and result is None:
            failure = "scheduler completed but raw result is not visible"
        return SlurmStatus(state, code, result, failure)

    def reconcile(self, submission_id, *, since, until, run):
        """One bounded accounting query. Absence or ambiguity never launches work.

        since/until are the saved submission search window, in the site's local
        time. They are explicit because sacct's default date window can omit jobs.
        """
        name = "nwqlib-" + _submission(submission_id)
        start, end = datetime.fromisoformat(since), datetime.fromisoformat(until)
        if start >= end:
            raise ValueError("reconciliation requires an increasing accounting window")
        output = self._command(["sacct", "--noheader", "--parsable2", "--allocations",
            f"--clusters={self.profile.cluster}", f"--accounts={self.profile.account}",
            f"--name={name}", f"--starttime={start.isoformat()}", f"--endtime={end.isoformat()}",
            "--format=JobIDRaw,JobName%80"], run)
        rows = [line.split("|") for line in output.splitlines() if line.strip()]
        if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1] != name:
            return None
        return self._locator(rows[0][0], submission_id)

    def cancel(self, locator, *, run):
        """Send one ``scancel`` for the original cluster and job. A later refresh reads the outcome."""
        self._directory(locator)
        self._command(["scancel", f"--clusters={locator.cluster}", locator.job_id], run)


class NWQSimSlurmBackend(Record):
    """NWQ-Sim run as a Slurm batch job on a cluster.

    Build it with keyword arguments, for example
    `NWQSimSlurmBackend(profile=..., accounting_since=..., accounting_until=..., max_input_bytes=..., max_output_bytes=..., max_buffer_bytes=..., max_response_bytes=..., timeout_seconds=...)`,
    and pass it as `backend=` to [`prepare`][nwqlib.scientist.prepare]. Every
    argument except `optimization_level` is required. The profile holds the
    runner, its build and the pending-job directory (`spool`). Submission calls `sbatch` once,
    and a job whose acknowledgement was lost is found again by its job name in
    Slurm accounting within the configured time window. NWQLib has tested this
    backend only offline, with scheduler and protocol checks, and has not tested
    a live site allocation, runner or hardware. The [Slurm guide](../slurm.md)
    shows a complete run.

    Attributes:
        kind: Fixed `"slurm"`, the backend type.
        profile: Required. The [`SlurmProfile`][nwqlib.backends.slurm.SlurmProfile].
            Its total rank count and GPU request are checked against its
            `backend` and `method`.
        accounting_since: Required. Start of the accounting search window, an ISO
            time without a time zone, read in the site's local time. Choose a
            window that covers the intended submission period.
        accounting_until: Required. End of that window, later than
            `accounting_since`.
        max_input_bytes: Required. Positive. Limit in bytes on each request file,
            as in [`NWQSimBackend`][nwqlib.backends.nwqsim.NWQSimBackend].
        max_output_bytes: Required. Positive. Limit in bytes on each result file.
        max_buffer_bytes: Required. Positive. Limit in bytes on the runner's known
            arrays per process, as in `NWQSimBackend`.
        max_response_bytes: Required. Positive. Limit in bytes on the output of
            each scheduler command. It must cover the `sbatch --parsable` reply,
            12 bytes plus the cluster name's length (22 bytes for `perlmutter`).
        timeout_seconds: Required. Positive deadline in seconds for each scheduler
            command.
        optimization_level: Default `0`. Qiskit transpiler level of the translation to
            U and CX gates, as in `NWQSimBackend`. A level other than 0 adds the
            same [roundoff exclusion](../glossary.md#roundoff-exclusion)
            `"optimization_level"` to the
            [preparation record](../glossary.md#preparation-record).

    Raises:
        ValueError: If the rank or GPU request does not fit the profile's
            `backend` and `method`, the accounting window has a time zone or is
            not increasing, or `timeout_seconds` is not positive.
    """

    kind: Literal["slurm"] = "slurm"
    supports_synchronous: ClassVar[bool] = False
    qualification_notice: ClassVar[str] = (
        "This Slurm site has offline scheduler/protocol checks only; the selected site allocation, "
        "runner and hardware have no live qualification. Qualify the selected build at the site "
        "before relying on its execution results.")
    profile: SlurmProfile
    accounting_since: Text
    accounting_until: Text
    max_input_bytes: PositiveInt
    max_output_bytes: PositiveInt
    max_buffer_bytes: PositiveInt
    max_response_bytes: PositiveInt
    timeout_seconds: Real
    optimization_level: Annotated[Count, Field(le=3)] = 0

    @model_validator(mode="after")
    def _configuration(self):
        """Check the rank and GPU request, the accounting window and the CLI deadline.

        The window must be naive, because ``sacct`` reads ``--starttime`` and
        ``--endtime`` in the site's local time, and it must be increasing.
        """
        self.profile.admit_execution()
        start, end = datetime.fromisoformat(self.accounting_since), datetime.fromisoformat(self.accounting_until)
        if start.tzinfo is not None or end.tzinfo is not None or start >= end:
            raise ValueError("accounting requires an increasing explicit site-local time window")
        if self.timeout_seconds <= 0:
            raise ValueError("Slurm CLI deadline must be positive")
        return self

    def native_job_id_length(self):
        """Return the 10-digit upper bound for a plain Slurm job identifier.

        Slurm's ``submit_response_msg_t.job_id`` is uint32_t (``slurm/slurm.h``,
        tag ``slurm-25-05-0-1``). Its positive decimal representation has at
        most 10 ASCII digits. ``SlurmLauncher._locator`` enforces that domain.
        """
        return 10

    def _runner(self):
        """The NWQ-Sim adapter for this profile, which owns preparation and result decoding."""
        from nwqlib.backends.nwqsim import NWQSimBackend
        return NWQSimBackend(executable=self.profile.executable, spool=self.profile.spool,
            backend=self.profile.backend, method=self.profile.method,
            ranks=self.profile.nodes * self.profile.ranks_per_node,
            max_input_bytes=self.max_input_bytes, max_output_bytes=self.max_output_bytes,
            max_buffer_bytes=self.max_buffer_bytes, optimization_level=self.optimization_level)

    def _launcher(self):
        return SlurmLauncher(self.profile, max_response_bytes=self.max_response_bytes,
                             timeout_seconds=self.timeout_seconds)

    def target_for(self, observation):
        return self._runner().target_for(observation)

    def prepare(self, circuit, *, observation, runtime, position, source_definitions, run, snapshot, views=None):
        """Prepare with the NWQ-Sim adapter after checking the runner's revision against build_identity.

        The target is the runner's (``target_for``), so a trajectory is
        admitted and refused exactly as the NWQ-Sim adapter admits it for the
        selected backend/method, and its views reach that adapter unchanged.
        """
        if observation.kind == "trajectory" and "trajectory" not in self.target_for(observation).readouts:
            observation.reject_unsupported_schedule("Slurm NWQ-Sim")
        context = run._state["backend_context"]
        runner = self._runner()
        if runner._build(context)["nwqsim_revision"] != self.profile.build_identity:
            raise ValueError("Slurm profile build identity differs from the actual selected runner")
        return runner.prepare(circuit, observation=observation, runtime=runtime, position=position,
            source_definitions=source_definitions, run=run, snapshot=snapshot, views=views)

    def restore_native(self, record, payload=None, *, run):
        """Restore through the NWQ-Sim adapter."""
        return self._runner().restore_native(record, payload, run=run)

    def admit_batch(self, natives):
        """Admit one runner request and run the launcher's configuration and payload checks.

        ``submit_detached`` calls this before it commits the submission intent,
        so the response floor, the rank and GPU request, a non-CPU build
        identity, the script rendering and the input byte allowance
        (``SlurmLauncher.admit``) refuse with no submission, event, exposure or
        output reservation left in the Run.
        """
        self._runner().admit_batch(natives)
        native, = natives
        self._launcher().admit(native.payload, max_input_bytes=self.max_input_bytes)

    def launch(self, natives, *, submission_id, run):
        """Submit the request that ``admit_batch`` accepted and return its locator.

        After the intent only the Run's data headroom checks, the spool writes and the one
        sbatch call remain.
        """
        self._runner().admit_batch(natives)
        native, = natives
        # The common owner has persisted the intent/exposure before this call.
        # Native output is written once within this admitted combined byte cap.
        run.check_data(self.max_output_bytes)
        return self._launcher()._submit(native.payload, submission_id=submission_id,
            max_input_bytes=self.max_input_bytes, run=run)

    def reconcile(self, submission, natives, *, run):
        """Search accounting for the original job name within the configured window."""
        self._runner().admit_batch(natives)
        return self._launcher().reconcile(submission.submission_id,
            since=self.accounting_since, until=self.accounting_until, run=run)

    def refresh(self, locator, natives, *, run):
        """Combine one scheduler observation with the native result decoder.

        A missing result file keeps the job refreshable, pending while the
        scheduler reports an active state and uncertain otherwise. An ambiguous
        scheduler response stays uncertain. Otherwise a visible
        packet is decoded by the NWQ-Sim owner, and a nonzero Slurm exit code
        or signal is passed to it as a failure. A valid packet can publish
        while accounting is still delayed.
        """
        from nwqlib.backends.connection import BackendRefresh
        from nwqlib.backends.nwqsim import NWQSimExecutionError
        self._runner().admit_batch(natives)
        native, = natives
        status = self._launcher().refresh(locator, run=run)
        provider_status = status.state
        if status.exit_code is not None:
            provider_status += "; ExitCode=" + status.exit_code
        if status.result_path is None:
            # Even a terminal allocation can precede visibility of its native
            # packet. Leave it refreshable and do not invent an evolution count.
            uncertain = status.failure is not None or status.state not in {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING"}
            return BackendRefresh("uncertain" if uncertain else "acknowledged",
                provider_status=provider_status, failure=status.failure or "native terminal result unavailable")
        if status.state == "uncertain":
            return BackendRefresh("uncertain", provider_status=provider_status, failure=status.failure)
        run.check_data(self.max_output_bytes)
        # A nonzero exit value or a signal reaches the NWQ-Sim result decoder as a failure.
        returncode = int(status.exit_code is not None and status.exit_code != "0:0")
        try:
            result = self._runner()._result(status.result_path, native=native,
                                            job_id=locator.job_id, returncode=returncode)
        except NWQSimExecutionError as error:
            failure = str(error)
            if returncode:
                failure += "; Slurm ExitCode=" + status.exit_code
            return BackendRefresh("cancelled" if status.state.startswith("CANCELLED") else "failed",
                provider_status=provider_status, failure=failure, native_simulations=error.native_simulations)
        return BackendRefresh("completed", results=(("0", result),),
            provider_status=provider_status, native_simulations=1)

    def cancel(self, locator, *, run):
        """Send one ``scancel`` for the original job. A later refresh reports the outcome."""
        self._launcher().cancel(locator, run=run)
