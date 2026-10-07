"""Native target wiring with supplied packets/commands; zero native evolution."""

import json
from math import isfinite
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
import pytest
from nwqlib.backends import NWQSimBackend
from nwqlib.backends import nwqsim
from nwqlib.core.planning import ObservationSpec, RuntimeOptions
from nwqlib._prepared_execution import (
    Run,
    prepare_experiment as prepare,
    refresh_submissions,
    submit_detached,
)
from nwqlib._run_archive import load as load_run
from test_nexus_backend import IO

REVISION = "efd02262ff9c5def4f2eac1df6416c1ee2023d5b"
ROUTES = (
    ("CPU", "SV"),
    ("CPU", "DM"),
    ("MPI", "SV"),
    ("NVGPU", "SV"),
    ("NVGPU", "DM"),
    ("AMDGPU", "SV"),
    ("AMDGPU", "DM"),
)


def selection(*, width=4, shots=4):
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import ingest_occupation

    problem = Expectation(
        state=ingest_occupation("1" + "0" * (width - 1), num_qubits=width),
        observable=ingest_pauli((("Z" + "I" * (width - 2) + "Z", 1.0),), num_qubits=width),
    )
    return nwqlib.plan(problem, method=ExpectationMethod(), shots=shots, seed=23)


def configured(tmp_path, backend="CPU", method="SV", **changes):
    fields = dict(
        executable="/supplied/runner",
        spool=str(tmp_path / "spool"),
        backend=backend,
        method=method,
        ranks=2 if backend == "MPI" else 1,
        max_input_bytes=65536,
        max_output_bytes=65536,
        max_buffer_bytes=65536,
    )
    fields.update(changes)
    return NWQSimBackend(**fields)


def description(backend, method, **changes):
    fields = dict(
        format="nwqlib.nwqsim/3",
        backends=[backend],
        methods=[method],
        readouts=["counts", "pauli", "probabilities", "amplitudes", "trajectory"]
        if (backend, method) == ("CPU", "SV")
        else ["counts"],
        distributed=backend == "MPI",
        gpu_error_check=backend in {"NVGPU", "AMDGPU"},
        nwqsim_revision=REVISION,
        compiler="supplied C++17",
        architecture="supplied",
    )
    fields.update(changes)
    return SimpleNamespace(stdout=json.dumps(fields))


def _phase_factor_reply(argv, factor, queries):
    """Stub runner: the supplied output phase factor for --phase-factor, the CPU/SV description otherwise."""
    if argv[1] == "--phase-factor":
        queries.append(argv[2])
        return SimpleNamespace(stdout=json.dumps(dict(format="nwqlib.nwqsim/3", phase_factor=list(factor))))
    return description("CPU", "SV", roundoff_base_revision=REVISION, roundoff_sources_identical=True)


@pytest.mark.parametrize("backend,method", ROUTES)
def test_selected_target_public_detached_resume_keeps_one_job_and_shot_population(
    tmp_path, monkeypatch, backend, method
):
    """Inject the native file protocol to test target/rank identity and one-shot-population
    resumption without simulation.
    """
    selected, config = (selection(), configured(tmp_path, backend, method))
    calls = []

    def describe(argv, **kwargs):
        assert argv == [config.executable, "--describe"]
        return description(backend, method)

    def supplied_process(argv, **kwargs):
        calls.append(argv)
        if backend == "MPI":
            assert argv[:3] == [config.mpi_launcher, "-n", "2"]
            argv = argv[3:]
        assert argv[0] == config.executable
        request = json.loads(Path(argv[1]).read_bytes())
        assert (request["backend"], request["method"], request["ranks"]) == (backend, method, config.ranks)
        Path(argv[2]).write_text(
            json.dumps(
                dict(
                    format=request["format"],
                    input_id=request["input_id"],
                    backend=backend,
                    method=method,
                    ranks=config.ranks,
                    nwqsim_revision=REVISION,
                    status="completed",
                    native_simulations=1,
                    kind="counts",
                    shots=4,
                    counts={"1000": 4},
                )
            )
        )
        return SimpleNamespace(pid=987654)

    monkeypatch.setattr(nwqsim.subprocess, "run", describe)
    monkeypatch.setattr(nwqsim.subprocess, "Popen", supplied_process)
    journal = tmp_path / "run"
    with Run(selected, backend=config, directory=journal) as run:
        handle = prepare(
            selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=23)
        )
        receipt = submit_detached((handle,), run=run)
        assert handle.record.target.name == f"nwqsim_{backend.lower()}_{method.lower()}"
    with load_run(journal, backend=config) as run:
        (chunk,) = refresh_submissions(run=run)
        assert chunk.job == receipt.locator.job_id and chunk.returned_shots == 4
        histogram = chunk.histogram()
        assert (histogram.width, histogram.index_list(), histogram.weights.tolist()) == (4, [8], [4])
        assert run.collect(chunk)
        assert run.trace.jobs == 1 and run.trace.submissions[0].native_simulations == 1
        assert sum((event.shots for event in run.trace.events)) == 4
    with load_run(journal, backend=config) as run:
        assert not refresh_submissions(run=run) and len(run.observations.chunks) == 1
    assert len(calls) == 1


@pytest.mark.parametrize("backend,method", ROUTES[1:])
@pytest.mark.parametrize("kind", ["amplitudes", "probabilities", "pauli_expectation"])
def test_non_cpu_sv_exact_readout_rejects_before_describe_or_lowering(
    tmp_path, monkeypatch, backend, method, kind
):
    from qiskit import QuantumCircuit

    def forbidden(*args, **kwargs):
        raise AssertionError("unsupported observation reached external work")

    monkeypatch.setattr(nwqsim.subprocess, "run", forbidden)
    monkeypatch.setattr("qiskit.transpile", forbidden)
    config = configured(tmp_path, backend, method)
    from test_amplitude_masses import readout

    spec = ObservationSpec(
        kind=kind,
        labels=("ZIIZ",) if kind == "pauli_expectation" else (),
        qubits=(0,) if kind == "probabilities" else (),
        amplitudes=readout(width=4, coordinates=(0, 1, 2, 3), success=(), conditions=())
        if kind == "amplitudes"
        else None,
    )
    with pytest.raises(ValueError, match="does not support"):
        config.prepare(
            QuantumCircuit(4),
            observation=spec,
            runtime=RuntimeOptions(seed=23),
            position=0,
            source_definitions=(),
            snapshot="unsupported",
            run=IO(),
        )


@pytest.mark.parametrize(
    "changes,match",
    [
        ({"backend": "MPI", "method": "DM"}, "silently substitutes"),
        ({"backend": "MPI", "ranks": 3}, "power-of-two"),
        ({"backend": "NVGPU", "ranks": 2}, "only the MPI backend"),
        ({"backend": "NVGPU_MPI"}, "uninitialized gpu_mem"),
    ],
)
def test_invalid_native_routes_reject_at_configuration(tmp_path, changes, match):
    with pytest.raises(ValueError, match=match):
        configured(tmp_path, **changes)


@pytest.mark.parametrize(
    "changes",
    [
        dict(backends=["CPU"]),
        dict(methods=["DM"]),
        dict(distributed=False),
        dict(readouts=["counts", "amplitudes"]),
    ],
)
def test_mpi_cannot_use_a_different_or_noncollective_described_binary(tmp_path, monkeypatch, changes):
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("MPI", "SV", **changes))
    with pytest.raises(ValueError, match="configured native target"):
        configured(tmp_path, "MPI")._build({})


@pytest.mark.parametrize("backend,method", [("NVGPU", "SV"), ("AMDGPU", "DM")])
def test_gpu_binary_must_enable_runtime_error_checking(tmp_path, monkeypatch, backend, method):
    config = configured(tmp_path, backend, method)
    monkeypatch.setattr(
        nwqsim.subprocess, "run", lambda *args, **kwargs: description(backend, method, gpu_error_check=False)
    )
    with pytest.raises(ValueError, match="configured native target"):
        config._build({})
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description(backend, method))
    assert config._build({})["gpu_error_check"] is True


@pytest.mark.parametrize(
    "backend,method,state,samples",
    [
        ("CPU", "SV", 392, 32),
        ("CPU", "DM", 4232, 32),
        ("MPI", "SV", 256, 96),
        ("NVGPU", "SV", 784, 128),
        ("NVGPU", "DM", 12304, 128),
        ("AMDGPU", "SV", 784, 128),
        ("AMDGPU", "DM", 12304, 128),
    ],
)
def test_buffer_admission_uses_actual_method_and_rank_populations(
    tmp_path, monkeypatch, backend, method, state, samples
):
    from qiskit import QuantumCircuit

    config = configured(tmp_path, backend, method, max_buffer_bytes=state + samples - 1)
    calls = []

    def describe(*args, **kwargs):
        calls.append(args)
        return description(backend, method)

    monkeypatch.setattr(nwqsim.subprocess, "run", describe)
    circuit = QuantumCircuit(4, 1)
    circuit.measure(3, 0)
    kwargs = dict(
        observation=ObservationSpec(kind="counts", shots=4),
        runtime=RuntimeOptions(seed=23),
        position=1,
        source_definitions=(),
        snapshot="arrays",
        run=IO(),
    )
    with pytest.raises(ValueError, match="byte limit"):
        config.prepare(circuit, **kwargs)
    assert not calls
    config.revise(max_buffer_bytes=state + samples).prepare(circuit, **kwargs)


@pytest.mark.parametrize(
    "backend,method,compiler,flags",
    [
        ("CPU", "DM", "c++", []),
        ("MPI", "SV", "mpic++", ["-DMPI_ENABLED"]),
        ("NVGPU", "SV", "nvcc", ["-DCUDA_ENABLED", "-DGPU_ERROR_CHECK", "-x", "cu"]),
        ("AMDGPU", "DM", "hipcc", ["-DHIP_ENABLED", "-DGPU_ERROR_CHECK", "-x", "hip"]),
    ],
)
def test_build_uses_actual_target_compiler_and_public_feature_flags(
    tmp_path, monkeypatch, backend, method, compiler, flags
):
    source = tmp_path / "source"
    (source / "include").mkdir(parents=True)
    (source / "include" / "backendManager.hpp").touch()
    commands = []
    monkeypatch.setattr(nwqsim.shutil, "which", lambda selected: "/supplied/" + selected)

    def command(argv, **kwargs):
        commands.append(argv)
        if argv[0] == "git":
            # returncode 1 answers merge-base --is-ancestor: not a descendant.
            return SimpleNamespace(stdout=REVISION if "rev-parse" in argv else "", returncode=1)
        Path(argv[-1]).write_bytes(b"supplied build output")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nwqsim.subprocess, "run", command)
    output = nwqsim.build_runner(source, tmp_path / "runner", backend=backend, method=method)
    assert output.read_bytes() == b"supplied build output"
    assert commands[-1][0] == "/supplied/" + compiler
    assert commands[-1][3 : 3 + len(flags)] == flags
    assert f'-DNWQLIB_BACKEND="{backend}"' in commands[-1]
    assert f'-DNWQLIB_METHOD="{method}"' in commands[-1]


@pytest.mark.parametrize("extra", [None, "untracked", "ignored"])
def test_build_rejects_headers_outside_the_recorded_revision(tmp_path, monkeypatch, extra):
    import subprocess

    source = tmp_path / "source"
    (source / "include").mkdir(parents=True)
    (source / "include" / "backendManager.hpp").touch()
    git = ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]
    for arguments in (["init", "-q"], ["add", "-A"], ["commit", "-q", "-m", "source"]):
        subprocess.run(git + arguments, check=True, capture_output=True)
    if extra == "ignored":
        (source / ".git" / "info" / "exclude").write_text("include/extra.hpp\n")
    if extra is not None:
        (source / "include" / "extra.hpp").touch()
    compiled = []
    real_run = subprocess.run

    def command(argv, **kwargs):
        if argv[0] == "git":
            return real_run(argv, **kwargs)
        compiled.append(argv)
        Path(argv[-1]).write_bytes(b"supplied build output")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nwqsim.shutil, "which", lambda selected: "/supplied/" + selected)
    monkeypatch.setattr(nwqsim.subprocess, "run", command)
    if extra is None:
        nwqsim.build_runner(source, tmp_path / "runner")
        assert len(compiled) == 1
    else:
        # An extra header under include/ would be compiled into this runner.
        with pytest.raises(ValueError, match="clean NWQ-Sim revision"):
            nwqsim.build_runner(source, tmp_path / "runner")
        assert compiled == []


@pytest.mark.parametrize(
    "history,identical",
    [("base", "1"), ("descendant", "1"), ("changed_kernel", "0"), ("unrelated", None)],
)
def test_build_compiles_the_roundoff_derivation_rule_into_the_runner(tmp_path, monkeypatch, history, identical):
    import subprocess

    # The rule is nwqsim._ROUNDOFF_BASE_REVISION's: descent from the base and
    # byte-identical derivation sources. The fixture's first commit plays the base.
    source = tmp_path / "source"
    for path in ("include/backendManager.hpp", *nwqsim._ROUNDOFF_SOURCES):
        (source / path).parent.mkdir(parents=True, exist_ok=True)
        (source / path).write_text("base\n")
    git = ["git", "-C", str(source), "-c", "user.name=t", "-c", "user.email=t@example.invalid"]

    def commit(message):
        subprocess.run(git + ["add", "-A"], check=True, capture_output=True)
        subprocess.run(git + ["commit", "-q", "-m", message], check=True, capture_output=True)

    subprocess.run(git + ["init", "-q"], check=True, capture_output=True)
    commit("base")
    base = subprocess.run(git + ["rev-parse", "HEAD"], check=True, capture_output=True, text=True).stdout.strip()
    if history == "unrelated":
        # The same files on a root commit that does not descend from the base.
        subprocess.run(git + ["checkout", "-q", "--orphan", "unrelated"], check=True, capture_output=True)
        commit("unrelated")
    elif history != "base":
        (source / "CMakeLists.txt").write_text("outside the derivation\n")
        if history == "changed_kernel":
            (source / "include/svsim/sv_cpu.hpp").write_text("changed kernel\n")
        commit("later")
    monkeypatch.setattr(nwqsim, "_ROUNDOFF_BASE_REVISION", base)
    compiled = []
    real_run = subprocess.run

    def command(argv, **kwargs):
        if argv[0] == "git":
            return real_run(argv, **kwargs)
        compiled.append(argv)
        Path(argv[-1]).write_bytes(b"supplied build output")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(nwqsim.shutil, "which", lambda selected: "/supplied/" + selected)
    monkeypatch.setattr(nwqsim.subprocess, "run", command)
    nwqsim.build_runner(source, tmp_path / "runner")
    roundoff = [argument for argument in compiled[0] if argument.startswith("-DNWQLIB_ROUNDOFF")]
    expected = [] if identical is None else [
        f'-DNWQLIB_ROUNDOFF_BASE_REVISION="{base}"', f"-DNWQLIB_ROUNDOFF_SOURCES_IDENTICAL={identical}"]
    assert roundoff == expected


def test_local_notice_identifies_host_lifetime_and_spool(tmp_path, monkeypatch):
    monkeypatch.setattr(nwqsim.platform, "node", lambda: "original-vm")
    notice = configured(tmp_path).qualification_notice
    assert "original-vm" in notice and str(tmp_path / "spool") in notice and ("remain running" in notice)


@pytest.mark.parametrize("state", ["exited", "absent", "running"])
def test_local_process_without_result_stays_uncertain_and_is_not_relaunched(tmp_path, monkeypatch, state):
    """Injected process state: only a live process without output remains acknowledged."""
    selected, config = selection(), configured(tmp_path)
    launches, signals = [], []
    process = SimpleNamespace(pid=987654, poll=lambda: 1 if state == "exited" else None)

    def signal(pid, number):
        signals.append((pid, number))
        if state == "absent":
            raise ProcessLookupError

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    monkeypatch.setattr(nwqsim.subprocess, "Popen", lambda argv, **kwargs: launches.append(argv) or process)
    monkeypatch.setattr(nwqsim.os, "kill", signal)

    def outcome(run):
        run.resume()
        with pytest.raises(TimeoutError):
            run.wait(timeout=0)
        (submission,) = run._state["submissions"].values()
        return submission.status, submission.failure, [event.status for event in run.trace.events]

    journal = tmp_path / "run"
    with Run(selected, backend=config, directory=journal) as run:
        run.resume()
        if state == "exited":  # The launching process still holds its child handle.
            observed = outcome(run)
    if state != "exited":  # A reopened run has only the saved process identity.
        with load_run(journal, backend=config) as run:
            observed = outcome(run)
    assert observed == {
        "exited": ("uncertain", "native process exited without a terminal result", ["uncertain"]),
        "absent": ("uncertain", "native process is absent and no terminal result is available", ["uncertain"]),
        "running": ("acknowledged", None, ["reserved"]),
    }[state]
    assert len(launches) == 1 and all(number == 0 for _, number in signals)


@pytest.mark.parametrize("stage", ["before_directory", "after_directory", "before_process"])
def test_local_interruption_before_launch_has_no_spool_recovery_route(tmp_path, monkeypatch, stage):
    from nwqlib.execution import RunFailed

    selected, config = selection(), configured(tmp_path)
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    monkeypatch.setattr(nwqsim.subprocess, "Popen", lambda *a, **k: pytest.fail("unexpected launch"))

    def interrupted(self, natives, *, submission_id, run):
        directory = Path(self.spool)/submission_id
        if stage != "before_directory":
            directory.mkdir(parents=True)
        if stage == "before_process":
            (directory/"input.json").write_bytes(natives[0].payload)
        raise KeyboardInterrupt("interrupted before starting the local process")

    monkeypatch.setattr(type(config), "launch", interrupted)
    journal = tmp_path/"run"
    with Run(selected, backend=config, directory=journal) as run:
        with pytest.raises(KeyboardInterrupt):
            run.resume()
    with load_run(journal, backend=config) as run:
        exposure, state = run.exposure, run.rng.snapshot()
        for continuation in (run.resume, lambda: run.wait(timeout=0)):
            with pytest.raises(RunFailed) as caught:
                continuation()
            assert caught.value.status == "uncertain" and caught.value.stage == "recovery"
            assert run.exposure == exposure and run.rng.snapshot() == state


def test_local_lost_acknowledgement_binds_only_the_original_output_without_relaunch(tmp_path, monkeypatch):
    """The launch is interrupted before its locator is stored; injected process, no native runner."""
    from nwqlib.execution import RunFailed

    selected, config = selection(), configured(tmp_path)
    launches = []
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    monkeypatch.setattr(nwqsim.subprocess, "Popen",
                        lambda argv, **kwargs: launches.append(argv) or SimpleNamespace(pid=987654, poll=lambda: None))

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt("interrupted before the acknowledgement was stored")

    journal = tmp_path / "run"
    with Run(selected, backend=config, directory=journal) as run:
        with monkeypatch.context() as patch:
            patch.setattr(Run, "_ack_submission", interrupted)
            with pytest.raises(KeyboardInterrupt):
                run.resume()
        run.resume()  # The original child is still observable in this process.
    with load_run(journal, backend=config) as run:
        assert not refresh_submissions(run=run)  # No output yet: nothing is bound or launched.
        (submission,) = run._state["submissions"].values()
        assert submission.locator is None and submission.status == "uncertain" and len(launches) == 1
        exposure = run.exposure
        with pytest.raises(RunFailed, match="uncertain"):
            run.resume()  # No child handle survived reopen and no output exists yet.
        assert run.exposure == exposure and len(launches) == 1
        ((_, source, output, _),) = launches
        request = json.loads(Path(source).read_bytes())
        Path(output).write_text(json.dumps(dict(
            format=request["format"], input_id=request["input_id"], backend="CPU", method="SV", ranks=1,
            nwqsim_revision=REVISION, status="completed", native_simulations=1, kind="counts", shots=4,
            counts={"1000": 4})))
        (chunk,) = refresh_submissions(run=run)
        (submission,) = run._state["submissions"].values()
        assert submission.status == "completed" and chunk.returned_shots == 4
        assert submission.locator.job_id == submission.submission_id == Path(output).parent.name
        assert not refresh_submissions(run=run) and len(launches) == 1
        run.resume()
        assert run.result is not None and len(launches) == 1


def test_mpi_native_input_rank_and_target_association_rejects_supplied_foreign_result(tmp_path):
    config = configured(tmp_path, "MPI")
    native = nwqsim._PreparedNWQSim(None, b"", "original", dict(nwqsim_revision=REVISION))
    output = tmp_path / "result.json"
    output.write_text(
        json.dumps(
            dict(
                format="nwqlib.nwqsim/3",
                input_id="original",
                status="completed",
                backend="MPI",
                method="SV",
                ranks=1,
                nwqsim_revision=REVISION,
                native_simulations=1,
            )
        )
    )
    with pytest.raises(ValueError, match="target/ranks/build"):
        config._packet(output, native)


def test_mpi_launcher_verifies_real_binary_before_slurm_spool_or_scheduler(tmp_path, monkeypatch):
    from nwqlib.backends.slurm import SlurmLauncher, SlurmProfile

    profile = SlurmProfile(
        cluster="site",
        account="account",
        walltime="00:01:00",
        nodes=1,
        ranks_per_node=2,
        threads=1,
        executable="/supplied/runner",
        build_identity=REVISION,
        backend="MPI",
        spool=str(tmp_path / "spool"),
    )
    calls = []
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    launcher = SlurmLauncher(
        profile, transport=lambda *args, **kwargs: calls.append(args), max_response_bytes=1024
    )
    with pytest.raises(ValueError, match="configured native target"):
        launcher.launch(b"{}", submission_id=str(uuid4()), max_input_bytes=100, run=IO())
    assert not calls and (not Path(profile.spool).exists())


@pytest.mark.parametrize(
    "width,clbits,shots,match",
    [(2, 1, 4, "two local qubits"), (4, 1, 2**31, "int-sized shot count"), (4, 64, 4, "63 classical bits")],
)
def test_mpi_native_shape_rejects_before_external_description(
    tmp_path, monkeypatch, width, clbits, shots, match
):
    from qiskit import QuantumCircuit

    def forbidden(*args, **kwargs):
        raise AssertionError("invalid native population reached external work")

    monkeypatch.setattr(nwqsim.subprocess, "run", forbidden)
    with pytest.raises(ValueError, match=match):
        configured(tmp_path, "MPI").prepare(
            QuantumCircuit(width, clbits),
            observation=ObservationSpec(kind="counts", shots=shots),
            runtime=RuntimeOptions(seed=23),
            position=0,
            source_definitions=(),
            snapshot="invalid-shape",
            run=IO(),
        )


def test_every_mpi_rank_input_read_is_admitted_before_launch(tmp_path, monkeypatch):
    """Four MPI readers plus one saved payload require five input copies to be admitted before
    launch.
    """
    from nwqlib.backends.slurm import SlurmLauncher, SlurmProfile

    config = configured(tmp_path, "MPI", ranks=4)
    payload = b"supplied existing native input"
    native = nwqsim._PreparedNWQSim(None, payload, "original", {})
    recorded, launches = ([], [])
    monkeypatch.setattr(
        nwqsim.subprocess, "Popen", lambda *args, **kwargs: launches.append(args) or SimpleNamespace(pid=1234)
    )
    run = IO()
    run.check_data = lambda size: recorded.append(size)
    config.launch((native,), submission_id=str(uuid4()), run=run)
    assert recorded == [5 * len(payload) + config.max_output_bytes]
    assert len(launches) == 1
    profile = SlurmProfile(
        cluster="site",
        account="account",
        walltime="00:01:00",
        nodes=2,
        ranks_per_node=2,
        threads=1,
        executable=config.executable,
        build_identity=REVISION,
        backend="MPI",
        spool=str(tmp_path / "slurm"),
    )
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("MPI", "SV"))
    launcher = SlurmLauncher(profile, transport=lambda *args, **kwargs: "73;site\n", max_response_bytes=1024)
    recorded.clear()
    sid = str(uuid4())
    launcher.launch(payload, submission_id=sid, max_input_bytes=100, run=run)
    script = (Path(profile.spool) / sid / "job.sh").read_bytes()
    assert recorded[0] == 5 * len(payload) + len(script)


@pytest.mark.parametrize(
    "ranks,gate,wire,admitted",
    [
        (2, "u", 31, False),
        (2, "cx_control", 31, False),
        (2, "cx_target", 31, False),
        (2, "u", 0, True),
        (2, "cx_target", 1, True),
        (2, None, None, True),
        (4, "u", 31, True),
    ],
)
def test_global_mpi_gate_transfer_count_rejects_before_state_work(
    tmp_path, monkeypatch, ranks, gate, wire, admitted
):
    """A 32-qubit metadata-only circuit distinguishes global MPI transfer overflow from valid
    local-only gates.
    """
    from qiskit import QuantumCircuit

    config = configured(tmp_path, "MPI", ranks=ranks, max_buffer_bytes=2**40)
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("MPI", "SV"))
    circuit = QuantumCircuit(32, 1, global_phase=0.25)
    if gate == "u":
        circuit.u(0.3, 0.2, 0.1, wire)
    elif gate == "cx_control":
        circuit.cx(wire, 0)
    elif gate == "cx_target":
        circuit.cx(0, wire)
    circuit.measure(31, 0)
    kwargs = dict(
        observation=ObservationSpec(kind="counts", shots=4),
        runtime=RuntimeOptions(seed=23),
        position=len(circuit.data),
        source_definitions=(),
        snapshot="transfer-count",
        run=IO(),
    )
    if admitted:
        prepared = config.prepare(circuit, **kwargs)
        assert json.loads(prepared.payload)["num_qubits"] == 32
    else:
        with pytest.raises(ValueError, match="int-sized local state transfer"):
            config.prepare(circuit, **kwargs)
    assert not Path(config.spool).exists()


def test_twenty_qubit_known_array_projection_is_scalar_only():
    spec = ObservationSpec(kind="counts", shots=1024)
    mib, tib = (1024**2, 1024**4)
    assert nwqsim._buffer_bytes("CPU", "SV", 1, 20, spec) == (24 * mib + 8, 8192)
    assert nwqsim._buffer_bytes("MPI", "SV", 4, 20, spec) == (8 * mib, 24576)
    assert nwqsim._buffer_bytes("NVGPU", "SV", 1, 20, spec) == (48 * mib + 16, 32768)
    assert nwqsim._buffer_bytes("CPU", "DM", 1, 20, spec) == (16 * tib + 8 * mib + 8, 8192)
    assert nwqsim._buffer_bytes("AMDGPU", "DM", 1, 20, spec) == (48 * tib + 16, 32768)


def test_prepare_lowers_dense_unitaries_exactly(tmp_path, monkeypatch):
    # Qiskit's own U/CX lowering of this near-identity unitary errs by about
    # 1e-6 per entry. The expected matrix is SciPy's exponential.
    import numpy as np
    from scipy.linalg import expm
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import Operator

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    rng = np.random.default_rng(3)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    unitary = expm(-1e-6j * (x + x.conj().T) / 2)
    circuit = QuantumCircuit(2, 2)
    circuit.append(UnitaryGate(unitary), [0, 1])
    circuit.measure([0, 1], [0, 1])
    prepared = configured(tmp_path).prepare(
        circuit, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=23),
        position=len(circuit.data), source_definitions=(), snapshot="dense", run=IO())
    body = prepared.native.circuit.remove_final_measurements(inplace=False)
    assert np.abs(Operator(body).data - unitary).max() <= 1e-13
    assert prepared.transformation.startswith("exact dense-unitary synthesis")


def test_prepare_admits_dense_synthesis_against_the_run_limit(tmp_path, monkeypatch):
    # U/CX lowering synthesizes the three-qubit unitary exactly. One work
    # unit below its law, the Run's max_synthesis_work refuses the
    # preparation before the synthesis starts.
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    from qiskit.quantum_info import random_unitary
    from nwqlib.execution import ExecutionLimits
    from nwqlib.subroutines import _dense_synthesis

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    circuit = QuantumCircuit(3, 3)
    circuit.append(UnitaryGate(random_unitary(8, seed=37).data), [0, 1, 2])
    circuit.measure([0, 1, 2], [0, 1, 2])
    from test_dense_synthesis import _dense_synthesis_work
    need = _dense_synthesis_work(3)
    started = []
    monkeypatch.setattr(_dense_synthesis, "dense_unitary_circuit",
                        lambda matrix: started.append(np.shape(matrix)) or pytest.fail("synthesis started"))

    class Limited(IO):
        limits = ExecutionLimits(max_data_bytes=1_000_000, max_synthesis_work=need - 1)

    with pytest.raises(ValueError, match="max_synthesis_work"):
        configured(tmp_path).prepare(
            circuit, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=23),
            position=len(circuit.data), source_definitions=(), snapshot="dense", run=Limited())
    assert started == []


def exact_selection():
    import nwqlib
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.operators.inputs import ingest_pauli
    from nwqlib.problems import Expectation
    from nwqlib.problems.inputs import ingest_occupation

    problem = Expectation(state=ingest_occupation("10", num_qubits=2),
                          observable=ingest_pauli((("ZZ", 1.0),), num_qubits=2))
    return nwqlib.plan(problem, method=ExpectationMethod(), seed=23)


@pytest.mark.parametrize("level", [0, 1])
def test_non_default_optimization_level_marks_the_receipt_uncertified(tmp_path, monkeypatch, level):
    # Level 1 merges the two Rz rotations and cancels the CX pair: 2 U and 2 CX
    # at level 0, 1 U and 0 CX at level 1. The receipt counts the compiled
    # circuit, and the level-1 receipt carries the exclusion that leaves it
    # without a state error; the default receipt keeps its derived one.
    from qiskit import QuantumCircuit

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description(
        "CPU", "SV", roundoff_base_revision=REVISION, roundoff_sources_identical=True))
    config = configured(tmp_path, optimization_level=level)
    circuit = QuantumCircuit(2)
    circuit.rz(0.1, 0)
    circuit.rz(0.2, 0)
    circuit.cx(0, 1)
    circuit.cx(0, 1)
    prepared = config.prepare(
        circuit, observation=ObservationSpec(kind="probabilities", qubits=(0, 1)), runtime=RuntimeOptions(seed=23),
        position=len(circuit.data), source_definitions=(), snapshot="merge", run=IO())
    counts = dict(prepared.native.circuit.count_ops())
    assert (counts.get("u", 0), counts.get("cx", 0)) == ((2, 2) if level == 0 else (1, 0))
    assert prepared.probability_window_exclusions == (() if level == 0 else ("optimization_level",))
    assert prepared.compiler.domain.startswith(f"optimization_level={level};")
    selected = exact_selection()
    with Run(selected, backend=config, directory=tmp_path / "run") as run:
        handle = prepare(selected.resolve(selected.experiments[0].name), run=run, runtime=RuntimeOptions(seed=23))
    delta, reason = handle.record.state_error()
    assert (delta is None) == (level == 1)
    assert reason == (None if level == 0 else "outside the roundoff derivation: optimization_level")


def test_optimization_level_two_keeps_the_logical_circuit_and_refuses_relabeled_exact_readouts(
        tmp_path, monkeypatch):
    # Levels 2 and 3 resynthesize two-qubit blocks, so the dense unitary is not
    # synthesized exactly first. Without a coupling map they also elide a SWAP
    # and relabel the wires after it; counts follow the relabeled
    # measurements and record the final layout, but an exact readout in
    # logical wire order is refused. They also remove diagonal gates before
    # terminal measurements, which keeps every computational-basis
    # probability and marginal (|d_x| = 1, ``NWQSimBackend._lower``) but can
    # change relative phases. A probability readout of a measured circuit is
    # therefore accepted and matches level 0 on a case whose lowered state
    # differs beyond a global phase, and an amplitude readout is refused.
    from qiskit import QuantumCircuit
    from qiskit.circuit.library import UnitaryGate
    import numpy as np
    from qiskit.quantum_info import Statevector, random_unitary
    from nwqlib._validation import NUMERICAL_RELATION_RTOL
    from test_amplitude_masses import readout

    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description("CPU", "SV"))
    synthesized = []

    class Counted(IO):
        def _exact_dense_unitaries(self, circuit):
            synthesized.append(circuit)
            return IO._exact_dense_unitaries(self, circuit)

    config = configured(tmp_path, optimization_level=2)
    dense = QuantumCircuit(2, 2)
    dense.append(UnitaryGate(random_unitary(4, seed=5).data), [0, 1])
    dense.measure([0, 1], [0, 1])
    prepared = config.prepare(
        dense, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=23),
        position=len(dense.data), source_definitions=(), snapshot="dense", run=Counted())
    assert synthesized == [] and prepared.transformation.startswith("Qiskit U/CX lowering")
    assert "optimization_level" in prepared.probability_window_exclusions
    swapped = QuantumCircuit(2, 2)
    swapped.x(0)
    swapped.swap(0, 1)
    swapped.measure([0, 1], [0, 1])
    counts = config.prepare(
        swapped, observation=ObservationSpec(kind="counts", shots=4), runtime=RuntimeOptions(seed=23),
        position=len(swapped.data), source_definitions=(), snapshot="swap", run=IO())
    native = counts.native.circuit
    measured = {native.find_bit(item.clbits[0]).index: native.find_bit(item.qubits[0]).index
                for item in native.data if item.operation.name == "measure"}
    flipped = {native.find_bit(item.qubits[0]).index for item in native.data if item.operation.name == "u"}
    assert flipped == {measured[1]} and counts.logical_to_native == (1, 0)
    body = QuantumCircuit(2, 2)
    body.h(0)
    body.h(1)
    body.cz(0, 1)
    body.rz(0.7, 0)
    # The requested probability bits need not be the measured subset.
    for measured_wires, qubits in (((0, 1), (0, 1)), ((0,), (0,)), ((0,), (1, 0))):
        circuit = body.copy()
        circuit.measure(list(measured_wires), list(measured_wires))
        states = {}
        for level in (0, 2):
            prepared = configured(tmp_path, optimization_level=level).prepare(
                circuit, observation=ObservationSpec(kind="probabilities", qubits=qubits),
                runtime=RuntimeOptions(seed=23), position=len(circuit.data), source_definitions=(),
                snapshot="measured", run=IO())
            states[level] = Statevector(prepared.native.circuit.remove_final_measurements(inplace=False))
        assert not states[2].equiv(states[0])
        np.testing.assert_allclose(states[2].probabilities(list(qubits)), states[0].probabilities(list(qubits)),
                                   rtol=0, atol=NUMERICAL_RELATION_RTOL)
    amplitudes = readout(width=2, coordinates=(0, 1), success=(), conditions=())
    with pytest.raises(ValueError, match="removes diagonal gates before measurements. Amplitude readouts"):
        config.prepare(
            dense, observation=ObservationSpec(kind="amplitudes", amplitudes=amplitudes),
            runtime=RuntimeOptions(seed=23), position=len(dense.data), source_definitions=(),
            snapshot="dense", run=IO())
    for _ in range(2):
        # A measured and then an unmeasured relabeled circuit both reach the layout refusal.
        with pytest.raises(ValueError, match="relabeled wires"):
            config.prepare(
                swapped, observation=ObservationSpec(kind="probabilities", qubits=(0, 1)),
                runtime=RuntimeOptions(seed=23), position=len(swapped.data), source_definitions=(),
                snapshot="swap", run=IO())
        swapped.remove_final_measurements()


@pytest.mark.parametrize("level", [0, 1])
def test_trajectory_boundaries_survive_the_selected_optimization_level(tmp_path, monkeypatch, level):
    # The boundary barriers keep every point's native position at level 1,
    # and a level other than 0 marks the trajectory receipt with the
    # exclusion "optimization_level".
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.core.planning import ObservationPoint, ReadoutView
    from test_observation_schedule import pauli, trajectory, view_plan

    monkeypatch.setattr(ExpectationMethod, "save_archive", lambda self, plan, files: {"format": "unreopened"})
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda *args, **kwargs: description(
        "CPU", "SV", roundoff_base_revision=REVISION, roundoff_sources_identical=True))
    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0, 1))
    plan = view_plan(trajectory(
        ObservationPoint(id="viewed", position=1, kind="probabilities", qubits=(0, 1), view=view),
        pauli("middle", 2, "ZZ", "XI"),
        ObservationPoint(id="end", kind="probabilities", qubits=(1, 0), view=view)))
    config = configured(tmp_path, optimization_level=level)
    with Run(plan, backend=config, directory=tmp_path / "run", progress=False) as run:
        receipt = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7)).record
    assert receipt.boundaries == (1, 2, 3)
    assert receipt.probability_window_exclusions == (() if level == 0 else ("optimization_level",))
    assert (receipt.state_error()[0] is None) == (level == 1)


@pytest.mark.parametrize("level", [0, 1])
@pytest.mark.parametrize("reduced", [True, False], ids=["reduction", "native"])
def test_trajectory_receipt_lists_amplitude_derived_masses_only_with_a_reduction_point(
        tmp_path, monkeypatch, reduced, level):
    # A reduction save hands the saved amplitudes to a host reducer, so the
    # trajectory receipt lists "amplitude-derived masses". A trajectory of
    # only probability and Pauli points carries no such label. Resolving that
    # label, as the projected kernel does, leaves the receipt's state budget
    # available at the default optimization level, and the level-1 marker
    # still keeps it unavailable. The final-boundary reduction state is
    # multiplied by the runner's reported output phase factor, and the
    # receipt charges that factor's envelope; a trajectory without such a
    # state has no product and the assessed zero envelope.
    import numpy as np
    from nwqlib._phase_product import phase_product_envelope
    from nwqlib._quantum_readout import PROJECTED_MASS_EXCLUSIONS
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.core import planning
    from nwqlib.core.planning import Reducer, ReducerOutput
    from test_observation_schedule import pauli, reduction, three_call_plan, trajectory

    monkeypatch.setattr(ExpectationMethod, "save_archive", lambda self, plan, files: {"format": "unreopened"})
    monkeypatch.setattr(ExpectationMethod, "reduction_allowance",
                        lambda self, plan, point, *, observation, width, run: 1, raising=False)
    factor, queries = (0.7443505388340905, 0.6677890949524401), []
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda argv, **kwargs: _phase_factor_reply(argv, factor, queries))
    monkeypatch.setitem(planning.READOUT_REDUCERS, "norm_first", Reducer(
        shape=lambda parameters: (ReducerOutput("float64", (1,)),), work=lambda parameters, width: 1,
        execute=lambda state, parameters, bindings: (np.ones(1),)))
    end = reduction("end", None) if reduced else pauli("end", None, "ZZ")
    plan = three_call_plan(trajectory(pauli("first", 1, "ZZ"), end))
    config = configured(tmp_path, optimization_level=level)
    with Run(plan, backend=config, directory=tmp_path / "run", progress=False) as run:
        handle = prepare(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
    receipt, request = handle.record, json.loads(handle._native.payload)
    # One query, sent with the lowered circuit's finite global phase, and its
    # pair on the final-boundary reduction state only.
    assert len(queries) == (1 if reduced else 0) and all(isfinite(json.loads(angle)) for angle in queries)
    assert [point["phase_factor"] for point in request["observation"]["points"]] == [
        None, list(factor) if reduced else None]
    expected = (PROJECTED_MASS_EXCLUSIONS if reduced else ()) + (("optimization_level",) if level else ())
    assert receipt.probability_window_exclusions == expected
    assert (receipt.state_error()[0] is None) == (reduced or level == 1)
    assert (receipt.state_error(PROJECTED_MASS_EXCLUSIONS)[0] is None) == (level == 1)
    assert receipt.statevector_roundoff == (
        phase_product_envelope(complex(*factor), 1 << len(receipt.logical_to_native)) if reduced else (0.0, 0.0))
    assert (receipt.saved_state_error(PROJECTED_MASS_EXCLUSIONS)[0] is None) == (level == 1)


@pytest.mark.parametrize("factor", [(0.7443505388340905, 0.6677890949524401), (1.0, -0.0)],
                         ids=["general", "identity"])
def test_amplitude_endpoint_request_carries_and_charges_the_runner_phase_factor(tmp_path, monkeypatch, factor):
    # An amplitude endpoint is multiplied by the output phase factor that the
    # runner reports during preparation. The request carries that pair and
    # the receipt's statevector_roundoff is its envelope
    # (_phase_product.phase_product_envelope). A represented (1.0, -0.0)
    # factor is an exact identity product with the assessed zero envelope.
    from qiskit import QuantumCircuit
    from nwqlib._phase_product import phase_product_envelope
    from test_amplitude_masses import readout

    queries = []
    monkeypatch.setattr(nwqsim.subprocess, "run", lambda argv, **kwargs: _phase_factor_reply(argv, factor, queries))
    circuit = QuantumCircuit(2, global_phase=0.25)
    circuit.h(0)
    prepared = configured(tmp_path).prepare(
        circuit, observation=ObservationSpec(kind="amplitudes",
                                             amplitudes=readout(width=2, coordinates=(0, 1), success=(),
                                                                conditions=())),
        runtime=RuntimeOptions(seed=23), position=len(circuit.data), source_definitions=(), snapshot="amplitudes",
        run=IO())
    request = json.loads(prepared.native.payload)
    assert len(queries) == 1 and request["phase_factor"] == list(factor)
    assert prepared.statevector_roundoff == (
        (0.0, 0.0) if factor == (1.0, -0.0) else phase_product_envelope(complex(*factor), 4))
