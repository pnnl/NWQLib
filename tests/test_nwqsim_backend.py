"""Optional real NWQ-Sim qualification; analytical oracles need no second simulator."""

import json
from math import cos, pi, sin, sqrt
import os
from pathlib import Path
import struct
import subprocess
import sys

import pytest


@pytest.fixture
def runner():
    executable = os.environ.get("NWQLIB_TEST_NWQSIM_EXECUTABLE")
    if executable is None:
        pytest.skip("set NWQLIB_TEST_NWQSIM_EXECUTABLE to a qualified native build")
    return str(Path(executable).resolve(strict=True))


def _request(*, qubits, gates, kind, measured=(), classical=(), labels=(), phase_factor=(1.0, 0.0), shots=0):
    return dict(
        format="nwqlib.nwqsim/3", input_id="analytical-native-witness", backend="CPU", method="SV", ranks=1,
        num_qubits=qubits, num_clbits=4 if kind == "counts" else 0,
        gates=[dict(name=name, qubits=wires, params=parameters) for name, wires, parameters in gates],
        phase_factor=list(phase_factor) if kind == "amplitudes" else None, seed=17, shots=shots, max_buffer_bytes=1024 * 1024,
        max_output_bytes=1024 * 1024,
        observation=dict(kind=kind, qubits=list(measured), clbits=list(classical), labels=list(labels)),
    )


def _execute(runner, tmp_path, request, *, succeeds=True):
    payload = json.dumps(request).encode()
    source, result = tmp_path / "input.json", tmp_path / "result.json"
    source.write_bytes(payload)
    completed = subprocess.run(
        [runner, str(source), str(result), str(len(payload))], capture_output=True, text=True,
        check=False,
    )
    assert (completed.returncode == 0) is succeeds, completed.stderr
    data = json.loads(result.read_bytes())
    assert data["input_id"] == request["input_id"]
    return data, result


def test_native_counts_project_wires_and_preserve_unwritten_classical_bits(runner, tmp_path):
    request = _request(
        qubits=3, gates=[("u", [0], [pi, 0, pi]), ("u", [1], [pi, 0, pi])],
        kind="counts", measured=(2, 0), classical=(0, 2), shots=64,
    )
    data, result = _execute(runner, tmp_path, request)
    # |q2 q1 q0> = |011>; only q0 reaches classical bit 2. q1 is unrequested.
    assert data["counts"] == {"0100": 64}
    assert data["native_simulations"] == 1
    previous = result.read_bytes()
    rerun = subprocess.run(
        [runner, str(tmp_path / "input.json"), str(result), "1048576"],
        capture_output=True, text=True, check=False,
    )
    assert rerun.returncode != 0
    assert result.read_bytes() == previous


def test_native_pauli_complex_bell_state_uses_y_phase_and_control_wire(runner, tmp_path):
    request = _request(
        qubits=2, gates=[("u", [0], [pi / 2, pi / 2, 0]), ("cx", [0, 1], [])],
        kind="pauli", labels=("II", "XY", "YX", "ZZ", "XX", "YY", "IZ"),
    )
    data, _ = _execute(runner, tmp_path, request)
    # (|00> + i|11>)/sqrt(2): XY=YX=ZZ=1, XX=YY=IZ=0. The non-real
    # phase distinguishes conjugation/Y-sign errors and a reversed CX.
    expected = dict(II=1, XY=1, YX=1, ZZ=1, XX=0, YY=0, IZ=0)
    assert data["pauli"] == pytest.approx(expected, rel=0, abs=1e-12)
    assert data["native_simulations"] == 1


def _marginal(result, data, bins):
    """The dense float64 marginal sidecar the result names, checked against its descriptor."""
    import numpy as np
    artifact = data["probabilities"]
    path = result.parent / artifact["file"]
    assert (artifact["dtype"], artifact["count"], artifact["bytes"]) == ("float64-native", bins, 8 * bins)
    assert path.stat().st_size == 8 * bins
    return np.fromfile(path, dtype=np.float64)


def test_native_probability_order_is_the_requested_register_order(runner, tmp_path):
    """Marginal bins follow the requested register: bit zero is the first observed qubit.

    The identity-ordered proper prefix sums unobserved high bits into bin
    x & (2**k - 1); a permuted register gathers each observed bit. Both are
    compared with qiskit.quantum_info.Statevector probabilities of the same
    U/CX circuit (qargs[0] is its least significant bit) at 1e-12.
    """
    import numpy as np
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Statevector

    request = _request(
        qubits=2, gates=[("u", [0], [pi / 3, 0, 0]), ("u", [1], [2 * pi / 3, 0, 0])],
        kind="probabilities", measured=(1, 0),
    )
    (tmp_path / "two").mkdir()
    data, result = _execute(runner, tmp_path / "two", request)
    # P(q0=1)=1/4 and P(q1=1)=3/4, independently. The requested register
    # has q1 at its least-significant position, which swaps the unequal bins.
    assert _marginal(result, data, 4) == pytest.approx([3 / 16, 9 / 16, 1 / 16, 3 / 16], rel=0, abs=1e-12)

    angles = np.random.default_rng(29).uniform(-pi, pi, size=(6, 3))
    gates = [("u", [wire], [float(value) for value in angles[wire]]) for wire in range(6)]
    gates += [("cx", [wire, (wire + 1) % 6], []) for wire in range(6)]
    gates += [("u", [wire], [float(angles[wire, 1]), 0.0, float(angles[wire, 2])]) for wire in (0, 3, 5)]
    circuit = QuantumCircuit(6)
    for name, wires, parameters in gates:
        if name == "u":
            circuit.u(*parameters, wires[0])
        else:
            circuit.cx(*wires)
    state = Statevector(circuit)
    for label, measured in (("prefix", (0, 1, 2, 3)), ("permuted", (5, 1, 3))):
        (tmp_path / label).mkdir()
        data, result = _execute(runner, tmp_path / label, _request(qubits=6, gates=gates, kind="probabilities",
                                                                    measured=measured))
        expected = state.probabilities(list(measured))
        np.testing.assert_allclose(_marginal(result, data, 1 << len(measured)), expected, rtol=0, atol=1e-12)
        assert data["native_simulations"] == 1


def test_native_amplitudes_keep_global_phase_without_a_second_evolution(runner, tmp_path):
    request = _request(
        qubits=1, gates=[("u", [0], [pi / 2, 0, 0])], kind="amplitudes", phase_factor=(cos(pi / 2), sin(pi / 2)),
    )
    data, result = _execute(runner, tmp_path, request)
    artifact = result.parent / data["amplitudes"]["file"]
    values = struct.unpack("=4d", artifact.read_bytes())
    assert values == pytest.approx((0, 1 / sqrt(2), 0, 1 / sqrt(2)), rel=0, abs=1e-12)
    assert data["amplitudes"]["bytes"] == 32
    assert data["native_simulations"] == 1


def test_native_output_admission_precedes_evolution(runner, tmp_path):
    request = _request(qubits=5, gates=[], kind="amplitudes")
    # A complete 32-amplitude artifact alone needs 512 bytes, before metadata.
    request["max_output_bytes"] = 256
    data, _ = _execute(runner, tmp_path, request, succeeds=False)
    assert data["native_simulations"] == 0
    assert "output envelope" in data["error"]
    assert not list(tmp_path.glob("*.amplitudes"))


@pytest.mark.parametrize("initial_one", [False, True])
def test_native_u_composes_finite_phases_without_overflow(runner, tmp_path, initial_one):
    if initial_one:
        gates = [("u", [0], [pi, 0, 0]), ("u", [0], [0, 0.31, -0.73])]
        expected = (0, 0, cos(-0.42), sin(-0.42))
    else:
        # Exactly |0> for every finite phi and lambda. NaN * 0 from the other
        # matrix entry still contaminates the answer in the unpatched kernel.
        gates = [("u", [0], [0, sys.float_info.max, sys.float_info.max])]
        expected = (1, 0, 0, 0)
    data, result = _execute(runner, tmp_path, _request(qubits=1, gates=gates, kind="amplitudes"))
    values = struct.unpack("=4d", (result.parent / data["amplitudes"]["file"]).read_bytes())
    assert values == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.parametrize("collision", ["amplitudes", "claim"])
def test_native_existing_artifact_or_active_claim_rejects_before_input(runner, tmp_path, collision):
    result = tmp_path / "result.json"
    owned = tmp_path / ("result.json." + collision)
    if collision == "claim":
        owned.mkdir()
        (owned / "owner").write_bytes(b"other attempt")
    else:
        owned.write_bytes(b"previous amplitudes")
    # Missing input distinguishes an early ownership refusal from later
    # validation. The other attempt's artifact/claim must stay untouched.
    completed = subprocess.run(
        [runner, str(tmp_path / "missing.json"), str(result), "1048576"],
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    assert "already" in completed.stderr
    assert not result.exists()
    if collision == "claim":
        assert (owned / "owner").read_bytes() == b"other attempt"
    else:
        assert owned.read_bytes() == b"previous amplitudes"
        assert not (tmp_path / "result.json.claim").exists()


@pytest.mark.skipif(os.name != "posix", reason="native runner qualification targets POSIX")
@pytest.mark.parametrize("file_limit", [300, 0])
def test_native_write_failure_cleans_only_this_attempt_and_preserves_outcome(runner, tmp_path, file_limit):
    import resource
    import signal

    request = _request(qubits=5, gates=[], kind="amplitudes")
    source, result = tmp_path / "input.json", tmp_path / "result.json"
    source.write_text(json.dumps(request))
    unrelated = tmp_path / "result.json.amplitudes.partial"
    unrelated.write_bytes(b"unrelated previous file")

    def limit_file_writes():
        signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
        resource.setrlimit(resource.RLIMIT_FSIZE, (file_limit, file_limit))

    completed = subprocess.run(
        [runner, str(source), str(result), "1048576"], preexec_fn=limit_file_writes,
        capture_output=True, text=True, check=False,
    )
    assert completed.returncode != 0
    if file_limit:
        data = json.loads(result.read_bytes())
        assert data["status"] == "failed"
        assert data["native_simulations"] == 1
        assert not (tmp_path / "result.json.claim").exists()
    else:
        assert not result.exists()
        claim = tmp_path / "result.json.claim"
        assert claim.is_dir() and not list(claim.iterdir())
        rerun = subprocess.run([runner, str(source), str(result), "1048576"],
                               capture_output=True, text=True, check=False)
        assert rerun.returncode != 0 and "already claimed" in rerun.stderr
    assert not (tmp_path / "result.json.amplitudes").exists()
    assert unrelated.read_bytes() == b"unrelated previous file"


def test_native_trajectory_is_one_evolution_that_restores_each_view(runner, tmp_path, monkeypatch):
    """One NWQ-Sim trajectory through the public detached workflow observes every point of one evolution.

    Points: a viewed two-qubit marginal at boundary 1, Pauli labels at boundary
    2 and a viewed marginal in reversed qubit order at the end. The first
    view's inverse must restore the state before boundary 2. Each point is
    compared with its separately constructed logical prefix
    (qiskit.quantum_info), the view's tail applied to it where the point has a
    view, every outcome included. The comparison uses the predeclared small-fixture
    regression threshold atol=2e-12, rtol=0: a regression threshold at two
    qubits, not a receipt bound; the independently evaluated prefix has no
    NWQ-Sim receipt. The Expectation archive stores built-in blocks only, and
    this Run is never reopened, so the test stubs its archive writer to hold
    the adjoint view block.
    """
    import numpy as np
    from nwqlib.algorithms.expectation import ExpectationMethod
    from qiskit.quantum_info import Pauli, Statevector
    from nwqlib._prepared_execution import Run, prepare_experiment, refresh_submissions, submit_detached
    from nwqlib.backends import NWQSimBackend
    from nwqlib.blocks.lowering import _lower_qiskit
    from nwqlib.core.planning import ObservationPoint, ReadoutView, RuntimeOptions
    from test_observation_schedule import pauli, trajectory, view_plan

    monkeypatch.setattr(ExpectationMethod, "save_archive", lambda self, plan, files: {"format": "unreopened"})
    decoded, decode = [], NWQSimBackend._result

    def recording_decode(self, *args, **kwargs):
        result = decode(self, *args, **kwargs)
        decoded.append(result.raw_output["trajectory"])
        return result

    monkeypatch.setattr(NWQSimBackend, "_result", recording_decode)
    view = ReadoutView(tail="tail", inverse="tail_inverse", wires=(0, 1))
    plan = view_plan(trajectory(
        ObservationPoint(id="viewed", position=1, kind="probabilities", qubits=(0, 1), view=view),
        pauli("middle", 2, "ZZ", "XI", "YY", "IZ"),
        ObservationPoint(id="end", kind="probabilities", qubits=(1, 0), view=view)))
    _, construction = plan.resolve("trajectory")._selected_construction(plan)
    selected = {record.content_id for record in construction.selections}
    blocks = tuple(block for block in plan.blocks if block.record.content_id in selected)
    body = _lower_qiskit(construction, definition=plan.construction.program.root, blocks=blocks).circuit
    tail = _lower_qiskit(construction, definition="tail", blocks=blocks).circuit
    backend = NWQSimBackend(executable=runner, spool=str(tmp_path / "spool"), max_input_bytes=1 << 20,
                            max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)
    with Run(plan, backend=backend, directory=tmp_path / "run", progress=False) as run:
        handle = prepare_experiment(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        receipt = handle.record
        submission = submit_detached((handle,), run=run)
        run._state["backend_context"]["processes"][submission.submission_id].wait()
        chunks = {chunk.point: chunk for chunk in refresh_submissions(run=run)}
        (record,) = run.trace.submissions
        assert record.native_simulations == 1 and run.trace.jobs == 1
        assert [chunk.boundary for chunk in chunks.values()] == list(receipt.boundaries) == [1, 2, 3]
    # The adapter hands each marginal over as the dense float64 buffer it
    # read, 2**k values with outcome j at position j, not as a mapping.
    (returned,) = decoded
    for point in ("viewed", "end"):
        marginal = returned[point]["probabilities"]
        assert type(marginal) is np.ndarray and marginal.dtype == np.float64 and marginal.shape == (4,)

    def prefix(boundary):
        circuit = body.copy_empty_like()
        circuit.data = body.data[:boundary]
        return Statevector(circuit)

    for point, boundary, qubits in (("viewed", 1, [0, 1]), ("end", 3, [1, 0])):
        expected = prefix(boundary).evolve(tail).probabilities(qubits)
        histogram = chunks[point].histogram()
        observed = dict(zip(histogram.indices().tolist(), histogram.weights.tolist(), strict=True))
        np.testing.assert_allclose([observed.get(index, 0.) for index in range(4)], expected, atol=2e-12, rtol=0)
    middle = prefix(2)
    for item in chunks["middle"].values:
        np.testing.assert_allclose(item.value, middle.expectation_value(Pauli(item.label)).real, atol=2e-12, rtol=0)


def test_native_trajectory_saved_states_carry_the_prefix_phase_only_at_the_last_boundary(runner, tmp_path,
                                                                                            monkeypatch):
    """Reductions read the saved NWQ-Sim state of their point through the detached workflow.

    A phase-invariant reducer (the squared norm and |psi_0|**2) runs before the
    last boundary on the raw saved state. A phase-sensitive reducer (psi_0)
    runs at the last boundary, whose saved state is multiplied, as an output
    copy, by the phase of the lowered prefix; the complex prepared state makes
    that phase nonzero. Both are compared with the
    separately constructed logical prefix (qiskit.quantum_info.Statevector),
    whose global phase is the logical circuit's, at the predeclared
    small-fixture regression threshold atol=2e-12, rtol=0 (two qubits).
    """
    import numpy as np
    from qiskit.quantum_info import Statevector
    from nwqlib._prepared_execution import Run, prepare_experiment, refresh_submissions, submit_detached
    from nwqlib.algorithms.expectation import ExpectationMethod
    from nwqlib.backends import NWQSimBackend
    from nwqlib.blocks.lowering import lower_qiskit
    from nwqlib.core import planning
    from nwqlib.core.planning import ObservationPoint, Reducer, ReducerOutput, RuntimeOptions
    from test_observation_schedule import three_call_plan, trajectory

    def invariant(state, parameters, bindings):
        return np.array([np.vdot(state, state).real]), np.array([abs(state[0]) ** 2])

    def first(state, parameters, bindings):
        return (np.array([state[0]]),)

    monkeypatch.setitem(planning.READOUT_REDUCERS, "norm_mass", Reducer(
        shape=lambda parameters: (ReducerOutput("float64", (1,)), ReducerOutput("float64", (1,))),
        work=lambda parameters, width: 2 << width, execute=invariant, phase_invariant=True))
    monkeypatch.setitem(planning.READOUT_REDUCERS, "first_amplitude", Reducer(
        shape=lambda parameters: (ReducerOutput("complex128", (1,)),),
        work=lambda parameters, width: 2 << width, execute=first))
    monkeypatch.setattr(ExpectationMethod, "save_archive", lambda self, plan, files: {"format": "unreopened"})
    plan = three_call_plan(trajectory(
        ObservationPoint(id="middle", position=1, kind="reduction", reducer="norm_mass", parameters={}),
        ObservationPoint(id="end", kind="reduction", reducer="first_amplitude", parameters={})),
        state=(1j, 2., 3., -4j))
    backend = NWQSimBackend(executable=runner, spool=str(tmp_path / "spool"), max_input_bytes=1 << 20,
                            max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)
    with Run(plan, backend=backend, directory=tmp_path / "run", progress=False) as run:
        handle = prepare_experiment(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7),
                                    reduction_allowance=4 * (2 << 2))
        submission = submit_detached((handle,), run=run)
        run._state["backend_context"]["processes"][submission.submission_id].wait()
        chunks = {chunk.point: chunk for chunk in refresh_submissions(run=run)}
    _, construction = plan.resolve("trajectory")._selected_construction(plan)
    selected = {record.content_id for record in construction.selections}
    body = lower_qiskit(construction, blocks=tuple(block for block in plan.blocks
                                                   if block.record.content_id in selected)).circuit

    def prefix(boundary):
        circuit = body.copy_empty_like()
        circuit.data = body.data[:boundary]
        return Statevector(circuit).data

    middle = prefix(1)
    norm, mass = (item.real[0] for item in chunks["middle"].values)
    np.testing.assert_allclose([norm, mass], [np.vdot(middle, middle).real, abs(middle[0]) ** 2], atol=2e-12, rtol=0)
    (value,) = chunks["end"].values
    np.testing.assert_allclose(complex(value.real[0], value.imaginary[0]), prefix(3)[0], atol=2e-12, rtol=0)


def test_native_trajectory_admits_its_completion_rows_before_the_runner_starts(runner, tmp_path, monkeypatch):
    """A fresh NWQ-Sim trajectory is admitted only when its detached outcome rows fit; a reopened Run still fetches.

    Without a forecast the early condition is ``58*K + 168 <= M`` for K Pauli
    points: each point's 36-character UUID job and empty values envelope
    costs 58 bytes (``NWQSimBackend.native_job_id_length``), and the fresh
    outcome transaction grows by at most 168 bytes
    (``_prepared_execution.detached_completion_growth``). One byte below it
    the Run refuses before the runner starts, and at it the result publishes.
    A Run launched under a smaller minimum and reopened is not refused by the
    fresh-event minimum at fetch: its outcome is checked against the actual
    rows at publication and publishes once the allowance is extended.
    """
    from nwqlib import _prepared_execution
    from nwqlib._prepared_execution import Run, prepare_experiment, refresh_submissions, submit_detached
    from nwqlib._run_archive import load as load_run
    from nwqlib.backends import NWQSimBackend
    from nwqlib.core.planning import RuntimeOptions
    from nwqlib.execution import ExecutionLimits
    from test_observation_schedule import pauli, three_call_plan, trajectory

    plan = three_call_plan(trajectory(pauli("a", 1, "ZZ", "XY"), pauli("b", 2, "XX"), pauli("end", None, "ZI")),
                           state=(1., 2j, -3., 4.))
    required = 58 * 3 + 168
    backend = NWQSimBackend(executable=runner, spool=str(tmp_path / "spool"), max_input_bytes=1 << 20,
                            max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)

    def run_at(directory, allowance):
        return Run(plan, backend=backend, directory=tmp_path / directory, progress=False,
                   limits=ExecutionLimits(max_completion_metadata_bytes=allowance))

    def launch(run):
        handle = prepare_experiment(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
        submission = submit_detached((handle,), run=run)
        run._state["backend_context"]["processes"][submission.submission_id].wait()

    with run_at("short", required - 1) as run:
        with pytest.raises(ValueError, match=f"need at least {required} bytes, more than "
                                             f"max_completion_metadata_bytes={required - 1}, before native work"):
            launch(run)
        assert run.trace.jobs == 0
    assert not (tmp_path / "spool").exists() or not any((tmp_path / "spool").iterdir())
    with run_at("exact", required) as run:
        launch(run)
        assert len(refresh_submissions(run=run)) == 3
    # A Run launched while the job text was priced at one character and the
    # completion rows at nothing, below the point envelopes 58*K of a fetch.
    launched = 58 * 3 - 1
    with monkeypatch.context() as patch, run_at("reopened", launched) as run:
        patch.setattr(NWQSimBackend, "native_job_id_length", None)
        patch.setattr(_prepared_execution, "_completion_minimum", lambda run, max_bytes: 0)
        launch(run)
    with load_run(tmp_path / "reopened", backend=backend) as reopened:
        with pytest.raises(ValueError, match=f"completion metadata exceeds.*{launched} allowed"):
            refresh_submissions(run=reopened)
        reopened.extend_limits(max_completion_metadata_bytes=required)
        assert len(refresh_submissions(run=reopened)) == 3


def test_native_probability_endpoint_admits_its_completion_rows_before_the_runner_starts(runner, tmp_path, monkeypatch):
    """A single-endpoint NWQ-Sim probability readout reserves its detached outcome rows before the runner starts.

    The endpoint's metadata check (``_probability_endpoint_minimum``) adds
    the fresh outcome transaction's upper growth, 168 bytes without a
    forecast (``_prepared_execution.detached_completion_growth``), to the
    endpoint's own chunk metadata, so its submission requirement lies 168
    bytes above the one without completion rows. One byte below it the Run
    refuses before the runner starts, and at it the result publishes.
    """
    import re

    from nwqlib import _prepared_execution
    from nwqlib._prepared_execution import Run, prepare_experiment, refresh_submissions, submit_detached
    from nwqlib.backends import NWQSimBackend
    from nwqlib.core.planning import ObservationSpec, RuntimeOptions
    from nwqlib.execution import ExecutionLimits
    from test_observation_schedule import three_call_plan

    plan = three_call_plan(ObservationSpec(kind="probabilities", qubits=(0, 1)), state=(1., 2j, -3., 4.))
    spool = tmp_path / "spool"
    backend = NWQSimBackend(executable=runner, spool=str(spool), max_input_bytes=1 << 20,
                            max_output_bytes=1 << 20, max_buffer_bytes=1 << 20)
    runs = iter(range(1 << 10))

    def attempt(allowance):
        """The requirement named by a refusal before native work, or the published chunks."""
        with Run(plan, backend=backend, directory=tmp_path / f"run{next(runs)}", progress=False,
                 limits=ExecutionLimits(max_completion_metadata_bytes=allowance)) as run:
            try:
                handle = prepare_experiment(plan.resolve("trajectory"), run=run, runtime=RuntimeOptions(seed=7))
                submission = submit_detached((handle,), run=run)
            except ValueError as error:
                assert run.trace.jobs == 0
                assert not spool.exists() or not any(spool.iterdir())
                return int(re.search(r"probability endpoint metadata and completion rows need at least "
                                     rf"(\d+) bytes, more than max_completion_metadata_bytes={allowance}, "
                                     "before native work", str(error)).group(1))
            run._state["backend_context"]["processes"][submission.submission_id].wait()
            return refresh_submissions(run=run)

    def submission_requirement():
        # Preparation names the requirement before lowering; at it, lowering
        # runs and the submission names the requirement of the actual receipt.
        late = attempt(attempt(1))
        assert isinstance(late, int), "the requirement before lowering already admitted the submission"
        return late

    with monkeypatch.context() as patch:
        patch.setattr(_prepared_execution, "_completion_minimum", lambda run, max_bytes: 0)
        without_completion_rows = submission_requirement()
    required = submission_requirement()
    assert required == without_completion_rows + 168
    assert attempt(required - 1) == required
    assert len(attempt(required)) == 1


def test_native_exact_lchs_completes_at_a_non_default_optimization_level(runner, tmp_path, monkeypatch):
    """An LCHS exact reduction on NWQ-Sim at level 1 completes under the omega convention.

    Both trajectory receipts list "amplitude-derived masses", and the
    projected kernel resolves only that label. At level 0 on a qualified
    runner the resolved saved-state budget delta (``saved_state_error``,
    which includes the envelope of the runner's output phase product) is
    available, and every
    LCHS mass check (``_quantum_readout.validate_saved_masses``) at
    acquisition and publication receives it. At level 1 the exclusion
    "optimization_level" remains after that resolution, so every check
    receives no delta and uses the propagated probability window omega. The value is
    compared with the default-level NWQ-Sim value at the predeclared
    small-fixture regression threshold atol=1e-12, not a receipt bound.
    """
    import numpy as np
    import nwqlib._quantum_readout as readout
    from nwqlib import LinearDynamics, NormalizedExpectation, plan, prepare, submit
    from nwqlib.algorithms.lchs import LCHS, ProviderConfig
    from nwqlib.backends import NWQSimBackend
    from nwqlib.operators import ingest_pauli

    terms = (("ZI", .5), ("IZ", .3), ("XX", .2), ("YY", -.1))
    chosen = plan(LinearDynamics(A=np.diag([.1, .2, .3, .4]), initial_state=[1., 2., 3., 4.], time=.1),
                  method=LCHS(hamiltonian_evolution_backend="trotter", k_quadrature=ProviderConfig(
                      implementation="signed_binary_uniform", parameters={"num_qubits": 2, "lsb_position": -1})),
                  output=NormalizedExpectation(observable=ingest_pauli(terms, num_qubits=2)),
                  execution="quantum", seed=7)
    validate = readout.validate_saved_masses
    received = []

    def recording(*args, **kwargs):
        received.append(kwargs.get("delta"))
        return validate(*args, **kwargs)

    monkeypatch.setattr(readout, "validate_saved_masses", recording)
    values = []
    for level in (0, 1):
        received.clear()
        backend = NWQSimBackend(executable=runner, spool=str(tmp_path / f"spool{level}"), max_input_bytes=1 << 22,
                                max_output_bytes=1 << 22, max_buffer_bytes=1 << 22, optimization_level=level)
        # The detached Run keeps its folder under tmp_path, not the default run directory.
        prepared = prepare(chosen, backend=backend, progress=False, directory=tmp_path / f"run{level}")
        with prepared.run:
            result = submit(prepared).wait()
        (receipt,) = result.data.receipts
        assert ("optimization_level" in receipt.probability_window_exclusions) == (level == 1)
        assert "amplitude-derived masses" in receipt.probability_window_exclusions
        resolved = receipt.saved_state_error(("amplitude-derived masses",))[0]
        assert (resolved is None) == (level == 1)
        # One call at acquisition and one at publication, each with that budget.
        assert len(received) >= 2 and set(received) == {resolved}
        values.append(result.value)
    assert abs(values[1] - values[0]) <= 1e-12
