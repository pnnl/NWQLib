"""Tests for backend capabilities, Aer preparation and execution, and QASM export."""

from __future__ import annotations


import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes import Decompose
from qiskit_aer import AerSimulator

from nwqlib.backends import (
    AER_COUNTS_TARGET,
    AER_STATEVECTOR_TARGET,
    export_qasm,
)
import nwqlib.backends.qiskit_aer as qiskit_aer_backend
from nwqlib.execution import ExecutionMode


def _bell_circuit(*, measured: bool = False) -> QuantumCircuit:
    circuit = QuantumCircuit(2, 2 if measured else 0)
    circuit.h(0)
    circuit.cx(0, 1)
    if measured:
        circuit.measure([0, 1], [0, 1])
    return circuit


def _execute(target, circuit, *, seed=7, shots=None, **readout):
    """Prepare and submit one circuit through the shared Aer owners."""
    prepared = qiskit_aer_backend._prepare_aer_execution(target, circuit, shots=shots, seed=seed, **readout)
    return qiskit_aer_backend._submit_aer_execution(prepared)


def _nonzero_marginal(probabilities, width):
    """The nonzero entries of a dense marginal by bit-string key, the first observed qubit rightmost."""
    return {format(index, f"0{width}b"): float(value) for index, value in enumerate(probabilities) if value}


def test_aer_small_readouts_have_no_configured_zero_cutoff(monkeypatch):
    configurations = []

    def simulator(**settings):
        configurations.append(settings)
        return AerSimulator(**settings)

    monkeypatch.setattr(qiskit_aer_backend, "AerSimulator", simulator)
    angle = 2e-13
    circuit = QuantumCircuit(1)
    circuit.ry(angle, 0)
    result = _execute(AER_STATEVECTOR_TARGET, circuit, probability_qubits=(0,))
    # With Aer 0.17.2 the default zero_threshold of 1e-10 removed a
    # probability of 1e-26 from save_probabilities_dict, so the configured
    # value detects a deleted setting.
    assert configurations[0].get("zero_threshold") == 0.0
    actual, expected = result.raw_output["probabilities"][1], np.sin(angle/2)**2
    # A relative window compares the small nonzero probability with its
    # analytic one-gate value, which lies far below an ordinary absolute
    # tolerance.
    assert actual == pytest.approx(expected, rel=2e-14, abs=0.)


@pytest.mark.parametrize("amplitudes", ([0.5, 0.25j, -0.5, 0.5j], [1, 2, 3, 4], [0, 1j, 0, 0]))
def test_public_preparation_circuit_preserves_amplitudes_on_native_aer(amplitudes) -> None:
    from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation

    expected = np.asarray(amplitudes, dtype=complex)
    expected /= np.linalg.norm(expected)
    circuit = build_qiskit_state_preparation(amplitudes).circuit.copy()
    circuit.save_statevector()
    backend = AerSimulator(method="statevector")
    compiled = transpile(circuit, backend=backend, optimization_level=0)
    actual = np.asarray(backend.run(compiled, shots=1).result().get_statevector())
    # Independent amplitudes include global phase; fidelity alone would hide its loss.
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("name", ("measure", "barrier", "x"))
def test_aer_custom_instruction_names_preserve_definitions(monkeypatch, name) -> None:
    from qiskit.circuit import Gate

    definition = QuantumCircuit(1)
    definition.h(0)
    inner = Gate(name, 1, [])
    inner.definition = definition
    nested = QuantumCircuit(1)
    nested.append(inner, [0])
    gate = Gate(name, 1, [])
    gate.definition = nested
    circuit = QuantumCircuit(1, 1)
    circuit.append(gate, [0])
    circuit.measure(0, 0)
    calls = {"remove_final_measurements": 0, "decompose": 0, "run": 0}
    original_remove = qiskit_aer_backend._remove_final_measurements
    original_decompose = QuantumCircuit.decompose
    original_run = qiskit_aer_backend.AerSimulator.run

    def remove_once(circuit, **kwargs):
        calls["remove_final_measurements"] += 1
        return original_remove(circuit, **kwargs)

    def decompose_once(self, *args, **kwargs):
        calls["decompose"] += 1
        return original_decompose(self, *args, **kwargs)

    def run_once(self, *args, **kwargs):
        calls["run"] += 1
        return original_run(self, *args, **kwargs)

    monkeypatch.setattr(qiskit_aer_backend, "_remove_final_measurements", remove_once)
    monkeypatch.setattr(QuantumCircuit, "decompose", decompose_once)
    monkeypatch.setattr(qiskit_aer_backend.AerSimulator, "run", run_once)
    result = _execute(AER_STATEVECTOR_TARGET, circuit)
    # H|0> = |+>.
    expected = np.array([1., 1.]) / np.sqrt(2.)
    np.testing.assert_allclose(result.raw_output["statevector"], expected, rtol=0., atol=1.e-14)
    assert calls["remove_final_measurements"] <= 1
    assert calls["decompose"] <= 1
    assert calls["run"] == 1
    assert result.metadata["removed_final_measurements"] is True
    assert result.metadata["execution_measurement_count"] == 0
    assert result.metadata["statevector_is_measurement_conditioned"] is False
    assert circuit.data[0].operation.name == name
    assert circuit.data[0].operation.definition == nested


def test_aer_lowering_budget_rejects_hidden_native_names_at_the_owner(monkeypatch):
    from qiskit.circuit import Gate

    submissions = []
    original_run = AerSimulator.run

    def observe_run(self, circuit, **kwargs):
        submissions.append(circuit.count_ops())
        return original_run(self, circuit, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)
    circuit = QuantumCircuit(1)
    circuit.h(0)
    for _ in range(qiskit_aer_backend.AER_EXECUTION_DECOMPOSE_REPS):
        gate = Gate("x", 1, [])
        gate.definition = circuit
        circuit = QuantumCircuit(1)
        circuit.append(gate, [0], copy=False)
    # Exactly ten expansions expose H: reaching the budget is not itself failure.
    result = _execute(AER_STATEVECTOR_TARGET, circuit)
    np.testing.assert_allclose(result.raw_output["statevector"], np.ones(2) / np.sqrt(2), rtol=0., atol=1.e-14)
    assert result.metadata["execution_preprocessing"]["decompose_reps"] == qiskit_aer_backend.AER_EXECUTION_DECOMPOSE_REPS
    assert submissions[0].get("h") == 1 and "x" not in submissions[0]
    gate = Gate("x", 1, [])
    gate.definition = circuit
    hidden = QuantumCircuit(1)
    hidden.append(gate, [0], copy=False)
    # One extra wrapper used to dispatch as native X and silently return |1>.
    with pytest.raises(ValueError, match="Aer lowering requires native operations"):
        _execute(AER_STATEVECTOR_TARGET, hidden)
    with pytest.raises(ValueError, match="Aer lowering requires native operations"):
        _execute(AER_STATEVECTOR_TARGET, hidden, preparation=qiskit_aer_backend._AerPreparation())
    branch = QuantumCircuit(1, 1)
    with branch.if_test((branch.clbits[0], 0)):
        branch.append(gate, [0], copy=False)
    with pytest.raises(ValueError, match="Aer lowering requires native operations"):
        _execute(AER_STATEVECTOR_TARGET, branch)
    assert len(submissions) == 1


@pytest.mark.parametrize("qubits", [(2, 0)])
def test_native_probability_readout_preserves_order_and_small_population(qubits) -> None:
    circuit = QuantumCircuit(3)
    circuit.ry(2.0e-6, 0)
    circuit.h(1)
    circuit.cx(1, 2)
    reference = _execute(AER_STATEVECTOR_TARGET, circuit, seed=17)
    expected = {}
    for index, amplitude in enumerate(reference.raw_output["statevector"]):
        probability = abs(amplitude) ** 2
        if probability:
            bitstring = format(index, "03b")
            key = "".join(bitstring[-1 - qubit] for qubit in reversed(qubits))
            expected[key] = expected.get(key, 0.0) + probability
    result = _execute(AER_STATEVECTOR_TARGET, circuit, seed=17, probability_qubits=qubits)
    assert set(result.raw_output) == {"probabilities"}
    returned = _nonzero_marginal(result.raw_output["probabilities"], len(qubits))
    assert set(returned) == set(expected)
    for key, probability in expected.items():
        # The same state evolution and a marginal sum differ only by roundoff.
        assert returned[key] == pytest.approx(
            probability, rel=1.0e-12, abs=0.0
        )
    operations = result.metadata["execution_preprocessing"]["execution_operation_names"]
    assert "save_probabilities" in operations
    assert "save_statevector" not in operations


@pytest.mark.parametrize("suffix", ("terminal", "mid"))
def test_aer_statevector_preserves_mid_circuit_measurements(monkeypatch, suffix) -> None:
    circuit = QuantumCircuit(1, 1)
    circuit.h(0)
    readout = QuantumCircuit(1, 1, name="readout")
    readout.measure(0, 0)
    circuit.append(readout.to_instruction(), [0], [0])
    if suffix != "terminal":
        circuit.x(0)
    if suffix == "mid_final":
        circuit.measure(0, 0)

    original = circuit.copy()
    submitted = []
    original_run = AerSimulator.run

    def observe_run(self, execution_circuit, **kwargs):
        submitted.append(execution_circuit.count_ops().get("measure", 0))
        return original_run(self, execution_circuit, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)

    result = _execute(AER_STATEVECTOR_TARGET, circuit, seed=123)

    assert result.metadata["removed_final_measurements"] is (suffix != "mid")
    assert submitted == [0 if suffix == "terminal" else 1]
    assert result.metadata["execution_measurement_count"] == submitted[0]
    assert result.metadata["statevector_is_measurement_conditioned"] is (suffix != "terminal")
    assert result.metadata["statevector_semantics"] == ("pre_final_measurement" if suffix == "terminal" else "measurement_conditioned")
    # Terminal readout removal preserves |+>. An actual mid-circuit measurement
    # collapses it; the following X only swaps the two possible outcomes.
    np.testing.assert_allclose(sorted(abs(np.asarray(result.raw_output["statevector"])) ** 2),
                               [.5, .5] if suffix == "terminal" else [0., 1.], rtol=0., atol=1.e-14)
    assert circuit == original


def test_aer_dynamic_measurements_preserve_classical_dependencies_and_unknown_counts(monkeypatch):
    circuit = QuantumCircuit(2, 1)
    circuit.x(0)
    circuit.measure(0, 0)
    # q0 has no later quantum operation, but its measurement controls q1.
    # Removing it would silently return |01> instead of |11>.
    with circuit.if_test((circuit.clbits[0], 1)):
        circuit.x(1)
        circuit.measure(1, 0)
    circuit.measure(1, 0)
    original = circuit.copy()
    submissions = []
    original_run = AerSimulator.run

    def observe_run(self, submitted, **kwargs):
        submissions.append(kwargs["shots"])
        return original_run(self, submitted, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)
    result = _execute(AER_STATEVECTOR_TARGET, circuit)
    assert submissions == [1]
    assert result.metadata["execution_measurement_count"] is None
    assert result.metadata["removed_final_measurements"] is True
    assert result.metadata["statevector_is_measurement_conditioned"] is None
    assert result.metadata["statevector_semantics"] == "unknown_control_flow"
    np.testing.assert_allclose(abs(np.asarray(result.raw_output["statevector"])) ** 2,
                               [0., 0., 0., 1.], rtol=0., atol=1.e-14)
    assert circuit == original


def test_aer_counts_return_measured_bell_counts(monkeypatch) -> None:
    shots = 200
    circuit = _bell_circuit(measured=True)
    original = circuit.copy()
    submitted, passes = [], []
    original_run = AerSimulator.run
    original_decompose = Decompose.run

    def observe_pass(self, dag):
        passes.append(None)
        return original_decompose(self, dag)

    def observe_run(self, execution_circuit, **kwargs):
        submitted.append((execution_circuit.count_ops().get("measure", 0), kwargs["shots"]))
        return original_run(self, execution_circuit, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)
    monkeypatch.setattr(Decompose, "run", observe_pass)
    result = _execute(AER_COUNTS_TARGET, circuit, shots=shots, seed=123)

    counts = result.raw_output["counts"]
    preprocessing = result.metadata["execution_preprocessing"]

    assert result.execution_mode is ExecutionMode.SHOTS
    assert result.backend_target == AER_COUNTS_TARGET
    assert set(counts).issubset({"00", "11"})
    assert sum(counts.values()) == shots
    assert preprocessing["preprocessing"] == "decompose"
    # Native Bell gates need one admission pass.
    assert preprocessing["decompose_reps"] == len(passes)
    assert preprocessing["decompose_reps"] <= 1
    assert preprocessing["save_statevector_added"] is False
    assert result.metadata["execution_measurement_count"] == 2
    assert submitted == [(2, shots)]
    assert circuit == original


def test_aer_preparation_reuses_unique_blocks_without_keeping_submissions(monkeypatch):
    import gc
    import weakref
    from qiskit.circuit import Gate

    theta, phase = 0.37, 0.19
    rotation = QuantumCircuit(1, global_phase=phase)
    rotation.ry(theta, 0)
    flip = QuantumCircuit(1)
    flip.x(0)
    # Same name, width and parameter signature deliberately identify different definitions.
    gates = [Gate("shared", 1, [theta]), Gate("shared", 1, [theta])]
    gates[0].definition, gates[1].definition = rotation, flip
    circuits = []
    for repetitions in (1, 2):
        circuit = QuantumCircuit(2, 2, global_phase=-phase)
        circuit.x(1)  # Native prefix must survive substitution starting at index one.
        for _ in range(repetitions):
            circuit.append(gates[0], [1], copy=False)
        circuit.append(gates[1], [0], copy=False)
        circuit.measure([1, 0], [0, 1])
        circuits.append(circuit)
    originals = [circuit.copy() for circuit in circuits]
    lowered_sources, submissions, executed = [], [], []
    original_lower = qiskit_aer_backend._lower_aer_circuit
    original_prepare = qiskit_aer_backend._AerPreparation.prepare
    original_run = AerSimulator.run

    def record_lower(circuit, decompose):
        if len(circuit.data) == 1 and any(circuit.data[0].operation is gate for gate in gates):
            lowered_sources.append(circuit.data[0].operation)
        return original_lower(circuit, decompose)

    def record_run(simulator, circuit, **kwargs):
        submissions.append(weakref.ref(circuit))
        executed.append((simulator.options.seed_simulator, kwargs["shots"]))
        return original_run(simulator, circuit, **kwargs)

    def record_prepare(owner, circuit, decompose):
        prepared = original_prepare(owner, circuit, decompose)
        for before, after in zip(circuit.data, prepared.data, strict=True):
            cached = owner._blocks.get(id(before.operation))
            expected = before.operation if cached is None else cached[1]
            # A new instruction container must not recursively copy shared
            # lowered definitions on every occurrence or submission.
            assert after.operation is expected
        return prepared

    monkeypatch.setattr(qiskit_aer_backend, "_lower_aer_circuit", record_lower)
    monkeypatch.setattr(qiskit_aer_backend._AerPreparation, "prepare", record_prepare)
    monkeypatch.setattr(AerSimulator, "run", record_run)
    preparation = qiskit_aer_backend._AerPreparation()
    for seed in (7, 19):
        for repetitions, circuit in enumerate(circuits, 1):
            result = _execute(AER_STATEVECTOR_TARGET, circuit, preparation=preparation, seed=seed)
            # RY(theta)^r|1> on q1, X|0> on q0, with every global phase kept.
            expected = np.exp(1j * (repetitions - 1) * phase) * np.array([
                0., -np.sin(repetitions * theta / 2), 0., np.cos(repetitions * theta / 2),
            ])
            np.testing.assert_allclose(result.raw_output["statevector"], expected,
                                       rtol=0., atol=1.e-14)
            assert result.metadata["execution_measurement_count"] == 0
        # Two unique source gates, despite growing repetitions and a second stage.
        assert len(lowered_sources) == 2
        assert all(actual is expected for actual, expected in zip(lowered_sources, gates, strict=True))
        assert len(preparation._blocks) == 2
        assert all(original is gates[index] for index, (original, _) in enumerate(preparation._blocks.values()))
    assert executed == [(seed, 1) for seed in (7, 19) for _ in circuits]
    # Aer has cyclic job references; after collection, the still-live preparation
    # owner must not keep any of these expanded submission circuits reachable.
    gc.collect()
    assert all(reference() is None for reference in submissions)
    assert circuits == originals


@pytest.mark.parametrize("partial,simplified,repeated", (
    (True, False, False), (False, True, True),
))
def test_aer_multiplexer_preserves_definition_variants(monkeypatch, partial, simplified, repeated):
    from qiskit.circuit.library import UCGate
    from qiskit.quantum_info import Statevector

    identity = np.eye(2)
    flip = np.array([[0., 1.], [1., 0.]])
    table = [flip, flip] if repeated else [identity, flip]
    gate = UCGate(table, up_to_diagonal=partial, mux_simp=simplified)
    state = np.array([1., 2.j, -3., 4.j]) / np.sqrt(30.)
    # Gate.definition is its declared action, including any omitted diagonal.
    # Qiskit's independent state evolution consumes that definition directly;
    # dispatching only its matrix params to Aer ignores the representation flags.
    expected = Statevector(state).evolve(gate.definition).data
    circuit = QuantumCircuit(2)
    circuit.initialize(state)
    circuit.append(gate, [0, 1], copy=False)
    submissions = []
    original_run = AerSimulator.run

    def observe_run(self, submitted, **kwargs):
        submissions.append(submitted.count_ops())
        return original_run(self, submitted, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)
    result = _execute(AER_STATEVECTOR_TARGET, circuit, preparation=qiskit_aer_backend._AerPreparation())
    # Four amplitudes and a bounded two-qubit decomposition: roundoff-only budget.
    np.testing.assert_allclose(result.raw_output["statevector"], expected, rtol=0., atol=2.e-14)
    assert len(submissions) == 1
    # Requesting simplification does not require expansion if the table stays full.
    assert submissions[0].get("multiplexer", 0) == int(not partial and not repeated)


@pytest.mark.parametrize("version", ("qasm3",))
def test_native_qasm_caps_preserve_destination_and_admit_before_export(tmp_path, monkeypatch, version):
    from qiskit import qasm2, qasm3

    path = tmp_path / "existing.qasm"
    path.write_text("previous result")
    circuit = _bell_circuit(measured=True)
    with pytest.raises(ValueError, match="max_text_bytes"):
        export_qasm(circuit, format=version, path=path, max_text_bytes=1)
    assert path.read_text() == "previous result"
    assert list(tmp_path.iterdir()) == [path]
    def forbidden(*args, **kwargs):
        pytest.fail("export started before operation admission")
    monkeypatch.setattr(qasm2, "dumps", forbidden)
    monkeypatch.setattr(qasm3, "dump", forbidden)
    with pytest.raises(ValueError, match="max_operations"):
        export_qasm(circuit, format=version, path=path, max_operations=1)
    assert path.read_text() == "previous result"


def test_qasm3_path_uses_stream_and_preserves_imported_action(tmp_path, monkeypatch):
    from qiskit import qasm3
    from qiskit.quantum_info import Operator
    import numpy as np

    monkeypatch.setattr(qasm3, "dumps", lambda *a, **k: pytest.fail("assembled full QASM string"))
    path = tmp_path / "bell.qasm"
    circuit = _bell_circuit()
    assert export_qasm(circuit, path=path) == path
    restored = qasm3.loads(path.read_text())
    np.testing.assert_allclose(Operator(restored).data, Operator(circuit).data, atol=1e-14, rtol=0)
