"""Tests for backend capabilities, Aer preparation and execution, and QASM export."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.transpiler.passes import Decompose
from qiskit_aer import AerSimulator

from nwqlib.backends import (
    AER_COUNTS_TARGET,
    AER_STATEVECTOR_TARGET,
    BackendCapability,
    BackendTarget,
    capability_set,
    export_qasm,
    require_backend_capabilities,
)
import nwqlib.backends.qiskit_aer as qiskit_aer_backend
from nwqlib.execution import ExecutionMode


def test_backend_capability_mismatch_names_selected_readout_and_target():
    target = BackendTarget(name="statevector_only", provider="test",
                          capabilities=capability_set(BackendCapability.STATEVECTOR))
    require_backend_capabilities(target, capability_set(BackendCapability.STATEVECTOR),
                                 context="selected amplitudes")
    with pytest.raises(ValueError, match="selected counts.*statevector_only.*counts"):
        require_backend_capabilities(target, capability_set(BackendCapability.COUNTS),
                                     context="selected counts")


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


@pytest.mark.parametrize("readout", ("amplitudes", "probabilities", "pauli"))
def test_aer_small_readouts_have_no_configured_zero_cutoff(monkeypatch, readout):
    configurations = []

    def simulator(**settings):
        configurations.append(settings)
        return AerSimulator(**settings)

    monkeypatch.setattr(qiskit_aer_backend, "AerSimulator", simulator)
    angle = 2e-13
    circuit = QuantumCircuit(1)
    circuit.ry(angle, 0)
    options = ({"probability_qubits": (0,)} if readout == "probabilities" else
               {"pauli_expectation_readout": (1, ("X",))} if readout == "pauli" else {})
    result = _execute(AER_STATEVECTOR_TARGET, circuit, **options)
    # With Aer 0.17.2 the default zero_threshold of 1e-10 removed a
    # probability of 1e-26 from save_probabilities_dict, but it left the
    # statevector and expectation-value readouts behind these results
    # unchanged. The configured value therefore detects a deleted setting.
    assert configurations[0].get("zero_threshold") == 0.0
    if readout == "amplitudes":
        actual, expected = result.raw_output["statevector"][1], np.sin(angle/2)
    elif readout == "probabilities":
        actual, expected = result.raw_output["probabilities"][1], np.sin(angle/2)**2
    else:
        actual, expected = result.raw_output["pauli_expectations"]["X"], np.sin(angle)
    # A relative window compares each small nonzero readout with its
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
@pytest.mark.parametrize("suffix", ("none", "z", "measurement"))
def test_aer_custom_instruction_names_preserve_definitions(monkeypatch, name, suffix) -> None:
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
    if suffix == "z":
        circuit.z(0)
    elif suffix == "measurement":
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
    # H|0> = |+>; the Z suffix changes only the relative sign.
    expected = np.array([1., -1. if suffix == "z" else 1.]) / np.sqrt(2.)
    np.testing.assert_allclose(result.raw_output["statevector"], expected, rtol=0., atol=1.e-14)
    assert calls["remove_final_measurements"] <= 1
    assert calls["decompose"] <= 1
    assert calls["run"] == 1
    assert result.metadata["removed_final_measurements"] is (suffix == "measurement")
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


def test_aer_observation_position_survives_custom_measure_name() -> None:
    from qiskit.circuit import Gate

    definition = QuantumCircuit(1)
    definition.h(0)
    gate = Gate("measure", 1, [])
    gate.definition = definition
    circuit = QuantumCircuit(1, 1)
    circuit.append(gate, [0])
    circuit.z(0)
    readout = QuantumCircuit(1, 1, name="readout")
    readout.measure(0, 0)
    circuit.append(readout.to_instruction(), [0], [0])
    original = circuit.copy()
    result = _execute(AER_STATEVECTOR_TARGET, circuit, pauli_expectation_readout=(1, ("X",)),
                      preparation=qiskit_aer_backend._AerPreparation())
    # The selected position sees |+>, before the following Z produces |->.
    assert result.raw_output["pauli_expectations"]["X"] == pytest.approx(1., rel=0., abs=1.e-14)
    assert result.metadata["removed_final_measurements"] is True
    assert result.metadata["execution_measurement_count"] == 0
    after = _execute(AER_STATEVECTOR_TARGET, circuit, pauli_expectation_readout=(3, ("Z",)),
                     preparation=qiskit_aer_backend._AerPreparation())
    # Observing after the wrapped readout must keep the collapse, rather than
    # moving the save ahead of it when the composite instruction is expanded.
    assert abs(after.raw_output["pauli_expectations"]["Z"]) == pytest.approx(1., rel=0., abs=1.e-14)
    assert after.metadata["execution_measurement_count"] == 1
    assert after.metadata["removed_final_measurements"] is False
    assert circuit == original


@pytest.mark.parametrize("qubits", [(0,), (2, 0), (0, 1, 2)])
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


@pytest.mark.parametrize("suffix", ("terminal", "mid", "mid_final"))
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


@pytest.mark.parametrize("mode,dependency,nested", ((ExecutionMode.STATEVECTOR, True, False),
                                                  (ExecutionMode.STATEVECTOR, False, False),
                                                  (ExecutionMode.SHOTS, False, False),
                                                  (ExecutionMode.SHOTS, False, True)))
def test_aer_dynamic_measurements_preserve_classical_dependencies_and_unknown_counts(monkeypatch, mode, dependency, nested):
    circuit = QuantumCircuit(2 if dependency else 1, 1)
    circuit.x(0)
    if nested:
        # Only nested blocks contain Measure; loop iterations are not a static
        # measurement count. The terminal Z keeps this a valid Aer control flow.
        with circuit.if_test((circuit.clbits[0], 0)):
            with circuit.for_loop(range(2)):
                circuit.measure(0, 0)
        circuit.z(0)
    elif dependency:
        circuit.measure(0, 0)
        # q0 has no later quantum operation, but its measurement controls q1.
        # Removing it would silently return |01> instead of |11>.
        with circuit.if_test((circuit.clbits[0], 1)):
            circuit.x(1)
            circuit.measure(1, 0)
        circuit.measure(1, 0)
    else:
        # No top-level Measure: SHOTS admission and STATEVECTOR conditioning
        # must not infer known zero from an empty top-level measurement count.
        with circuit.if_test((circuit.clbits[0], 0)):
            circuit.measure(0, 0)
        final_readout = QuantumCircuit(1, 1, name="readout")
        final_readout.measure(0, 0)
        circuit.append(final_readout.to_instruction(), [0], [0])
    original = circuit.copy()
    submissions = []
    original_run = AerSimulator.run

    def observe_run(self, submitted, **kwargs):
        submissions.append(kwargs["shots"])
        return original_run(self, submitted, **kwargs)

    monkeypatch.setattr(AerSimulator, "run", observe_run)
    result = (_execute(AER_STATEVECTOR_TARGET, circuit) if mode is ExecutionMode.STATEVECTOR
              else _execute(AER_COUNTS_TARGET, circuit, shots=4))
    assert submissions == [1 if mode is ExecutionMode.STATEVECTOR else 4]
    assert result.metadata["execution_measurement_count"] is None
    if mode is ExecutionMode.STATEVECTOR:
        assert result.metadata["removed_final_measurements"] is True
        assert result.metadata["statevector_is_measurement_conditioned"] is None
        assert result.metadata["statevector_semantics"] == "unknown_control_flow"
        np.testing.assert_allclose(abs(np.asarray(result.raw_output["statevector"])) ** 2,
                                   [0., 0., 0., 1.] if dependency else [0., 1.], rtol=0., atol=1.e-14)
    else:
        assert result.raw_output["counts"] == {"1": 4}
    assert circuit == original


@pytest.mark.parametrize(
    ("readout", "selector", "expected_keys"),
    (
        ("amplitudes", {}, {"statevector"}),
        ("probabilities", {"probability_qubits": (0,)}, {"probabilities"}),
        ("pauli", {"pauli_expectation_readout": (3, ("Z",))}, {"pauli_expectations"}),
    ),
)
def test_aer_statevector_readouts_execute_one_seeded_trajectory(
    monkeypatch, readout, selector, expected_keys,
) -> None:
    circuit = QuantumCircuit(1, 1)
    block = QuantumCircuit(1)
    block.h(0)
    circuit.append(block.to_gate(), [0], copy=False)
    circuit.measure(0, 0)
    circuit.x(0)
    observed_shots = []
    original_run = qiskit_aer_backend.AerSimulator.run

    def record_run(simulator, execution_circuit, **kwargs):
        observed_shots.append(kwargs.get("shots"))
        assert execution_circuit.count_ops()["measure"] == 1
        return original_run(simulator, execution_circuit, **kwargs)

    monkeypatch.setattr(qiskit_aer_backend.AerSimulator, "run", record_run)
    preparation = qiskit_aer_backend._AerPreparation()
    results = [
        _execute(AER_STATEVECTOR_TARGET, circuit, seed=123, preparation=preparation, **selector)
        for _ in range(2)
    ]
    assert observed_shots == [1, 1]
    first, second = (result.raw_output for result in results)
    if readout == "amplitudes":
        np.testing.assert_array_equal(first["statevector"], second["statevector"])
    elif readout == "probabilities":
        np.testing.assert_array_equal(first["probabilities"], second["probabilities"])
    else:
        assert first == second
    for result in results:
        assert set(result.raw_output) == expected_keys
        assert result.metadata["statevector_is_measurement_conditioned"] is True
        assert result.metadata["execution_measurement_count"] == 1
        assert result.metadata["simulation_trajectory_count"] == 1
        if "statevector" in expected_keys:
            state = np.asarray(result.raw_output["statevector"])
            assert np.count_nonzero(np.abs(state) > 1.0e-14) == 1
            assert np.vdot(state, state).real == pytest.approx(1.0, rel=0.0, abs=1.0e-14)
        if "probabilities" in expected_keys:
            probabilities = _nonzero_marginal(result.raw_output["probabilities"], 1)
            assert set(probabilities).issubset({"0", "1"})
            # Decomposed rotations may leave roundoff-scale probability in the other branch.
            np.testing.assert_allclose(
                sorted(probabilities.get(bit, 0.0) for bit in ("0", "1")),
                [0.0, 1.0], rtol=0.0, atol=1.0e-14,
            )
        if "pauli_expectations" in expected_keys:
            assert abs(result.raw_output["pauli_expectations"]["Z"]) == pytest.approx(
                1.0, rel=0.0, abs=1.0e-14,
            )

@pytest.mark.parametrize("shots", (200, 3, np.int64(3)))
def test_aer_counts_return_measured_bell_counts(shots, monkeypatch) -> None:
    circuit = _bell_circuit(measured=True)
    if shots == 3:
        # Composite-only readout and mixed composite/native readout must both
        # reach the same native measurement admission owner.
        readout = QuantumCircuit(1, 1, name="readout")
        readout.measure(0, 0)
        instruction = readout.to_instruction()
        circuit.data[2] = circuit.data[2].replace(operation=instruction)
        if isinstance(shots, np.integer):
            circuit.data[3] = circuit.data[3].replace(operation=instruction)
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
    # Native Bell gates need one admission pass; wrapped readout one expansion.
    assert preprocessing["decompose_reps"] == len(passes)
    assert preprocessing["decompose_reps"] <= (2 if shots == 3 else 1)
    assert preprocessing["save_statevector_added"] is False
    assert result.metadata["execution_measurement_count"] == 2
    assert submitted == [(2, shots)]
    assert circuit == original


def test_aer_counts_require_measurements_and_positive_shots(monkeypatch) -> None:
    """A gate merely named measure is not measurement, and invalid shots must fail before
    preparation.
    """
    with pytest.raises(ValueError, match="requires at least one measurement"):
        _execute(AER_COUNTS_TARGET, _bell_circuit(), shots=100)

    from qiskit.circuit import Gate

    definition = QuantumCircuit(1)
    definition.x(0)
    named_measure = Gate("measure", 1, [])
    named_measure.definition = definition
    unmeasured = QuantumCircuit(1, 1)
    with unmeasured.if_test((unmeasured.clbits[0], 0)):
        with unmeasured.for_loop(range(2)):
            unmeasured.append(named_measure, [0], copy=False)
    original = unmeasured.copy()
    submissions = []

    def unexpected_submission(*args, **kwargs):
        submissions.append(None)
        raise AssertionError("provably measurement-free control flow was submitted")

    monkeypatch.setattr(AerSimulator, "run", unexpected_submission)
    with pytest.raises(ValueError, match="requires at least one measurement"):
        _execute(AER_COUNTS_TARGET, unmeasured, shots=4)
    assert not submissions
    assert unmeasured == original

    def unexpected_work(*_args, **_kwargs):
        raise AssertionError("invalid shots must reject before lowering or execution")

    monkeypatch.setattr(qiskit_aer_backend, "_prepare_execution_circuit", unexpected_work)
    for shots in (float("nan"), float("inf"), 0, -1):
        with pytest.raises(ValueError, match="shots must be an integer at least 1"):
            _execute(AER_COUNTS_TARGET, _bell_circuit(measured=True), shots=shots)


@pytest.mark.parametrize("mode", (ExecutionMode.SHOTS, ExecutionMode.STATEVECTOR))
def test_aer_preparation_reuses_unique_blocks_without_keeping_submissions(monkeypatch, mode):
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
        # Reverse qarg -> carg order: the fixed q0=1 must be the high count bit.
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
    measured = mode is ExecutionMode.SHOTS
    for seed, shots in ((7, 16), (19, 23)):
        for repetitions, circuit in enumerate(circuits, 1):
            result = _execute(AER_COUNTS_TARGET if measured else AER_STATEVECTOR_TARGET, circuit,
                              preparation=preparation, seed=seed, shots=shots if measured else None)
            if mode is ExecutionMode.STATEVECTOR:
                # RY(theta)^r|1> on q1, X|0> on q0, with every global phase kept.
                expected = np.exp(1j * (repetitions - 1) * phase) * np.array([
                    0., -np.sin(repetitions * theta / 2), 0., np.cos(repetitions * theta / 2),
                ])
                np.testing.assert_allclose(result.raw_output["statevector"], expected,
                                           rtol=0., atol=1.e-14)
                assert result.metadata["execution_measurement_count"] == 0
            else:
                counts = result.raw_output["counts"]
                assert set(counts) <= {"10", "11"}
                assert sum(counts.values()) == shots
                assert result.metadata["execution_measurement_count"] == 2
        # Two unique source gates, despite growing repetitions and a second stage.
        assert len(lowered_sources) == 2
        assert all(actual is expected for actual, expected in zip(lowered_sources, gates, strict=True))
        assert len(preparation._blocks) == 2
        assert all(original is gates[index] for index, (original, _) in enumerate(preparation._blocks.values()))
    assert executed == [(seed, shots if mode is ExecutionMode.SHOTS else 1)
                        for seed, shots in ((7, 16), (19, 23)) for _ in circuits]
    # Aer has cyclic job references; after collection, the still-live preparation
    # owner must not keep any of these expanded submission circuits reachable.
    gc.collect()
    assert all(reference() is None for reference in submissions)
    assert circuits == originals


@pytest.mark.parametrize("partial,simplified,repeated", (
    (False, False, False), (True, False, False), (False, True, True), (False, True, False),
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


def test_qasm_export_rejects_unknown_format() -> None:
    with pytest.raises(ValueError, match="qasm2"):
        export_qasm(_bell_circuit(), format="bad")


def test_qasm2_export_writes_selected_format_to_path(tmp_path) -> None:
    path = tmp_path / "bell.qasm"
    written = export_qasm(_bell_circuit(measured=True), format="qasm2", path=path)
    qasm_text = path.read_text()

    assert "OPENQASM 2.0" in qasm_text
    assert "OPENQASM 3" not in qasm_text
    assert written == path


@pytest.mark.parametrize("version", ("qasm2", "qasm3"))
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


@pytest.mark.parametrize("kind", ["pauli_expectation", "probabilities", "amplitudes"])
@pytest.mark.parametrize("qpy_loaded", [False, True])
def test_aer_restore_admits_actual_payload_and_ordered_wires(kind, qpy_loaded, monkeypatch):
    from io import BytesIO
    from qiskit import qpy
    from qiskit.quantum_info import Pauli
    from qiskit_aer.library import SaveExpectationValue, SaveProbabilities, SaveStatevector
    from nwqlib.backends.qiskit_aer import _restore_aer_readout

    def forbidden(*args, **kwargs):
        pytest.fail("readout restoration ran a simulator")

    monkeypatch.setattr(AerSimulator, "run", forbidden)
    circuit = QuantumCircuit(2, global_phase=.3)
    circuit.x(0)
    qubits = (1, 0) if kind == "probabilities" else (0, 1)
    observation = SimpleNamespace(kind=kind, labels=("IZ",), qubits=qubits)
    save = (SaveExpectationValue(Pauli("IZ"), label="nwqlib_exp_0") if kind == "pauli_expectation"
            else SaveProbabilities(2) if kind == "probabilities" else SaveStatevector(2))
    circuit.append(save, qubits)
    circuit.h(1)  # The save must stay before this later operation.
    if qpy_loaded:
        stream = BytesIO()
        qpy.dump(circuit, stream)
        stream.seek(0)
        circuit, = qpy.load(stream)
    if kind == "pauli_expectation":
        # Equivalent container/scalar forms are not a different observable.
        circuit.data[1].operation.params = [["IZ", [complex(1.), np.int64(0)]]]
    with pytest.raises(ValueError, match="does not match"):
        wrong = circuit.copy()
        wrong.data[1] = wrong.data[1].replace(qubits=tuple(reversed(wrong.data[1].qubits)))
        _restore_aer_readout(wrong, observation)
    if kind == "pauli_expectation":
        with pytest.raises(ValueError, match="does not match"):
            _restore_aer_readout(circuit.copy(), SimpleNamespace(kind=kind, labels=("ZI",), qubits=qubits))
        wrong = circuit.copy()
        wrong.data[1].operation.params = [("IZ", (.5, 0))]
        with pytest.raises(ValueError, match="does not match"):
            _restore_aer_readout(wrong, observation)
    _restore_aer_readout(circuit, observation)
    restored = circuit.data[1].operation
    assert type(restored) is type(save)
    assert tuple(circuit.find_bit(q).index for q in circuit.data[1].qubits) == qubits
    assert circuit.data[2].operation.name == "h" and circuit.global_phase == .3
    _restore_aer_readout(circuit, observation)
    assert circuit.data[1].operation is restored
