"""Unchanged coherent QPE register/power/phase contract.

The selected statistical methods are covered by test_qpe_workflow.py and
 test_qpe_selected.py; this native test is not part of the J22/J23 run.
"""

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.quantum_info import Statevector
from nwqlib.subroutines.qpe import build_coherent_qpe_circuit


def test_coherent_qpe_register_order_for_exact_phase(monkeypatch) -> None:
    from nwqlib.subroutines.qpe import coherent
    phase_qubits = 6
    phase_integer = 24  # Phase3/8; reversing011000 gives000110.
    unitary = np.diag([1.0, np.exp(2.0j * np.pi * phase_integer / (2**phase_qubits))])
    products = []
    matmul = np.matmul
    def counted(left,right):
        assert left is right
        products.append(1)
        return matmul(left,right)
    monkeypatch.setattr(coherent.np,'matmul',counted)
    qpe = build_coherent_qpe_circuit(unitary, num_phase_qubits=phase_qubits)
    assert len(products)==5
    monkeypatch.setattr(coherent.np,'matmul',matmul)
    phase = QuantumRegister(phase_qubits, "phase")
    system = QuantumRegister(1, "system")
    circuit = QuantumCircuit(phase, system)
    circuit.x(system[0])
    circuit.append(qpe.circuit.to_gate(), [*phase[:], system[0]])

    state = Statevector(circuit.decompose(reps=5))
    phase_probabilities: dict[int, float] = {}
    for index, amplitude in enumerate(state.data):
        phase_index = index & (2**phase_qubits - 1)
        phase_probabilities[phase_index] = phase_probabilities.get(phase_index, 0.0) + float(
            abs(amplitude) ** 2
        )

    assert max(phase_probabilities, key=phase_probabilities.get) == phase_integer
    # The exact phase has unit probability; allow accumulated gate roundoff.
    assert phase_probabilities[phase_integer] == pytest.approx(1.0, rel=0.0, abs=1.0e-12)

    result = build_coherent_qpe_circuit(np.eye(2), num_phase_qubits=np.int64(2))
    assert result.num_phase_qubits == 2
    with pytest.raises(ValueError, match="num_phase_qubits"):
        build_coherent_qpe_circuit(np.eye(2), num_phase_qubits=0)


def _controlled(unitary):
    """Controlled matrix with the control as qubit 0, the low-order bit."""
    dimension = unitary.shape[0]
    matrix = np.eye(2 * dimension, dtype=complex)
    matrix[1::2, 1::2] = unitary
    return matrix


def test_coherent_qpe_controls_near_identity_powers_exactly() -> None:
    # Qiskit 2.5.2's UnitaryGate.control kept a synthesis of the first
    # controlled power with entry errors of 3.3e-7 for t = 1e-6. Each appended
    # power must equal the controlled matrix of exp(-i 2**q t G) to rounding.
    from scipy.linalg import expm
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(21)
    x = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    generator = (x + x.conj().T) / 2
    qpe = build_coherent_qpe_circuit(expm(-1e-6j * generator), num_phase_qubits=3)
    powers = [item.operation for item in qpe.circuit.data if item.operation.num_qubits == 2]
    assert len(powers) == 3
    for index, gate in enumerate(powers):
        expected = _controlled(expm(-1e-6j * 2**index * generator))
        assert np.abs(Operator(gate).data - expected).max() <= 64 * 2.0**-53 * 4


def test_coherent_qpe_accepts_powers_of_an_admitted_near_unitary_input() -> None:
    # The input passes the 1e-8 admission window with defect 8e-9. Repeated
    # squaring doubles the defect, and with Qiskit 2.5.2 the UnitaryGate
    # constructor rejected the higher powers with
    # ValueError("Input matrix is not unitary.").
    from scipy.linalg import expm
    from qiskit.quantum_info import Operator

    rng = np.random.default_rng(22)
    x = rng.normal(size=(4, 4)) + 1j * rng.normal(size=(4, 4))
    exact = expm(-1j * (x + x.conj().T) / 2)
    defective = exact @ (np.eye(4) + 4e-9 * np.diag([1.0, -1.0, 1.0, -1.0]))
    qpe = build_coherent_qpe_circuit(defective, num_phase_qubits=4)
    powers = [item.operation for item in qpe.circuit.data if item.operation.num_qubits == 3]
    # The synthesized power is unitary and differs from U^8 by about the
    # defect of U^8, about 6e-8.
    expected = _controlled(np.linalg.matrix_power(exact, 8))
    assert np.abs(Operator(powers[3]).data - expected).max() <= 1e-6
