"""Independent NumPy/SciPy oracles for sorted-Pauli product formulas."""

from __future__ import annotations

from itertools import product

import numpy as np
import scipy.linalg


_SINGLE_QUBIT_PAULIS = {
    "I": np.eye(2, dtype=complex),
    "X": np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex),
    "Y": np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex),
    "Z": np.diag([1.0, -1.0]).astype(complex),
}



def pauli_matrix(label: str) -> np.ndarray:
    matrix = np.array([[1.0]], dtype=complex)
    for character in label:
        matrix = np.kron(matrix, _SINGLE_QUBIT_PAULIS[character])
    return matrix


def sorted_pauli_suzuki_2_reference(
    l_part: np.ndarray,
    h_part: np.ndarray,
    *,
    k_value: float,
    final_time: float,
    reps: int,
) -> np.ndarray:
    """Build ``exp(-i T c_I) S_2(T / reps)**reps`` independently."""

    combined = np.asarray(h_part, dtype=complex) + k_value * np.asarray(
        l_part,
        dtype=complex,
    )
    dimension = combined.shape[0]
    num_qubits = int(np.log2(dimension))
    identity_label = "I" * num_qubits
    identity_coefficient = 0.0j
    traceless_terms: list[tuple[np.ndarray, complex]] = []
    for characters in product("IXYZ", repeat=num_qubits):
        label = "".join(characters)
        matrix = pauli_matrix(label)
        coefficient = complex(np.trace(matrix.conj().T @ combined) / dimension)
        if label == identity_label:
            identity_coefficient = coefficient
        elif coefficient != 0.0:
            traceless_terms.append((matrix, coefficient))

    step_time = final_time / reps
    half_steps = [
        scipy.linalg.expm(-0.5j * step_time * coefficient * matrix)
        for matrix, coefficient in traceless_terms
    ]
    step = np.eye(dimension, dtype=complex)
    for half_step in half_steps:
        step = half_step @ step
    for half_step in reversed(half_steps):
        step = half_step @ step
    return np.exp(-1.0j * final_time * identity_coefficient) * np.linalg.matrix_power(
        step,
        reps,
    )


def sorted_pauli_lie_1_reference(
    l_part: np.ndarray,
    h_part: np.ndarray,
    *,
    k_value: float,
    final_time: float,
    reps: int,
) -> np.ndarray:
    """Build ``exp(-i T c_I) S_1(T / reps)**reps`` independently."""

    combined = np.asarray(h_part, dtype=complex) + k_value * np.asarray(
        l_part,
        dtype=complex,
    )
    dimension = combined.shape[0]
    num_qubits = int(np.log2(dimension))
    identity_label = "I" * num_qubits
    identity_coefficient = 0.0j
    factors: list[np.ndarray] = []
    for characters in product("IXYZ", repeat=num_qubits):
        label = "".join(characters)
        matrix = pauli_matrix(label)
        coefficient = complex(np.trace(matrix.conj().T @ combined) / dimension)
        if label == identity_label:
            identity_coefficient = coefficient
        elif coefficient != 0.0:
            factors.append(
                scipy.linalg.expm(-1.0j * final_time * coefficient * matrix / reps)
            )
    step = np.eye(dimension, dtype=complex)
    for factor in factors:
        step = factor @ step
    return np.exp(-1.0j * final_time * identity_coefficient) * np.linalg.matrix_power(
        step,
        reps,
    )


def fixed_trotter_product_circuit(operator, *, num_qubits, final_time, reps, order=2):
    """Build a fixed-step sorted-Pauli Lie-1 or Suzuki-2 product circuit."""
    from nwqlib.subroutines.hamiltonian_evolution.sparse_pauli_product import build_sparse_pauli_product_circuit

    return build_sparse_pauli_product_circuit(
        operator.to_list(),
        num_qubits=num_qubits,
        time_step=final_time,
        evolution_synthesis="lie_trotter" if order == 1 else "suzuki_trotter",
        reps=reps,
        order=order,
    )


def fixed_trotter_term_unitary_with_record(*, final_time, k_value, method, trotter_paulis):
    """One fixed-step sorted-Pauli node matrix and its PF accounting record, built from its circuit."""
    from qiskit import QuantumCircuit
    from qiskit.quantum_info import Operator, SparsePauliOp
    from nwqlib.algorithms.lchs.time_independent_terms import _LCHSTrotterNodeRecord, _combined_trotter_terms

    combined = _combined_trotter_terms(trotter_paulis, k_value=k_value)
    if combined.traceless_terms:
        step_circuit = fixed_trotter_product_circuit(
            SparsePauliOp.from_list(combined.traceless_terms),
            num_qubits=trotter_paulis.num_qubits,
            final_time=final_time / method.trotter_steps,
            reps=1,
            order=method.trotter_order,
        )
        step_count = int(method.trotter_steps)
    else:
        step_circuit = QuantumCircuit(trotter_paulis.num_qubits)
        step_count = 0
    step_product = np.asarray(Operator(step_circuit.decompose()).data)
    product = np.linalg.matrix_power(step_product, method.trotter_steps)
    phase = np.exp(-1.0j * final_time * combined.identity_coefficient)
    return phase * product, _LCHSTrotterNodeRecord(
        pf_bound_value=None,
        pruned_l1_mass=combined.pruned_l1_mass,
        combined_bound_value=None,
        step_count=step_count,
        selection=None,
    )
