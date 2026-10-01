"""Direct state-preparation helpers.

Exact basis, full-uniform, and supported prefix-uniform states use native fast paths. General vectors use conditional magnitude rotations and a phase diagonal without synthesis cutoffs.

The general path follows Mottonen et al., quant-ph/0407010v1, Sec. III. It applies a binary tree of uniformly controlled RY rotations with the angles of their Eq. (8) and then the phases. The paper interleaves phase-equalizing RZ multiplexors (Eqs. (4) and (5)) with the RY levels and cancels one CX per level, which reaches ``2**(n+1) - 2n - 2`` CX when one end is a basis state (half of the general ``2**(n+2) - 4n - 4`` stated on p. 4). Here all phases form one diagonal after the magnitude tree. Tree and diagonal cost ``2**n - 2`` CX each, ``2 * (2**n - 2)`` in total for a complex state, and a nonnegative vector, such as an LCU coefficient state, omits the diagonal entirely.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library.data_preparation import UniformSuperpositionGate

from nwqlib._numerics import normalize_state_vector
from nwqlib.subroutines._multiplexors import (
    append_control_diagonal_phases,
    append_uniformly_controlled_ry,
)


@dataclass(frozen=True, kw_only=True)
class DirectStatePreparation:
    """Direct state-preparation circuit.

    Args:
        circuit: Circuit that prepares ``normalized_state`` from ``|0...0>``.
        normalized_state: Normalized target-state amplitudes.
        input_norm: 2-norm of the original input vector.
        preparation_l2_error: Phase-sensitive construction error in the ideal-gate model. Zero means no algorithmic approximation. Floating-point synthesis and execution roundoff are not measured by circuit construction.
        num_qubits: Number of qubits in the prepared register.
        method: State-preparation backend identifier.
    """

    circuit: QuantumCircuit
    normalized_state: np.ndarray
    input_norm: float
    num_qubits: int
    preparation_l2_error: float = 0.0
    method: str = "qiskit_state_preparation"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like preparation summary."""

        return {
            "method": self.method,
            "input_norm": self.input_norm,
            "num_qubits": self.num_qubits,
            "dimension": int(self.normalized_state.shape[0]),
            "fidelity_to_target": None,
            "preparation_error_model": "ideal_gate_construction",
            "bond_dimensions": None,
            "num_layers": None,
            "preparation_l2_error": self.preparation_l2_error,
            "circuit_depth": self.circuit.depth(),
            "gate_counts": {
                name: int(count) for name, count in self.circuit.count_ops().items()
            },
        }


def build_qiskit_state_preparation(vector: Any) -> DirectStatePreparation:
    """Build a direct state-preparation circuit with exact native fast paths.

    General vectors use a binary magnitude tree and a phase diagonal. Pairwise hypot reductions avoid squared-magnitude underflow. The construction uses ``O(n * 2**n)`` classical arithmetic and ``O(2**n)`` live numerical storage, without a target unitary or state simulation.

    Args:
        vector: State amplitudes. The input may be unnormalized; the returned
            metadata records its norm so algorithms can rescale scientific
            outputs after quantum-state normalization.

    Returns:
        DirectStatePreparation containing the circuit and normalization data.
    """

    normalized_state, input_norm = normalize_state_vector(vector)
    return _build_normalized_state_preparation(
        normalized_state,
        input_norm=input_norm,
        register_name="system",
    )


def _build_normalized_state_preparation(
    normalized_state: np.ndarray,
    *,
    input_norm: float,
    register_name: str,
) -> DirectStatePreparation:
    """Build the shared exact-dispatch circuit for an already normalized state.

    Fast paths apply only to exact structure. A single basis state uses X
    gates, equal amplitudes on every index use H gates, and equal amplitudes
    on a prefix ``0, ..., m - 1`` use Qiskit's ``UniformSuperpositionGate``.
    Each carries the common amplitude phase as the circuit global phase, so a
    controlled preparation keeps it as a relative phase. The general path
    computes block norms with pairwise ``hypot``. At qubit ``t`` the RY angle
    for a block is ``2 * atan2(n1, n0)``, where ``n0`` and ``n1`` are the
    norms of its entries with bit ``t`` equal to zero and one. This equals
    Mottonen et al., quant-ph/0407010v1, Eq. (8), and gives angle zero for an
    all-zero block. Qubit
    ``t`` is controlled by qubits ``t + 1, ..., n - 1`` with qubit ``t + 1``
    least significant, which matches Qiskit's little-endian index.
    """

    num_qubits = int(np.log2(normalized_state.shape[0]))
    register = QuantumRegister(num_qubits, register_name)
    circuit = QuantumCircuit(register, name="state_preparation")
    nonzero = np.flatnonzero(normalized_state != 0.0)
    common_amplitude = normalized_state[nonzero[0]]
    common_phase = float(np.angle(common_amplitude))
    if nonzero.size == 1:
        basis_index = int(nonzero[0])
        for qubit in range(num_qubits):
            if basis_index & (1 << qubit):
                circuit.x(register[qubit])
        circuit.global_phase = common_phase
    elif nonzero.size == normalized_state.size and np.all(
        normalized_state == common_amplitude
    ):
        circuit.h(register)
        circuit.global_phase = common_phase
    elif (
        1 < nonzero.size < normalized_state.size
        and np.array_equal(nonzero, np.arange(nonzero.size))
        and np.all(normalized_state[nonzero] == common_amplitude)
    ):
        circuit.append(
            UniformSuperpositionGate(int(nonzero.size), num_qubits=num_qubits),
            list(register),
        )
        circuit.global_phase = common_phase
    else:
        norms = [np.abs(normalized_state)]
        for _ in range(num_qubits):
            pairs = norms[-1].reshape(-1, 2)
            norms.append(np.hypot(pairs[:, 0], pairs[:, 1]))
        for target in reversed(range(num_qubits)):
            pairs = norms[target].reshape(-1, 2)
            angles = 2.0 * np.arctan2(pairs[:, 1], pairs[:, 0])
            if np.any(angles != 0.0):
                append_uniformly_controlled_ry(
                    circuit, register[target], list(register)[target + 1 :], angles
                )
        phases = np.angle(normalized_state)
        if np.any(phases != 0.0):
            append_control_diagonal_phases(circuit, list(register), phases)
    return DirectStatePreparation(
        circuit=circuit,
        normalized_state=normalized_state,
        input_norm=input_norm,
        num_qubits=num_qubits,
    )




__all__ = [
    "DirectStatePreparation",
    "build_qiskit_state_preparation",
]
