"""Qiskit circuits for three blocks that several algorithms share.

For a real Pauli sum ``A = sum_j c_j P_j`` with ``alpha = sum_j |c_j|``:

- ``signed_pauli_select`` builds ``SELECT = sum_j |j><j| (x) s_j P_j``
  with ``s_j = +1`` or ``-1`` the sign of ``c_j``. With the PREP state
  ``|G> = sum_j sqrt(|c_j| / alpha) |j>`` it gives the block encoding
  ``<G| SELECT |G> = A / alpha``.
- ``pauli_readout_basis`` rotates each system qubit, controlled by the
  index register, so that a Z-basis measurement of the system reads the
  Pauli word ``P_j`` of the index held.
- ``zero_reflection`` builds ``I - 2|0><0|`` or ``2|0><0| - I``.

The formulas write the index register first, as the papers do. The
circuits place it on the low-order qubits, starting at qubit 0, so a
Qiskit dense matrix of SELECT, whose qubit 0 is the least significant
index, reads ``sum_j s_j P_j (x) |j><j|``.

The blocks ``select_signed_pauli``, ``select_pauli_readout`` and
``select_zero_reflection`` in ``nwqlib.blocks.selection`` bind these
functions as the constructors that lowering calls, and the CX laws of those
blocks describe these circuits. The Lanczos method uses all three for the
qubitized walk of Kirby et al., arXiv:2208.00567v4, and its readout,
FixedGCIM's sampled route uses the readout of single Pauli words, one for
each ancilla axis and one for each qubit-wise-commuting group, and QSP
evolution uses ``zero_reflection`` for oblivious amplitude amplification.
Keeping one construction per block makes the lowered circuit match the
counted one. This module imports Qiskit when loaded, so
``blocks.selection`` imports it only inside those constructors and
planning stays free of Qiskit.
"""

import numpy as np
from qiskit import QuantumCircuit
from qiskit.circuit.library import ZGate

from ._multiplexors import append_control_diagonal_phases, append_dependency_projected_unitaries
from .pauli_decomposition import _LOCAL_PAULI_FACTORS
from .qiskit_compat import controlled


def signed_pauli_select(labels, coefficients, num_index_qubits, num_system_qubits):
    """Return the signed SELECT circuit on ``a + n`` qubits without forming a dense matrix.

    ``labels`` are Qiskit Pauli labels of length n (qubit 0 last) and
    ``coefficients`` their real coefficients. Addresses beyond
    ``len(labels)`` act as the identity. When the index register holds
    ``j``, the circuit applies ``+P_j`` for ``c_j > 0`` and ``-P_j`` for
    ``c_j < 0``. A zero coefficient gets ``-P_j`` when some coefficient is
    negative and ``+P_j`` otherwise, which does not matter because PREP gives
    its address zero amplitude. The signs become a ``pi`` phase diagonal on
    the index register, which occupies the first ``a`` circuit qubits (the
    low-order bits), and each system qubit then receives one
    dependency-projected UCG. Every branch is a Hermitian unitary, so SELECT
    squares to the identity, which the qubitized iterate
    ``((2|G><G| - I) (x) I) SELECT`` requires (Low and Chuang,
    arXiv:1610.06546v3, Corollary 9).
    """
    a, n = num_index_qubits, num_system_qubits
    circuit = QuantumCircuit(a + n, name="signed_SELECT")
    index = circuit.qubits[:a]
    if any(c < 0 for c in coefficients):
        append_control_diagonal_phases(circuit, index, [0.0 if c > 0 else np.pi for c in coefficients])
    for q in range(n):
        append_dependency_projected_unitaries(
            circuit, circuit.qubits[a + q], index,
            [_LOCAL_PAULI_FACTORS[label[-1 - q]] for label in labels],
        )
    return circuit


def pauli_readout_basis(labels, num_index_qubits, num_system_qubits):
    """Return the index-controlled basis change that turns each ``P_j`` into a Z word.

    On ``a + n`` qubits, index register first, each system qubit receives
    H where the label of the held index has X, ``H S†`` where it has Y, and
    the identity for Z, I and padded addresses. Conjugation maps each
    factor to Z (``H X H = Z`` and ``(H S†) Y (H S†)† = Z``). After this
    index-controlled rotation, the parity of the Z-basis outcomes on the
    non-identity positions of the held index's label gives the eigenvalue
    of that Pauli string.
    """
    a, n = num_index_qubits, num_system_qubits
    circuit = QuantumCircuit(a + n, name="SELECT_basis")
    identity = np.eye(2, dtype=complex)
    hadamard = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
    rotations = {
        "I": identity,
        "X": hadamard,
        "Y": hadamard @ np.diag([1, -1j]),
        "Z": identity,
    }
    for q in range(n):
        append_dependency_projected_unitaries(
            circuit, circuit.qubits[a + q], circuit.qubits[:a],
            [rotations[label[-1 - q]] for label in labels],
        )
    return circuit


def zero_reflection(num_qubits, *, positive_zero=False):
    """Return ``I - 2|0><0|`` on ``num_qubits`` qubits, or ``2|0><0| - I`` for ``positive_zero``.

    The circuit flips every qubit, applies Z controlled on all but one of
    them and flips back, so only ``|0...0>`` acquires the sign ``-1``.
    Conjugated by a PREP unitary ``G``, ``2|0><0| - I`` becomes the reflection
    ``2|G><G| - I`` of the qubitized iterate (Low and Chuang,
    arXiv:1610.06546v3, Corollary 9). On zero qubits the projector is the
    scalar one, so the result is the global phase ``-1``, or ``+1`` for
    ``positive_zero``.
    """
    circuit = QuantumCircuit(num_qubits, name="2_zero_minus_I" if positive_zero else "oaa_reflection")
    if num_qubits:
        circuit.x(range(num_qubits))
        if num_qubits == 1:
            circuit.z(0)
        else:
            circuit.append(controlled(ZGate(), num_qubits - 1), range(num_qubits))
        circuit.x(range(num_qubits))
    else:
        circuit.global_phase = np.pi
    if positive_zero:
        circuit.global_phase += np.pi
    return circuit
