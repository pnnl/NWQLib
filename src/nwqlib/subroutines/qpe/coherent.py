"""Coherent quantum phase-estimation circuit helpers.

The construction is the QFT-based phase estimation of Cleve, Ekert,
Macchiavello and Mosca, arXiv:quant-ph/9708016v1, Sec. 5. The builder
docstring gives the equation numbers of that arXiv version and the
bit-order convention.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import QFTGate, UnitaryGate

from nwqlib._limits import DEFAULT_MAX_BYTES
from nwqlib._validation import integer
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines._dense_synthesis import controlled_synthesis_size
from nwqlib.subroutines._power_of_two import is_power_of_two as _is_power_of_two
from nwqlib.subroutines._matrix_checks import is_unitary
from nwqlib.subroutines.qiskit_compat import controlled


@dataclass(frozen=True, kw_only=True)
class CoherentQPECircuit:
    """A coherent phase-estimation circuit, without measurement, and its register sizes.

    [`build_coherent_qpe_circuit`][nwqlib.subroutines.qpe.coherent.build_coherent_qpe_circuit]
    returns it. The circuit is `circuit`. The fields below are read-only.

    Attributes:
        circuit: Circuit on the `phase` register followed by the `system`
            register.
        num_phase_qubits: Number m of phase-estimation qubits.
        num_system_qubits: Number of target-system qubits.
        powers: Exponents `2**q` of the controlled powers `U^(2**q)`, in
            phase-qubit order `q = 0, ..., m - 1`.
        bit_order: Integer convention for the phase register,
            `"little_endian_phase_integer"`: phase qubit 0 holds the least
            significant bit.
    """

    circuit: QuantumCircuit
    num_phase_qubits: int
    num_system_qubits: int
    powers: tuple[int, ...]
    bit_order: str = "little_endian_phase_integer"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-like QPE circuit summary."""

        return {
            "num_phase_qubits": self.num_phase_qubits,
            "num_system_qubits": self.num_system_qubits,
            "powers": list(self.powers),
            "bit_order": self.bit_order,
            "circuit_depth": self.circuit.depth(),
            "gate_counts": {
                name: int(count) for name, count in self.circuit.count_ops().items()
            },
        }


def build_coherent_qpe_circuit(
    unitary: Any,
    *,
    num_phase_qubits: int,
    name: str = "coherent_qpe",
    max_work: int = 1_000_000_000,
    max_bytes: int = DEFAULT_MAX_BYTES,
) -> CoherentQPECircuit:
    """Build the coherent phase-estimation circuit of a dense unitary, without measurement.

    The circuit is the QFT-based phase estimation of Cleve, Ekert,
    Macchiavello and Mosca, "Quantum algorithms revisited", Proc. R. Soc.
    Lond. A 454, 339 (1998), arXiv:quant-ph/9708016v1, Sec. 5. Section,
    figure, equation and page numbers below refer to that arXiv version.
    Hadamards on the phase
    register and the controlled powers of Fig. 6 (p. 10) are followed by the
    inverse of the QFT of Eq. (4.1) (p. 8). The circuit has no measurement,
    and its register order is ``phase`` followed by ``system``.

    Let m = num_phase_qubits. Phase qubit q controls U^(2**q). For an
    eigenvector with U v = exp(2*pi*i*theta) v, the phase register after the
    controlled powers is 2**(-m/2) * sum_k exp(2*pi*i*theta*k) |k>, with k
    read little-endian. This is Eq. (5.1), whose phase phi is theta here,
    with the factor 2**(-m/2) included. Qiskit's QFTGate maps |j> to
    2**(-m/2) * sum_k exp(2*pi*i*j*k/2**m) |k>, which is the normalized
    Eq. (4.1), so its inverse returns |j> exactly when theta = j/2**m
    modulo 1. For any other theta, a measurement of the register gives an
    integer nearest to 2**m*theta (modulo 2**m) with probability at least
    4/pi**2, Eqs. (5.2)-(5.4) (p. 11).

    Conventions relative to the paper. The eigenvalue exp(2*pi*i*theta)
    has the same sign. The paper takes 0 <= phi < 1. Here theta may be any
    real number, and theta + 1 gives the same eigenvalue and the same
    register state, so the register integer estimates 2**m*theta modulo
    2**m. The paper writes y = y_1...y_m with the most significant bit y_1
    on its top qubit, which controls U^(2**(m-1)) (Fig. 6 labels it U^(2^j)
    with j = m - 1). Qiskit puts the least
    significant bit on phase qubit 0. Both registers hold the same integer,
    so the state is the same vector. The paper's QFT network of Fig. 5
    leaves its output qubits in reverse order, while QFTGate is the whole
    map of Eq. (4.1) with that reversal included. As in the paper, the
    inverse QFT acts after all controlled powers.

    Powers are formed by repeated squaring of the dense matrix, m - 1
    products in total. The 1e-8 unitarity check is an absolute input
    tolerance, not an accuracy statement. Each squaring can double
    the unitarity defect, so the powers skip Qiskit's own constructor check
    of ``U^dagger U`` (``numpy.allclose`` with the identity, atol 1e-8 and
    rtol 1e-5), which would reject high powers of an admitted input.
    The unitary polar factor of each controlled power is synthesized to
    binary64 rounding from its controlled matrix on n + 1 qubits, for n
    system qubits (see
    [Controlled dense unitaries](../../development/dense_synthesis.md#controlled-dense-unitaries)),
    so the realized power differs from the computed one by the order of its
    unitarity defect. That takes at most
    `(25/96) 4**(n+1) - 2**(n+1) + 4/3` CX for n >= 2 and order
    `8**(n+1)` classical arithmetic per power.

    Before the unitarity check, the work of the check and the m - 1
    squarings, `D**3` units each for dimension D, plus m times the work of
    one exact controlled synthesis on n + 1 qubits is compared with
    `max_work`. The bytes of six D-square complex128 arrays, as for the
    dense powers of the QPE Methods, plus the working bytes of one
    synthesis and the kept bytes of all m are compared with `max_bytes`.
    The default work limit is the QPE Methods' `max_work`.

    Args:
        unitary (array_like): Dense unitary matrix of power-of-two
            dimension D, unitary to the absolute tolerance 1e-8.
        num_phase_qubits (int): Number m of qubits used for phase
            estimation.
        name (str): Default `"coherent_qpe"`. Circuit name.
        max_work (int): Default `1_000_000_000`. Limit on the work of the
            unitarity check, the powers and their exact controlled
            synthesis.
        max_bytes (int): Default 10 GB (decimal, `10_000_000_000` bytes).
            Limit on the bytes of the power arrays and of the syntheses.

    Returns:
        qpe (CoherentQPECircuit): The circuit, with no measurements, in
            `qpe.circuit`.

    Raises:
        ValueError: If dimensions or qubit counts are invalid, if the
            matrix is not unitary to 1e-8, or if the powers and their
            synthesis exceed `max_work` or `max_bytes`.

    Examples:
        With `U = diag(1, exp(2 pi i 5/8))` and the system qubit in the
        eigenvector `|1>`, three phase qubits read the integer
        `2**3 * 5/8 = 5` with probability 1:

        >>> import numpy as np
        >>> from qiskit import QuantumCircuit
        >>> from qiskit.quantum_info import Statevector
        >>> from nwqlib.subroutines.qpe import build_coherent_qpe_circuit
        >>> U = np.diag([1.0, np.exp(2j * np.pi * 5 / 8)])
        >>> qpe = build_coherent_qpe_circuit(U, num_phase_qubits=3)
        >>> circuit = QuantumCircuit(4)
        >>> _ = circuit.x(3)
        >>> circuit = circuit.compose(qpe.circuit)
        >>> probabilities = Statevector(circuit).probabilities([0, 1, 2])
        >>> print(int(np.argmax(probabilities)), round(probabilities.max(), 10))
        5 1.0
    """

    num_phase_qubits = integer(num_phase_qubits, "num_phase_qubits", 1)
    matrix = np.asarray(unitary, dtype=complex)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("unitary must be a square matrix")
    if not _is_power_of_two(matrix.shape[0]):
        raise ValueError("unitary dimension must be a power of two")
    dimension = matrix.shape[0]
    synthesis_work, working_bytes, kept_bytes = controlled_synthesis_size(dimension.bit_length())
    _check_bytes(96 * dimension**2 + working_bytes + num_phase_qubits * kept_bytes, max_bytes,
                 "coherent QPE powers and their controlled synthesis")
    if num_phase_qubits * (dimension**3 + synthesis_work) > integer(max_work, "max_work", 1):
        raise ValueError("coherent QPE powers and their controlled synthesis exceed max_work")
    # Entrywise absolute window on U^dagger U - I, equal to the is_unitary
    # default registered in docs/ENGINEERING_CONSTANTS.md.
    if not is_unitary(matrix, atol=1.0e-8):
        raise ValueError("unitary must be unitary")

    num_system_qubits = int(round(math.log2(matrix.shape[0])))
    phase = QuantumRegister(num_phase_qubits, "phase")
    system = QuantumRegister(num_system_qubits, "system")
    circuit = QuantumCircuit(phase, system, name=name)
    circuit.h(phase)

    powers: list[int] = []
    powered = matrix
    for phase_index in range(num_phase_qubits):
        power = 2**phase_index
        powers.append(power)
        if phase_index:
            powered = np.matmul(powered, powered)
        circuit.append(
            controlled(UnitaryGate(powered, label=f"U^{power}", check_input=False), 1),
            [phase[phase_index], *system[:]],
        )

    circuit.append(QFTGate(num_phase_qubits).inverse(), phase[:])
    return CoherentQPECircuit(
        circuit=circuit,
        num_phase_qubits=num_phase_qubits,
        num_system_qubits=num_system_qubits,
        powers=tuple(powers),
    )


__all__ = [
    "CoherentQPECircuit",
    "build_coherent_qpe_circuit",
]
