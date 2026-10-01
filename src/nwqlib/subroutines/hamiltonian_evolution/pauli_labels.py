"""Compact Pauli-label helpers for sparse Hamiltonian evolution.

A compact label such as ``"x0z2"`` names X on qubit 0 and Z on qubit 2, with
strictly increasing qubit indices. Qiskit's dense label puts qubit 0 in the
rightmost character, so the conversion reverses positions.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit.quantum_info import SparsePauliOp

PAULI_LABEL_PATTERN = re.compile(r"([xyz])(\d+)")


def make_pauli_label(ops: Mapping[int, str]) -> str:
    """Return a compact label such as ``"x0z2"`` from qubit-indexed Paulis."""

    if not ops:
        # Every consumer (parse_pauli_label, apply_pauli_rotation, ...) rejects
        # the empty label, so fail at creation instead of far from the origin.
        raise ValueError("ops must be non-empty; identity terms have no Pauli label")
    pieces: list[str] = []
    for qubit, op in sorted(ops.items()):
        qubit_index = int(qubit)
        if qubit_index < 0:
            raise ValueError("qubit indices must be non-negative")
        normalized = str(op).upper()
        if normalized not in {"X", "Y", "Z"}:
            raise ValueError("Pauli operators must be X, Y, or Z")
        pieces.append(f"{normalized.lower()}{qubit_index}")
    return "".join(pieces)


def parse_pauli_label(label: str) -> dict[int, str]:
    """Parse a compact Pauli label into ``{qubit: op}``.

    The compact convention uses zero-based qubit indices and requires strictly
    increasing qubit order so that labels have a stable canonical form.
    """

    if not label:
        raise ValueError("Pauli label must be non-empty")

    position = 0
    ops: dict[int, str] = {}
    last_qubit = -1
    for match in PAULI_LABEL_PATTERN.finditer(label):
        if match.start() != position:
            raise ValueError(f"invalid Pauli label: {label!r}")
        op = match.group(1).upper()
        qubit = int(match.group(2))
        if qubit <= last_qubit:
            raise ValueError("Pauli label qubits must be strictly increasing")
        ops[qubit] = op
        last_qubit = qubit
        position = match.end()

    if position != len(label):
        raise ValueError(f"invalid Pauli label: {label!r}")
    return ops


def pauli_label_to_qiskit_string(label: str, num_qubits: int) -> str:
    """Convert compact labels to Qiskit's full-string Pauli convention."""

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")

    pauli_chars = ["I"] * num_qubits
    for qubit, op in parse_pauli_label(label).items():
        if qubit >= num_qubits:
            raise ValueError("Pauli label references a qubit outside num_qubits")
        pauli_chars[num_qubits - 1 - qubit] = op
    return "".join(pauli_chars)


def sparse_pauli_op_from_terms(
    terms: Sequence[Mapping[str, object]],
    num_qubits: int,
) -> SparsePauliOp:
    """Build a Qiskit ``SparsePauliOp`` from compact Pauli-term records."""

    from qiskit.quantum_info import SparsePauliOp

    if num_qubits <= 0:
        raise ValueError("num_qubits must be positive")
    if not terms:
        return SparsePauliOp.from_list([("I" * num_qubits, 0.0)])

    qiskit_terms = [
        (
            pauli_label_to_qiskit_string(str(term["pauli"]), num_qubits),
            complex(term["coefficient"]),
        )
        for term in terms
    ]
    return SparsePauliOp.from_list(qiskit_terms)


__all__ = [
    "PAULI_LABEL_PATTERN",
    "make_pauli_label",
    "parse_pauli_label",
    "pauli_label_to_qiskit_string",
    "sparse_pauli_op_from_terms",
]
