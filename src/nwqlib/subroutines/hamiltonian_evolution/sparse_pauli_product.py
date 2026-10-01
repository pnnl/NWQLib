"""Product-formula circuit for a validated Pauli summand list.

Single conversion owner from dense qiskit labels (qubit 0 rightmost) to the
compact Pauli-evolution block form, delegating the synthesis to
``build_pauli_evolution_circuit``. This internal helper is not exported by
``nwqlib.subroutines.hamiltonian_evolution``; call sites import it from this
module.
"""

from __future__ import annotations

from typing import Any, Sequence

from qiskit import QuantumCircuit

from nwqlib._validation import integer
from nwqlib.subroutines.hamiltonian_evolution.pauli_evolution import (
    build_pauli_evolution_circuit,
)
from nwqlib.subroutines.hamiltonian_evolution.pauli_labels import make_pauli_label


def build_sparse_pauli_product_circuit(
    terms: Sequence[tuple[str, complex]],
    *,
    num_qubits: int,
    time_step: float,
    evolution_synthesis: str,
    reps: int,
    order: int | None = None,
) -> QuantumCircuit:
    """Build ``S_order(time_step / reps)**reps`` over the given summands.

    Args:
        terms: ``(dense_label, coefficient)`` summands in application order.
            Callers pass identity-free Hermitian summand lists (an identity
            summand only shifts the global phase, which the compact
            Pauli-evolution representation cannot carry); coefficient
            realness is enforced by the delegated rotation path.
        num_qubits: Width of the dense labels.
        time_step: Total evolution time realized by the product formula.
        evolution_synthesis: Delegated synthesis name
            (``"lie_trotter"`` or ``"suzuki_trotter"``).
        reps: Product-formula step count.
        order: Suzuki order (one or a positive even integer), forwarded to
            its synthesis. Lie-Trotter permits only its intrinsic order one.
    """

    synthesis_options: dict[str, Any] = {"reps": reps}
    if order is not None:
        order = integer(order, "product-formula order", 1)
        if evolution_synthesis == "suzuki_trotter":
            if order > 1 and order % 2:
                raise ValueError("Suzuki product-formula order must be one or a positive even integer")
            synthesis_options["order"] = order
        elif evolution_synthesis != "lie_trotter" or order != 1:
            raise ValueError("explicit product-formula order requires Suzuki synthesis or Lie-Trotter order one")
    compact_terms = []
    for label, coefficient in terms:
        # Qiskit dense labels put qubit 0 rightmost; compact labels index
        # qubits explicitly, so reverse before sorting by qubit.
        ops = {
            num_qubits - 1 - position: character
            for position, character in enumerate(label)
            if character != "I"
        }
        compact_terms.append(
            {"pauli": make_pauli_label(ops), "coefficient": coefficient}
        )

    return build_pauli_evolution_circuit(
        [
            {
                "terms": compact_terms,
                "time_step": time_step,
            }
        ],
        num_qubits=num_qubits,
        evolution_synthesis=evolution_synthesis,
        evolution_synthesis_options=synthesis_options,
    )
