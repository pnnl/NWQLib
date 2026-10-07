"""Records of Pauli terms and Hamiltonian-evolution blocks for circuit construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, kw_only=True)
class PauliEvolutionTerm:
    """One Pauli term `c P` of an evolution block, with `P` given by a compact label.

    Build it with keyword arguments, for example
    `PauliEvolutionTerm(pauli="x0z2", coefficient=0.5)` for `0.5 X_0 Z_2`.
    Both arguments are required.

    Args:
        pauli: Compact indexed Pauli label such as `"x0z2"`, with
            strictly increasing zero-based qubit indices, as
            [`parse_pauli_label`][nwqlib.subroutines.hamiltonian_evolution.pauli_labels.parse_pauli_label]
            accepts.
        coefficient: Coefficient in the block generator. A
            Hermitian generator needs it real.
    """

    pauli: str
    coefficient: complex


@dataclass(frozen=True, kw_only=True)
class PauliEvolutionBlock:
    """One Hamiltonian-evolution step `exp(-i * time_step * sum_j c_j P_j)` or one of its exact special forms.

    Build it with keyword arguments, for example
    `PauliEvolutionBlock(terms=(term,), time_step=0.1)`, and pass a sequence
    of blocks to
    [`build_pauli_evolution_circuit`][nwqlib.subroutines.hamiltonian_evolution.pauli_evolution.build_pauli_evolution_circuit].
    `terms` and `time_step` are required. For the default kind the block is
    synthesized as one product-formula application in term order, which is
    exact when the terms commute. The `"number_projector"` and `"kinetic"`
    kinds have the exact forms that the
    [Pauli-evolution builders][nwqlib.subroutines.hamiltonian_evolution.pauli_evolution]
    describe.

    Args:
        terms: Ordered Pauli terms of the
            generator.
        time_step: Duration multiplying the generator for this
            block.
        time: Default `None`. Optional schedule time, distinct
            from the step duration.
        kind: Default `"hamiltonian_evolution"`. Block meaning:
            `"hamiltonian_evolution"`, `"number_projector"` or `"kinetic"`.
        support: Default `None`. Qubits of a
            `"number_projector"` block.
        angle: Default `None`. Angle of a `"number_projector"`
            block.
        metadata: Default empty. Additional construction
            information.
    """

    terms: tuple[PauliEvolutionTerm, ...]
    time_step: float
    time: float | None = None
    kind: str = "hamiltonian_evolution"
    support: tuple[int, ...] | None = None
    angle: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)


def coerce_pauli_evolution_term(term: PauliEvolutionTerm | Mapping[str, Any]) -> PauliEvolutionTerm:
    """Return ``term`` as a ``PauliEvolutionTerm``."""

    if isinstance(term, PauliEvolutionTerm):
        return term
    return PauliEvolutionTerm(pauli=str(term["pauli"]), coefficient=complex(term["coefficient"]))


def coerce_pauli_evolution_block(
    block: PauliEvolutionBlock | Mapping[str, Any],
) -> PauliEvolutionBlock:
    """Return ``block`` as a ``PauliEvolutionBlock``."""

    if isinstance(block, PauliEvolutionBlock):
        return block

    support = block.get("support")
    return PauliEvolutionBlock(
        terms=tuple(coerce_pauli_evolution_term(term) for term in block["terms"]),
        time_step=float(block["time_step"]),
        time=None if block.get("time") is None else float(block["time"]),
        kind=str(block.get("kind", "hamiltonian_evolution")),
        support=None if support is None else tuple(int(index) for index in support),
        angle=None if block.get("angle") is None else float(block["angle"]),
        metadata=dict(block.get("metadata", {})),
    )


def pauli_terms_to_dicts(terms: tuple[PauliEvolutionTerm, ...]) -> list[dict[str, Any]]:
    """Return the terms as `{"pauli": label, "coefficient": c}` records, the input of `sparse_pauli_op_from_terms`."""

    return [{"pauli": term.pauli, "coefficient": term.coefficient} for term in terms]


__all__ = [
    "PauliEvolutionBlock",
    "PauliEvolutionTerm",
    "coerce_pauli_evolution_block",
    "coerce_pauli_evolution_term",
    "pauli_terms_to_dicts",
]
