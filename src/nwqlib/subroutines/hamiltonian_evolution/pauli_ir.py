"""Typed Pauli-evolution intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, kw_only=True)
class PauliEvolutionTerm:
    """One compact Pauli term in an evolution block.

    Attributes:
        pauli: Compact indexed Pauli label accepted by the evolution label parser.
        coefficient: Coefficient in the block generator; Hermitian execution imposes its real-coefficient premise.
    """

    pauli: str
    coefficient: complex


@dataclass(frozen=True, kw_only=True)
class PauliEvolutionBlock:
    """One Hamiltonian-evolution instruction.

    For the default kind the block means exp(-i * time_step * sum_j c_j P_j),
    synthesized as one product-formula application in term order. Special
    kinds are described in pauli_evolution.

    Attributes:
        terms: Ordered compact Pauli terms of the selected generator.
        time_step: Duration multiplying the generator for this evolution block.
        time: Optional schedule-time metadata; distinct from the step duration.
        kind: Instruction meaning, normally hamiltonian_evolution.
        support: Optional explicit wire positions for a specialized compact instruction.
        angle: Optional rotation-angle metadata for a specialized compact instruction.
        metadata: Additional selected construction information; not a second executed circuit.
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
    """Return compact dictionary records accepted by Qiskit conversion helpers."""

    return [{"pauli": term.pauli, "coefficient": term.coefficient} for term in terms]


__all__ = [
    "PauliEvolutionBlock",
    "PauliEvolutionTerm",
    "coerce_pauli_evolution_block",
    "coerce_pauli_evolution_term",
    "pauli_terms_to_dicts",
]
