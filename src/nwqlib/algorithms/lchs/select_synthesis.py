"""Homogeneous LCHS product-formula SELECT plans and compilers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

import numpy as np

from nwqlib.core.records import FrozenArray

from qiskit import QuantumCircuit, QuantumRegister
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit.synthesis import LieTrotter, SuzukiTrotter

from nwqlib.subroutines._multiplexors import (
    AffineAngleTable,
    append_affine_rotation,
    append_control_diagonal_phases,
    append_uniformly_controlled_rz,
)
from nwqlib.subroutines.hamiltonian_evolution.sparse_pauli_product import (
    build_sparse_pauli_product_circuit,
)


@dataclass(frozen=True, kw_only=True)
class LCHSProductFormulaSelectPlan:
    """SELECT tables in consecutive schedule blocks with repetition multiplicities.

    Attributes:
        pauli_labels: Ordered distinct nonidentity Pauli labels needed by the branch
            generators; label characters follow Qiskit's high-to-low qubit convention.
        occurrence_schedule: Ordered Pauli occurrences within one product-formula step,
            including repeated labels in a symmetric second-order formula.
        occurrence_angle_tables: Read-only float64 ``FrozenArray`` of shape
            (B*W, padded_node_count): block-major rows, one per scheduled occurrence
            in each of the B compressed repetition blocks of W occurrences. Each row
            holds the rotation angles in radians of every address; completed or
            inactive branches have zero angles. Consumers read its ``array``. The
            selected identity digests the array as one byte block.
        occurrence_block_repetitions: Number of consecutive formula steps sharing each
            block of angle tables. Emission expands these counts; planning keeps them compact.
        branch_step_counts: Actual per-address product-formula step counts, including
            zero for inactive/padding branches and node-dependent budgeted selections.
        max_step_count: Maximum of the stored branch step counts, or zero if all are inactive.
        identity_phases: Per-address scalar phases in radians from the retained identity
            Hamiltonian term, -elapsed_time*identity_coefficient; zero on padding.
        coefficient_phases: Per-address arguments of the complex LCU coefficients,
            carried in SELECT separately from identity evolution phases.
        physical_node_count: Number of branches mapped to mathematical nodes, excluding
            padding; different source-time branches can map to the same k-node index.
        padded_node_count: Power-of-two address-table size, including identity padding.
        branch_to_node: Mathematical k-node index for each address, or None for identity
            padding. This mapping preserves branch order rather than sorting by k.
        pruning_decisions: Per-address Pauli labels omitted by coefficient pruning;
            padding has an empty tuple.
        pruned_l1_mass: Per-address finite upper bound on the summed moduli of the
            omitted Pauli coefficients (error_budget.upper_dropped_mass), before
            multiplication by elapsed time; not a complete product-formula error bound.
        formula_order: Selected Lie-Trotter order 1 or symmetric Suzuki order 2.
        repetitions: Total repetitions represented by occurrence_block_repetitions;
            equals max_step_count in the generated compact schedule.
        structure_certificate: Optional affine-address structural relation used to admit
            structured SELECT; None means no such relation is attached. It does not
            certify total dynamical accuracy.
        structured_generator_payload: Optional affine occurrence tables consumed by
            structured lowering; None selects no structured payload.
    """

    pauli_labels: tuple[str, ...]
    occurrence_schedule: tuple[str, ...]
    occurrence_angle_tables: FrozenArray
    occurrence_block_repetitions: tuple[int, ...]
    branch_step_counts: tuple[int, ...]
    max_step_count: int
    identity_phases: tuple[float, ...]
    coefficient_phases: tuple[float, ...]
    physical_node_count: int
    padded_node_count: int
    branch_to_node: tuple[int | None, ...]
    pruning_decisions: tuple[tuple[str, ...], ...]
    pruned_l1_mass: tuple[float, ...]
    formula_order: int
    repetitions: int
    structure_certificate: Mapping[str, Any] | None
    structured_generator_payload: Mapping[str, Any] | None



def _validate_affine_pauli_domain(terms: Sequence[tuple[str, complex]]) -> None:
    """Admit real Pauli coefficients on equal-width labels for affine lowering."""
    terms = tuple(terms)
    if not terms:
        raise ValueError("affine_pauli requires at least one Pauli term")
    width: int | None = None
    for term in terms:
        if not isinstance(term, tuple) or len(term) != 2:
            raise ValueError("affine_pauli terms must be (label, coefficient) pairs")
        label, coefficient = term
        if not isinstance(label, str) or not label or set(label) - set("IXYZ"):
            raise ValueError("affine_pauli labels must be nonempty I/X/Y/Z strings")
        if width is None:
            width = len(label)
        elif len(label) != width:
            raise ValueError("affine_pauli labels must have equal width")
        value = complex(coefficient)
        if not np.isfinite(value.real) or not np.isfinite(value.imag):
            raise ValueError("affine_pauli coefficients must be finite")
        if value.imag != 0.0:
            raise ValueError("affine_pauli coefficients must be real")


def product_formula_occurrence_template(
    pauli_labels: Sequence[str],
    *,
    order: int,
) -> tuple[tuple[str, float], ...]:
    """Return Qiskit's one-step Lie-1 or Suzuki-2 occurrence schedule."""

    labels = tuple(pauli_labels)
    if not labels:
        return ()
    num_qubits = len(labels[0])
    if any(len(label) != num_qubits for label in labels):
        raise ValueError("all Pauli labels must have the same width")
    operator = SparsePauliOp.from_list([(label, 1.0) for label in labels])
    if order == 1:
        synthesis = LieTrotter()
    elif order == 2:
        synthesis = SuzukiTrotter(order=2)
    else:
        raise ValueError("product-formula order must be 1 or 2")
    expansion = synthesis.expand(PauliEvolutionGate(operator, time=1.0))
    schedule: list[tuple[str, float]] = []
    for compact_pauli, qubit_indices, angle in expansion:
        dense = ["I"] * num_qubits
        for pauli, qubit in zip(compact_pauli, qubit_indices, strict=True):
            dense[num_qubits - 1 - int(qubit)] = pauli
        schedule.append(("".join(dense), float(angle)))
    return tuple(schedule)


def _product_formula_branch_terms(
    plan: LCHSProductFormulaSelectPlan, branch: int
) -> tuple[tuple[str, float], ...]:
    """Read the branch's per-step Pauli coefficients from its compact schedule.

    Table entries are Qiskit rotation angles, twice the exponent, and a
    Suzuki label can occur twice per step. Half the sum over one step's
    occurrences is the label's exponent coefficient times elapsed/steps.
    """

    width = len(plan.occurrence_schedule)
    if plan.branch_step_counts[branch] == 0:
        return ()
    angles: dict[str, float] = {}
    for label, table in zip(
        plan.occurrence_schedule, plan.occurrence_angle_tables.array[:width], strict=True
    ):
        angles[label] = angles.get(label, 0.0) + float(table[branch]) / 2.0
    return tuple((label, angle) for label, angle in sorted(angles.items()) if angle != 0.0)


def _build_product_formula_branch(
    plan: LCHSProductFormulaSelectPlan,
    branch: int,
    *,
    num_system_qubits: int,
) -> QuantumCircuit:
    """Realize the selected elementary branch using its existing step decision."""

    terms = _product_formula_branch_terms(plan, branch)
    steps = plan.branch_step_counts[branch]
    if not terms or steps == 0:
        return QuantumCircuit(num_system_qubits)
    # The recovered coefficients already equal the per-step exponent
    # c_P*elapsed/steps. A total time of `steps` split into `steps`
    # repetitions therefore gives each repetition time 1, which applies
    # exactly the selected per-step angles.
    return build_sparse_pauli_product_circuit(
        terms,
        num_qubits=num_system_qubits,
        time_step=float(steps),
        evolution_synthesis="lie_trotter" if plan.formula_order == 1 else "suzuki_trotter",
        reps=steps,
        order=plan.formula_order,
    )


def _product_formula_branch_phase_count(plan: LCHSProductFormulaSelectPlan) -> int:
    """Count physical phases after the circuit's ordered global-phase assignments.

    The branch builder assigns the identity phase before LCU adds the coefficient
    phase. Qiskit normalizes each assignment modulo 2*pi; combining them first
    can erase a small coefficient added after an identity phase of a full turn.
    """

    return sum(
        node is not None and (identity % (2 * np.pi) + coefficient) % (2 * np.pi) != 0.0
        for node, coefficient, identity in zip(
            plan.branch_to_node, plan.coefficient_phases, plan.identity_phases, strict=True
        )
    )


def _branch_controlled_product_formula_resource_law(
    plan: LCHSProductFormulaSelectPlan,
) -> dict[str, Any]:
    """Upper bound on the CX of the branch-controlled product-formula SELECT, all repetitions included.

    native.construct_select builds each physical branch with
    ``_build_product_formula_branch`` and controls it on the c address bits
    with ``qiskit_compat.controlled``, so Qiskit's add_control
    unrolls the branch and controls each of its gates at once. Qiskit 2.5.2
    synthesizes an occurrence of a one-qubit Pauli label as one RZ, RX or RY.
    A longer label becomes an H pair for each X factor, an SX and SXdg pair
    for each Y factor, ``2*(support - 1)`` parity CX and one RZ. Each gate
    costs its price in ``compiled_selection._controlled_kind_cx``, where an
    SX or SXdg costs the phase price, and the branch's global phase costs
    Qiskit's MCPhase on c qubits (``_mcphase_cx``).
    Without address bits the single branch keeps only its parity CX. The
    value is recorded as an estimate, because another Qiskit version can
    synthesize or control these gates differently.
    """
    from .compiled_selection import _controlled_kind_cx

    controls = plan.padded_node_count.bit_length() - 1
    cost = _controlled_kind_cx(controls)
    rotations = parity = basis = occurrences = rotation_cx = basis_cx = 0
    for branch, node in enumerate(plan.branch_to_node):
        if node is None:
            continue
        labels = tuple(label for label, _ in _product_formula_branch_terms(plan, branch))
        # Lie has one occurrence per label. Suzuki has two except for its middle label.
        schedule = labels if plan.formula_order == 1 else (*labels[:-1], *labels[::-1])
        steps = plan.branch_step_counts[branch]
        occurrences += len(schedule) * steps
        for label in schedule:
            support = sum(pauli != "I" for pauli in label)
            rotations += steps
            if support == 1:
                rotation_cx += steps * (cost["rz"] if "Z" in label else cost["rotation"])
                continue
            rotation_cx += steps * cost["rz"]
            parity += 2 * (support - 1) * steps
            basis += 2 * (label.count("X") + label.count("Y")) * steps
            basis_cx += 2 * (label.count("X") * cost["h_or_x"] + label.count("Y") * cost["phase"]) * steps
    parity_cx = cost["cx"] * parity
    phase_cx = cost["global_phase"] * _product_formula_branch_phase_count(plan)
    total = rotation_cx + basis_cx + parity_cx + phase_cx
    return {
        "resolved_lcu_select_implementation": "branch_controlled",
        "select_formula_order": plan.formula_order,
        "select_formula_repetitions": plan.repetitions,
        "select_formula_occurrence_count": occurrences,
        "select_stored_occurrence_count": len(plan.occurrence_angle_tables),
        "select_occurrence_block_repetitions": list(plan.occurrence_block_repetitions),
        "select_branch_step_counts": list(plan.branch_step_counts),
        "select_branch_controlled_full_unitary_count": 0,
        "select_branch_controlled_gate_count": plan.physical_node_count,
        "select_branch_controlled_rotation_count": rotations,
        "select_parity_network_basis_cx_count": parity_cx,
        "select_control_diagonal_entry_count": plan.padded_node_count,
        "select_control_diagonal_basis_cx_count": phase_cx,
        "select_basis_cx_count": total,
        "select_serial_depth_model": total + basis + rotations,
        "resource_gate_model_scope": "Elementary branch-controlled Pauli gates with the selected per-branch repetitions. CX uses Qiskit 2.5.2's multi-controlled gate constructions without ancillas. The serial depth model adds one layer per basis-change and rotation gate to that CX count. Neither has a cross-gate cancellation model or a certified uncertainty interval.",
    }


def _pauli_support(label: str) -> tuple[tuple[int, str], ...]:
    """Return (qubit, factor) for each non-identity factor, little-endian qubit index.

    The label's last character is qubit 0, as in Qiskit.
    """
    num_qubits = len(label)
    return tuple(
        (num_qubits - 1 - position, pauli)
        for position, pauli in enumerate(label)
        if pauli != "I"
    )


def _enter_pauli_rotation_frame(
    circuit: QuantumCircuit,
    *,
    system_qubits: Sequence[Any],
    label: str,
) -> tuple[Any, ...]:
    """Apply one shared Pauli basis/parity network and return active qubits.

    H maps X to Z and Sdg followed by H maps Y to Z, so after the basis change
    the Pauli string is a product of Z on the active qubits. The CX ladder
    then accumulates their parity on the last active qubit, where
    exp(-i*theta/2*Z...Z) becomes a single RZ(theta). The caller applies that
    rotation and then _leave_pauli_rotation_frame.
    """

    support = _pauli_support(label)
    if not support:
        raise ValueError("product-formula occurrence labels must be non-identity")
    active = tuple(system_qubits[qubit] for qubit, _ in support)
    for qubit, (_, pauli) in zip(active, support, strict=True):
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.sdg(qubit)
            circuit.h(qubit)
    for source, target in zip(active[:-1], active[1:]):
        circuit.cx(source, target)
    return active


def _leave_pauli_rotation_frame(
    circuit: QuantumCircuit,
    *,
    active: Sequence[Any],
    label: str,
) -> None:
    """Undo the shared Pauli parity network and basis change."""

    support = _pauli_support(label)
    for source, target in reversed(tuple(zip(active[:-1], active[1:]))):
        circuit.cx(source, target)
    for qubit, (_, pauli) in reversed(tuple(zip(active, support, strict=True))):
        if pauli == "X":
            circuit.h(qubit)
        elif pauli == "Y":
            circuit.h(qubit)
            circuit.s(qubit)


def _append_multiplexed_pauli_rotation(
    circuit: QuantumCircuit,
    *,
    system_qubits: Sequence[Any],
    control_qubits: Sequence[Any],
    label: str,
    angles: Sequence[float],
    gate=None,
):
    """Append basis, parity, one complete-table UCRZ, and their inverse.

    ``gate`` is the UCRZ already built for the same occurrence and angle
    table, which is appended again instead of being rebuilt. Returns the
    UCRZ gate.
    """

    active = _enter_pauli_rotation_frame(
        circuit,
        system_qubits=system_qubits,
        label=label,
    )
    if gate is None:
        gate = append_uniformly_controlled_rz(
            circuit,
            active[-1],
            control_qubits,
            angles,
        )
    else:
        circuit.append(gate, [active[-1], *control_qubits])
    _leave_pauli_rotation_frame(circuit, active=active, label=label)
    return gate


def _append_structured_pauli_rotation(
    circuit: QuantumCircuit,
    *,
    system_qubits: Sequence[Any],
    control_qubits: Sequence[Any],
    label: str,
    affine: AffineAngleTable,
) -> None:
    """Append basis/parity plus one certified affine Pauli rotation."""

    active = _enter_pauli_rotation_frame(
        circuit,
        system_qubits=system_qubits,
        label=label,
    )
    append_affine_rotation(
        circuit,
        active[-1],
        control_qubits,
        affine,
        axis="z",
    )
    _leave_pauli_rotation_frame(circuit, active=active, label=label)


def build_multiplexed_product_formula_select(
    plan: LCHSProductFormulaSelectPlan,
    *,
    num_system_qubits: int,
) -> QuantumCircuit:
    """Build the plan's branch-diagonal multiplexed SELECT circuit.

    One control-diagonal phase carries every branch's coefficient argument and
    identity-evolution phase. Each Pauli occurrence is then one basis change,
    one parity network, one uniformly controlled RZ over the full address
    table and their inverse, so all branches advance together through the
    shared Lie or Suzuki schedule. A branch with fewer steps gets zero angles
    after its last step.
    """

    num_control_qubits = plan.padded_node_count.bit_length() - 1
    system = QuantumRegister(num_system_qubits, "system")
    control = (
        QuantumRegister(num_control_qubits, "lcu_control")
        if num_control_qubits
        else None
    )
    control_qubits = list(control) if control is not None else []
    select = (
        QuantumCircuit(control, system, name="lcu_select_multiplexor")
        if control is not None
        else QuantumCircuit(system, name="lcu_select_multiplexor")
    )
    combined_phases = tuple(
        coefficient + identity
        for coefficient, identity in zip(
            plan.coefficient_phases,
            plan.identity_phases,
            strict=True,
        )
    )
    append_control_diagonal_phases(select, control_qubits, combined_phases)
    # Each repetition block has one angle table per occurrence, so its UCRZ
    # gates are built once and appended for every repetition of the block.
    width = len(plan.occurrence_schedule)
    for block, repetitions in enumerate(plan.occurrence_block_repetitions):
        tables = plan.occurrence_angle_tables.array[block * width : (block + 1) * width]
        gates = [None] * width
        for _ in range(repetitions):
            for index, (label, angles) in enumerate(zip(plan.occurrence_schedule, tables, strict=True)):
                gates[index] = _append_multiplexed_pauli_rotation(
                    select,
                    system_qubits=list(system),
                    control_qubits=control_qubits,
                    label=label,
                    angles=angles,
                    gate=gates[index],
                )
    return select


def build_structured_product_formula_select(
    plan: LCHSProductFormulaSelectPlan,
    *,
    num_system_qubits: int,
) -> QuantumCircuit:
    """Build the plan's certified affine-address structured SELECT."""

    certificate = plan.structure_certificate
    payload = plan.structured_generator_payload
    if not certificate or not certificate.get("eligible", False):
        raise ValueError("structured SELECT requires an eligible structure certificate")
    if not payload:
        raise ValueError("structured SELECT requires a structured generator payload")
    affine_tables = tuple(payload["occurrence_affine_tables"])
    if len(affine_tables) != len(plan.occurrence_angle_tables):
        raise ValueError("structured payload must cover every formula occurrence")

    num_control_qubits = plan.padded_node_count.bit_length() - 1
    system = QuantumRegister(num_system_qubits, "system")
    control = (
        QuantumRegister(num_control_qubits, "lcu_control")
        if num_control_qubits
        else None
    )
    control_qubits = list(control) if control is not None else []
    select = (
        QuantumCircuit(control, system, name="lcu_select_structured")
        if control is not None
        else QuantumCircuit(system, name="lcu_select_structured")
    )
    combined_phases = tuple(
        coefficient + identity
        for coefficient, identity in zip(
            plan.coefficient_phases,
            plan.identity_phases,
            strict=True,
        )
    )
    append_control_diagonal_phases(select, control_qubits, combined_phases)
    width = len(plan.occurrence_schedule)
    for block, repetitions in enumerate(plan.occurrence_block_repetitions):
        tables = affine_tables[block * width : (block + 1) * width]
        for _ in range(repetitions):
            for label, affine in zip(plan.occurrence_schedule, tables, strict=True):
                _append_structured_pauli_rotation(
                    select,
                    system_qubits=list(system),
                    control_qubits=control_qubits,
                    label=label,
                    affine=affine,
                )
    return select


def build_product_formula_select(
    plan: LCHSProductFormulaSelectPlan,
    *,
    num_system_qubits: int,
    implementation: Literal["multiplexor", "structured"],
) -> QuantumCircuit:
    """Build one resolved scalable product-formula SELECT."""

    if implementation == "multiplexor":
        return build_multiplexed_product_formula_select(
            plan,
            num_system_qubits=num_system_qubits,
        )
    if implementation == "structured":
        return build_structured_product_formula_select(
            plan,
            num_system_qubits=num_system_qubits,
        )
    raise ValueError("implementation must be 'multiplexor' or 'structured'")


__all__ = [
    "LCHSProductFormulaSelectPlan",
    "build_product_formula_select",
    "build_multiplexed_product_formula_select",
    "build_structured_product_formula_select",
    "product_formula_occurrence_template",
]
