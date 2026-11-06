"""Balanced one-ancilla pair circuit of a sampled LCHS observable.

A pair row of `sampled_rows` holds two branches x and y. Branch b has an
input kind (0 initial, 1 source), elapsed time t_b, kernel node k_b and
unit direction ``|chi_b> = U(t_b, k_b) |psi_{kind_b}>`` with
``U(t, k) = exp(-i*t*(k*L_c + H_c))`` on the system, where L_c and H_c are
the Hermitian Pauli tables of the centered children. The row's branch
phase is ``theta = arg w_y - arg w_x``. The scalar phase
``phi(t, k) = -t*(k*(c_L + s) + c_H)`` of the centering is part of the
weights and therefore of theta, so this circuit never applies it. An
identity term still present in L_c or H_c is a real part of the generator.
It multiplies branch b by ``exp(-i*t_b*(k_b*l_I + h_I))``, which the circuit
applies once, as the plan's identity phase on the branch qubit.

Pair state and readout (An, Liu and Lin, arXiv:2303.01029v2, Supplement IV;
Chakraborty, arXiv:2302.13555v4, Sec. 3.2). H on the branch qubit and the
branch-selected input preparation give ``(|0>|psi_x> + |1>|psi_y>)/sqrt(2)``.
Equal kinds prepare their direction once without control. Different kinds
prepare each direction under the branch value of its slot, so the global
phase of each preparation circuit becomes the physical relative phase of
the two directions. The branch-selected evolution
``|0><0| (x) U_x + |1><1| (x) U_y`` then gives
``(|0>|chi_x> + |1>|chi_y>)/sqrt(2)``, and the readout phase
``diag(1, exp(i*theta))`` gives
``|Psi> = (|0>|chi_x> + exp(i*theta)|1>|chi_y>)/sqrt(2)``. For a Pauli term P,

    <Psi| X_a (x) P |Psi> = (exp(i*theta) m + conj(exp(i*theta) m))/2
                          = Re(exp(i*theta) <chi_x|P|chi_y>),

with ``m = <chi_x|P|chi_y>`` and P Hermitian. H on the branch qubit maps X_a
to Z_a, and the basis change of P (H on each X site, S-dagger then H on
each Y site, as `quantum._rotations` does) maps P to Z on its support. The
branch outcome s_a = (-1)**branch_bit and the parity
s_P = (-1)**popcount(system_bits & support) therefore have
``E[s_a s_P] = Re(exp(i*theta) m)`` and, with P = I,
``E[s_a] = Re(exp(i*theta) <chi_x|chi_y>)``. These are the shot values that
`sampled_moments.row_scores` scales into Z_Q and Z_N.

Branch-selected product formula. Both branches use the same
step count r and the same ordered Lie-1 or Suzuki-2 schedule of the
nonidentity labels of L_c and H_c, sorted, with coefficient
``c_P(k) = k*l_P + h_P``. Occurrence (P, mu) of the Qiskit schedule rotates
by the exponent ``a_b = (mu/2)*(t_b/r)*c_P(k_b)``, where mu is 2 for every
Lie label and for the middle Suzuki label and 1 for each of the two other
Suzuki occurrences. The SELECT tables store the RZ angle 2*a_b. For one
occurrence,

    |0><0| (x) exp(-i*a_x*P) + |1><1| (x) exp(-i*a_y*P)
        = exp(-i*[(a_x + a_y)*P + (a_x - a_y)*Z_a P]/2),

as the two sides agree on Z_a = +1 and Z_a = -1. The two generators
commute, so this is exact, and the product of these factors in schedule
order is exactly the block selection of the two complete product formulas,
with no further splitting error. Different elapsed times give different H
angles, so H is selected by the branch as well as L. The multiplexed SELECT
emits each factor as a one-control uniformly controlled RZ inside the
shared basis and parity network, which keeps its two CX also when
a_x = a_y. The structured SELECT writes the two-entry table as the exact
one-bit affine formula ``a(b) = a_x + b*(a_y - a_x)``. Its controlled
rotation exists only when a_y - a_x is nonzero, which in binary64 holds
exactly when a_x != a_y, so an exactly equal pair (H labels of a pair with
t_x = t_y, or k_x = k_y) costs one RZ and no CX.

Distributed branch controls. With r_c control
copies, a CX tree after the input preparation gives
``Fan |b>|0...0> = |b>^(r_c)``. Every SELECT operation on the branch qubit
is block diagonal in its computational basis, ``sum_b |b><b| (x) V_b``, so on
``sum_b |b>^(r_c) (x) phi_b`` it acts the same on any copy. After the SELECT
the inverse tree returns ``sum_b |b>|0...0> (x) V_b phi_b``, with clean
copies and the branch cross phase kept. The tree is uncomputed before the
readout phase and H, which act on the root only. It costs 2*(r_c - 1) CX
and r_c - 1 qubits and has 2*ceil(log2 r_c) CX layers on all-to-all
connectivity. Parallel Pauli layers need system supports that allow them,
so no depth reduction is claimed.

Row templates. A backend that binds parameterized circuits (IBM SamplerV2
PUBs) runs many rows from one compiled circuit. `pair_key` decides the
gates `build_pair_circuit` emits for one row without building a circuit,
as the builders do, including the rotations they omit for an exactly zero
angle, and puts each emitted continuous angle in a slot: the Walsh angles
of a multiplexed RZ (`gray_code_rotation_angles`), the offset and
coefficient of a structured one, the RZ angle and phase of the
branch-phase diagonal (`append_control_diagonal_phases`), the PTSC
correction angles and theta. It walks the gates outside the SELECT and
keys each SELECT copy by the choices that decide its gates, so rows with
equal keys have the same gates in the same order without a pass over every
repeated occurrence. `build_pair_template` walks the same gates for
one row and puts a parameter in each slot, so binding a row's slot values
gives the gates and angles of its own pair circuit. No angle relation is
new: every value comes from the producer that the row builder calls.
`pair_template_size` counts that circuit's entries from the key alone.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

import numpy as np
from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from nwqlib._limits import check_memory
from nwqlib._validation import integer
from nwqlib.core.records import FrozenArray
from nwqlib.subroutines._multiplexors import (
    certify_affine_angle_table,
    gray_code_rotation_angles,
    pad_angle_table,
    product_formula_select_resource_law,
)
from nwqlib.subroutines.qiskit_compat import controlled

from .sampled_recipes import low_order_sign, mask_label
from .select_synthesis import (
    LCHSProductFormulaSelectPlan,
    _combined_branch_phases,
    _enter_pauli_rotation_frame,
    _leave_pauli_rotation_frame,
    _select_circuit_bytes,
    admit_product_formula_select,
    build_product_formula_select,
    product_formula_occurrence_template,
)

_KIND_NAMES = ("initial", "source")

# Gates of the product-formula SELECT that act on its address qubit. Each is
# block diagonal in that qubit's computational basis, the premise that lets
# any clean copy of the branch value control it (module docstring).
_ADDRESS_DIAGONAL_GATES = frozenset({"ucrz", "crz", "nwqlib_phase_diagonal"})


@dataclass(frozen=True)
class LCHSSampledPairCircuit:
    """One shot circuit of a pair row and the classical bits its scores read.

    Attributes:
        circuit: Qubits are the branch qubit (index 0), the r_c - 1 control
            copies (indices 1 to r_c - 1) and the system qubits j at index
            r_c + j. The classical register ``readout`` has q + 1 bits: bit 0
            holds the branch qubit measured in X after the readout phase,
            and bit 1 + j holds system qubit j measured in the basis of the
            row's Pauli term. A circuit built with ``measure=False`` has no
            classical register, and its caller measures in this bit order.
        branch_mask: Outcome-integer mask of the branch bit, 1, so that
            s_a = (-1)**popcount(outcome & branch_mask).
        parity_mask: Outcome-integer mask of the term's support,
            s_P = (-1)**popcount(outcome & parity_mask). It is 0 for the
            identity term.
    """

    circuit: QuantumCircuit
    branch_mask: int
    parity_mask: int


def _real_pauli_table(terms: Mapping[str, Any], name: str) -> dict[str, float]:
    """Return a Hermitian Pauli table as real coefficients after checking labels and values."""
    table = {}
    for label, value in terms.items():
        if not isinstance(label, str) or not label or set(label) - set("IXYZ"):
            raise ValueError(f"{name} labels must be nonempty I/X/Y/Z strings")
        value = complex(value)
        if value.imag != 0.0 or not isfinite(value.real):
            raise ValueError(f"{name} coefficients of a Hermitian Pauli table must be finite and real")
        table[label] = value.real
    return table


def _pair_tables(l_terms: Mapping[str, Any], h_terms: Mapping[str, Any]):
    """Check the L_c and H_c Pauli tables of `pair_select_plan` and return them with their non-identity labels.

    Returns ``(l_table, h_table, labels, identity)``: read-only real tables
    of one label width, the sorted labels other than the identity, and the
    identity label, as `_pair_select_plan` reads them. A Plan that builds
    many rows of one generator checks its tables once here
    (`sampled_plan._pair_plan`).
    """
    l_table = _real_pauli_table(l_terms, "L_c")
    h_table = _real_pauli_table(h_terms, "H_c")
    widths = {len(label) for label in (*l_table, *h_table)}
    if len(widths) > 1:
        raise ValueError("L_c and H_c labels must have one width")
    identity = "I" * (widths.pop() if widths else 0)
    labels = tuple(sorted((set(l_table) | set(h_table)) - {identity}))
    return MappingProxyType(l_table), MappingProxyType(h_table), labels, identity


def pair_select_plan(
    l_terms: Mapping[str, Any],
    h_terms: Mapping[str, Any],
    *,
    times: Sequence[float],
    nodes: Sequence[float],
    steps: int,
    order: int,
    implementation: str,
    template: Sequence[tuple[str, float]] | None = None,
    repetitions: int | None = None,
    occurrence_coefficients: Sequence[tuple[float, float]] | None = None,
) -> LCHSProductFormulaSelectPlan:
    """Return the two-slot product-formula SELECT plan of one pair row for one SELECT implementation.

    Address 0 is branch x and address 1 is branch y, so the branch qubit is
    the SELECT address, ``padded_node_count = 2``, and ``branch_to_node``
    names the slots (0, 1) rather than kernel-grid indices. Both slots take
    the row's step count, which makes one repetition block of
    ``repetitions`` (default ``steps``) copies of the occurrence word.
    The angle of occurrence (P, mu) in slot b is ``mu*t_b*c_P(k_b)/steps``,
    the formula of `time_independent_terms._build_product_formula_select_plan`,
    and the identity phase is ``-t_b*(k_b*l_I + h_I)``. The coefficient
    phases are zero because theta is applied at the readout. A structured
    plan attaches the structure certificate, which compares every two-entry
    table with its one-bit affine formula (module docstring) under the
    binary64 window of `certify_affine_angle_table`. Both SELECT
    implementations can build a structured plan. A multiplexor plan reads
    only the angle tables and carries no certificate, so only the
    multiplexor builds it. A recipe word of `sampled_recipes` passes its own template:
    the MPF macrostep with ``steps`` = r and ``repetitions`` = r/2, or one
    PTSC Lie step of the unmerged H then L lists, whose occurrences of a
    shared label take their own list's coefficient through
    ``occurrence_coefficients``.

    Args:
        l_terms: Pauli table {label: real coefficient} of L_c, in Qiskit
            high-to-low label order. An identity label is a generator term,
            not the removed centering scalar.
        h_terms: Pauli table of H_c in the same form.
        times: Elapsed times (t_x, t_y).
        nodes: Kernel nodes (k_x, k_y).
        steps: Product-formula steps r of both branches, at least 1.
        order: 1 for Lie-Trotter or 2 for symmetric Suzuki.
        implementation: "multiplexor" or "structured", the SELECT that
            `build_pair_circuit` will build from the plan.
        template: Default None. The occurrence schedule
            `product_formula_occurrence_template` returns for the sorted
            labels and order, passed by a caller that plans many rows of one
            generator, or a recipe word. None computes it.
        repetitions: Default None, which repeats the word ``steps`` times.
        occurrence_coefficients: Default None, which takes each
            occurrence's coefficients from the tables by its label. Otherwise
            one ``(l, h)`` pair per template occurrence, so that the angle is
            ``mu*t_b*(k_b*l + h)/steps``.

    Returns:
        plan (LCHSProductFormulaSelectPlan): Tables, phases and, for the
            structured SELECT, the certificate.
    """
    if implementation not in ("multiplexor", "structured"):
        raise ValueError("implementation must be 'multiplexor' or 'structured'")
    return _pair_select_plan(_pair_tables(l_terms, h_terms), times=times, nodes=nodes, steps=steps, order=order,
                             template=template, repetitions=repetitions,
                             occurrence_coefficients=occurrence_coefficients,
                             structured=implementation == "structured")


def _pair_select_plan(checked, *, times, nodes, steps, order, template, repetitions, occurrence_coefficients,
                      structured):
    """Build `pair_select_plan` from tables that `_pair_tables` checked.

    With ``structured=False`` (a multiplexor SELECT) the plan carries no
    structure certificate or affine tables, which only the structured
    SELECT reads.
    """
    steps = integer(steps, "pair row steps", 1)
    l_table, h_table, labels, identity = checked
    if template is None:
        template = product_formula_occurrence_template(labels, order=order)
    repetitions = steps if repetitions is None else repetitions
    repetitions = integer(repetitions, "pair row repetitions", 1)
    if occurrence_coefficients is None:
        occurrence_coefficients = [(l_table.get(label, 0.0), h_table.get(label, 0.0)) for label, _ in template]
    branches = tuple(zip((float(time) for time in times), (float(node) for node in nodes), strict=True))
    if len(branches) != 2:
        raise ValueError("a pair row has exactly two branches")
    angles = np.array(
        [[multiplier * time * (node * l_value + h_value) / steps for time, node in branches]
         for (_, multiplier), (l_value, h_value) in zip(template, occurrence_coefficients, strict=True)],
        dtype=float,
    ).reshape(len(template), 2)
    tables = FrozenArray(angles)
    certificate = generator = None
    if structured:
        affine = tuple(
            certify_affine_angle_table(row, offset=float(row[0]), coefficients=(float(row[1] - row[0]),))
            for row in tables.array
        )
        eligible = all(table.max_residual <= table.tolerance for table in affine)
        certificate = MappingProxyType({
            "eligible": eligible,
            "rejection_reasons": () if eligible else ("a one-bit affine table exceeds its binary64 window",),
            "reconstructed_table_max_residual": max((table.max_residual for table in affine), default=0.0),
            "reconstruction_tolerance": max((table.tolerance for table in affine), default=0.0),
        })
        generator = MappingProxyType({"occurrence_affine_tables": affine})
    return LCHSProductFormulaSelectPlan(
        pauli_labels=labels,
        occurrence_schedule=tuple(label for label, _ in template),
        occurrence_angle_tables=tables,
        occurrence_block_repetitions=(repetitions,),
        branch_step_counts=(repetitions, repetitions),
        identity_phases=tuple(
            float(-time * (node * l_table.get(identity, 0.0) + h_table.get(identity, 0.0))) + 0.0
            for time, node in branches
        ),
        coefficient_phases=(0.0, 0.0),
        physical_node_count=2,
        padded_node_count=2,
        branch_to_node=(0, 1),
        pruning_decisions=((), ()),
        pruned_l1_mass=(0.0, 0.0),
        formula_order=order,
        repetitions=repetitions,
        structure_certificate=certificate,
        structured_generator_payload=generator,
    )


def _fanout_pairs(count: int) -> tuple[tuple[int, int], ...]:
    """Return the (control, target) CX of a doubling tree that copies wire 0 to wires 1..count-1.

    Layer l copies each wire j < 2**l to j + 2**l, so the tree has count - 1
    CX in ceil(log2 count) layers.
    """
    pairs, width = [], 1
    while width < count:
        pairs.extend((source, source + width) for source in range(width) if source + width < count)
        width *= 2
    return tuple(pairs)


def _check_widths(plan: LCHSProductFormulaSelectPlan, term_label: str, num_system_qubits: int) -> None:
    """Reject plan or readout labels of another width, which would act on the wrong system qubits."""
    if any(len(label) != num_system_qubits for label in plan.pauli_labels):
        raise ValueError("plan labels must cover the prepared system register")
    if (not isinstance(term_label, str) or len(term_label) != num_system_qubits
            or set(term_label) - set("IXYZ")):
        raise ValueError("term_label must be an I/X/Y/Z label over the system qubits")


def build_pair_circuit(
    plan: LCHSProductFormulaSelectPlan,
    *,
    preparations: Sequence[QuantumCircuit | None] | Mapping[int, QuantumCircuit] | None,
    kinds: Sequence[int],
    theta: float,
    term_label: str,
    implementation: Literal["multiplexor", "structured"],
    control_copies: int = 1,
    measure: bool = True,
    corrections: Sequence[tuple[tuple, tuple]] | None = None,
    max_bytes=None,
) -> LCHSSampledPairCircuit:
    """Build the pair circuit of one row (module docstring).

    The gate order is H on the branch qubit, input preparation, fanout to
    the control copies, the plan's SELECT from `build_product_formula_select`
    with each address-qubit gate moved to a copy in turn, the inverse
    fanout, the readout phase ``P(theta)`` and H on the branch qubit, the
    basis change of ``term_label`` and, with ``measure``, the measurement of
    the branch and all system qubits.

    With ``corrections`` (PTSC, `sampled_recipes.LCHSPairCompensation`)
    the plan is one Lie step, and each segment appends that SELECT and then
    the segment's two corrections, x under branch value 0 and y under
    branch value 1 (`_append_correction`). A correction is block diagonal in
    the branch qubit's computational basis, so it is controlled by the root
    branch qubit while the copies hold the same value; the X pair that
    selects value 0 acts between gates that do not touch the copies.

    Args:
        plan: The row's `pair_select_plan`.
        preparations: Input preparation circuits on the system, indexed by
            kind (0 initial, 1 source). Each prepares its unit direction,
            and its global phase is the physical phase of that direction.
            ``None`` is allowed for equal kinds only. The caller then
            prepares the shared direction on the system before this
            circuit, which equals the uncontrolled preparation here because
            it acts on the system alone and commutes with H on the branch
            qubit.
        kinds: Input kinds (kind_x, kind_y).
        theta: Branch phase arg w_y - arg w_x in radians.
        term_label: Pauli term P of the row's system readout, or the
            identity label for a norm-only row.
        implementation: SELECT construction, ``"multiplexor"`` or
            ``"structured"`` (module docstring).
        control_copies: Copies r_c >= 1 of the branch value that control the
            SELECT. 1 keeps every control on the branch qubit.
        measure: Default True. False returns the unitary circuit without the
            classical register, for a caller that measures the branch qubit
            and then the system qubits in that bit order itself.
        corrections: Default None. One ``(x descriptor, y descriptor)``
            pair per PTSC segment, as `LCHSPairCompensation.corrections`
            returns them, applied after each Lie step of a one-step plan.
        max_bytes: Default None, the physical memory. Limit checked before
            the SELECT is built, for the SELECT, its copy in the pair circuit
            for every segment and at most 2*(7q + 3) instructions of the two
            corrections of each segment (`select_synthesis.admit_product_formula_select`).

    Returns:
        pair (LCHSSampledPairCircuit): Circuit and branch/parity masks.
    """
    control_copies = integer(control_copies, "control_copies", 1)
    kind_x, kind_y = (int(kind) for kind in kinds)
    if preparations is None and kind_x != kind_y:
        raise ValueError("branches of different kinds need their controlled input preparations")
    q = len(term_label) if preparations is None else preparations[kind_x].num_qubits
    _check_widths(plan, term_label, q)

    branch = QuantumRegister(1, "branch")
    copies = [QuantumRegister(control_copies - 1, "branch_copy")] if control_copies > 1 else []
    system = QuantumRegister(q, "system")
    readout = [ClassicalRegister(q + 1, "readout")] if measure else []
    circuit = QuantumCircuit(branch, *copies, system, *readout, name="lchs_sampled_pair")
    root = branch[0]
    controls = [root, *(qubit for register in copies for qubit in register)]

    circuit.h(root)
    if preparations is not None and kind_x == kind_y:
        circuit.append(_input_gate(preparations, kind_x), list(system))
    elif preparations is not None:
        for slot, kind in enumerate((kind_x, kind_y)):
            circuit.append(_input_gate(preparations, kind, slot), [root, *system])

    fanout = _fanout_pairs(control_copies)
    for source, target in fanout:
        circuit.cx(controls[source], controls[target])
    # This circuit builds a separate SELECT and copies it once per segment. A
    # correction is at most an X pair, a Pauli frame of 2(2q + q - 1) gates, q
    # controlled Paulis or one CRZ, and one phase: 7q + 3 instructions.
    segments = 1 if corrections is None else len(corrections)
    admit_product_formula_select(plan, implementation=implementation, max_bytes=max_bytes, copies=segments,
                                 extra_instructions=0 if corrections is None else 2 * segments * (7 * q + 3))
    select = build_product_formula_select(plan, num_system_qubits=q, implementation=implementation,
                                          max_bytes=max_bytes)
    address = select.qubits[0]
    wires = dict(zip(select.qubits[1:], system, strict=True))
    uses = 0
    if corrections is not None and plan.repetitions != 1:
        raise ValueError("PTSC corrections follow a one-step Lie plan")
    for segment in (None,) if corrections is None else corrections:
        for instruction in select.data:
            if address in instruction.qubits:
                if instruction.operation.name not in _ADDRESS_DIAGONAL_GATES:
                    raise ValueError(f"SELECT gate {instruction.operation.name!r} is not diagonal on the branch qubit")
                wires[address] = controls[uses % control_copies]
                uses += 1
            circuit.append(instruction.operation, [wires[qubit] for qubit in instruction.qubits])
        circuit.global_phase += select.global_phase
        for value, descriptor in enumerate(() if segment is None else segment):
            _append_correction(circuit, root, system, descriptor, value)
    for source, target in reversed(fanout):
        circuit.cx(controls[source], controls[target])

    circuit.p(float(theta), root)
    circuit.h(root)
    for qubit, axis in enumerate(reversed(term_label)):
        if axis == "Y":
            circuit.sdg(system[qubit])
        if axis in "XY":
            circuit.h(system[qubit])
    if measure:
        circuit.measure(root, readout[0][0])
        for qubit in range(q):
            circuit.measure(system[qubit], readout[0][1 + qubit])
    parity_mask = sum(1 << (1 + qubit) for qubit, axis in enumerate(reversed(term_label)) if axis != "I")
    return LCHSSampledPairCircuit(circuit=circuit, branch_mask=1, parity_mask=parity_mask)


def _append_correction(circuit: QuantumCircuit, root, system, descriptor, value: int) -> None:
    """Append one PTSC correction ``(order, phase, x, z, angle)`` under branch value ``value``.

    The correction (`sampled_recipes`) is the identity for order 0,
    C = i**phase W for order s >= 4 and exp(i*sign*angle*W) for s = 2, 3,
    with W the label word of the masks and sign that of P' = Q or -iQ
    (`sampled_recipes.low_order_sign`). Under branch value 1:
    a controlled C = i**phase W is the controlled Pauli of each support
    site (CX, CY, CZ) and P(phase*pi/2) on the branch qubit, since
    |0><0| (x) I + |1><1| (x) e^{i phi} W = P(phi) times the controlled W;
    W = I leaves only the phase, phase*pi/2 or sign*angle, which is relative
    between the branches. exp(i*phi*W) for W != I is the frame of
    `select_synthesis._enter_pauli_rotation_frame` around CRZ(-2*phi),
    because RZ(lambda) = exp(-i*lambda*Z/2). Value 0 conjugates the branch
    qubit by X.
    """
    order, phase, x, z, angle = descriptor
    if not order:
        return
    label = mask_label(x, z, len(system))
    if value == 0:
        circuit.x(root)
    if not (x | z):
        circuit.p(phase * np.pi / 2 if order >= 4 else low_order_sign(phase) * angle, root)
    elif order >= 4:
        gates = dict(X=circuit.cx, Y=circuit.cy, Z=circuit.cz)
        for qubit, axis in enumerate(reversed(label)):
            if axis != "I":
                gates[axis](root, system[qubit])
        if phase:
            circuit.p(phase * np.pi / 2, root)
    else:
        active = _enter_pauli_rotation_frame(circuit, system_qubits=list(system), label=label)
        circuit.crz(-2.0 * low_order_sign(phase) * angle, root, active[-1])
        _leave_pauli_rotation_frame(circuit, active=active, label=label)
    if value == 0:
        circuit.x(root)


def _preparation_gate(preparation: QuantumCircuit, kind: int):
    """Return a preparation circuit as a gate named after its kind, keeping its global phase."""
    gate = preparation.to_gate(label=f"prep_{_KIND_NAMES[kind]}")
    gate.name = f"prep_{_KIND_NAMES[kind]}"
    return gate


class _InputPreparations(dict):
    """Plan-owned input circuits with their lazily constructed controlled gates."""

    def __init__(self, preparations):
        super().__init__(preparations)
        self._gates = {}

    def gate(self, kind, state=None):
        """Return one input direction, optionally controlled on the branch value ``state``."""
        key = (kind, state)
        if key not in self._gates:
            self._gates[key] = (_preparation_gate(self[kind], kind) if state is None
                                else controlled(self.gate(kind), 1, ctrl_state=state))
        return self._gates[key]


def _input_gate(preparations, kind, state=None):
    """Reuse Plan-owned gates or build from the direct caller's preparation circuit."""
    if isinstance(preparations, _InputPreparations):
        return preparations.gate(kind, state)
    gate = _preparation_gate(preparations[kind], kind)
    return gate if state is None else controlled(gate, 1, ctrl_state=state)


class _PairWalk:
    """The gates of one pair circuit, recorded, or appended with one parameter per continuous angle.

    It offers the `QuantumCircuit` methods that `_walk_pair`,
    `select_synthesis._enter_pauli_rotation_frame`, its inverse and
    `_append_correction` call, on integer qubit positions: 0 is the branch
    qubit, 1 to r_c - 1 the control copies and r_c + j system qubit j, the
    qubit order of `build_pair_circuit`. ``p`` and ``crz`` take an angle,
    which opens a slot. Without a circuit the walk only records:
    ``values`` holds the slot values in order and ``skeleton`` every gate
    with its qubits and slot. With a circuit it also appends each gate, with
    the slot's parameter in place of its value.
    """

    def __init__(self, circuit=None, parameters=None, preparations=None):
        self.circuit, self._parameters, self._preparations = circuit, parameters, preparations
        self.skeleton, self.values = [], []

    def angle(self, value):
        """Open a slot holding ``value`` and return its index."""
        self.values.append(float(value))
        return len(self.values) - 1

    def append(self, name, qubits, slot=None):
        """Record one gate and, when building, append it.

        ``name`` is a `QuantumCircuit` method, ``"global_phase"`` (added to
        the circuit's phase) or ``("prep", kind, ctrl_state)`` for an input
        preparation, controlled on the branch value ``ctrl_state`` unless it
        is None.
        """
        self.skeleton.append((name, tuple(qubits), slot))
        if self.circuit is None:
            return
        wires = [self.circuit.qubits[qubit] for qubit in qubits]
        angle = () if slot is None else (self._parameters[slot],)
        if name == "global_phase":
            self.circuit.global_phase += angle[0]
        elif isinstance(name, tuple):
            _, kind, state = name
            self.circuit.append(_input_gate(self._preparations, kind, state), wires)
        else:
            getattr(self.circuit, name)(*angle, *wires)

    def h(self, qubit):
        self.append("h", (qubit,))

    def x(self, qubit):
        self.append("x", (qubit,))

    def s(self, qubit):
        self.append("s", (qubit,))

    def sdg(self, qubit):
        self.append("sdg", (qubit,))

    def cx(self, control, target):
        self.append("cx", (control, target))

    def cy(self, control, target):
        self.append("cy", (control, target))

    def cz(self, control, target):
        self.append("cz", (control, target))

    def p(self, value, qubit):
        self.append("p", (qubit,), self.angle(value))

    def crz(self, value, control, target):
        self.append("crz", (control, target), self.angle(value))


def _select_slots(walk, plan, implementation):
    """Open the slots of the plan's SELECT once and return ``(diagonal, occurrences)``.

    The SELECT of a PTSC plan repeats once per segment, and a repetition
    block repeats one angle table per occurrence, so each value takes one
    slot. ``diagonal`` is None when `append_control_diagonal_phases`
    emits no gate (both branch phases exactly 0), else the slots of its RZ
    angle (None when that angle is exactly 0) and of its global phase, the
    difference and mean of the wrapped phases that it computes.
    ``occurrences`` holds per occurrence table the slots of its two Walsh
    angles (multiplexor) or of its offset and coefficient (structured),
    None where the row builder omits the rotation.
    """
    if implementation == "structured":
        if not plan.structure_certificate or not plan.structure_certificate.get("eligible", False):
            raise ValueError("structured SELECT requires an eligible structure certificate")
        affine = plan.structured_generator_payload["occurrence_affine_tables"]
    elif implementation != "multiplexor":
        raise ValueError("implementation must be 'multiplexor' or 'structured'")
    padded = pad_angle_table(_combined_branch_phases(plan), control_qubits=1)
    diagonal = None
    if any(padded):
        wrapped = np.angle(np.exp(1.0j * np.asarray(padded, dtype=float)))
        difference = wrapped[1] - wrapped[0]
        diagonal = (walk.angle(difference) if difference != 0.0 else None,
                    walk.angle(wrapped[0] / 2 + wrapped[1] / 2))
    occurrences = []
    for index, table in enumerate(plan.occurrence_angle_tables.array):
        if implementation == "multiplexor":
            pair = gray_code_rotation_angles(pad_angle_table(table, control_qubits=1))
        else:
            pair = (affine[index].offset, affine[index].coefficients[0])
        occurrences.append(tuple(walk.angle(value) if value != 0.0 else None for value in pair))
    return diagonal, occurrences


def _walk_select(walk, plan, slots, *, implementation, system, controls, uses):
    """Walk one SELECT of the plan with its address gates moved to the control copies in turn, and return the next use.

    The gates are those of `build_product_formula_select`: the diagonal
    on the address, then per occurrence the Pauli frame, the address RZ
    multiplexor (RZ, CX from the address, RZ, CX, by
    `_multiplexors._rotation_multiplexor`) or the structured RZ and CRZ
    (`append_affine_rotation`), and the inverse frame. Every gate that the
    row builder places on the address takes the next copy, as in
    `build_pair_circuit`.
    """
    diagonal, occurrences = slots
    if diagonal is not None:
        address = controls[uses % len(controls)]
        uses += 1
        if diagonal[0] is not None:
            walk.append("rz", (address,), diagonal[0])
        walk.append("global_phase", (), diagonal[1])
    width = len(plan.occurrence_schedule)
    for block, repetitions in enumerate(plan.occurrence_block_repetitions):
        for _ in range(repetitions):
            for index, label in enumerate(plan.occurrence_schedule):
                first, second = occurrences[block * width + index]
                active = _enter_pauli_rotation_frame(walk, system_qubits=list(system), label=label)
                target = active[-1]
                if first is not None:
                    walk.append("rz", (target,), first)
                if implementation == "multiplexor":
                    address = controls[uses % len(controls)]
                    uses += 1
                    walk.cx(address, target)
                    if second is not None:
                        walk.append("rz", (target,), second)
                    walk.cx(address, target)
                elif second is not None:
                    address = controls[uses % len(controls)]
                    uses += 1
                    walk.append("crz", (address, target), second)
                _leave_pauli_rotation_frame(walk, active=active, label=label)
    return uses


def _walk_pair(walk, plan, *, kinds, theta, term_label, implementation, control_copies, corrections, prepares,
               select=True):
    """Walk the gates of `build_pair_circuit` with ``measure=False`` for one row, in its order, and return its SELECT slots.

    With ``select=False`` each copy of the SELECT is recorded as one
    ``("select",)`` marker instead of its gates, which `_select_slots` and
    the plan's occurrence structure determine (`pair_key`).
    """
    control_copies = integer(control_copies, "control_copies", 1)
    kind_x, kind_y = (int(kind) for kind in kinds)
    if not prepares and kind_x != kind_y:
        raise ValueError("branches of different kinds need their controlled input preparations")
    q = len(term_label)
    _check_widths(plan, term_label, q)
    if corrections is not None and plan.repetitions != 1:
        raise ValueError("PTSC corrections follow a one-step Lie plan")
    root, controls = 0, tuple(range(control_copies))
    system = tuple(range(control_copies, control_copies + q))
    walk.h(root)
    if prepares and kind_x == kind_y:
        walk.append(("prep", kind_x, None), system)
    elif prepares:
        for slot, kind in enumerate((kind_x, kind_y)):
            walk.append(("prep", kind, slot), (root, *system))
    fanout = _fanout_pairs(control_copies)
    for source, target in fanout:
        walk.cx(controls[source], controls[target])
    slots = _select_slots(walk, plan, implementation)
    uses = 0
    for segment in (None,) if corrections is None else corrections:
        if select:
            uses = _walk_select(walk, plan, slots, implementation=implementation, system=system,
                                controls=controls, uses=uses)
        else:
            walk.skeleton.append(("select",))
        for value, descriptor in enumerate(() if segment is None else segment):
            _append_correction(walk, root, system, descriptor, value)
    for source, target in reversed(fanout):
        walk.cx(controls[source], controls[target])
    walk.p(theta, root)
    walk.h(root)
    for qubit, axis in enumerate(reversed(term_label)):
        if axis == "Y":
            walk.sdg(system[qubit])
        if axis in "XY":
            walk.h(system[qubit])
    return slots


def pair_key(
    plan: LCHSProductFormulaSelectPlan,
    *,
    kinds: Sequence[int],
    theta: float,
    term_label: str,
    implementation: Literal["multiplexor", "structured"],
    control_copies: int = 1,
    corrections: Sequence[tuple[tuple, tuple]] | None = None,
    prepares: bool = True,
) -> tuple[tuple, tuple[float, ...]]:
    """Return a key of the gates of one row's pair circuit and its slot values, without building or walking its SELECT.

    The arguments are those of `build_pair_circuit`, with ``prepares``
    stating whether the circuit prepares the row's inputs (its
    ``preparations`` is not None). The key holds the gates outside the
    SELECT copies (input preparations, control fanout, PTSC corrections,
    branch phase and readout basis change), each with its qubit positions
    and slot, a marker in place of each SELECT copy, and what decides the
    SELECT's gates: the implementation, the widths, the occurrence schedule
    and block repetitions, and which diagonal and occurrence angles open a
    slot rather than being omitted as exactly 0. It costs one pass over the
    occurrence tables instead of one over every repeated occurrence.

    Returns:
        key (tuple): Rows with equal keys have the same gates and slots.
        values (tuple[float, ...]): The slot values in slot order.
    """
    walk = _PairWalk()
    diagonal, occurrences = _walk_pair(walk, plan, kinds=kinds, theta=theta, term_label=term_label,
                                       implementation=implementation, control_copies=control_copies,
                                       corrections=corrections, prepares=prepares, select=False)
    key = (tuple(walk.skeleton), implementation, control_copies, len(term_label),
           tuple(plan.occurrence_schedule), tuple(plan.occurrence_block_repetitions), diagonal, tuple(occurrences))
    return key, tuple(walk.values)


def pair_template_size(key: tuple) -> tuple[int, int, int]:
    """Return the instruction entries of a `pair_key`'s template, its CRZ entries and its declared bytes.

    The key fixes the walk of `build_pair_template`, so the count comes from
    its tables, with repetitions multiplied in, in time independent of the step
    counts. Take S SELECT markers, O other outside entries of which C_out are
    CRZ, and for label l of block b the repetitions r_b, the frame
    F_l = 2x + 4y + 2(w - 1) of its x X letters, y Y letters and support w, and
    the indicators f and g of its first and second slots. Then

        P = d + sum_b r_b sum_l (F_l + f + multiplexor*(2 + g))
        C_sel = structured*sum_b r_b sum_l g
        N = O + S*(P + C_sel)
        C = C_out + S*C_sel

    where d is 1 when the SELECT diagonal keeps its RZ. A slot is kept when it
    is not None, so slot index 0 is kept. The frame and its inverse are emitted
    even when both rotations are omitted, a multiplexor always emits its two
    CX, and a global phase adds no entry. O already holds every PTSC segment's
    corrections, whose gates are emitted even for an angle of 0, and each input
    preparation as its one wrapper entry.

    N equals ``len(build_pair_template(...).data)`` whatever values are bound
    later. It counts abstract instruction entries, not compiled native gates.

    The bytes 16384 + 256*N + 1024*C are a coarse allowance declared for one
    template, not a physical bound. The coefficients keep their meaning from
    `select_synthesis.product_formula_select_size`, measured there for standard
    gates. A CRZ entry takes both its 256 and its further 1024. The allowance
    does not price the shared body or the controlled synthesis behind a
    preparation entry, which stay outside it, nor the walk's temporary skeleton
    and parameter expressions, which are unmeasured, nor the copies made when a
    Run converts and compiles the template, which are the Run's to account for.

    Returns:
        entries (int): N.
        crz (int): C.
        bytes (int): The allowance.
    """
    skeleton, implementation, _, _, schedule, repetitions, diagonal, occurrences = key
    names = [entry[0] for entry in skeleton]
    segments = names.count("select")
    outside = len(names) - segments - names.count("global_phase")
    frames = [2*label.count("X") + 4*label.count("Y") + 2*(len(label) - label.count("I") - 1) for label in schedule]
    multiplexor = implementation == "multiplexor"
    plain = int(diagonal is not None and diagonal[0] is not None)
    crz = 0
    for block, count in enumerate(repetitions):
        for index, frame in enumerate(frames):
            first, second = occurrences[block*len(frames) + index]
            kept = second is not None
            plain += count*(frame + (first is not None) + (2 + kept if multiplexor else 0))
            crz += 0 if multiplexor else count*kept
    entries = outside + segments*(plain + crz)
    controlled = names.count("crz") + segments*crz
    return entries, controlled, _select_circuit_bytes(entries, controlled)


def build_pair_template(
    plan: LCHSProductFormulaSelectPlan,
    *,
    parameters: Sequence[Any],
    preparations: Sequence[QuantumCircuit | None] | Mapping[int, QuantumCircuit] | None,
    kinds: Sequence[int],
    theta: float,
    term_label: str,
    implementation: Literal["multiplexor", "structured"],
    control_copies: int = 1,
    corrections: Sequence[tuple[tuple, tuple]] | None = None,
    max_bytes=None,
) -> QuantumCircuit:
    """Build the parameterized pair circuit of one row's gates (module docstring).

    The circuit has the registers of `build_pair_circuit` with
    ``measure=False`` and its gates for this row, with ``parameters[i]`` in
    place of slot i. Binding slot values of `pair_key` for any row of
    the same key gives that row's gates and angles.

    Args:
        plan: The row's `pair_select_plan`.
        parameters: One Qiskit parameter per slot of the row's `pair_key`.
        preparations: As in `build_pair_circuit`.
        kinds: As in `build_pair_circuit`.
        theta: As in `build_pair_circuit`.
        term_label: As in `build_pair_circuit`.
        implementation: As in `build_pair_circuit`.
        control_copies: As in `build_pair_circuit`.
        corrections: As in `build_pair_circuit`.
        max_bytes: Default None, the physical memory. Limit for the allowance
            of `pair_template_size`, checked before the SELECT is walked. A
            template holds no separate SELECT, unlike `build_pair_circuit`.

    Returns:
        template (QuantumCircuit): The parameterized circuit.
    """
    key, _ = pair_key(plan, kinds=kinds, theta=theta, term_label=term_label, implementation=implementation,
                      control_copies=control_copies, corrections=corrections, prepares=preparations is not None)
    check_memory(pair_template_size(key)[2], max_bytes, "sampled pair template")
    q = len(term_label)
    copies = [QuantumRegister(control_copies - 1, "branch_copy")] if control_copies > 1 else []
    circuit = QuantumCircuit(QuantumRegister(1, "branch"), *copies, QuantumRegister(q, "system"),
                             name="lchs_sampled_pair_template")
    walk = _PairWalk(circuit, parameters, preparations)
    _walk_pair(walk, plan, kinds=kinds, theta=theta, term_label=term_label, implementation=implementation,
               control_copies=control_copies, corrections=corrections, prepares=preparations is not None)
    if len(walk.values) != len(parameters):
        raise ValueError("a pair template takes one parameter per slot of its row's skeleton")
    return circuit


def pair_circuit_resource_law(
    plan: LCHSProductFormulaSelectPlan,
    *,
    kinds: Sequence[int],
    term_label: str,
    implementation: Literal["multiplexor", "structured"],
    control_copies: int = 1,
) -> dict[str, Any]:
    """Return the gates `build_pair_circuit` emits outside the input preparations, from the plan alone.

    ``gate_counts`` are the counts after one-level expansion of the SELECT's
    ``ucrz``, ``crz`` and ``nwqlib_phase_diagonal`` gates and before any
    transpilation. With W scheduled occurrences, support size w_o of
    occurrence o, r steps and E occurrences (over all r steps) whose two
    angles are exactly equal,

        multiplexor CX = r*sum_o 2*w_o + 2*(r_c - 1),
        structured CX  = multiplexor CX - 2*E,

    because every occurrence has the 2*(w_o - 1) parity CX, the
    multiplexed RZ always has 2 CX, and the structured rotation has 2 CX
    only for unequal angles (`product_formula_select_resource_law` gives
    the SELECT part). RZ counts drop exactly-zero rotations as the builders
    do: a multiplexed table emits its nonzero Walsh angles
    (`gray_code_rotation_angles`), a structured table one RZ for a nonzero
    offset and the two RZ of Qiskit's CRZ for a nonzero coefficient, and the
    identity phase one RZ when its wrapped branch phases differ. Each X
    site of an occurrence costs two H and each Y site two H, one S-dagger
    and one S. The branch qubit has two H and one P, and the readout basis
    change of term_label one H per X or Y site and one S-dagger per Y site.

    The input preparations are counted as gate instances: one uncontrolled
    for equal kinds, two singly controlled for mixed kinds. Their
    elementary counts belong to their preparation constructors and Qiskit's
    control synthesis and are not derived here, so
    ``preparation_gate_counts`` is None. No circuit depth is given beyond
    the fanout layers.
    """
    control_copies = integer(control_copies, "control_copies", 1)
    q = len(term_label)
    _check_widths(plan, term_label, q)
    select = product_formula_select_resource_law(plan, implementation=implementation)
    width = len(plan.occurrence_schedule)
    tables = plan.occurrence_angle_tables.array
    affine = plan.structured_generator_payload["occurrence_affine_tables"] if implementation == "structured" else ()
    rz = equal = 0
    for block, repetitions in enumerate(plan.occurrence_block_repetitions):
        for index in range(block * width, (block + 1) * width):
            equal += repetitions * int(tables[index, 0] == tables[index, 1])
            if implementation == "multiplexor":
                emitted = int(np.count_nonzero(gray_code_rotation_angles(tables[index])))
            else:
                emitted = int(affine[index].offset != 0.0) + 2 * int(affine[index].coefficients[0] != 0.0)
            rz += repetitions * emitted
    phases = np.add(plan.coefficient_phases, plan.identity_phases)
    if np.any(phases != 0.0):
        wrapped = np.angle(np.exp(1.0j * phases))
        rz += int(wrapped[1] != wrapped[0])
    x_sites = plan.repetitions * sum(label.count("X") for label in plan.occurrence_schedule)
    y_sites = plan.repetitions * sum(label.count("Y") for label in plan.occurrence_schedule)
    mixed = int(kinds[0]) != int(kinds[1])
    gate_counts = {
        "cx": int(select["select_basis_cx_count"]) + 2 * (control_copies - 1),
        "rz": rz,
        "h": 2 + 2 * (x_sites + y_sites) + term_label.count("X") + term_label.count("Y"),
        "s": y_sites,
        "sdg": y_sites + term_label.count("Y"),
        "p": 1,
        "measure": q + 1,
    }
    return {
        "qubits": q + control_copies,
        "clbits": q + 1,
        "gate_counts": {name: count for name, count in gate_counts.items() if count},
        "preparation_count": 2 if mixed else 1,
        "controlled_preparation_count": 2 if mixed else 0,
        "preparation_gate_counts": None,
        "control_copies": control_copies,
        "fanout_cx_count": 2 * (control_copies - 1),
        "fanout_layer_count": 2 * (control_copies - 1).bit_length(),
        "equal_angle_occurrence_count": equal,
        "select": select,
    }


__all__ = [
    "LCHSSampledPairCircuit",
    "build_pair_circuit",
    "build_pair_template",
    "pair_circuit_resource_law",
    "pair_select_plan",
    "pair_key",
    "pair_template_size",
]
