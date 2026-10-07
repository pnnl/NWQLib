"""Internal shared owners for single-target multiplexors.

A multiplexor with ``k`` select qubits applies ``U_j`` to one target qubit
when the select register holds ``j`` (Shende, Bullock and Markov,
quant-ph/0406176v5, Sec. 3). State preparation, Pauli SELECT, banded block
encodings and LCHS SELECT share three constructions from this module:

* Uniformly controlled RY and RZ rotations follow Mottonen et al.,
  quant-ph/0407010v1, Sec. II, Fig. 2 and Eq. (3). This is Shende et al.,
  quant-ph/0406176v5, Theorem 8, after the CX cancellations of their Fig. 2,
  and it uses ``2**k`` CX for ``k >= 1`` controls.
* A diagonal of phases on a control register follows the recursion of
  Shende et al., quant-ph/0406176v5, Theorem 7, with one RZ multiplexor per
  qubit.
* Uniformly controlled single-qubit unitaries use Qiskit's ``UCGate`` after
  an exact projection onto the address bits on which the table depends.

Qiskit's uniformly controlled rotations and ``UCGate`` use target-first
qubit order: ``[target, control_0, ..., control_{a-1}]``, where
``control_0`` is the least-significant table-index bit.  The append helpers
below own that convention, validate the participating qubits, and pad short
tables to the full control-basis size.

The pinned exact arbitrary-unitary path is
``UCGate(gate_list, up_to_diagonal=False, mux_simp=False)``.  The default
``up_to_diagonal=False`` is already exact, so no compensation diagonal is
added by this module.  ``mux_simp=False`` prevents Qiskit's repeated-operator
search from simplifying identity padding and making structural counts depend
on table contents.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal, Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate, Qubit
    from qiskit.circuit.library import UCGate

import numpy as np



@dataclass(frozen=True)
class AffineAngleTable:
    """Affine angle law for a complete multiplexor table and its check.

    The law is ``angle(b) = offset + sum_q coefficients[q] * b_q`` over the
    address bits ``b_q`` of basis index ``b``, with ``q = 0`` the
    least-significant control. Rotations about one axis add their angles, so
    such a table needs no multiplexor. ``append_affine_rotation`` realizes it
    with one unconditional rotation and one singly controlled rotation per
    nonzero coefficient.

    Attributes:
        offset: Angle applied for every address, in radians.
        coefficients: Angle added when address bit ``q`` is one, in radians.
        max_residual: Largest absolute difference between the supplied
            complete table and the law, in radians.
        tolerance: Binary64 roundoff window for that comparison. The law
            describes the table only when ``max_residual <= tolerance``, and
            the consumer makes that decision.
    """

    offset: float
    coefficients: tuple[float, ...]
    max_residual: float
    tolerance: float


@dataclass(frozen=True)
class ProjectedUnitaryTable:
    """Exact dependency projection of a padded single-qubit unitary table.

    Attributes:
        support: Original address bits on which the table depends, in
            increasing order.
        projected_table: Table indexed by the supported bits only, with
            ``support[0]`` as its least-significant bit.
    """

    support: tuple[int, ...]
    projected_table: tuple[np.ndarray, ...]

    @property
    def effective_control_count(self) -> int:
        """Number of address bits on which the table actually depends."""

        return len(self.support)


def _basis_size(control_qubits: int) -> int:
    """Return ``2**control_qubits``, the number of table entries a complete multiplexor needs."""
    if control_qubits < 0:
        raise ValueError("control_qubits must be nonnegative")
    return 1 << control_qubits


def _validated_qubits(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
) -> tuple[Qubit, ...]:
    """Return ``controls`` as a tuple after checking distinctness and membership in ``circuit``.

    ``circuit.find_bit`` raises for a qubit that the circuit does not own.
    """
    controls = tuple(controls)
    ordered = (target, *controls)
    if len(set(ordered)) != len(ordered):
        raise ValueError("target and controls must be distinct qubits")
    for qubit in ordered:
        circuit.find_bit(qubit)
    return controls


def pad_angle_table(
    angles: Sequence[float],
    *,
    control_qubits: int,
) -> tuple[float, ...]:
    """Pad a nonempty finite angle table with identity angles (zeros)."""

    values = tuple(float(angle) for angle in angles)
    size = _basis_size(control_qubits)
    if not values:
        raise ValueError("angle table must not be empty")
    if len(values) > size:
        raise ValueError("angle table exceeds the control-register basis size")
    if not np.all(np.isfinite(values)):
        raise ValueError("angle table entries must be finite")
    return values + (0.0,) * (size - len(values))


def pad_unitary_table(
    unitaries: Sequence[np.ndarray],
    *,
    control_qubits: int,
) -> tuple[np.ndarray, ...]:
    """Pad a nonempty single-qubit-unitary table with identity matrices."""

    values = tuple(np.asarray(unitary, dtype=complex) for unitary in unitaries)
    size = _basis_size(control_qubits)
    if not values:
        raise ValueError("unitary table must not be empty")
    if len(values) > size:
        raise ValueError("unitary table exceeds the control-register basis size")
    for unitary in values:
        if unitary.shape != (2, 2):
            raise ValueError("unitary table entries must be 2 by 2 matrices")
        if not np.all(np.isfinite(unitary)):
            raise ValueError("unitary table entries must be finite")
    identities = tuple(np.eye(2, dtype=complex) for _ in range(size - len(values)))
    return tuple(unitary.copy() for unitary in values) + identities


def append_uniformly_controlled_ry(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    angles: Sequence[float],
) -> Gate:
    """Append a padded Gray-code RY multiplexor.

    Every CX is kept, and only exact-zero transformed rotations are omitted.
    """

    controls = _validated_qubits(circuit, target, controls)
    padded = pad_angle_table(angles, control_qubits=len(controls))
    gate = _rotation_multiplexor(padded, axis="y")
    circuit.append(gate, [target, *controls])
    return gate


def append_uniformly_controlled_rz(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    angles: Sequence[float],
) -> Gate:
    """Append a padded Gray-code RZ multiplexor.

    Every CX is kept, and only exact-zero transformed rotations are omitted.
    """

    controls = _validated_qubits(circuit, target, controls)
    padded = pad_angle_table(angles, control_qubits=len(controls))
    gate = _rotation_multiplexor(padded, axis="z")
    circuit.append(gate, [target, *controls])
    return gate


def _rotation_multiplexor(angles: Sequence[float], *, axis: Literal["y", "z"]) -> Gate:
    """Lower a complete angle table by the Walsh transform and cyclic Gray code.

    Implements Mottonen et al. (quant-ph/0407010v1), Sec. II, Fig. 2 and
    Eq. (3). The target receives ``2**k`` rotations separated by CX gates.
    Rotation ``i`` uses the angle
    ``theta_i = 2**-k * sum_j (-1)**popcount(j & g(i)) * alpha_j`` with the
    binary reflected Gray code ``g(i) = i ^ (i >> 1)``, and the butterfly
    loop evaluates this scaled Walsh-Hadamard transform in ``O(k 2**k)``
    operations. After rotation ``i`` the CX control is the bit in which
    ``g(i)`` and ``g(i + 1)`` differ, and the last CX closes the cycle on the
    top control. For address ``c``, rotation ``i`` therefore acts with sign
    ``(-1)**popcount(c & g(i))``, and the signed angles sum to ``alpha_c``.
    The sign flip ``X R(theta) X = R(-theta)`` holds for the Y and Z axes
    only, which is why only RY and RZ tables use this lowering. An exact-zero
    transformed angle omits its rotation but keeps the CX skeleton, so a
    table with ``k >= 1`` controls always costs ``2**k`` CX (Shende et al.,
    quant-ph/0406176v5, Theorem 8).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate


    schedule = gray_code_rotation_schedule(angles)
    control_count = len(schedule).bit_length() - 1
    definition = QuantumCircuit(control_count + 1)
    rotation = definition.ry if axis == "y" else definition.rz
    for angle, control in schedule:
        if angle != 0.0:
            rotation(angle, 0)
        if control is not None:
            definition.cx(1 + control, 0)
    gate = Gate(f"ucr{axis}", control_count + 1, list(angles))
    gate.definition = definition
    return gate


def gray_code_rotation_schedule(angles: Sequence[float]) -> tuple[tuple[float, int | None], ...]:
    """Return the rotation angles and CX controls of the cyclic Gray-code multiplexor.

    Entry ``i`` is ``(theta_i, c_i)``. The target receives the rotation by
    ``theta_i`` and then a CX from control ``c_i``, counted from zero among
    the controls. ``_rotation_multiplexor`` gives the angle formula and the
    control rule. The last entry closes the cycle on the top control, and
    ``c_i`` is None when the table has no controls. The quantum Shannon
    decomposition of ``_dense_synthesis`` omits that closing CX where the
    merge of Krol and Al-Ars, arXiv:2403.13692v2, Sec. 5.2, moves it into the
    middle block as a CZ.
    """
    values = gray_code_rotation_angles(angles)
    return tuple(zip(values.tolist(), gray_code_controls(len(values))))


def gray_code_rotation_angles(angles: Any) -> np.ndarray:
    """Return the angles ``theta_i`` of :func:`gray_code_rotation_schedule` for every table on the last axis of ``angles``.

    The quantum Shannon decomposition of ``_dense_synthesis`` passes the
    tables of all multiplexors of one kind at one recursion depth as one
    array, and each butterfly level below updates all of them at once.
    """
    values = np.array(angles, dtype=float)
    size = values.shape[-1]
    # In-place butterfly: after the k halving levels, values[..., j] equals
    # 2**-k * sum_c (-1)**popcount(j & c) * alpha_c.
    stride = 1
    while stride < size:
        blocks = values.reshape(*values.shape[:-1], -1, 2 * stride)
        left = blocks[..., :stride].copy()
        right = blocks[..., stride:].copy()
        blocks[..., :stride] = left / 2 + right / 2
        blocks[..., stride:] = left / 2 - right / 2
        stride *= 2
    return values[..., [index ^ (index >> 1) for index in range(size)]]


@lru_cache(maxsize=None)
def gray_code_controls(size: int) -> tuple[int | None, ...]:
    """Return the CX controls ``c_i`` of :func:`gray_code_rotation_schedule` for a table of ``size`` angles."""
    control_count = size.bit_length() - 1
    if not control_count:
        return (None,) * size
    # g(i) and g(i + 1) differ in the lowest set bit of i + 1, which
    # (i + 1) & -(i + 1) isolates. The last g(2**k - 1) differs from
    # g(0) = 0 in the top bit, which closes the cycle.
    return tuple(((index + 1) & -(index + 1)).bit_length() - 1 for index in range(size - 1)) + (control_count - 1,)


def append_uniformly_controlled_unitaries(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    unitaries: Sequence[np.ndarray],
) -> UCGate:
    """Append the pinned exact, unsimplified ``UCGate`` construction."""
    from qiskit.circuit.library import UCGate


    controls = _validated_qubits(circuit, target, controls)
    padded = pad_unitary_table(unitaries, control_qubits=len(controls))
    gate = UCGate(list(padded), up_to_diagonal=False, mux_simp=False)
    circuit.append(gate, [target, *controls])
    return gate


def project_unitary_table_dependencies(
    unitaries: Sequence[np.ndarray],
    *,
    control_qubits: int,
) -> ProjectedUnitaryTable:
    """Project a padded table onto the address bits it exactly depends on.

    Bit ``q`` belongs to the support exactly when at least one pair of table
    entries whose indices differ only in bit ``q`` are not entrywise equal.
    The projected table is ordered by the supported bits in increasing
    original-bit order, preserving Qiskit's least-significant-control
    convention.

    A multiplexor whose operation does not change with some select bits need
    not read them (Shende et al., quant-ph/0406176v5, Sec. 3). The test is
    exact entrywise equality, so the projected multiplexor is the same
    operator. Pauli SELECT benefits because the local factor table of one
    system qubit often depends on few address bits, and each removed control
    roughly halves that qubit's UCG cost (``projected_unitary_resource_law``).
    """

    padded = pad_unitary_table(unitaries, control_qubits=control_qubits)
    support = tuple(
        bit
        for bit in range(control_qubits)
        if any(
            not np.array_equal(padded[index], padded[index ^ (1 << bit)])
            for index in range(len(padded))
            if not (index & (1 << bit))
        )
    )
    projected = tuple(
        padded[
            sum(
                ((projected_index >> position) & 1) << original_bit
                for position, original_bit in enumerate(support)
            )
        ].copy()
        for projected_index in range(1 << len(support))
    )
    return ProjectedUnitaryTable(
        support=support,
        projected_table=projected,
    )


def local_pauli_dependencies(x: np.ndarray, z: np.ndarray, system_bit: int) -> tuple[int, ...]:
    """Return the address bits on which one system qubit's local Pauli letters depend.

    ``x`` and ``z`` are the packed Pauli masks of an admitted table, uint64 of
    shape ``(m, words)``, qubit zero first (bit j of word j // 64 is system
    qubit j, as ``operators._pauli._pauli_masks`` numbers them). For address
    k < m define ``c_k = x_k,j + 2 z_k,j``, representing I, X, Z, Y by 0, 1, 2,
    3, and pad to P = 2**a addresses, a = bit_length(m - 1), with zero. The
    map from these codes to canonical local Pauli matrices is injective under
    entrywise equality, so

        exists k with k_b = 0 and c_k != c_(k xor 2**b)
        iff exists k with k_b = 0 and U_k != U_(k xor 2**b),

    whose right side is the support predicate of
    ``project_unitary_table_dependencies``, and the ordered support agrees.
    Coefficient phases stay in their separate diagonal, and the equivalence
    does not cover a local table that incorporates them. One term has no
    address bits, identical nonidentity letters can acquire dependencies through
    identity padding, and all-identity tables have empty support. An empty
    table is rejected, as the entrywise projection rejects it.
    """
    m = len(x)
    if m == 0:
        raise ValueError("unitary table must not be empty")
    a = (m - 1).bit_length()
    codes = np.zeros(1 << a, dtype=np.uint8)
    word, bit = divmod(system_bit, 64)
    xx = ((x[:, word] >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
    zz = ((z[:, word] >> np.uint64(bit)) & np.uint64(1)).astype(np.uint8)
    codes[:m] = xx | (zz << 1)
    return tuple(b for b in range(a)
                 if np.any(codes.reshape(-1, 2, 1 << b)[:, 0, :]
                           != codes.reshape(-1, 2, 1 << b)[:, 1, :]))


def project_local_pauli_table(
    labels: Sequence[str], system_bit: int, support: tuple[int, ...],
) -> ProjectedUnitaryTable:
    """Return the projected local Pauli table of one system qubit on its known support.

    Entry p of the projected table is the canonical local Pauli matrix of
    the label at the address whose supported bits are the bits of p, in
    increasing supported-bit order, and whose unsupported bits are zero.
    An address at or beyond ``len(labels)`` is identity padding. Flipping an
    unsupported bit leaves the entry unchanged, so fixing all unsupported
    bits to zero gives the same projected table as
    ``project_unitary_table_dependencies`` (see ``local_pauli_dependencies``).
    ``label[-1 - system_bit]`` is the letter on that qubit.
    """
    from nwqlib.subroutines.pauli_decomposition import _LOCAL_PAULI_FACTORS

    table = []
    for projected_index in range(1 << len(support)):
        address = sum(((projected_index >> position) & 1) << bit for position, bit in enumerate(support))
        letter = labels[address][-1 - system_bit] if address < len(labels) else "I"
        table.append(_LOCAL_PAULI_FACTORS[letter].copy())
    return ProjectedUnitaryTable(support=support, projected_table=tuple(table))


def append_projected_unitary_table(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    projection: ProjectedUnitaryTable,
) -> ProjectedUnitaryTable:
    """Append the exact UCG of an already projected table on its support controls."""
    from qiskit.circuit.library import UnitaryGate

    controls = _validated_qubits(circuit, target, controls)
    if projection.effective_control_count:
        append_uniformly_controlled_unitaries(
            circuit,
            target,
            [controls[index] for index in projection.support],
            projection.projected_table,
        )
    elif not np.array_equal(projection.projected_table[0], np.eye(2, dtype=complex)):
        circuit.append(UnitaryGate(projection.projected_table[0]), [target])
    return projection


def append_dependency_projected_unitaries(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    unitaries: Sequence[np.ndarray],
) -> ProjectedUnitaryTable:
    """Append the exact UCG on only the table's dependency-support controls."""
    controls = _validated_qubits(circuit, target, controls)
    projection = project_unitary_table_dependencies(
        unitaries,
        control_qubits=len(controls),
    )
    return append_projected_unitary_table(circuit, target, controls, projection)


def append_control_diagonal_phases(
    circuit: QuantumCircuit,
    controls: Sequence[Qubit],
    phases: Sequence[float],
) -> Gate | None:
    """Append one padded control-register diagonal of phase angles.

    Exact-zero tables construct no gate. Nonzero tables use the Gray-code RZ
    multiplexors of ``append_uniformly_controlled_rz``, which keep every CX.
    With no controls the single branch phase is applied as the circuit
    global phase.

    The lowering is the recursion of Shende et al. (quant-ph/0406176v5),
    Theorem 7. On the least-significant remaining qubit,
    ``diag(exp(i phi_0), exp(i phi_1)) = exp(i (phi_0 + phi_1) / 2) RZ(phi_1 - phi_0)``
    in Qiskit's ``RZ(theta) = exp(-i theta Z / 2)`` convention. Each level is
    therefore one RZ multiplexor on the pairwise differences, the pairwise
    means form the diagonal on the remaining qubits, and the last mean is the
    global phase. With ``a`` controls this costs ``2**a - 2`` CX
    (``multiplexor_resource_law``).
    """
    from qiskit import QuantumCircuit
    from qiskit.circuit import Gate


    controls = tuple(controls)
    if len(set(controls)) != len(controls):
        raise ValueError("controls must be distinct qubits")
    for qubit in controls:
        circuit.find_bit(qubit)
    padded = pad_angle_table(phases, control_qubits=len(controls))
    if not any(padded):
        return None
    if not controls:
        circuit.global_phase += padded[0]
        return None
    definition = QuantumCircuit(len(controls))
    # Wrapping each phase to (-pi, pi] leaves the diagonal unchanged and keeps
    # every derived RZ angle below 2*pi in magnitude.
    values = np.angle(np.exp(1.0j * np.asarray(padded, dtype=float)))
    for target in range(len(controls)):
        pairs = values.reshape(-1, 2)
        append_uniformly_controlled_rz(
            definition,
            definition.qubits[target],
            definition.qubits[target + 1 :],
            pairs[:, 1] - pairs[:, 0],
        )
        values = pairs[:, 0] / 2 + pairs[:, 1] / 2
    definition.global_phase = float(values[0])
    # Aer reserves "diagonal" for complex entries, not the phase-angle parameters here.
    gate = Gate("nwqlib_phase_diagonal", len(controls), list(padded))
    gate.definition = definition
    circuit.append(gate, list(controls))
    return gate


def affine_angle_table_values(
    *,
    offset: float,
    coefficients: Sequence[float],
) -> tuple[float, ...]:
    """Reconstruct a complete affine bit table in basis-index order."""

    offset = float(offset)
    values = tuple(float(value) for value in coefficients)
    if not np.isfinite(offset) or not np.all(np.isfinite(values)):
        raise ValueError("affine offset and coefficients must be finite")
    return tuple(
        offset
        + sum(
            coefficient
            for qubit, coefficient in enumerate(values)
            if branch & (1 << qubit)
        )
        for branch in range(1 << len(values))
    )


def _affine_angle_tolerance(offset: float, coefficients: Sequence[float]) -> float:
    """Return the binary64 window for comparing a table with its affine law.

    The window is ``gamma_m * S`` with ``S = |offset| + sum_q |coefficients[q]|``,
    ``gamma_m = m * eps / (1 - m * eps)``, ``m = 2k + 1`` for ``k`` controls
    and machine epsilon ``eps = 2**-52``, twice the unit roundoff ``u``. S
    bounds the magnitude of every entry of the law.

    ``affine_angle_table_values`` forms an entry as the offset plus the sum
    of the coefficients of its set bits. That is at most k rounded additions
    and no products, so the reconstruction differs from the exact law by at
    most ``k u S / (1 - k u)``. The window is more than four times that bound.
    The remainder absorbs the rounding in the supplied table, which its
    producer computed by its own arithmetic. For the LCHS product-formula
    tables, the registry row in docs/ENGINEERING_CONSTANTS.md counts both
    computations' roundings and shows that the window covers them to first
    order in u. A larger residual rejects the affine law. Revisit with
    changed arithmetic, another table producer or supported precision.
    """
    control_qubits = len(coefficients)
    operations = max(1, 2 * control_qubits + 1)
    epsilon = np.finfo(float).eps
    gamma = operations * epsilon / (1.0 - operations * epsilon)
    scale = abs(offset) + sum(abs(value) for value in coefficients)
    return float(gamma * scale)


def certify_affine_angle_table(
    angles: Sequence[float],
    *,
    offset: float,
    coefficients: Sequence[float],
) -> AffineAngleTable:
    """Compare an independently derived affine law with a complete table.

    The result reports the residual and its roundoff window without deciding.
    The LCHS structured SELECT accepts the law only when
    ``max_residual <= tolerance`` for every occurrence table.
    """

    values = tuple(float(angle) for angle in angles)
    coefficients = tuple(float(value) for value in coefficients)
    reconstructed = affine_angle_table_values(
        offset=offset,
        coefficients=coefficients,
    )
    if len(values) != len(reconstructed):
        raise ValueError("affine certificate width must match the complete angle table")
    if not np.all(np.isfinite(values)):
        raise ValueError("affine angle table entries must be finite")
    max_residual = float(np.max(np.abs(np.asarray(values) - np.asarray(reconstructed))))
    tolerance = _affine_angle_tolerance(float(offset), coefficients)
    return AffineAngleTable(
        offset=float(offset),
        coefficients=coefficients,
        max_residual=max_residual,
        tolerance=tolerance,
    )


def append_affine_rotation(
    circuit: QuantumCircuit,
    target: Qubit,
    controls: Sequence[Qubit],
    affine: AffineAngleTable,
    *,
    axis: Literal["y", "z"],
) -> None:
    """Lower an affine table to one rotation plus active bit controls.

    Rotations about one axis commute and add their angles, so
    ``R(offset + sum_q c_q b_q) = R(offset) * prod_q R(c_q)**b_q``. The
    ``2**k``-entry multiplexor therefore becomes at most one unconditional
    rotation (no CX) and one singly controlled rotation per nonzero
    coefficient (two CX each, ``affine_rotation_resource_law``). The caller supplies a law whose
    residual it has already accepted.
    """
    from nwqlib.subroutines.qiskit_compat import controlled
    from qiskit.circuit.library import RYGate, RZGate


    controls = _validated_qubits(circuit, target, controls)
    if len(affine.coefficients) != len(controls):
        raise ValueError("affine coefficient count must match the control register")
    rotation_type = RYGate if axis == "y" else RZGate
    if axis not in ("y", "z"):
        raise ValueError("axis must be 'y' or 'z'")
    if affine.offset != 0.0:
        circuit.append(rotation_type(affine.offset), [target])
    for control, angle in zip(controls, affine.coefficients):
        if angle != 0.0:
            circuit.append(controlled(rotation_type(angle), 1), [control, target])


def multiplexor_resource_law(
    kind: Literal["ucry", "ucrz", "diagonal"],
    *,
    control_qubits: int,
) -> dict[str, int]:
    """Return native rotation-multiplexor/diagonal slots and basis-CX counts.

    The counts describe the constructors in this module before transpiler
    optimization. A rotation multiplexor with ``k >= 1`` controls uses
    ``2**k`` CX (Shende et al., quant-ph/0406176v5, Theorem 8 and Fig. 2,
    p. 11, and Mottonen et al., quant-ph/0407010v1, Sec. II, p. 2). A
    diagonal on ``a`` qubits sums that count over the Theorem 7 recursion
    (p. 10), whose last level has no control, giving
    ``sum_{k=1}^{a-1} 2**k = 2**a - 2``.
    """

    if control_qubits < 0:
        raise ValueError("control_qubits must be nonnegative")
    table_size = 1 << control_qubits
    if kind in ("ucry", "ucrz"):
        return {
            "table_entries": table_size,
            "multiplexor_gates": 1,
            "control_diagonal_gates": 0,
            "basis_cx_gates": 0 if control_qubits == 0 else table_size,
        }
    if kind == "diagonal":
        return {
            "table_entries": table_size,
            "multiplexor_gates": 0,
            "control_diagonal_gates": int(control_qubits > 0),
            "basis_cx_gates": max(0, table_size - 2),
        }
    raise ValueError("kind must be 'ucry', 'ucrz', or 'diagonal'")


def projected_unitary_resource_law(effective_control_count: int) -> dict[str, int]:
    """Return the exact pinned UCG core/completion CX split.

    These are the CX counts of Qiskit's pinned
    ``UCGate(up_to_diagonal=False, mux_simp=False)`` synthesis with ``k``
    effective controls. The multiplexor up to a diagonal uses ``2**k - 1``
    CX, and the diagonal that completes it uses
    ``2 * (2**k - 1) = 2**(k + 1) - 2`` CX, the diagonal law of
    ``multiplexor_resource_law`` at width ``k + 1``. They describe the SDK
    construction rather than a paper bound, so a change in Qiskit's UCGate
    synthesis requires rechecking them.
    """

    if effective_control_count < 0:
        raise ValueError("effective_control_count must be nonnegative")
    core = (1 << effective_control_count) - 1
    completion = 2 * core
    return {
        "effective_control_count": effective_control_count,
        "ucg_core_cx": core,
        "ucg_completion_diagonal_cx": completion,
        "basis_cx_gates": core + completion,
    }


def affine_rotation_resource_law(affine: AffineAngleTable) -> dict[str, int]:
    """Return structural and exact basis-CX counts for affine lowering.

    The offset is one unconditional rotation without CX. Each nonzero
    coefficient is one singly controlled rotation, the singly multiplexed
    rotation ``R(0) (+) R(c)``, which uses two CX (Shende et al.,
    quant-ph/0406176v5, Theorem 4, p. 9).
    """

    controlled_rotations = sum(angle != 0.0 for angle in affine.coefficients)
    return {
        "unconditional_rotations": int(affine.offset != 0.0),
        "controlled_rotations": controlled_rotations,
        "basis_cx_gates": 2 * controlled_rotations,
    }


def product_formula_select_resource_law(
    plan: Any,
    *,
    implementation: Literal["multiplexor", "structured"] = "multiplexor",
) -> dict[str, Any]:
    """Return the resolved product-formula SELECT structural census.

    The input is the immutable LCHS product-formula plan.  ``Any`` avoids a
    module cycle: the plan type lives in ``algorithms.lchs.select_synthesis``,
    whose circuit builders already depend on this multiplexor owner.

    Counts describe SELECT only.  In particular, ``arbitrary_rotation_count``
    is the raw SELECT rotation-slot count (UCRZ table entries plus independent
    control-diagonal phases), not a fault-tolerant T-count and not a PREP
    count.  PREP is priced separately by the resource estimator.

    Each occurrence of a Pauli rotation with support ``w`` adds ``2 (w - 1)``
    CX for the parity ladder that the SELECT builder enters before the
    rotation and undoes after it. The ``"structured"`` census applies only to
    a plan whose affine certificate is eligible and replaces each complete
    UCRZ table with its affine lowering.
    """

    padded_branches = int(plan.padded_node_count)
    if padded_branches < 1 or padded_branches & (padded_branches - 1):
        raise ValueError("padded SELECT branch count must be a positive power of two")
    physical_branches = int(plan.physical_node_count)
    if not 0 < physical_branches <= padded_branches:
        raise ValueError("physical SELECT branch count must fit the padded table")

    schedule_width = len(plan.occurrence_schedule)
    block_repetitions = tuple(plan.occurrence_block_repetitions)
    stored_occurrence_count = len(plan.occurrence_angle_tables)
    if (
        stored_occurrence_count != schedule_width * len(block_repetitions)
        or any(type(value) is not int or value < 1 for value in block_repetitions)
        or sum(block_repetitions) != int(plan.repetitions)
    ):
        raise ValueError("SELECT occurrence blocks must cover the repeated schedule")
    occurrence_count = schedule_width * int(plan.repetitions)
    if any(len(table) != padded_branches for table in plan.occurrence_angle_tables.array):
        raise ValueError("every SELECT angle table must span the padded branch table")

    control_qubits = padded_branches.bit_length() - 1
    labels = plan.occurrence_schedule
    support_sizes = tuple(sum(pauli != "I" for pauli in label) for label in labels)
    if any(size < 1 for size in support_sizes):
        raise ValueError("product-formula occurrence labels must be non-identity")

    ucr_law = multiplexor_resource_law("ucrz", control_qubits=control_qubits)
    diagonal_law = multiplexor_resource_law(
        "diagonal", control_qubits=control_qubits
    )
    multiplexed_cx = occurrence_count * ucr_law["basis_cx_gates"]
    parity_cx = sum(2 * (support_size - 1) for support_size in support_sizes) * int(plan.repetitions)
    has_diagonal = any(
        coefficient + identity != 0.0
        for coefficient, identity in zip(
            plan.coefficient_phases, plan.identity_phases, strict=True
        )
    )
    diagonal_cx = diagonal_law["basis_cx_gates"] if has_diagonal else 0
    independent_diagonal_phases = max(0, padded_branches - 1) if has_diagonal else 0
    if implementation == "structured":
        certificate = plan.structure_certificate
        payload = plan.structured_generator_payload
        if not certificate or not certificate.get("eligible", False):
            raise ValueError("structured SELECT resource law requires an eligible certificate")
        if not payload:
            raise ValueError("structured SELECT resource law requires a generator payload")
        affine_tables = tuple(payload.get("occurrence_affine_tables", ()))
        if len(affine_tables) != stored_occurrence_count:
            raise ValueError("structured payload must cover every formula occurrence")
        rotation_laws = tuple(affine_rotation_resource_law(table) for table in affine_tables)
        unconditional_rotations = sum(
            law["unconditional_rotations"] * block_repetitions[index // schedule_width]
            for index, law in enumerate(rotation_laws)
        )
        controlled_rotations = sum(
            law["controlled_rotations"] * block_repetitions[index // schedule_width]
            for index, law in enumerate(rotation_laws)
        )
        structured_cx = sum(
            law["basis_cx_gates"] * block_repetitions[index // schedule_width]
            for index, law in enumerate(rotation_laws)
        )
        return {
            "select_physical_branch_count": physical_branches,
            "select_padded_branch_count": padded_branches,
            "select_formula_order": int(plan.formula_order),
            "select_formula_occurrence_count": occurrence_count,
            "select_stored_occurrence_count": stored_occurrence_count,
            "select_occurrence_block_repetitions": list(block_repetitions),
            "select_multiplexed_rotation_gate_count": 0,
            "select_multiplexed_angle_slot_count": 0,
            "select_structured_unconditional_rotation_count": unconditional_rotations,
            "select_structured_controlled_rotation_count": controlled_rotations,
            "select_max_structured_control_degree": int(controlled_rotations > 0),
            "select_control_diagonal_entry_count": padded_branches,
            "select_branch_controlled_full_unitary_count": 0,
            "select_multiplexed_rotation_basis_cx_count": 0,
            "select_parity_network_basis_cx_count": parity_cx,
            "select_control_diagonal_basis_cx_count": diagonal_cx,
            "select_basis_cx_count": structured_cx + parity_cx + diagonal_cx,
            "arbitrary_rotation_count": (
                unconditional_rotations
                + controlled_rotations
                + independent_diagonal_phases
            ),
            "rotation_synthesis_precision": None,
        }
    if implementation != "multiplexor":
        raise ValueError("implementation must be 'multiplexor' or 'structured'")
    return {
        "select_physical_branch_count": physical_branches,
        "select_padded_branch_count": padded_branches,
        "select_formula_order": int(plan.formula_order),
        "select_formula_occurrence_count": occurrence_count,
        "select_stored_occurrence_count": stored_occurrence_count,
        "select_occurrence_block_repetitions": list(block_repetitions),
        "select_multiplexed_rotation_gate_count": occurrence_count,
        "select_multiplexed_angle_slot_count": occurrence_count * padded_branches,
        "select_structured_unconditional_rotation_count": 0,
        "select_structured_controlled_rotation_count": 0,
        "select_max_structured_control_degree": 0,
        "select_control_diagonal_entry_count": padded_branches,
        "select_branch_controlled_full_unitary_count": 0,
        "select_multiplexed_rotation_basis_cx_count": multiplexed_cx,
        "select_parity_network_basis_cx_count": parity_cx,
        "select_control_diagonal_basis_cx_count": diagonal_cx,
        "select_basis_cx_count": multiplexed_cx + parity_cx + diagonal_cx,
        "arbitrary_rotation_count": (
            occurrence_count * padded_branches + independent_diagonal_phases
        ),
        "rotation_synthesis_precision": None,
    }
