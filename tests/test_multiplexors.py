"""Independent matrix and structural tests for internal multiplexor owners.

Tolerance class: these are construction-exact circuit identities.  Each
matrix allowance is derived from matrix dimension, the closed-form basis-CX
count, and binary64 epsilon; no synthesis or method approximation is present.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit import ControlledGate
from qiskit.quantum_info import Operator

from nwqlib.subroutines._multiplexors import (
    AffineAngleTable,
    affine_rotation_resource_law,
    append_affine_rotation,
    append_control_diagonal_phases,
    append_uniformly_controlled_ry,
    append_uniformly_controlled_rz,
    append_uniformly_controlled_unitaries,
    certify_affine_angle_table,
    multiplexor_resource_law,
    projected_unitary_resource_law,
)


def _matrix_tolerance(dimension: int, basis_cx_gates: int) -> float:
    # Machine-precision construction class: an eight-operation rounding
    # envelope per matrix entry and decomposed basis operation.
    return (
        8.0
        * dimension
        * max(1, basis_cx_gates + 1)
        * np.finfo(float).eps
    )


def _rotation(axis: str, angle: float) -> np.ndarray:
    cosine = np.cos(angle / 2.0)
    sine = np.sin(angle / 2.0)
    if axis == "y":
        return np.array([[cosine, -sine], [sine, cosine]], dtype=complex)
    return np.diag([np.exp(-0.5j * angle), np.exp(0.5j * angle)])


def test_tiny_uniform_rotations_survive_execution_decomposition() -> None:
    control_qubits = 3
    angles = [0.0] * (1 << control_qubits)
    angles[-1] = 1.0e-11
    circuit = QuantumCircuit(control_qubits + 1)
    append_uniformly_controlled_rz(circuit, circuit.qubits[0], circuit.qubits[1:], angles)
    actual = Operator(circuit.decompose(reps=10)).data[-2:, -2:]
    observed = (actual[1, 1] / actual[0, 0]).imag / 2
    assert observed / 5.0e-12 == pytest.approx(1.0, rel=1.0e-12, abs=0.0)


def _single_qubit_unitary(index: int) -> np.ndarray:
    return _rotation("y", 0.19 + 0.037 * index)


def _block_diagonal(blocks: list[np.ndarray]) -> np.ndarray:
    return scipy.linalg.block_diag(*blocks)


def _embedded_reference(
    blocks: list[np.ndarray],
    *,
    num_qubits: int,
    target: int,
    controls: tuple[int, ...],
) -> np.ndarray:
    """Explicit little-endian dense embedding, independent of Qiskit gates."""

    dimension = 1 << num_qubits
    reference = np.zeros((dimension, dimension), dtype=complex)
    for column in range(dimension):
        branch = sum(
            ((column >> control) & 1) << table_bit
            for table_bit, control in enumerate(controls)
        )
        input_target = (column >> target) & 1
        for output_target in (0, 1):
            row = (column & ~(1 << target)) | (output_target << target)
            reference[row, column] = blocks[branch][output_target, input_target]
    return reference


@pytest.mark.parametrize("control_qubits", range(1, 5))
def test_constant_unitary_tables_keep_the_declared_unsimplified_cx_law(control_qubits):
    circuit = QuantumCircuit(control_qubits + 1)
    append_uniformly_controlled_unitaries(
        circuit, circuit.qubits[0], circuit.qubits[1:],
        [np.eye(2, dtype=complex)] * (1 << control_qubits))
    # A content-based simplifier removes these controls. The selected fixed
    # construction must still realize its 3*(2**a-1) CX decomposition slots.
    assert _recursive_operation_counts(circuit)["cx"] == 3*((1 << control_qubits)-1)


@pytest.mark.parametrize("control_qubits", (3,))
@pytest.mark.parametrize("axis", ("y", "z"))
def test_uniformly_controlled_rotations_match_independent_dense_blocks(
    control_qubits: int,
    axis: str,
) -> None:
    rng = np.random.default_rng(100 + 10 * control_qubits + ord(axis))
    angles = rng.normal(size=1 << control_qubits)
    circuit = QuantumCircuit(control_qubits + 1)
    append = (
        append_uniformly_controlled_ry
        if axis == "y"
        else append_uniformly_controlled_rz
    )
    append(circuit, circuit.qubits[0], circuit.qubits[1:], angles)

    reference = _block_diagonal([_rotation(axis, angle) for angle in angles])
    law = multiplexor_resource_law(f"ucr{axis}", control_qubits=control_qubits)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= _matrix_tolerance(
        reference.shape[0], law["basis_cx_gates"]
    )


@pytest.mark.parametrize("control_qubits", (3,))
def test_uniformly_controlled_unitaries_match_independent_dense_blocks(
    control_qubits: int,
) -> None:
    unitaries = [_single_qubit_unitary(index) for index in range(1 << control_qubits)]
    circuit = QuantumCircuit(control_qubits + 1)
    gate = append_uniformly_controlled_unitaries(
        circuit,
        circuit.qubits[0],
        circuit.qubits[1:],
        unitaries,
    )

    assert gate.up_to_diagonal is False
    assert gate.simp_contr[0] is False
    reference = _block_diagonal(unitaries)
    law = projected_unitary_resource_law(control_qubits)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= _matrix_tolerance(
        reference.shape[0], law["basis_cx_gates"]
    )


@pytest.mark.parametrize("control_qubits", (3,))
def test_control_diagonal_phases_match_independent_dense_diagonal(
    control_qubits: int,
) -> None:
    # An affine table gives every recursive RZ multiplexor a constant angle
    # table, which hides a permuted control order. Irregular phases do not.
    phases = np.random.default_rng(700 + control_qubits).uniform(
        -np.pi, np.pi, size=1 << control_qubits
    )
    circuit = QuantumCircuit(control_qubits)
    append_control_diagonal_phases(circuit, circuit.qubits, phases)

    reference = np.diag(np.exp(1.0j * np.asarray(phases)))
    law = multiplexor_resource_law("diagonal", control_qubits=control_qubits)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= _matrix_tolerance(
        reference.shape[0], law["basis_cx_gates"]
    )


def test_padding_and_permuted_qubit_order_match_explicit_embedding() -> None:
    """A permuted control/target layout and identity padding expose errors hidden by contiguous
    register tests.
    """
    circuit = QuantumCircuit(4)
    target = circuit.qubits[2]
    controls = (circuit.qubits[3], circuit.qubits[0])
    angles = (0.21, -0.38, 0.57)
    append_uniformly_controlled_ry(circuit, target, controls, angles)
    padded_angles = [*angles, 0.0]
    reference = _embedded_reference(
        [_rotation("y", angle) for angle in padded_angles],
        num_qubits=4,
        target=2,
        controls=(3, 0),
    )
    law = multiplexor_resource_law("ucry", control_qubits=2)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= _matrix_tolerance(
        reference.shape[0], law["basis_cx_gates"]
    )

    unitary_circuit = QuantumCircuit(4)
    unitaries = [_single_qubit_unitary(index) for index in range(3)]
    append_uniformly_controlled_unitaries(
        unitary_circuit,
        unitary_circuit.qubits[2],
        (unitary_circuit.qubits[3], unitary_circuit.qubits[0]),
        unitaries,
    )
    unitary_reference = _embedded_reference(
        [*unitaries, np.eye(2, dtype=complex)],
        num_qubits=4,
        target=2,
        controls=(3, 0),
    )
    uc_law = projected_unitary_resource_law(2)
    assert np.max(
        np.abs(Operator(unitary_circuit).data - unitary_reference)
    ) <= _matrix_tolerance(unitary_reference.shape[0], uc_law["basis_cx_gates"])


@pytest.mark.parametrize("control_qubits", (3,))
@pytest.mark.parametrize("axis", ("y",))
def test_affine_certificate_and_lowering_match_independent_dense_blocks(
    control_qubits: int,
    axis: str,
) -> None:
    """Independent dense Y-rotation blocks check the affine angle construction with three
    controls.
    """
    offset = 0.13
    coefficients = tuple(
        (-1.0) ** qubit * (0.2 + 0.04 * qubit)
        for qubit in range(control_qubits)
    )
    angles = tuple(
        offset
        + sum(
            coefficient
            for qubit, coefficient in enumerate(coefficients)
            if branch & (1 << qubit)
        )
        for branch in range(1 << control_qubits)
    )
    affine = certify_affine_angle_table(
        angles,
        offset=offset,
        coefficients=coefficients,
    )
    assert affine.max_residual <= affine.tolerance

    circuit = QuantumCircuit(control_qubits + 1)
    append_affine_rotation(
        circuit,
        circuit.qubits[0],
        circuit.qubits[1:],
        affine,
        axis=axis,
    )
    reference = _block_diagonal([_rotation(axis, angle) for angle in angles])
    law = affine_rotation_resource_law(affine)
    assert np.max(np.abs(Operator(circuit).data - reference)) <= _matrix_tolerance(
        reference.shape[0], law["basis_cx_gates"]
    )


def _recursive_operation_counts(circuit: QuantumCircuit) -> Counter[str]:
    """Count structural gates and basis leaves through nested definitions."""

    counts: Counter[str] = Counter()
    structural = {"ucry", "ucrz", "multiplexer", "diagonal"}
    basis = {"cx", "ry", "rz", "u", "unitary"}

    def visit(operation) -> None:
        name = "diagonal" if operation.name == "nwqlib_phase_diagonal" else operation.name
        if name in structural:
            counts[name] += 1
        if operation.name in basis:
            counts[operation.name] += 1
            return
        definition = getattr(operation, "definition", None)
        if definition is not None:
            for instruction in definition.data:
                visit(instruction.operation)

    for instruction in circuit.data:
        visit(instruction.operation)
    return counts


@pytest.mark.parametrize("control_qubits", range(6))
def test_multiplexor_resource_law_matches_recursive_decomposition(
    control_qubits: int,
) -> None:
    """Count recursively decomposed UCR, unitary multiplexor and diagonal operations against
    their scoped resource laws.
    """
    rng = np.random.default_rng(900 + control_qubits)
    angles = rng.normal(size=1 << control_qubits)
    for axis, append in (
        ("y", append_uniformly_controlled_ry),
        ("z", append_uniformly_controlled_rz),
    ):
        circuit = QuantumCircuit(control_qubits + 1)
        gate = append(circuit, circuit.qubits[0], circuit.qubits[1:], angles)
        counts = _recursive_operation_counts(circuit)
        law = multiplexor_resource_law(
            f"ucr{axis}", control_qubits=control_qubits
        )
        assert counts[f"ucr{axis}"] == law["multiplexor_gates"]
        assert len(gate.params) == law["table_entries"]
        assert counts["cx"] == law["basis_cx_gates"]

    uc_circuit = QuantumCircuit(control_qubits + 1)
    append_uniformly_controlled_unitaries(
        uc_circuit,
        uc_circuit.qubits[0],
        uc_circuit.qubits[1:],
        [_single_qubit_unitary(index) for index in range(1 << control_qubits)],
    )
    uc_counts = _recursive_operation_counts(uc_circuit)
    uc_law = projected_unitary_resource_law(control_qubits)
    assert uc_counts["cx"] == uc_law["basis_cx_gates"]

    diagonal_circuit = QuantumCircuit(control_qubits)
    append_control_diagonal_phases(
        diagonal_circuit,
        diagonal_circuit.qubits,
        rng.normal(size=1 << control_qubits),
    )
    diagonal_counts = _recursive_operation_counts(diagonal_circuit)
    diagonal_law = multiplexor_resource_law(
        "diagonal", control_qubits=control_qubits
    )
    assert diagonal_counts["diagonal"] == diagonal_law["control_diagonal_gates"]
    assert diagonal_counts["cx"] == diagonal_law["basis_cx_gates"]


def test_affine_rotation_resource_law_matches_lowering() -> None:
    affine = AffineAngleTable(
        offset=0.3,
        coefficients=(0.0, -0.2, 0.4),
        max_residual=0.0,
        tolerance=0.0,
    )
    circuit = QuantumCircuit(4)
    append_affine_rotation(
        circuit,
        circuit.qubits[0],
        circuit.qubits[1:],
        affine,
        axis="z",
    )
    counts = _recursive_operation_counts(circuit)
    law = affine_rotation_resource_law(affine)
    assert circuit.count_ops()["rz"] == law["unconditional_rotations"]
    assert sum(isinstance(item.operation, ControlledGate) for item in circuit.data) == law[
        "controlled_rotations"
    ]
    assert counts["cx"] == law["basis_cx_gates"]
