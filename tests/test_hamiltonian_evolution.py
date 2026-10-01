"""Tests for Hamiltonian-evolution subroutines."""

import numpy as np
import pytest
import scipy.linalg
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import Operator

from nwqlib.subroutines.hamiltonian_evolution import (
    PauliEvolutionBlock,
    PauliEvolutionTerm,
    apply_pauli_rotation,
    build_pauli_evolution_circuit,
    make_evolution_synthesis,
    make_pauli_label,
    parse_pauli_label,
    pauli_label_to_qiskit_string,
    sparse_pauli_op_from_terms,
)


from conftest import dense_evolution_copy as _dense_evolution_copy


def test_pauli_labels_round_trip() -> None:
    ops = {8: "Z", 0: "X", 17: "Y"}
    label = make_pauli_label(ops)

    assert label == "x0z8y17"
    assert parse_pauli_label(label) == {0: "X", 8: "Z", 17: "Y"}


def test_invalid_pauli_label_raises() -> None:
    with pytest.raises(ValueError):
        parse_pauli_label("x2x1")


def test_pauli_label_to_qiskit_string() -> None:
    assert pauli_label_to_qiskit_string("x0z2", 4) == "IZIX"


def test_make_evolution_synthesis_rejects_unknown_method() -> None:
    with pytest.raises(ValueError):
        make_evolution_synthesis("unknown")  # type: ignore[arg-type]


def test_sparse_product_forwards_order_to_the_actual_suzuki_action():
    from nwqlib.subroutines.hamiltonian_evolution.sparse_pauli_product import build_sparse_pauli_product_circuit

    x, z = np.array([[0., 1.], [1., 0.]]), np.diag([1., -1.])
    def rotation(pauli, t):
        return np.cos(t)*np.eye(2)-1j*np.sin(t)*pauli
    def second(t):
        return rotation(x, .7*t/2) @ rotation(z, -.4*t) @ rotation(x, .7*t/2)
    p = 1/(4-4**(1/3))
    t = .5
    first = rotation(z, -.4*t) @ rotation(x, .7*t)
    fourth = second(p*t) @ second(p*t) @ second((1-4*p)*t) @ second(p*t) @ second(p*t)
    for order, expected in ((1, first), (2, second(t)), (4, fourth)):
        actual = build_sparse_pauli_product_circuit([("X", .7), ("Z", -.4)], num_qubits=1,
            time_step=t, evolution_synthesis="suzuki_trotter", reps=1, order=order)
        # A raw PauliEvolutionGate's matrix is exact; realize its selected
        # synthesis before checking the product-formula action.
        np.testing.assert_allclose(Operator(actual.decompose()).data, expected, rtol=0, atol=2e-14)
    assert np.linalg.norm(fourth-second(t)) > 1e-5
    for order in (None, 1):
        lie = build_sparse_pauli_product_circuit([("X", .7), ("Z", -.4)], num_qubits=1,
            time_step=t, evolution_synthesis="lie_trotter", reps=1, order=order)
        np.testing.assert_allclose(Operator(lie.decompose()).data, first,
                                   rtol=0, atol=2e-14)


@pytest.mark.parametrize("synthesis, order", [("suzuki_trotter", 3), ("suzuki_trotter", 0),
    ("suzuki_trotter", 2.5), ("suzuki_trotter", True), ("lie_trotter", 2)])
def test_sparse_product_order_rejects_before_consuming_terms(synthesis, order):
    from nwqlib.subroutines.hamiltonian_evolution.sparse_pauli_product import build_sparse_pauli_product_circuit

    class Unread:
        def __iter__(self):
            pytest.fail("invalid product order consumed terms")
    with pytest.raises(ValueError, match="product-formula order"):
        build_sparse_pauli_product_circuit(Unread(), num_qubits=1, time_step=.1,
            evolution_synthesis=synthesis, reps=1, order=order)


def test_apply_pauli_rotation_matches_pauli_evolution_gate() -> None:
    labels = ("x0", "y0", "z0", "x0y1", "y0z2", "x0y1z2", "y0y1y2", "x0z1y2")
    angles = np.random.default_rng(20260702).uniform(-1.2, 1.2, size=len(labels))

    for label, angle in zip(labels, angles):
        num_qubits = max(parse_pauli_label(label)) + 1
        native = QuantumCircuit(num_qubits)
        apply_pauli_rotation(native, label, float(angle))
        if label == "x0z1y2":
            assert {"cx", "h", "rz"} <= {
                instruction.operation.name for instruction in native.data
            }

        reference = QuantumCircuit(num_qubits)
        operator = sparse_pauli_op_from_terms(
            [{"pauli": label, "coefficient": 1.0}],
            num_qubits=num_qubits,
        )
        reference.append(PauliEvolutionGate(operator, time=float(angle) / 2.0), reference.qubits)

        # Machine-precision identity: both circuits implement exp(-i angle * P / 2).
        assert (
            np.max(np.abs(Operator(native).data - Operator(_dense_evolution_copy(reference)).data))
            <= 1.0e-12
        )


def test_pauli_evolution_circuit_matches_independent_matrix_exponential() -> None:
    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    identity = np.eye(2, dtype=complex)

    def on_qubits(q0, q1, q2):
        # Qiskit operators keep qubit 0 in the rightmost Kronecker factor.
        return np.kron(q2, np.kron(q1, q0))

    time_step = 0.31
    blocks = (
        PauliEvolutionBlock(
            terms=(PauliEvolutionTerm(pauli="x0y1z2", coefficient=0.7),),
            time_step=time_step,
        ),
        # Disjoint supports commute, so the default Lie-Trotter step is exact.
        PauliEvolutionBlock(
            terms=(
                PauliEvolutionTerm(pauli="z0z2", coefficient=-0.25),
                PauliEvolutionTerm(pauli="x1", coefficient=0.4),
            ),
            time_step=time_step,
        ),
    )
    first = 0.7 * on_qubits(x, y, z)
    second = -0.25 * on_qubits(z, identity, z) + 0.4 * on_qubits(identity, x, identity)
    expected = scipy.linalg.expm(-1j * time_step * second) @ scipy.linalg.expm(
        -1j * time_step * first
    )

    circuit = build_pauli_evolution_circuit(blocks, num_qubits=3)
    # Realize the selected synthesis; a raw PauliEvolutionGate's matrix is exact.
    # A few 8x8 unitary products: binary64 roundoff stays far below 1e-12.
    np.testing.assert_allclose(Operator(circuit.decompose()).data, expected, rtol=0.0, atol=1.0e-12)


def test_complex_coefficients_raise() -> None:
    block = {"terms": [{"pauli": "z0", "coefficient": 1.0 + 1.0j}], "time_step": 0.5}
    with pytest.raises(ValueError, match="complex"):
        build_pauli_evolution_circuit([block], num_qubits=1)


def test_make_pauli_label_rejects_empty_ops() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        make_pauli_label({})
