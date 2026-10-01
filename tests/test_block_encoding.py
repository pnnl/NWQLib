"""Tests for the block-encoding subroutines.

The embedding-identity test checks the convention in FRAMEWORK.md,
"Block-Encoding and QSP Conventions": for every construction and
registered instance shape, the circuit's all-zero-ancilla block reproduces
``A / alpha`` at 1e-12. No wall-clock assertions anywhere; integer anchors are
construction-exact quantities only (term counts, ancilla counts, gate counts
of the built circuit).
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import XGate
from qiskit.quantum_info import Operator, SparsePauliOp

from conftest import assert_no_block_encoding_metadata_mirrors

from nwqlib.subroutines.block_encoding import (
    BLOCK_ENCODING_IMPLEMENTATIONS,
    BandSpecification,
    block_encoding_top_left,
    build_block_encoding,
    build_block_encoding_from_plan,
    projector_complement_matrix,
)
from nwqlib.subroutines._multiplexors import (
    append_dependency_projected_unitaries,
    projected_unitary_resource_law,
)
from nwqlib.backends.resources import (
    block_encoding_per_query_cx,
    dense_unitary_cx_qsd_upper_bound,
)
from nwqlib.subroutines.block_encoding import plan_block_encoding
from nwqlib.subroutines.qiskit_compat import controlled, inverse_realized_gate


def _second_difference(num_qubits,*,diffusivity=1.,spacing=1.,boundary="periodic"):
    size=1<<num_qubits
    shift=np.roll(np.eye(size),1,axis=0) if boundary=="periodic" else np.eye(size,k=1)
    return diffusivity/spacing**2*(2*np.eye(size)-shift-shift.T)


def test_selected_numeric_and_structured_encodings_do_not_import_sdk():
    """Known plans are usable before their native circuit dependency exists."""
    import subprocess
    import sys
    code = '''
import importlib.abc
import sys
import numpy as np
class NoSDK(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "qiskit" or fullname.startswith("qiskit."):
            raise AssertionError("planning imported " + fullname)
sys.meta_path.insert(0, NoSDK())
from nwqlib.subroutines.block_encoding import BandSpecification, plan_block_encoding
from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm
band = plan_block_encoding(BandSpecification(offsets=(0,1),coefficients=(.5,.25),num_qubits=4))
assert band.alpha == .75
pauli = PauliDecomposition(terms=(PauliTerm(label="Z",coefficient=.5),),
    input_dimension=2,operator_dimension=2,num_qubits=1,atol=0.)
assert plan_block_encoding(pauli,implementation="pauli_lcu").alpha == .5
dense = plan_block_encoding(np.diag([.5,.25]),implementation="dense_dilation")
assert dense.alpha == .5
assert not any(name == "qiskit" or name.startswith("qiskit.") for name in sys.modules)
'''
    subprocess.run([sys.executable, '-c', code], check=True, capture_output=True, text=True)


def _dense_from_bands(specification: BandSpecification) -> np.ndarray:
    """Materialize the circulant a band specification describes (test oracle)."""

    dimension = 2**specification.num_qubits
    matrix = np.zeros((dimension, dimension), dtype=complex)
    columns = np.arange(dimension)
    for offset, coefficient in zip(specification.offsets, specification.coefficients):
        matrix[(columns + offset) % dimension, columns] = coefficient
    return matrix


def _encoded_block(record) -> np.ndarray:
    """Return the all-zero-ancilla block of the built circuit's unitary."""

    unitary = Operator(record.circuit).data
    return block_encoding_top_left(unitary, num_ancillas=record.num_ancillas)


def _hermitian_2x2() -> np.ndarray:
    return np.array([[1.0, 2.0 - 1.0j], [2.0 + 1.0j, -1.0]], dtype=complex)


def _random_dense(dimension: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(dimension, dimension)) + 1.0j * rng.normal(size=(dimension, dimension))


def test_controlled_lowering_matches_direct_nondefault_control_matrix() -> None:
    num_controls = 2
    ctrl_state = 0b01
    actual = Operator(controlled(XGate(), num_controls, ctrl_state=ctrl_state)).data
    direct = np.zeros_like(actual)
    control_mask = (1 << num_controls) - 1
    target_mask = 1 << num_controls
    for column in range(direct.shape[1]):
        row = column ^ target_mask if column & control_mask == ctrl_state else column
        direct[row, column] = 1.0

    # exact: controlled X and the direct oracle are permutation matrices.
    assert np.array_equal(actual, direct)
    assert not np.array_equal(actual, Operator(controlled(XGate(), num_controls)).data)


@pytest.mark.parametrize("control_state", (0, 1))
def test_realized_inverse_preserves_open_control_and_relative_phase(monkeypatch, control_state) -> None:
    from qiskit.circuit.library import UnitaryGate

    unitary = np.exp(0.23j) * np.asarray([[np.cos(0.31), -np.sin(0.31)], [np.sin(0.31), np.cos(0.31)]])
    forward = controlled(UnitaryGate(unitary), 1, ctrl_state=control_state)

    def forbidden(*args, **kwargs):
        raise AssertionError("inverse entered control synthesis")

    monkeypatch.setattr(UnitaryGate, "control", forbidden)
    inverse = inverse_realized_gate(forward)
    expected = np.eye(4, dtype=complex)
    indices = np.arange(control_state, 4, 2)
    expected[np.ix_(indices, indices)] = unitary.conj().T
    np.testing.assert_allclose(Operator(inverse).data, expected, rtol=0.0, atol=2.0e-14)


def test_admitted_unitary_inverse_does_not_repeat_matrix_validation(monkeypatch) -> None:
    from qiskit.circuit.library import UnitaryGate
    import qiskit.circuit.library.generalized_gates.unitary as unitary_module

    matrix = np.exp(0.17j) * np.asarray([[np.cos(0.31), -np.sin(0.31)], [np.sin(0.31), np.cos(0.31)]])
    gate = UnitaryGate(matrix)
    calls = []
    original = unitary_module.is_unitary_matrix

    def observed(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(unitary_module, "is_unitary_matrix", observed)
    inverse = inverse_realized_gate(gate)
    assert calls == []
    np.testing.assert_allclose(inverse.to_matrix(), matrix.conj().T, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("axis", ("y", "z"))
def test_native_controlled_inverse_can_be_controlled_and_translated(axis) -> None:
    from qiskit.circuit.library import CRYGate, CRZGate

    gate = (CRYGate if axis == "y" else CRZGate)(0.31)
    inverse = inverse_realized_gate(gate)
    assert inverse.base_class is gate.base_class
    assert inverse.params == [-0.31]
    circuit = QuantumCircuit(3)
    circuit.append(controlled(inverse, 1), circuit.qubits)
    translated = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    if axis == "y":
        cosine, sine = np.cos(0.31 / 2), np.sin(0.31 / 2)
        rotation = np.asarray([[cosine, sine], [-sine, cosine]], dtype=complex)
    else:
        rotation = np.diag(np.exp(1j * np.asarray([0.31 / 2, -0.31 / 2])))
    expected = np.eye(8, dtype=complex)
    expected[np.ix_([3, 7], [3, 7])] = rotation
    np.testing.assert_allclose(Operator(translated).data, expected, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("outer_control", (False, True))
def test_generic_controlled_inverse_has_distinct_compiler_name(outer_control) -> None:
    from qiskit.circuit.library import C3SXGate

    forward = C3SXGate()
    inverse = inverse_realized_gate(forward)
    expected_inverse = Operator(forward).data.conj().T
    gate = controlled(inverse, 1) if outer_control else inverse
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits)
    translated = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    expected = expected_inverse
    if outer_control:
        expected = np.eye(32, dtype=complex)
        expected[1::2, 1::2] = expected_inverse
    np.testing.assert_allclose(Operator(translated).data, expected, rtol=0.0, atol=1.0e-12)


@pytest.mark.parametrize("subclass", (False, True))
def test_generic_inverse_preserves_symbolic_parameter_binding(subclass) -> None:
    from qiskit.circuit import Gate, Parameter

    class SelectedDefinitionGate(Gate):
        pass

    theta = Parameter("theta")
    definition = QuantumCircuit(1)
    definition.ry(theta, 0)
    definition.global_phase = 0.23 * theta
    gate = definition.to_gate()
    if subclass:
        gate = SelectedDefinitionGate("selected", 1, [theta])
        gate.definition = definition
    circuit = QuantumCircuit(1)
    circuit.append(inverse_realized_gate(gate), circuit.qubits)
    assert set(circuit.parameters) == {theta}
    bound = circuit.assign_parameters({theta: 0.31})
    translated = transpile(bound, basis_gates=["u", "cx"], optimization_level=0)
    cosine, sine = np.cos(0.31 / 2), np.sin(0.31 / 2)
    expected = np.exp(-0.23j * 0.31) * np.asarray([[cosine, sine], [-sine, cosine]])
    np.testing.assert_allclose(Operator(translated).data, expected, rtol=0.0, atol=2.0e-14)


@pytest.mark.parametrize("outer_control", (False, True))
def test_diagonal_inverse_uses_selected_definition_payload(monkeypatch, outer_control) -> None:
    from qiskit.circuit.library import DiagonalGate

    diagonal = np.exp(1j * np.asarray([0.17, -0.31, 0.53, 0.79]))
    forward = DiagonalGate(diagonal)

    def forbidden(*args, **kwargs):
        raise AssertionError("inverse delegated the subclass payload")

    monkeypatch.setattr(DiagonalGate, "inverse", forbidden)
    inverse = inverse_realized_gate(forward)
    assert inverse.params == []
    gate = controlled(inverse, 1) if outer_control else inverse
    circuit = QuantumCircuit(gate.num_qubits)
    circuit.append(gate, circuit.qubits)
    translated = transpile(circuit, basis_gates=["u", "cx"], optimization_level=0)
    expected = np.diag(diagonal.conj())
    if outer_control:
        expected = np.eye(8, dtype=complex)
        expected[1::2, 1::2] = np.diag(diagonal.conj())
    np.testing.assert_allclose(Operator(translated).data, expected, rtol=0.0, atol=2.0e-14)


_FIVE_BAND_SPEC = BandSpecification(
    offsets=(-2, -1, 0, 1, 2),
    coefficients=(0.3, -1.0, 2.5, -1.0, 0.3 + 0.1j),
    num_qubits=3,
)

# Every construction and registered instance shape (plan Section 7 item 1).
# pauli_lcu shapes are registered up to 2 system qubits: the reused lcu/core
# SELECT synthesizes eagerly through Qiskit controlled-unitary definitions,
# whose rounding accumulates to ~3e-12 at 8 terms x 3 system qubits — above
# the machine-precision class. banded and dense_dilation cover 3-qubit shapes
# at 1e-12; the cross-construction test still reconstructs the 3-qubit
# instance through pauli_lcu under a documented synthesis-noise margin.
_EMBEDDING_CASES = {
    "pauli_lcu_hermitian_1q": lambda: (
        build_block_encoding(_hermitian_2x2(), implementation="pauli_lcu"),
        _hermitian_2x2(),
    ),
    "pauli_lcu_heat_2q": lambda: (
        build_block_encoding(_second_difference(2), implementation="pauli_lcu"),
        _second_difference(2),
    ),
    "banded_spec_tridiagonal_2q": lambda: (
        build_block_encoding(
            BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2)
        ),
        _dense_from_bands(
            BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2)
        ),
    ),
    "banded_dense_detected_heat_3q": lambda: (
        build_block_encoding(
            _second_difference(3, diffusivity=0.7, spacing=0.25),
            implementation="banded",
        ),
        _second_difference(3, diffusivity=0.7, spacing=0.25),
    ),
    "banded_single_band_complex_3q": lambda: (
        build_block_encoding(BandSpecification(offsets=(1,), coefficients=(0.5j,), num_qubits=3)),
        _dense_from_bands(BandSpecification(offsets=(1,), coefficients=(0.5j,), num_qubits=3)),
    ),
    "banded_five_band_complex_3q": lambda: (
        build_block_encoding(_FIVE_BAND_SPEC),
        _dense_from_bands(_FIVE_BAND_SPEC),
    ),
    "dense_dilation_heat_2q": lambda: (
        build_block_encoding(_second_difference(2), implementation="dense_dilation"),
        _second_difference(2),
    ),
    "dense_dilation_random_nonhermitian_2q": lambda: (
        build_block_encoding(_random_dense(4, seed=7), implementation="dense_dilation"),
        _random_dense(4, seed=7),
    ),
    "dense_dilation_random_1q": lambda: (
        build_block_encoding(_random_dense(2, seed=11), implementation="dense_dilation"),
        _random_dense(2, seed=11),
    ),
}


@pytest.mark.parametrize("case", sorted(_EMBEDDING_CASES))
def test_embedding_identity_reproduces_scaled_matrix(case: str) -> None:
    """Machine-precision identity: the encoded block equals A / alpha at 1e-12."""

    record, matrix = _EMBEDDING_CASES[case]()
    block = _encoded_block(record)
    assert np.max(np.abs(block - matrix / record.alpha)) <= 1.0e-12
    assert record.error_bound <= 1.0e-12


def test_cross_construction_agreement_on_tridiagonal() -> None:
    """All three constructions reconstruct the same A; alphas hit analytic values.

    2-qubit periodic heat matrix (kappa=h=1): all three encoded blocks agree
    with A/alpha at machine precision. At N=4 the analytic alphas provably
    coincide at 4.0: banded = |2|+|-1|+|-1| = 4 kappa/h^2; pauli_lcu = 4 from
    II(2), IX(-1), XX(-1); dense_dilation = ||A||_2 = 2 - 2 cos(pi) = 4.

    3-qubit instance: the alphas genuinely differ — banded stays 4, pauli_lcu
    is 5 from III(2), IIX(-1), IXX/IYY/XXX/XYY(+-1/2) -> 2+1+4*(1/2), and
    dense_dilation stays 4 (max_k 2 - 2 cos(2 pi k/8) at k=4).
    """

    small = _second_difference(2)
    for name in ("pauli_lcu", "banded", "dense_dilation"):
        record = build_block_encoding(small, implementation=name)
        assert record.alpha == pytest.approx(4.0, rel=1.0e-12, abs=0)
        assert np.max(np.abs(_encoded_block(record) - small / record.alpha)) <= 1.0e-12

    matrix = _second_difference(3)
    records = {
        name: build_block_encoding(matrix, implementation=name)
        for name in ("pauli_lcu", "banded", "dense_dilation")
    }
    assert records["banded"].alpha == pytest.approx(4.0, rel=1.0e-15, abs=0)
    assert records["pauli_lcu"].alpha == pytest.approx(5.0, rel=1.0e-12, abs=0)
    assert records["pauli_lcu"].implementation == "multiplexed_pauli"
    assert records["dense_dilation"].alpha == pytest.approx(4.0, rel=1.0e-12, abs=0)
    assert records["pauli_lcu"].alpha != records["banded"].alpha
    for name in ("banded", "dense_dilation"):
        block = _encoded_block(records[name])
        assert np.max(np.abs(block - matrix / records[name].alpha)) <= 1.0e-12
    # pauli_lcu at 8 terms x 3 system qubits carries ~3e-12 of Qiskit eager
    # controlled-unitary synthesis rounding (measured); 1e-10 is a ~30x-margin
    # tripwire on the same exact identity, not a method tolerance.
    pauli_block = _encoded_block(records["pauli_lcu"])
    assert np.max(np.abs(pauli_block - matrix / records["pauli_lcu"].alpha)) <= 1.0e-10


def test_auto_resolution_order() -> None:
    """Auto uses band structure, then the shared predicted per-query costs."""

    spec = BandSpecification(offsets=(0, 1), coefficients=(1.0, 0.5), num_qubits=2)
    banded = build_block_encoding(spec)
    assert banded.implementation == "banded"
    assert build_block_encoding(_second_difference(2)).implementation == "banded"
    diagonal = np.diag([1.0, 2.0, 3.0, 4.0]).astype(complex)  # 3 Pauli terms, not circulant
    assert build_block_encoding(diagonal).implementation == "dense_dilation"
    dense = _random_dense(16, seed=3)
    record = build_block_encoding(dense)
    assert record.implementation == "dense_dilation"
    for built in (banded, record):
        assert built.metadata["requested_block_encoding_implementation"] == "auto"
    assert set(record.metadata["predicted_per_query_cx"]) == {
        "multiplexed_pauli",
        "dense_dilation",
    }
    assert record.metadata["selection_reason"].startswith("auto selected dense_dilation")


def test_auto_skips_strictly_unselected_dense_norm_and_preserves_real_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    strict = SparsePauliOp.from_list([("ZII", 0.7), ("IIX", 0.2)]).to_matrix()
    tied = SparsePauliOp.from_list(
        [("II", 1.0), ("IX", 0.7), ("XX", 0.3), ("XY", 0.2)]
    ).to_matrix()
    norm = np.linalg.norm
    norm_calls = []

    def counted_norm(array, *args, **kwargs):
        norm_calls.append(array.shape)
        return norm(array, *args, **kwargs)

    monkeypatch.setattr(np.linalg, "norm", counted_norm)
    plan = plan_block_encoding(strict)
    assert plan.implementation == "multiplexed_pauli"
    assert norm_calls == []
    assert "dense_dilation" not in plan.detail["normalization_records"]

    from nwqlib.subroutines.block_encoding import core

    tied_cost = plan_block_encoding(tied, implementation="pauli_lcu").detail[
        "predicted_per_query_cx"
    ]["multiplexed_pauli"]
    monkeypatch.setattr(core, "_dense_dilation_predicted_cx", lambda _qubits: tied_cost)
    tie = plan_block_encoding(tied)
    assert tie.detail["predicted_per_query_cx"] == {
        "multiplexed_pauli": tied_cost, "dense_dilation": tied_cost
    }
    assert tie.implementation == "dense_dilation"
    assert norm_calls == [(4, 4)]
    assert tie.alpha <= tie.detail["normalization_records"]["multiplexed_pauli"]

    # A legal supplied dense normalization can be worse than Pauli's L1
    # normalization. Equal CX costs must then select the smaller alpha.
    oversized = plan_block_encoding(tied, normalization=10.0)
    assert oversized.implementation == "multiplexed_pauli"
    assert oversized.alpha < 10.0


def test_f3_qsd_bound_selects_the_realized_lower_cx_route_near_tie() -> None:
    matrix = SparsePauliOp.from_list(
        [
            ("XX", 1.1804 + 0.0995j),
            ("IY", 0.8638 - 0.001j),
            ("YY", 0.29 + 0.2632j),
        ]
    ).to_matrix()

    selected = build_block_encoding(matrix)
    assert selected.implementation == "dense_dilation"
    # Two address qubits: 18 SELECT CX, 2 coefficient-phase CX, and
    # 2 * (2**2 - 2) positive-amplitude PREP CX.
    assert selected.metadata["predicted_per_query_cx"] == {
        "multiplexed_pauli": 24.0,
        "dense_dilation": 20.0,
    }

    built = {
        name: build_block_encoding(matrix, implementation=name)
        for name in ("pauli_lcu", "dense_dilation")
    }
    cx_counts = {
        name: transpile(
            encoding.circuit,
            basis_gates=["rz", "sx", "x", "cx"],
            optimization_level=0,
        )
        .count_ops()
        .get("cx", 0)
        for name, encoding in built.items()
    }
    # Realized QSD output is content- and platform-dependent; the contract is
    # the registered upper bound plus the ordering below.
    assert cx_counts["dense_dilation"] <= int(
        selected.metadata["predicted_per_query_cx"]["dense_dilation"]
    )
    assert cx_counts["dense_dilation"] < cx_counts["pauli_lcu"]


def test_dense_unitary_cx_qsd_upper_bound_values_and_domain() -> None:
    assert [dense_unitary_cx_qsd_upper_bound(n) for n in range(1, 6)] == [
        0,
        3,
        20,
        100,
        444,
    ]
    with pytest.raises(ValueError, match="total_qubits must be at least 1"):
        dense_unitary_cx_qsd_upper_bound(0)


def test_auto_cost_routing_distinguishes_same_term_count_repetition() -> None:
    """Sixteen-term tables can route differently solely from dependency support."""

    characters = "IXYZ"
    repeated = [f"I{left}{right}" for left in characters for right in characters]
    full = [
        "IZY",
        "ZXZ",
        "ZIX",
        "YYZ",
        "IXY",
        "YXI",
        "ZXX",
        "ZXY",
        "YZX",
        "YYX",
        "ZZI",
        "ZYI",
        "YII",
        "ZYZ",
        "ZYY",
        "XXI",
    ]

    def _matrix(labels: list[str]) -> np.ndarray:
        return SparsePauliOp.from_list(
            [(label, 1.0 + 0.125j * (index + 1)) for index, label in enumerate(labels)]
        ).to_matrix()

    repeated_record = build_block_encoding(_matrix(repeated))
    full_record = build_block_encoding(_matrix(full))
    assert repeated_record.implementation == "multiplexed_pauli"
    assert full_record.implementation == "dense_dilation"
    assert repeated_record.metadata["pauli_term_count"] == 16
    assert full_record.metadata["term_count"] == 16
    assert repeated_record.metadata["effective_control_counts"] == [2, 2, 0]
    assert full_record.metadata["effective_control_counts"] == [4, 4, 4]
    # Both four-qubit PREP pairs cost 2 * (2**4 - 2) CX, plus 14 phase CX.
    # The dependency supports give 18 versus 135 SELECT CX.
    assert repeated_record.metadata["predicted_per_query_cx"] == {
        "multiplexed_pauli": 60.0,
        "dense_dilation": 100.0,
    }
    assert full_record.metadata["predicted_per_query_cx"] == {
        "multiplexed_pauli": 177.0,
        "dense_dilation": 100.0,
    }
    assert "predicted per-query CX" in repeated_record.metadata["selection_reason"]
    assert "predicted per-query CX" in full_record.metadata["selection_reason"]


@pytest.mark.parametrize(
    ("labels", "expected_controls"),
    (
        (("YI",), [0, 0]),
        (("II", "IX"), [1, 0]),
        (("II", "XX", "YY", "ZZ"), [2, 2]),
    ),
)
def test_pauli_effective_control_regimes_and_exact_complex_block(
    labels: tuple[str, ...], expected_controls: list[int]
) -> None:
    """Constant, single-bit, and full tables include exact complex Y factors."""

    coefficients = [1.0 + 0.2j * (index + 1) for index in range(len(labels))]
    matrix = SparsePauliOp.from_list(list(zip(labels, coefficients, strict=True))).to_matrix()
    record = build_block_encoding(matrix, implementation="pauli_lcu")
    assert record.implementation == "multiplexed_pauli"
    assert record.metadata["effective_control_counts"] == expected_controls
    np.testing.assert_allclose(
        record.alpha * _encoded_block(record),
        matrix,
        rtol=0.0,
        atol=1.0e-12,
    )


def test_pauli_top_left_is_exact_with_complex_phases_and_identity_padding() -> None:
    """Three physical terms exercise the fourth address state's identity pad."""

    terms = (("II", 0.5j), ("XY", -1.0 + 0.25j), ("YZ", 0.75 - 0.5j))
    matrix = SparsePauliOp.from_list(list(terms)).to_matrix()
    record = build_block_encoding(matrix, implementation="pauli_lcu")
    assert record.metadata["pauli_term_count"] == 3
    assert record.metadata["padded_table_size"] == 4
    np.testing.assert_allclose(
        record.alpha * _encoded_block(record), matrix, rtol=0.0, atol=1.0e-12
    )


def test_explicit_pauli_error_metadata_uses_no_dense_residual(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The supplied matrix is encoded without a new pruning or residual pass."""

    matrix = SparsePauliOp.from_list([("IY", 1.0), ("ZX", 0.25j)]).to_matrix()
    plan = plan_block_encoding(matrix, implementation="pauli_lcu")

    def _reject_reconstruction(*_args, **_kwargs):
        raise AssertionError("explicit Pauli construction rebuilt a dense residual")

    monkeypatch.setattr(
        type(plan.decomposition),
        "to_sparse_pauli_op",
        _reject_reconstruction,
    )
    record = build_block_encoding_from_plan(plan)
    assert record.error_bound == 0.
    assert "no coefficient pruning" in record.metadata["error_evaluation"]


@pytest.mark.parametrize("atol, removed", [(.2, .05), (.01, 0.)])
def test_supplied_pauli_encoding_bound_targets_retained_terms(atol, removed):
    from nwqlib.subroutines.pauli_decomposition import decompose_matrix_to_pauli

    original = np.array([[1., .05], [.05, -1.]])
    decomposition = decompose_matrix_to_pauli(original, atol=atol)
    retained = decomposition.to_sparse_pauli_op().to_matrix()
    record = build_block_encoding_from_plan(plan_block_encoding(decomposition,
        implementation="pauli_lcu"))
    realized = record.alpha * _encoded_block(record)
    np.testing.assert_allclose(realized, retained, rtol=0, atol=2e-14)
    assert np.linalg.norm(original-realized, 2) == pytest.approx(removed, rel=0, abs=2e-14)
    assert record.error_bound == 0.
    assert "supplied Pauli terms" in record.metadata["error_evaluation"]
    assert "upstream pruning error excluded" in record.metadata["error_evaluation"]


def test_default_pauli_and_circulant_selection_keep_small_nonzero_terms():
    matrix = np.diag([1e-13, 2e-13]).astype(complex)
    matrix[0, 1] = 3e-14j
    record = build_block_encoding(matrix)
    np.testing.assert_allclose(block_encoding_top_left(Operator(record.circuit).data,
        num_ancillas=record.num_ancillas), matrix/record.alpha, rtol=2e-13, atol=2e-13)
    near = np.diag([1., 1.+1e-13]).astype(complex)
    plan = plan_block_encoding(near)
    assert plan.implementation != "banded"
    assert plan.decomposition.atol == 0
    assert any(0 < abs(term.coefficient) < 1e-12 for term in plan.decomposition.terms)


@pytest.mark.parametrize("implementation", ["auto", "dense_dilation"])
def test_full_dense_build_reuses_one_svd_for_norm_and_completion(monkeypatch, implementation):
    from nwqlib.subroutines.block_encoding import core
    matrix = np.array([[1., .3j], [.2, -.5]])
    original = np.linalg.svd
    calls = []
    def svd(value, *args, **kwargs):
        calls.append(value.shape)
        assert kwargs.get("compute_uv", True)
        return original(value, *args, **kwargs)
    monkeypatch.setattr(np.linalg, "svd", svd)
    monkeypatch.setattr(core, "_dense_dilation_predicted_cx", lambda _q: 0.)
    record = build_block_encoding(matrix, implementation=implementation)
    assert calls == [(2, 2)]
    np.testing.assert_allclose(block_encoding_top_left(Operator(record.circuit).data,
        num_ancillas=1), matrix/record.alpha, rtol=2e-14, atol=2e-14)


def test_dense_dilation_planning_admits_the_singular_values_of_its_norm(monkeypatch):
    from nwqlib.subroutines.block_encoding import core

    d = 64
    rng = np.random.default_rng(11)
    matrix = rng.standard_normal((d, d)) + 1j*rng.standard_normal((d, d))
    # The intake scan, 16 d**2 = 65,536 units, and the singular values of
    # np.linalg.norm(A, 2): ceil(4 d**3/3) = 349,526 for the bidiagonal
    # reduction and 32 d (d + 1) = 133,120 for the rest of the SVD without
    # vectors. A full SVD with both frames would be charged 8 d**3.
    law = 548_182
    plan = plan_block_encoding(matrix, implementation="dense_dilation", max_work=law)
    assert plan.alpha == pytest.approx(np.linalg.norm(matrix, 2), rel=1e-14, abs=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted spectral norm")

    with monkeypatch.context() as guarded:
        guarded.setattr(np.linalg, "norm", forbidden)
        with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
            plan_block_encoding(matrix, implementation="dense_dilation", max_work=law - 1)
        # The bidiagonal reduction alone makes ceil(4 d**3/3) = 349,526 multiply-adds.
        with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
            plan_block_encoding(matrix, implementation="dense_dilation", max_work=300_000)
    # A caller's full SVD, whose frames the completion reuses, stays charged 8 d**3.
    with pytest.raises(ValueError, match="block-encoding normalization exceeds max_work"):
        core._plan_block_encoding(matrix, implementation="dense_dilation", max_bytes=10**9,
            max_work=16*d*d + 8*d**3 - 1, dense_norm=forbidden)


def test_pauli_dense_dilation_admits_the_singular_values_of_its_norm(monkeypatch):
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    d = 64
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="XYZIIX", coefficient=.5), PauliTerm(label="ZZIYII", coefficient=.3j),
               PauliTerm(label="IIIIII", coefficient=.2)),
        input_dimension=d, operator_dimension=d, num_qubits=6, atol=0.)
    table = core._admit_pauli_plan(decomposition, held_bytes=0, max_bytes=10**9, max_work=10**9, classify=False)
    # The Pauli tables, the expansion of 3 terms to the dense matrix (3 d**2)
    # and the singular values of np.linalg.norm(A, 2), ceil(4 d**3/3) +
    # 32 d (d + 1) = 482,646 units.
    law = table + 3*d*d + 482_646
    plan = plan_block_encoding(decomposition, implementation="dense_dilation", max_work=law)
    dense = decomposition.to_sparse_pauli_op().to_matrix()
    assert plan.alpha == pytest.approx(np.linalg.norm(dense, 2), rel=1e-14, abs=0)

    def forbidden(*args, **kwargs):
        raise AssertionError("unadmitted dense conversion")

    monkeypatch.setattr(PauliDecomposition, "to_sparse_pauli_op", forbidden)
    with pytest.raises(ValueError, match="block-encoding Pauli dense conversion exceeds max_work"):
        plan_block_encoding(decomposition, implementation="dense_dilation", max_work=law - 1)
    # The bidiagonal reduction alone makes ceil(4 d**3/3) = 349,526 multiply-adds.
    with pytest.raises(ValueError, match="block-encoding Pauli dense conversion exceeds max_work"):
        plan_block_encoding(decomposition, implementation="dense_dilation", max_work=300_000)


def test_pauli_classification_gate_admits_the_classifier_populations():
    """``_admit_pauli_plan`` with classification charges C(n,m,b) work and B_inner bytes.

    Owner: ``core._admit_pauli_plan`` and ``core._circulant_classification_work``.
    Expected values by hand, b = min(m, 2**n), P = 2**ceil(log2 m):
    work = 32 P n**2 + (n+32)m + 4(n+1)mb + 16b + 2b ceil(log2 b) + 32;
    bytes = held + 64P(n+16) + [128 L(2J) + 512](m+b+n+1) + 65536 with
    J = 4208 + 2n + 2 ceil(log2(b+1)) + ceil(log2(m+b+1)) and
    L(k) = 32 + 4 ceil(k/30). An empty decomposition skips classification
    and is charged only the tables, work 32 P n**2 and bytes held + 64P(n+16).
    """
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    # (n, m) = (2, 3): P = 4, b = 3, J = 4208+4+4+3 = 4219, L(8438) = 32+4*282 = 1160.
    # (n, m) = (3, 10): P = 16, b = 8, J = 4208+6+8+5 = 4227, L(8454) = 1160.
    # (n, m) = (1, 2): P = 2, b = 2, J = 4208+2+4+3 = 4217, L(8434) = 32+4*282 = 1160.
    # 2J is 4 above a multiple of 30, so each of the two smaller terms of J
    # moves L(2J): without ceil(log2(m+b+1)) L(8428) = 1156, and with
    # ceil(log2 b) in place of ceil(log2(b+1)) L(8430) = 1156.
    # (n, m) = (2, 0): P = 1, no classifier.
    cases = ((2, 3, 1000, 512 + 102 + 108 + 48 + 12 + 32, 1000 + 4608 + 148992 * 9 + 65536),
             (3, 10, 7, 4608 + 350 + 1280 + 128 + 48 + 32, 7 + 19456 + 148992 * 22 + 65536),
             (1, 2, 11, 64 + 66 + 32 + 32 + 4 + 32, 11 + 2176 + 148992 * 6 + 65536),
             (2, 0, 5, 128, 5 + 1152))
    for n, m, held, work, size in cases:
        labels = [format(index, f"0{2 * n}b") for index in range(m)]
        decomposition = PauliDecomposition(
            terms=tuple(PauliTerm(label="".join("IXZY"[int(label[2 * k:2 * k + 2], 2)] for k in range(n)),
                                  coefficient=1.0) for label in labels),
            input_dimension=1 << n, operator_dimension=1 << n, num_qubits=n, atol=0.)
        admit = lambda **limits: core._admit_pauli_plan(  # noqa: E731
            decomposition, held_bytes=held, classify=True, **limits)
        assert admit(max_bytes=size, max_work=work) == work
        with pytest.raises(ValueError, match="block-encoding Pauli work exceeds max_work"):
            admit(max_bytes=size, max_work=work - 1)
        with pytest.raises(ValueError, match="classification/table arrays needs"):
            admit(max_bytes=size - 1, max_work=work)


def test_direct_block_controls_precede_conversion_and_heavy_work(monkeypatch):
    from nwqlib.subroutines.block_encoding import core
    matrix = np.array([[1., .3j], [.2, -.5]])
    plan = plan_block_encoding(matrix, implementation="dense_dilation", normalization=2.)
    def forbidden(*args, **kwargs):
        raise AssertionError("short cap reached materialization or heavy work")
    with monkeypatch.context() as guarded:
        guarded.setattr(core, "_as_power_of_two_matrix", forbidden)
        guarded.setattr(np.linalg, "svd", forbidden)
        guarded.setattr(core, "pauli_coefficients", forbidden)
        for limit in ({"max_bytes":1}, {"max_work":1}):
            with pytest.raises(ValueError, match="max_"):
                plan_block_encoding(matrix, **limit)
            with pytest.raises(ValueError, match="max_"):
                build_block_encoding(matrix, **limit)
            with pytest.raises(ValueError, match="max_"):
                build_block_encoding_from_plan(plan, **limit)
    # Larger explicit allowances reach both nested owners unchanged. The
    # actual two-dimensional construction still uses only kilobytes.
    admit = core._admit_dense_input
    controls = []
    def observed(value, **kwargs):
        controls.append((kwargs["max_bytes"], kwargs["max_work"]))
        return admit(value, **kwargs)
    monkeypatch.setattr(core, "_admit_dense_input", observed)
    build_block_encoding(matrix, implementation="dense_dilation", max_bytes=20_000_000_000,
        max_work=200_000_000)
    assert controls and set(controls) == {(20_000_000_000, 200_000_000)}


@pytest.mark.parametrize(
    ("table", "expected_controls"),
    (
        (("I", "Y"), 1),
        (("I", "X", "Y", "Z"), 2),
    ),
)
def test_dependency_projected_ucg_basis_scaling(
    table: tuple[str, ...], expected_controls: int
) -> None:
    """Pinned UCG law equals Qiskit-2.4 basis CX at two support widths."""

    local = {
        "I": np.eye(2, dtype=complex),
        "X": np.array([[0, 1], [1, 0]], dtype=complex),
        "Y": np.array([[0, -1j], [1j, 0]], dtype=complex),
        "Z": np.diag([1, -1]).astype(complex),
    }
    controls = QuantumRegister(expected_controls, "control")
    target = QuantumRegister(1, "target")
    circuit = QuantumCircuit(controls, target)
    projection = append_dependency_projected_unitaries(
        circuit,
        target[0],
        list(controls),
        [local[label] for label in table],
    )
    assert projection.effective_control_count == expected_controls
    basis = transpile(
        circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
    )
    law = projected_unitary_resource_law(expected_controls)
    assert basis.count_ops().get("cx", 0) == law["basis_cx_gates"]


def test_every_implementation_reachable_and_unknown_rejected() -> None:
    """No dead options: each registry name builds; unknown names fail loudly."""

    matrix = _second_difference(2)
    for name in BLOCK_ENCODING_IMPLEMENTATIONS:
        record = build_block_encoding(matrix, implementation=name)
        assert record.implementation == name
        assert record.metadata["implementation_metadata"]["name"] == name
    with pytest.raises(ValueError, match="must be one of.*'auto'.*'banded'"):
        build_block_encoding(matrix, implementation="fable")


def test_band_specification_routes_only_to_banded() -> None:
    spec = BandSpecification(offsets=(0, 1), coefficients=(1.0, 0.5), num_qubits=2)
    for name in ("pauli_lcu", "dense_dilation"):
        with pytest.raises(ValueError, match="requires a dense matrix"):
            build_block_encoding(spec, implementation=name)


def test_dense_dirichlet_falls_back_to_dense_dilation() -> None:
    # Dense Dirichlet matrices are not circulant: explicit banded rejects with
    # the alternatives named; auto routes them elsewhere instead.
    dirichlet = _second_difference(2, boundary="dirichlet")
    with pytest.raises(ValueError, match="periodic banded Toeplitz"):
        build_block_encoding(dirichlet, implementation="banded")
    assert build_block_encoding(dirichlet).implementation == "dense_dilation"


def test_metadata_records_required_keys_and_honesty_labels() -> None:
    matrix = _second_difference(2)
    spec = BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2)
    labeled = {
        "pauli_lcu": (
            build_block_encoding(matrix, implementation="pauli_lcu"),
            "small_dense_validation",
        ),
        "banded_spec": (build_block_encoding(spec), "scalable_oracle"),
        "banded_dense": (
            build_block_encoding(matrix, implementation="banded"),
            "small_dense_validation",
        ),
        "dense_dilation": (
            build_block_encoding(matrix, implementation="dense_dilation"),
            "small_dense_validation",
        ),
    }
    for record, label in labeled.values():
        metadata = record.metadata
        assert_no_block_encoding_metadata_mirrors(record)
        assert metadata["preprocessing_label"] == label


def test_banded_gate_counts_match_recorded_closed_forms() -> None:
    """Construction-exact integer anchors for the analytic tier's gate formulas.

    These are structural op counts of the built multiplexor construction.
    """

    diffusivity, spacing = 0.7, 0.25
    matrix = _second_difference(3, diffusivity=diffusivity, spacing=spacing)
    record = build_block_encoding(matrix, implementation="banded")
    # Closed-form heat normalization: |2c| + |-c| + |-c| = 4 kappa/h^2.
    assert record.alpha == pytest.approx(4.0 * diffusivity / spacing**2, rel=1.0e-15, abs=0)
    assert record.metadata["band_offsets"] == [-1, 0, 1]
    assert record.metadata["toeplitz"] is True
    transpiled = transpile(record.circuit, basis_gates=["u", "cx"], optimization_level=1)
    block = block_encoding_top_left(Operator(transpiled).data, num_ancillas=record.num_ancillas)
    assert np.max(np.abs(block - matrix / record.alpha)) <= 1.0e-12
    counts = record.circuit.count_ops()
    formulas = record.metadata["periodic_gate_counts"]
    assert formulas["ucrz_multiplexor_gates"] == 3
    assert formulas["ucrz_table_entries_per_gate"] == 4
    assert formulas["ucrz_basis_cx_per_gate"] == 4
    assert formulas["control_diagonal_gates"] == 1
    assert counts["ucrz"] == formulas["ucrz_multiplexor_gates"]
    assert counts["nwqlib_phase_diagonal"] == formulas["control_diagonal_gates"]
    assert counts["qft"] == 1 and counts["qft_dg"] == 1
    assert counts["state_preparation"] == 1 and counts["state_preparation_dg"] == 1

    single = build_block_encoding(
        BandSpecification(offsets=(1,), coefficients=(0.5j,), num_qubits=3)
    )
    single_counts = single.circuit.count_ops()
    assert single.num_ancillas == 0
    assert single_counts["ucrz"] == 3
    assert "state_preparation" not in single_counts


@pytest.mark.parametrize(
    "specification",
    (
        BandSpecification(offsets=(-1, 0, 1), coefficients=(-1.0, 2.0, -1.0), num_qubits=2),
        _FIVE_BAND_SPEC,
    ),
)
def test_banded_multiplexor_law_tracks_measured_basis_scaling(
    specification: BandSpecification,
) -> None:
    """The structural law stays within 4% at address widths two and three."""

    plan = plan_block_encoding(specification)
    record = build_block_encoding(specification)
    basis = transpile(
        record.circuit,
        basis_gates=["rz", "sx", "x", "cx"],
        optimization_level=0,
    )
    measured = basis.count_ops().get("cx", 0)
    predicted = block_encoding_per_query_cx(
        plan.to_dict(),
        implementation="banded",
        system_qubits=plan.system_qubits,
    )
    assert abs(predicted - measured) / measured <= 0.04


def test_invalid_band_specifications_rejected() -> None:
    with pytest.raises(ValueError, match="distinct modulo"):
        build_block_encoding(
            BandSpecification(offsets=(1, -3), coefficients=(1.0, 1.0), num_qubits=2)
        )
    with pytest.raises(ValueError, match="nonzero"):
        build_block_encoding(
            BandSpecification(offsets=(0, 1), coefficients=(1.0, 0.0), num_qubits=2)
        )
    with pytest.raises(ValueError, match="one coefficient per offset"):
        build_block_encoding(BandSpecification(offsets=(0, 1), coefficients=(1.0,), num_qubits=2))


def test_zero_matrix_rejected_by_dense_constructions() -> None:
    zero = np.zeros((4, 4), dtype=complex)
    for implementation in ("pauli_lcu", "dense_dilation"):
        with pytest.raises(ValueError, match="zero matrix"):
            build_block_encoding_from_plan(plan_block_encoding(zero, implementation=implementation))


def test_projector_complement_dimension_boundary() -> None:
    with pytest.raises(ValueError, match="larger than the target dimension"):
        projector_complement_matrix([1.0, 0.0], dimension=0)

    assert projector_complement_matrix([1.0, 0.0]).shape == (2, 2)


@pytest.mark.parametrize("normalization", [0.0, -1.0, np.inf, np.nan, 1.0e-15])
def test_invalid_dense_normalization_rejects_before_synthesis(
    normalization: float, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = np.diag([1.0e-15, 2.0e-15])

    def reject_svd(*_args, **_kwargs):
        raise AssertionError("invalid normalization reached dense synthesis")

    monkeypatch.setattr(np.linalg, "svd", reject_svd)
    with pytest.raises(ValueError, match="normalization|below"):
        build_block_encoding_from_plan(
            plan_block_encoding(matrix, implementation="dense_dilation", normalization=normalization)
        )


def test_dense_scaling_loss_bound_survives_plan_and_build_without_residual_svd(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An extreme diagonal scale exposes underflow loss while spies prohibit an extra residual
    SVD.
    """
    from nwqlib.subroutines.block_encoding import core

    matrix = np.diag([1.0e-200, 1.0e200]).astype(complex)
    supplied = matrix.copy()
    svd = np.linalg.svd
    svd_calls = []
    normalization_bound = core._normalization_error_bound
    bound_calls = []

    def counted_svd(array, *args, **kwargs):
        svd_calls.append(array.shape)
        return svd(array, *args, **kwargs)

    def reject_norm(*_args, **_kwargs):
        raise AssertionError("construction requested a residual spectral decomposition")

    def counted_bound(*args):
        bound_calls.append(args[0].shape)
        return normalization_bound(*args)

    monkeypatch.setattr(np.linalg, "svd", counted_svd)
    monkeypatch.setattr(np.linalg, "norm", reject_norm)
    monkeypatch.setattr(core, "_normalization_error_bound", counted_bound)
    plan = plan_block_encoding(matrix, implementation="dense_dilation", normalization=1.0e200)
    record = build_block_encoding_from_plan(plan)
    assert bound_calls == [(2, 2)]
    assert svd_calls == [(2, 2)]
    assert record.error_bound == plan.error_bound
    assert 1.0e-200 <= record.error_bound <= 1.0e-199
    assert np.array_equal(matrix, supplied)
    assert _encoded_block(record)[0, 0] == 0.0


def test_dense_normalization_bound_covers_rounded_zero_residual() -> None:
    from fractions import Fraction

    matrix = np.diag([0.1, 0.2]).astype(complex)
    alpha = 0.3
    plan = plan_block_encoding(matrix, implementation="dense_dilation", normalization=alpha)
    record = build_block_encoding_from_plan(plan)
    block = _encoded_block(record)
    assert np.array_equal(matrix - alpha * block, np.zeros_like(matrix))
    exact_error = max(
        abs(Fraction(float(matrix[i, i].real)) - Fraction(alpha) * Fraction(float(block[i, i].real)))
        for i in range(2)
    )
    assert exact_error > 0
    assert Fraction(record.error_bound) >= exact_error
    assert record.error_bound == plan.error_bound

    exact = plan_block_encoding(np.diag([1.0, 2.0]), implementation="dense_dilation", normalization=2.0)
    assert build_block_encoding_from_plan(exact).error_bound == 0.0


def test_dense_contraction_check_uses_the_required_singular_values() -> None:
    matrix = np.full((2, 2), 1.0e-15)

    def build(normalization, source=matrix):
        return build_block_encoding_from_plan(
            plan_block_encoding(source, implementation="dense_dilation", normalization=normalization)
        )

    with pytest.raises(ValueError, match="below"):
        build(1.5e-15)

    legal = build(3.0e-15)
    np.testing.assert_allclose(
        _encoded_block(legal), matrix / legal.alpha, atol=np.finfo(float).eps, rtol=0.0
    )

    # A computed SVD norm can round a few ulps below the largest entry
    # magnitude, the exact norm here. An alpha one ulp below it is admitted
    # by the entry tests and the singular-value test alike, and the dilation
    # stays exact. A deficit of 1e-11, beyond rounding, is refused with both
    # raw values.
    diagonal = np.diag([1.0, 0.5]).astype(complex)
    rounded = build(float(np.nextafter(1.0, 0.0)), diagonal)
    np.testing.assert_allclose(
        rounded.alpha * _encoded_block(rounded), diagonal, atol=2 * np.finfo(float).eps, rtol=0.0
    )
    unitary = Operator(rounded.circuit).data
    assert np.max(np.abs(unitary @ unitary.conj().T - np.eye(4))) <= 4 * np.finfo(float).eps
    with pytest.raises(ValueError, match=r"alpha 0\.99999999999 is below .* entry magnitude is 1\.0"):
        build(1.0 - 1.0e-11, diagonal)


@pytest.mark.parametrize("scale", [1.0, 1.0e100])
def test_dense_banded_detection_bound_has_one_owner(scale: float) -> None:
    from dataclasses import replace

    matrix = scale * np.diag([2.0, 2.0 + 1.0e-13]).astype(complex)
    matrix[1, 0] = scale * 1.0e-14
    plan = plan_block_encoding(matrix, implementation="banded")
    ideal = _dense_from_bands(plan.source)
    independent_error = float(np.linalg.norm(matrix - ideal, 2))
    assert independent_error > 0.0
    assert independent_error <= plan.error_bound
    assert build_block_encoding_from_plan(plan).error_bound == plan.error_bound
    # Composition may add a child error in the same operator frame.
    augmented = replace(plan, error_bound=2.0 * plan.error_bound)
    assert build_block_encoding_from_plan(augmented).error_bound == augmented.error_bound

    exact = plan_block_encoding(ideal)
    assert exact.error_bound == 0.0
    assert build_block_encoding_from_plan(exact).error_bound == 0.0


def test_alpha_plan_matches_builder_validation() -> None:
    with pytest.raises(ValueError, match="distinct modulo"):
        plan_block_encoding(
            BandSpecification(offsets=(1, 5), coefficients=(1.0, -0.5), num_qubits=2)
        )
    with pytest.raises(ValueError, match="zero matrix"):
        plan_block_encoding(np.zeros((4, 4), dtype=complex))


@pytest.mark.parametrize("perturbed", [False, True])
def test_circulant_frobenius_certificate_rounds_outward(perturbed):
    from fractions import Fraction
    from nwqlib.subroutines.block_encoding.core import _detect_banded_pauli_structure
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    eps = np.finfo(float).eps
    coefficient = complex(1+3*eps, 18*eps) if perturbed else 1.
    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="IX", coefficient=1.), PauliTerm(label="XX", coefficient=coefficient)),
        input_dimension=4, operator_dimension=4, num_qubits=2, atol=0.)
    bands, bound = _detect_banded_pauli_structure(decomposition)
    x = np.array([[0., 1.], [1., 0.]])
    target = np.kron(np.eye(2), x) + coefficient * np.kron(x, x)
    implied = _dense_from_bands(bands)
    # Disjoint Pauli supports and disjoint shifts make these entries exact.
    # Rational squared differences give an independent Frobenius comparison.
    square = sum(((Fraction(float(a.real))-Fraction(float(b.real)))**2
                  + (Fraction(float(a.imag))-Fraction(float(b.imag)))**2
                  for a, b in zip(target.flat, implied.flat)), Fraction())
    assert Fraction(bound)**2 >= square
    assert (bound == 0.) is (not perturbed)


@pytest.mark.parametrize("labels", [("XZ", "IX"), ("XZZZZ", "IXYZI"), ("xzxz",), ("XQXZ",)])
def test_pauli_plan_refuses_labels_outside_the_declared_qubits_and_alphabet(labels):
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    # A plan-only consumer reads the control census without building, so a
    # malformed label must not reach a plan.
    with pytest.raises(ValueError, match="must be 4 letters from IXYZ"):
        plan_block_encoding(PauliDecomposition(
            terms=tuple(PauliTerm(label=label, coefficient=1.0) for label in labels),
            input_dimension=16, operator_dimension=16, num_qubits=4, atol=0.0,
        ), implementation="multiplexed_pauli")


@pytest.mark.parametrize("implementation", ["auto", "banded"])
@pytest.mark.parametrize("failure", ["overflow", "nonfinite"])
def test_banded_certificate_arithmetic_failure_does_not_select_generic_encoding(
    monkeypatch, implementation, failure,
) -> None:
    from nwqlib.subroutines.block_encoding import core
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    # IX+XX is the four-site nearest-neighbor cycle. A one-ulp coefficient
    # perturbation is admitted by structural detection but needs a nonzero bound.
    decomposition = PauliDecomposition(
        terms=(
            PauliTerm(label="IX", coefficient=1.0),
            PauliTerm(label="XX", coefficient=np.nextafter(1.0, 2.0)),
        ),
        input_dimension=4, operator_dimension=4, num_qubits=2, atol=0.0,
    )
    admitted = plan_block_encoding(decomposition, implementation=implementation)
    assert admitted.implementation == ("banded" if implementation == "banded" else "multiplexed_pauli")
    if implementation == "banded":
        assert 0.0 < admitted.error_bound < 1.0e-12
    else:
        assert admitted.error_bound == 0.0
    original_error = OverflowError("injected certificate conversion overflow")

    def fail_bound_conversion(*args, **kwargs):
        if failure == "overflow":
            raise original_error
        return np.inf

    def forbidden_generic_plan(*args, **kwargs):
        raise AssertionError("certificate arithmetic failure selected a generic encoding")

    monkeypatch.setattr(core.np, "nextafter", fail_bound_conversion)
    monkeypatch.setattr(core, "_pauli_plan_detail", forbidden_generic_plan)
    expected_error = OverflowError if failure == "overflow" else FloatingPointError
    with pytest.raises(expected_error, match="circulant certificate error bound") as caught:
        plan_block_encoding(decomposition, implementation=implementation)
    if failure == "overflow":
        assert caught.value.__cause__ is original_error


def test_auto_block_encoding_keeps_noncirculant_structural_fallback() -> None:
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition, PauliTerm

    decomposition = PauliDecomposition(
        terms=(PauliTerm(label="Z", coefficient=1.0),),
        input_dimension=2, operator_dimension=2, num_qubits=1, atol=0.0,
    )
    plan = plan_block_encoding(decomposition)
    assert plan.implementation != "banded"
    assert plan.decomposition is decomposition
    with pytest.raises(ValueError, match="not a certified circulant"):
        plan_block_encoding(decomposition, implementation="banded")


def test_block_encoding_metadata_is_json_safe_and_detached() -> None:
    matrix = np.array([[1.0, 0.2], [0.2, 0.7]], dtype=complex)
    encodings = [
        build_block_encoding(matrix, implementation="dense_dilation"),
        build_block_encoding(matrix, implementation="pauli_lcu"),
        build_block_encoding(
            BandSpecification(offsets=(0, 1), coefficients=(1.0, 0.25j), num_qubits=1),
            implementation="banded",
        ),
    ]

    for encoding in encodings:
        assert_no_block_encoding_metadata_mirrors(encoding)
        payload = encoding.to_dict()
        serialized = json.loads(json.dumps(payload))
        assert serialized["alpha"] == encoding.alpha
        assert serialized["error_bound"] == encoding.error_bound
        assert serialized["implementation"] == encoding.implementation
        payload["metadata"]["detached_test_key"] = "mutated"
        assert "detached_test_key" not in encoding.metadata


def test_scalar_metadata_domains_and_omitted_zero_plan():
    from dataclasses import replace
    from nwqlib.subroutines.block_encoding import BlockEncoding, BlockEncodingPlan
    from nwqlib.subroutines.pauli_decomposition import PauliDecomposition
    from nwqlib.algorithms.lchs.compiled_selection import _kept_pauli_part
    from nwqlib.algorithms.lchs.native import _build_compiled_qsp_part_encoding

    # No circuit is inspected or built: these records own scalar metadata only.
    actual = BlockEncoding(circuit=None, alpha=1., error_bound=0., num_ancillas=0,
                           system_qubits=0, implementation="supplied", metadata={})
    plan = BlockEncodingPlan(requested_implementation="auto", implementation="supplied",
        alpha=1., error_bound=0., num_ancillas=0, system_qubits=0,
        source=None, decomposition=None, detail={})
    for record in (actual, plan):
        for field, value in (("alpha", 0.), ("alpha", float("nan")), ("error_bound", -1.),
                             ("error_bound", float("inf")), ("num_ancillas", 1.9),
                             ("system_qubits", -1)):
            with pytest.raises(ValueError):
                replace(record, **{field: value})
        assert replace(record, error_bound=.25).error_bound == .25
    decomposition = PauliDecomposition(terms=(), input_dimension=2, operator_dimension=2,
                                      num_qubits=1, atol=0.)
    zero, zero_work = _kept_pauli_part(decomposition, set(), pruned_mass_bound=0.)
    assert zero_work == 0
    assert zero.alpha == 0 and zero.implementation == "exact_zero"
    with pytest.raises(ValueError, match="no block encoding"):
        _build_compiled_qsp_part_encoding(zero)
    for changes in ({"error_bound": .1}, {"alpha": 1.}, {"num_ancillas": 1}, {"source": object()}):
        with pytest.raises(ValueError):
            replace(zero, **changes)
