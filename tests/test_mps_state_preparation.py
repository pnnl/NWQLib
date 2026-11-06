"""Tests for MPS state-preparation helpers."""

from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector

from nwqlib.subroutines.state_preparation import mps_circuit as mps_circuit_module
from nwqlib.subroutines.state_preparation import mps as mps_module
from nwqlib.subroutines.state_preparation import (
    analyze_mps_state_compression,
    build_mps_circuit_state_preparation,
    build_qiskit_state_preparation,
    decompose_state_to_mps,
    validate_mps_circuit_state_preparation,
)


# Exact circuit classes differ only by ideal-gate floating evaluation at the
# scale of a short statevector simulation (FRAMEWORK machine-precision class).
EXACT_PREPARATION_ATOL = 1.0e-14


@pytest.mark.parametrize("num_layers", (2,))
def test_mps_circuit_backend_prepares_small_entangled_state_when_tensor_is_available(
    monkeypatch, num_layers
) -> None:
    pytest.importorskip("scikit_tt")
    _, tt_class = mps_circuit_module._require_scikit_tt()
    original_dot, original_ortho = tt_class.dot, tt_class.ortho
    residual_calls = {"dot": 0, "ortho": 0}

    def recording_dot(self, *args, **kwargs):
        residual_calls["dot"] += 1
        return original_dot(self, *args, **kwargs)

    def recording_ortho(self, *args, **kwargs):
        if kwargs.get("threshold") == 1.0e-12:
            residual_calls["ortho"] += 1
        return original_ortho(self, *args, **kwargs)

    monkeypatch.setattr(tt_class, "dot", recording_dot)
    monkeypatch.setattr(tt_class, "ortho", recording_ortho)
    state = np.zeros(4, dtype=complex)
    state[0] = np.exp(0.37j) * np.sqrt(0.6)
    state[3] = 1j * np.exp(0.37j) * np.sqrt(0.4)

    preparation = build_mps_circuit_state_preparation(state, num_layers=num_layers)
    validated = validate_mps_circuit_state_preparation(preparation)

    # Two extracted local unitaries need a residual only when another layer follows.
    expected_updates = 2 * (num_layers - 1)
    assert residual_calls == {"dot": expected_updates, "ortho": expected_updates}
    assert preparation.method == "mps_disentangling_circuit"
    assert preparation.fidelity_to_target is None
    assert validated.circuit_fidelity_status == "evaluated"
    assert validated.fidelity_to_target == pytest.approx(1.0, abs=1.0e-10)
    np.testing.assert_allclose(validated.prepared_state, state, atol=1e-12, rtol=0)
    assert validated.preparation_l2_error == pytest.approx(
        np.linalg.norm(validated.prepared_state - validated.target_state),
        rel=1.0e-12,
        abs=1.0e-15,
    )


def test_mps_product_preparation_preserves_controlled_phase_and_endian_order():
    state = np.exp(0.37j) * np.kron(
        np.array([1, 1j]) / np.sqrt(2),
        np.kron(np.array([0, 1]), np.array([np.sqrt(0.3), np.sqrt(0.7)])),
    )
    preparation = build_mps_circuit_state_preparation(state, max_bond_dim=1)
    circuit = QuantumCircuit(4)
    circuit.h(0)
    circuit.append(preparation.circuit.to_gate().control(1), range(4))
    actual = Statevector.from_instruction(circuit).data
    expected = np.zeros(16, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[1::2] = state / np.sqrt(2)
    np.testing.assert_allclose(actual, expected, atol=2e-14, rtol=0)
    assert preparation.circuit.count_ops() == {"u": 3}


@pytest.mark.parametrize(
    ("state", "expected_operation"),
    [
        (
            np.array(
                [
                    np.exp(-0.23j) / np.sqrt(3.0),
                    np.exp(-0.23j) / np.sqrt(3.0),
                    np.exp(-0.23j) / np.sqrt(3.0),
                    0.0,
                ]
            ),
            "USup",
        ),
    ],
)
def test_exact_direct_fast_paths_preserve_independent_reference(
    state: np.ndarray,
    expected_operation: str,
) -> None:
    preparation = build_qiskit_state_preparation(state)
    actual = np.asarray(Statevector.from_instruction(preparation.circuit).data)
    expected = np.asarray(state, dtype=complex) / np.linalg.norm(state)

    assert expected_operation in preparation.circuit.count_ops()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )
    assert abs(np.vdot(expected, actual)) ** 2 == pytest.approx(
        1.0,
        rel=0.0,
        abs=EXACT_PREPARATION_ATOL,
    )


@pytest.mark.parametrize(
    "state",
    [
        [1.0, 1.0e-12j, -1.0e-12, -1.0e-12j],
    ],
)
def test_direct_preparation_preserves_small_complex_components(state) -> None:
    expected = np.asarray(state, dtype=complex) / np.hypot.reduce(np.abs(state))
    preparation = build_qiskit_state_preparation(state)
    actual = Statevector.from_instruction(preparation.circuit.decompose(reps=10)).data
    np.testing.assert_allclose(actual, expected, rtol=0.0, atol=EXACT_PREPARATION_ATOL)
    small = (np.abs(expected) < 1.0e-10) & (expected != 0.0)
    # Scaling separates a missing amplitude from binary64 roundoff in the circuit.
    np.testing.assert_allclose(actual[small] / expected[small], 1.0, rtol=0.0, atol=0.01)


def test_direct_preparation_controlled_relative_phase_and_inverse() -> None:
    state = np.asarray([1.0j, 1.0e-8, -0.3, 0.2j])
    target = state / np.hypot.reduce(np.abs(state))
    preparation = build_qiskit_state_preparation(state)
    gate = preparation.circuit.to_gate()
    circuit = QuantumCircuit(3)
    circuit.h(0)
    circuit.append(gate.control(1, annotated=False), circuit.qubits)
    expected = np.zeros(8, dtype=complex)
    expected[0] = 1 / np.sqrt(2)
    expected[1::2] = target / np.sqrt(2)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.decompose(reps=10)).data,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )
    circuit.append(gate.inverse().control(1, annotated=False), circuit.qubits)
    expected[:] = 0.0
    expected[:2] = 1 / np.sqrt(2)
    np.testing.assert_allclose(
        Statevector.from_instruction(circuit.decompose(reps=10)).data,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )


@pytest.mark.parametrize(
    "state",
    [
        np.array(
            [0.5, 0.5, 0.5, np.nextafter(0.5, 1.0)],
            dtype=complex,
        ),
        np.array([0.5, 0.5j, -0.5, -0.5j], dtype=complex),
    ],
    ids=("nearly_equal_full_support", "relative_phases"),
)
def test_direct_preparation_exact_classifier_rejects_approximate_uniform_states(
    state: np.ndarray,
) -> None:
    preparation = build_qiskit_state_preparation(state)
    actual = np.asarray(Statevector.from_instruction(preparation.circuit).data)
    expected = np.asarray(state, dtype=complex) / np.linalg.norm(state)

    assert "h" not in preparation.circuit.count_ops()
    assert "USup" not in preparation.circuit.count_ops()
    np.testing.assert_allclose(
        actual,
        expected,
        rtol=0.0,
        atol=EXACT_PREPARATION_ATOL,
    )


def test_existing_mps_analysis_metrics_reuse_order_and_before_svd_cost(monkeypatch):
    state = np.array([np.sqrt(3)/2, 0, 0, .5], dtype=complex)
    truncated = decompose_state_to_mps(state, max_bond_dim=1)
    product = decompose_state_to_mps(np.array([1, 2, 2, 4])/5, max_bond_dim=1)
    reordered = decompose_state_to_mps(np.array([1, 2, 4, 2])/5)
    assert product.bond_dimensions == (1, 1, 1)
    assert reordered.bond_dimensions == (1, 2, 1)
    assert product.input_id != reordered.input_id
    assert all(not core.flags.writeable for core in truncated.cores)
    with pytest.raises(ValueError):
        truncated.cores[0].setflags(write=True)
    original = mps_module._reconstruct_cores
    calls = []
    monkeypatch.setattr(np.linalg, "svd", Mock(side_effect=AssertionError("unexpected SVD")))
    monkeypatch.setattr(mps_module, "_reconstruct_cores", lambda *a, **k: (calls.append(1), original(*a, **k))[1])
    scalar = analyze_mps_state_compression(truncated)
    scalar.to_dict()
    assert calls == []
    assert scalar.decomposition is truncated
    assert scalar.estimated_raw_l2 == pytest.approx(.5, rel=0, abs=1e-14)
    assert scalar.estimated_normalized_fidelity == pytest.approx(.75, rel=0, abs=1e-14)
    comparison = analyze_mps_state_compression(truncated, reference=state)
    assert calls == [1]
    assert comparison.reference_matches_input
    assert comparison.measured_raw_l2 == pytest.approx(.5, rel=0, abs=1e-14)
    assert comparison.measured_normalized_fidelity == pytest.approx(.75, rel=0, abs=1e-14)
    assert analyze_mps_state_compression(product).estimated_raw_l2 == pytest.approx(0, rel=0, abs=1e-14)
    with pytest.raises(ValueError, match="max_svd_work"):
        decompose_state_to_mps(np.ones(8), max_svd_work=1)
    with pytest.raises(ValueError, match="max_bytes"):
        truncated.to_statevector(max_bytes=1)
    # One contraction step of 2*2 output entries times the contracted rank 1.
    np.testing.assert_allclose(truncated.to_statevector(), [np.sqrt(3)/2, 0, 0, 0], rtol=0, atol=1e-15)
    with pytest.raises(ValueError, match="needs 4 scalar products, exceeding max_products=3"):
        truncated.to_statevector(max_products=3)


def test_selected_mps_reuses_the_hash_of_immutable_storage(monkeypatch):
    # The TT-SVD hashes the bytes-backed target it compresses, and builds from
    # those cores and the same array reuse that hash.
    from nwqlib.operators.inputs import _freeze_array

    hashed = []
    original = mps_module._digest
    monkeypatch.setattr(mps_module, "_digest", lambda *args: hashed.append(args[1]) or original(*args))
    target = _freeze_array(np.kron(np.array([.6, .8]), np.full(4, .5)).astype(complex))
    decomposition = mps_module._decompose_normalized_state(target, max_bond_dim=None, threshold=1e-14)
    for _ in range(2):
        circuit = build_mps_circuit_state_preparation(target, _selected_decomposition=decomposition).circuit
        np.testing.assert_allclose(Statevector(circuit).data, target, rtol=0, atol=EXACT_PREPARATION_ATOL)
    assert hashed == [(8,)]


def test_fidelity_kernel_accepts_extreme_but_finite_entry_magnitudes():
    """Vectors whose largest modulus is subnormal or overflows keep their fidelity.

    Dividing by a subnormal scale overflows NumPy's reciprocal, and the
    modulus of ``1.5e308 + 1.5e308j`` overflows although both parts are
    finite. Exact power-of-two rescaling avoids both, so the values below
    are the exact fidelities 1/2 and 1 up to the returned roundoff window.
    """
    from nwqlib._numerics import normalized_fidelity_with_window

    tiny = np.array([1e-310, 0.0], dtype=complex)
    huge = np.array([1.5e308 + 1.5e308j, 1.5e308 + 1.5e308j])
    for vector, other, expected in (
        (tiny, np.array([1.0, 1.0], dtype=complex), 0.5),
        (huge, np.array([1.0 + 1.0j, 1.0 + 1.0j]), 1.0),
        (huge, np.array([1.0, 0.0], dtype=complex), 0.5),
    ):
        fidelity, window = normalized_fidelity_with_window(vector, other)
        assert abs(fidelity - expected) <= window


@pytest.mark.parametrize("distance", [1e-9])
def test_mps_two_qubit_gates_are_exact_near_a_product_state(distance) -> None:
    # A two-qubit state has bond dimension two, so one layer prepares it
    # exactly apart from the synthesis of its local unitaries. Qiskit 2.6.0's
    # TwoQubitBasisDecomposer replaced the nearly local unitary of this state
    # by a local one and prepared |00> alone, an error equal to the distance.
    pytest.importorskip("scikit_tt")
    state = np.array([1.0, 0.0, 0.0, distance], dtype=complex)
    state /= np.linalg.norm(state)
    preparation = build_mps_circuit_state_preparation(state, num_layers=1)
    prepared = Statevector.from_instruction(preparation.circuit).data
    np.testing.assert_allclose(prepared, state, atol=EXACT_PREPARATION_ATOL, rtol=0)


def test_mps_builder_admits_its_whole_construction_before_the_first_svd(monkeypatch) -> None:
    # The builder compares once, before the first SVD, the TT-SVD sweep, the
    # at most layers * (n - 1) exact two-qubit syntheses and the layered
    # construction at the largest bonds the sweep can keep, (1, 2, 4, 2, 1)
    # for 16 amplitudes. One unit below their sum refuses before any SVD or
    # synthesis, and the sum admits.
    pytest.importorskip("scikit_tt")
    from test_dense_synthesis import _dense_synthesis_work
    from nwqlib.subroutines.state_preparation.mps import layered_construction_size

    rng = np.random.default_rng(9)
    state = rng.normal(size=16) + 1j * rng.normal(size=16)
    state /= np.linalg.norm(state)
    calls = []
    original = mps_circuit_module.dense_unitary_circuit
    monkeypatch.setattr(mps_circuit_module, "dense_unitary_circuit",
                        lambda matrix: calls.append(np.shape(matrix)) or original(matrix))
    svds = []
    original_svd = np.linalg.svd
    monkeypatch.setattr(np.linalg, "svd", lambda matrix, *a, **k: svds.append(matrix.shape) or original_svd(matrix, *a, **k))
    sweep = 2*8*2 + 4*4*4 + 8*2*2
    need = sweep + 2*3*_dense_synthesis_work(2) + layered_construction_size((1, 2, 4, 2, 1), 2)[0]
    with pytest.raises(ValueError, match="MPS circuit needs .* max_svd_work"):
        build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=need - 1)
    assert calls == [] and svds == []
    build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=need)
    assert calls and set(calls) == {(4, 4)} and len(calls) <= 6

    # After the TT-SVD, which runs NumPy's SVD, scikit_tt orthonormalizes the
    # MPS again after every extracted gate with SciPy's SVD. For ten random
    # qubits a limit one below the traced total of those SVDs alone,
    # m*c*min(m, c) each, must refuse the construction before scikit_tt
    # starts. The default, no limit, builds it.
    import scipy.linalg

    state = rng.normal(size=1024) + 1j * rng.normal(size=1024)
    traced = []
    original_svd = scipy.linalg.svd

    def svd(matrix, *args, **kwargs):
        traced.append(matrix.shape[0] * matrix.shape[1] * min(matrix.shape))
        return original_svd(matrix, *args, **kwargs)

    monkeypatch.setattr(scipy.linalg, "svd", svd)
    build_mps_circuit_state_preparation(state, num_layers=2)
    started = Mock(wraps=mps_circuit_module._scikit_tt_mps_from_cores)
    monkeypatch.setattr(mps_circuit_module, "_scikit_tt_mps_from_cores", started)
    with pytest.raises(ValueError, match="MPS circuit needs .* max_svd_work"):
        build_mps_circuit_state_preparation(state, num_layers=2, max_svd_work=sum(traced) - 1)
    assert started.call_count == 0


def test_mps_admission_prices_the_scalar_and_product_routes_they_take(monkeypatch) -> None:
    # A scalar is prepared by a global phase and cores whose bonds are all
    # one by one gate per qubit, so an explicit limit counts no two-qubit
    # synthesis for either. With bond cap one, four qubits need the sweep
    # 2*8*2 + 2*4*2 + 2*2*2 = 56 and 64 per qubit for the product gates, 312
    # in all, and one unit less refuses before any SVD.
    scalar = build_mps_circuit_state_preparation(np.array([1j]), max_bond_dim=1, max_svd_work=1)
    assert scalar.circuit.num_qubits == 0 and scalar.circuit.size() == 0
    svds = []
    original_svd = np.linalg.svd
    monkeypatch.setattr(np.linalg, "svd", lambda matrix, *a, **k: svds.append(matrix.shape) or original_svd(matrix, *a, **k))
    state = np.full(16, .25)
    with pytest.raises(ValueError, match="needs 312 work"):
        build_mps_circuit_state_preparation(state, max_bond_dim=1, max_svd_work=311)
    assert svds == []
    product = build_mps_circuit_state_preparation(state, max_bond_dim=1, max_svd_work=312)
    assert product.decomposition.bond_dimensions == (1,) * 5
    assert {item.operation.num_qubits for item in product.circuit.data} == {1}


def test_reused_cores_compare_their_construction_with_max_svd_work_before_building():
    """Cores supplied with the target, as a reopened LCHS Plan supplies them, are compared with max_svd_work.

    The 3-qubit GHZ cores have bonds (1, 2, 2, 1), so the default two layers
    need 4 two-qubit syntheses and the layered construction at those bonds.
    """
    from nwqlib.operators.inputs import _freeze_array
    from nwqlib.subroutines._dense_synthesis import dense_synthesis_size

    ghz = np.zeros(8, dtype=complex)
    ghz[[0, 7]] = np.sqrt(.5)
    target = _freeze_array(ghz)
    decomposition = mps_module._decompose_normalized_state(target, max_bond_dim=None, threshold=1e-14)
    assert decomposition.bond_dimensions == (1, 2, 2, 1)
    work = 4 * dense_synthesis_size(2)[0] + mps_module.layered_construction_size((1, 2, 2, 1), 2)[0]
    with pytest.raises(ValueError, match=f"needs {work} work .* exceeding max_svd_work={work - 1}"):
        build_mps_circuit_state_preparation(target, _selected_decomposition=decomposition, max_svd_work=work - 1)
    circuit = build_mps_circuit_state_preparation(target, _selected_decomposition=decomposition,
                                                  max_svd_work=work).circuit
    np.testing.assert_allclose(Statevector(circuit).data, target, rtol=0, atol=EXACT_PREPARATION_ATOL)
