"""Tests for LCU and direct state-preparation subroutines."""

from __future__ import annotations

import math
from decimal import Decimal, localcontext
from types import SimpleNamespace

import numpy as np
import pytest
from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Operator, Statevector

from nwqlib.subroutines.lcu import (
    build_lcu_circuit,
    build_lcu_prepare,
    build_lcu_select,
    prepare_lcu_data,
)
from nwqlib.subroutines.state_preparation import build_qiskit_state_preparation
from nwqlib._numerics import normalize_state_vector
from nwqlib.subroutines.lcu import core as lcu_core


def test_lcu_prepare_amplitudes_match_coefficient_magnitudes() -> None:
    coefficients = np.array([1.0, -4.0j, 3.0], dtype=complex)
    unitaries = [np.eye(2, dtype=complex) for _ in coefficients]
    data = prepare_lcu_data(coefficients, unitaries)

    state = Statevector.from_instruction(build_lcu_prepare(data))
    expected = np.sqrt(np.array([1.0, 4.0, 3.0, 0.0]) / 8.0)

    assert data.num_control_qubits == 2
    assert data.coefficient_l1_norm == 8.0  # The integer magnitudes sum exactly.
    np.testing.assert_allclose(state.data, expected, rtol=2e-14, atol=2e-14)

    # Independent high-precision ratio before sqrt: the binary64 probability
    # underflows, but the amplitude must remain nonzero in both public paths.
    coefficients = [1e-300, 3e30]
    with localcontext() as context:
        context.prec = 60
        small, large = map(Decimal.from_float, coefficients)
        amplitude = float((small / (small + large)).sqrt())
    for data in (
        prepare_lcu_data(coefficients, [np.eye(2), np.eye(2)]),
        lcu_core.prepare_lcu_gate_data(coefficients, system_dimension=2),
    ):
        # Three rounded sqrt/division operations, with ample relative headroom
        # and zero absolute tolerance so loss to zero cannot pass.
        assert data.prep_amplitudes[0] == pytest.approx(amplitude, rel=2e-15, abs=0)
    for coefficients in ([np.nan, 1.], [np.inf, 1.], [1e308, 1e308]):
        with pytest.raises(ValueError, match='finite'):
            lcu_core.prepare_lcu_gate_data(coefficients, system_dimension=2)


@pytest.mark.parametrize("scale", (1., 1e-300, 1e300))
def test_lcu_nonzero_global_scale_preserves_preparation_and_branch_phase(scale):
    coefficients = np.array([-1., 3.]) * scale
    data = lcu_core.prepare_lcu_gate_data(coefficients, system_dimension=2)
    np.testing.assert_allclose(data.prep_amplitudes, [.5, np.sqrt(3)/2], rtol=2e-15, atol=0.)
    assert data.coefficient_l1_norm == pytest.approx(4*scale, rel=2e-15, abs=0.)
    assert data.original_coefficients == tuple(complex(value) for value in coefficients)
    if scale == 1e-300:
        with pytest.raises(ValueError, match="1-norm must be nonzero"):
            lcu_core.prepare_lcu_gate_data(coefficients, system_dimension=2, coefficient_atol=1e-14)


def test_lcu_direct_prepare_uses_shared_exact_fast_path() -> None:
    coefficients = np.ones(4, dtype=complex)
    unitaries = [np.eye(2, dtype=complex) for _ in coefficients]
    data = prepare_lcu_data(coefficients, unitaries)

    preparation = build_lcu_prepare(data)

    assert preparation.count_ops() == {"h": 2}
    # The ideal direct PREP keeps the selected coefficient-state phase.
    assert build_lcu_circuit(coefficients, unitaries).preparation_l2_error == 0.0


def test_lcu_prepare_supports_mps_backend_for_product_amplitudes() -> None:
    # sqrt(c/12) for c=(1,2,3,6) is the product ((1,sqrt(3))/2) x ((1,sqrt(2))/sqrt(3))
    # in little-endian index order. Swapping the two qubits changes this state,
    # which a uniform state cannot reveal.
    coefficients = np.array([1.0, 2.0, 3.0, 6.0], dtype=complex)
    unitaries = [np.eye(2, dtype=complex) for _ in coefficients]
    data = prepare_lcu_data(coefficients, unitaries)

    state = Statevector.from_instruction(
        build_lcu_prepare(data, preparation_backend="mps_circuit", mps_max_bond_dim=1)
    )

    # Two single-qubit u gates contribute only ideal-gate floating-point roundoff.
    np.testing.assert_allclose(
        state.data, np.sqrt(np.array([1.0, 2.0, 3.0, 6.0]) / 12.0), rtol=0, atol=2e-14
    )


def test_lcu_circuit_reuses_one_mps_preparation_artifact(monkeypatch) -> None:
    calls = []

    def fake_mps_builder(vector, **_kwargs):
        calls.append(np.asarray(vector, dtype=complex).copy())
        return SimpleNamespace(
            circuit=QuantumCircuit(2),
            method="mps_disentangling_circuit",
            fidelity_to_target=None,
            to_dict=lambda: {
                "method": "mps_disentangling_circuit",
                "circuit_fidelity_status": "not_evaluated",
                "fidelity_to_target": None,
                "preparation_l2_error": None,
            },
        )

    monkeypatch.setattr(
        lcu_core,
        "build_mps_circuit_state_preparation",
        fake_mps_builder,
    )
    coefficients = np.ones(4, dtype=complex)
    unitaries = [np.eye(2, dtype=complex) for _ in coefficients]
    lcu = build_lcu_circuit(
        coefficients,
        unitaries,
        preparation_backend="mps_circuit",
    )

    assert len(calls) == 1
    assert lcu.preparation_metadata["preparation_l2_error"] is None
    assert lcu.preparation_l2_error is None


def test_layered_mps_lcu_error_is_unknown_even_without_tt_discard(monkeypatch):
    coefficients = np.random.default_rng(913).uniform(.1, 2., 16)
    signs = np.where(np.arange(16) % 3 == 0, -1., 1.)
    target = np.dot(coefficients, signs)/np.sum(coefficients)
    unitaries = [s*np.eye(2) for s in signs]
    for backend in ("direct", "mps_circuit"):
        with monkeypatch.context() as guard:
            guard.setattr(Statevector, "from_instruction",
                lambda *a, **k: pytest.fail("LCU construction/report simulated PREP"))
            lcu = build_lcu_circuit(coefficients, unitaries, preparation_backend=backend)
            metadata = lcu.to_dict()["preparation_metadata"]
        # One five-qubit column suffices because every selected branch is +/-I.
        actual = Statevector.from_instruction(lcu.circuit).data[0]
        if backend == "direct":
            assert actual == pytest.approx(target, rel=0, abs=2e-13)
        else:
            assert abs(actual-target) > 1e-3
            assert metadata["decomposition"]["discarded_weight"] == 0.
            assert metadata["circuit_fidelity_status"] == "not_evaluated"
            assert metadata["preparation_l2_error"] is None
            assert lcu.preparation_l2_error is None


def test_lcu_absorbs_complex_coefficient_phases_into_unitaries() -> None:
    identity = np.eye(2, dtype=complex)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    data = prepare_lcu_data([1.0, 2.0j], [identity, x_gate])

    assert data.positive_coefficients[:2] == pytest.approx((1.0, 2.0), rel=0, abs=2e-13)
    assert np.allclose(data.phase_adjusted_unitaries[1], 1.0j * x_gate)

    floor = np.nextafter(0.0, 1.0)
    data = prepare_lcu_data([1.0, complex(floor, floor)], [identity, x_gate])
    # Equal positive real/imaginary parts have phase pi/4 at every nonzero scale.
    np.testing.assert_allclose(
        data.phase_adjusted_unitaries[1], (1.0 + 1.0j) / math.sqrt(2.0) * x_gate,
        rtol=2.0e-15, atol=0.0,
    )


@pytest.mark.parametrize("scale", [1.0, 6.0e-15])
@pytest.mark.parametrize("coefficients", [(1.0, 1.0j), (1.0, -1.0), (1.0j, -1.0j)])
def test_lcu_circuit_all_zero_block_matches_linear_combination(scale, coefficients) -> None:
    identity = np.eye(2, dtype=complex)
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    unitaries = [identity, x_gate if coefficients == (1.0, 1.0j) else identity]
    lcu = build_lcu_circuit(scale * np.asarray(coefficients), unitaries)
    native = transpile(lcu.circuit, basis_gates=["cx", "u"], optimization_level=0)
    postselected_block = Operator(native).data[::2, ::2]
    expected = sum(c * unitary for c, unitary in zip(coefficients, unitaries)) / 2.0

    assert lcu.data.coefficient_l1_norm == pytest.approx(2.0 * scale, rel=1.0e-15, abs=0.0)
    # Two-qubit native synthesis roundoff; zero-block cancellation needs an absolute budget.
    np.testing.assert_allclose(postselected_block, expected, rtol=1.0e-12, atol=1.0e-12)


def test_lcu_single_term_does_not_allocate_control_register() -> None:
    x_gate = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)

    lcu = build_lcu_circuit([2.0], [x_gate])
    state = Statevector.from_instruction(lcu.circuit)

    assert lcu.data.num_control_qubits == 0
    assert lcu.circuit.num_qubits == 1
    assert state.probabilities_dict()["1"] == pytest.approx(1.0, rel=0, abs=2e-13)


def test_state_preparation_norm_is_scaling_safe() -> None:
    tiny = build_qiskit_state_preparation(np.array([1.0e-200, 0.0], dtype=complex))
    assert tiny.input_norm == 1.0e-200
    np.testing.assert_array_equal(tiny.normalized_state, np.array([1.0, 0.0]))

    large = build_qiskit_state_preparation(np.array([1.0e300, -1.0e300], dtype=complex))
    assert np.isfinite(large.input_norm)
    assert large.input_norm == pytest.approx(np.sqrt(2.0) * 1.0e300, rel=2e-15, abs=0)
    assert np.linalg.norm(large.normalized_state) == pytest.approx(1.0, rel=0, abs=2e-13)

    for scale in (1.0e-320, np.nextafter(0.0, 1.0)):
        normalized, input_norm = normalize_state_vector(scale * np.array([1.0, 1.0j]))
        assert input_norm == math.hypot(scale, scale)
        np.testing.assert_allclose(
            normalized, np.array([1.0, 1.0j]) / math.sqrt(2.0), rtol=2.0e-15, atol=0.0,
        )

    for phase in (1.0, 1.0j):
        vector = np.array([1.0e-161, phase * 2.0e-161], dtype=complex)
        preparation = build_qiskit_state_preparation(vector)
        # Direct construction has zero ideal-gate error in the record and its summary.
        assert preparation.preparation_l2_error == 0.0 == preparation.to_dict()["preparation_l2_error"]
        expected_norm = math.hypot(1.0e-161, 2.0e-161)
        # BLAS norm and two divisions each contribute ordinary binary64 rounding.
        assert preparation.input_norm == pytest.approx(expected_norm, rel=8 * np.finfo(float).eps, abs=0.0)
        expected = np.array([1.0, phase * 2.0]) / math.sqrt(5.0)
        np.testing.assert_allclose(preparation.normalized_state, expected, rtol=2.0e-15, atol=0.0)
        np.testing.assert_allclose(
            Statevector.from_instruction(preparation.circuit).data, expected,
            rtol=1.0e-12, atol=1.0e-12,
        )

    with pytest.raises(ValueError, match="nonzero norm"):
        build_qiskit_state_preparation(np.array([0.0, 0.0], dtype=complex))


def test_build_lcu_select_rejects_gate_level_data() -> None:
    from nwqlib.subroutines.lcu.core import build_lcu_select, prepare_lcu_gate_data

    data = prepare_lcu_gate_data([1.0, 1.0], system_dimension=2)

    with pytest.raises(ValueError, match="prepare_lcu_gate_data"):
        build_lcu_select(data)


def test_build_lcu_select_rejects_partial_dense_unitaries() -> None:
    from dataclasses import replace

    from nwqlib.subroutines.lcu.core import build_lcu_select, prepare_lcu_data

    data = prepare_lcu_data([1.0, 1.0, 1.0], [np.eye(2)] * 3)
    partial = replace(data, phase_adjusted_unitaries=data.phase_adjusted_unitaries[:1])

    with pytest.raises(ValueError, match="length.*term_count"):
        build_lcu_select(partial)


def test_build_lcu_select_keeps_supplied_branches_and_implicit_padding() -> None:
    from nwqlib.subroutines.lcu.core import build_lcu_select, prepare_lcu_data

    x = np.array([[0, 1], [1, 0]], dtype=complex)
    y = np.array([[0, -1j], [1j, 0]], dtype=complex)
    z = np.diag([1.0, -1.0]).astype(complex)
    coefficients = [1.0, -2.0j, 0.0]
    data = prepare_lcu_data(coefficients, [x, y, z])
    select = build_lcu_select(data)

    assert len(data.phase_adjusted_unitaries) == data.term_count == 3
    assert data.padded_term_count == 4
    # The zero-weight supplied Z remains a real SELECT branch; only the
    # construction-known fourth identity has no matrix or controlled gate.
    assert sum(select.count_ops().values()) == data.term_count
    # The address is in the two low bits: system rows for address j are j, j+4.
    # Distinct branches expose bit reversal; the fourth branch must be identity.
    expected = np.zeros((8, 8), dtype=complex)
    for address, unitary in enumerate((x, -1j * y, z, np.eye(2))):
        indices = [address, address + 4]
        expected[np.ix_(indices, indices)] = unitary
    native = transpile(select, basis_gates=["u", "cx"], optimization_level=0)
    np.testing.assert_allclose(Operator(native).data, expected, rtol=0.0, atol=2e-13)

    # PREPARE and SELECT must use the same addresses and coefficient phases.
    lcu = build_lcu_circuit(coefficients, [x, y, z])
    actual = Operator(lcu.circuit).data[::4, ::4]
    np.testing.assert_allclose(actual, (x - 2j * y) / 3, rtol=0.0, atol=2e-13)


def test_lcu_intake_rejects_before_unitary_conversion_and_honors_caps(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("invalid coefficients or short cap reached matrix conversion")
    with monkeypatch.context() as guard:
        guard.setattr(lcu_core,"_as_unitary_array",forbidden)
        for coefficients in ([0.,0.],[np.nan,1.],[np.inf,1.]):
            with pytest.raises(ValueError):
                prepare_lcu_data(coefficients,[np.eye(2),np.eye(2)])
        for controls in ({'max_bytes':1},{'max_work':1}):
            with pytest.raises(ValueError,match='max_'):
                prepare_lcu_data([1.,1.],[np.eye(2),np.eye(2)],**controls)
    data = prepare_lcu_data([1.,0.],[np.eye(2),np.diag([1.,2.])])
    with pytest.raises(ValueError,match='unitary'):
        build_lcu_select(data)
    seen = []
    admit = lcu_core._admit_lcu
    def observed(data,**kwargs):
        seen.append((kwargs['max_bytes'],kwargs['max_work']))
        return admit(data,**kwargs)
    monkeypatch.setattr(lcu_core,'_admit_lcu',observed)
    build_lcu_circuit([1.,1.],[np.eye(2),np.eye(2)],max_bytes=20_000_000_000,max_work=200_000_000)
    assert seen and set(seen)=={(20_000_000_000,200_000_000)}


def test_dense_select_admission_counts_branch_synthesis_before_it_starts(monkeypatch):
    """The admission counts the exact synthesis of each dense branch before it runs.

    With N = P = 4 branches on D = 64, each controlled branch is a 256-square
    matrix on m = 8 qubits. The admission charges each branch
    11 * 256**3 + (8**2 + 5*8 + 256) * 256**2 = 2.08e8 work units, so the
    four need 8.3e8, above an explicit max_work of 1e8, although every array
    fits easily. N = P = 2 on D = 64 needs 5.8e7 and is admitted. For N = P = 16
    on D = 256 and a raised work limit, the kept circuits need
    16 * (176 * 4096**2 + 16384) = 4.7e10 bytes, above the default 1e10.
    A small SELECT still builds.
    """

    def forbidden(*args, **kwargs):
        raise AssertionError("dense branch synthesis started before admission")

    with monkeypatch.context() as guard:
        guard.setattr(lcu_core, "_controlled_branch_gate", forbidden)
        with pytest.raises(ValueError, match="max_work"):
            build_lcu_circuit([1.0] * 4, [np.eye(64)] * 4, max_work=10**8)
        with pytest.raises(ValueError, match="max_bytes"):
            build_lcu_circuit([1.0] * 16, [np.eye(256)] * 16, max_work=10**16)
        prepare_lcu_data([1.0, 1.0], [np.eye(64)] * 2, max_work=10**8)

    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    lcu = build_lcu_circuit([0.25, 0.75], [np.eye(2), x])
    actual = Operator(lcu.circuit).data[::2, ::2]
    # Machine-precision class: ideal gates evaluated in binary64.
    np.testing.assert_allclose(actual, 0.25 * np.eye(2) + 0.75 * x, rtol=0.0, atol=1e-13)


def test_dense_select_branches_fit_their_admitted_slots():
    """Each synthesized branch fits the instruction slots and CX law of its admission.

    Two random branches on D = 8 give controlled branches on m = 4 qubits.
    The closed definition of each keeps at most (11/16) 4**4 = 176
    instructions, the slots that the byte law of ``_admit_lcu`` charges, and
    at most (25/96) 4**4 - 2**4 + 4/3 = 52 CX, the CX law of the controlled
    census. The SELECT applies each supplied branch on its address to
    rounding.
    """
    from qiskit.quantum_info import random_unitary

    branches = [np.asarray(random_unitary(8, seed=seed).data) for seed in (3, 4)]
    select = build_lcu_select(prepare_lcu_data([1.0, 1.0], branches))
    for instruction in select.data:
        closed = instruction.operation._definition
        assert len(closed.data) <= 11 * 4**4 // 16
        assert closed.count_ops().get("cx", 0) <= 52
    expected = sum(
        np.kron(branch, np.diag(np.eye(2)[j])) for j, branch in enumerate(branches)
    )
    # 64 unit roundoffs times the dimension, the bound of tests/test_dense_synthesis.py.
    np.testing.assert_allclose(Operator(select).data, expected, rtol=0.0, atol=64 * 2.0**-53 * 16)


def test_dense_select_is_exact_for_small_angle_branches():
    """Near-identity branches keep their small rotations exactly.

    Qiskit 2.5.2's ``UnitaryGate.control`` kept a one-control synthesis of
    this ``expm(-i 1e-6 G)`` with entry errors of 6.4e-11, and for
    ``RX(1e-7)`` with three controls its ``Isometry`` fallback raised
    ``ValueError("Input matrix is not unitary.")``.
    """
    from scipy.linalg import expm

    rng = np.random.default_rng(5)
    x = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
    small = expm(-1e-6j * (x + x.conj().T) / 2)
    select = build_lcu_select(prepare_lcu_data([1.0, 1.0], [small, np.eye(2)]))
    expected = np.kron(small, np.diag([1.0, 0.0])) + np.kron(np.eye(2), np.diag([0.0, 1.0]))
    np.testing.assert_allclose(Operator(select).data, expected, rtol=0.0, atol=64 * 2.0**-53 * 4)

    angle = 1e-7
    rx = np.array([[np.cos(angle / 2), -1j * np.sin(angle / 2)],
                   [-1j * np.sin(angle / 2), np.cos(angle / 2)]])
    select = build_lcu_select(prepare_lcu_data([1.0] * 8, [rx] * 8))
    np.testing.assert_allclose(Operator(select).data, np.kron(rx, np.eye(8)), rtol=0.0,
                               atol=64 * 2.0**-53 * 16)
