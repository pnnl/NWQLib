"""Tests for LCU and direct state-preparation subroutines."""

from __future__ import annotations

from decimal import Decimal, localcontext

import numpy as np
import pytest
from qiskit import transpile
from qiskit.quantum_info import Operator, Statevector

from nwqlib.subroutines.lcu import (
    build_lcu_circuit,
    build_lcu_prepare,
    build_lcu_select,
    prepare_lcu_data,
)
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
