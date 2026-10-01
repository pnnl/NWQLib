"""Actual compiled source SELECT preserves its independent physical grid."""

import numpy as np
import pytest
from scipy.linalg import expm

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms.lchs import LCHS, ProviderConfig


def test_actual_qsp_source_binds_elapsed_diagonals_recovery_and_physical_phase(monkeypatch):
    matrix = np.array([[-.2, .15+.1j], [.04-.06j, .3]])
    initial, source = np.array([.8, .6j]), np.array([.3, -.2j])
    elapsed = .5
    method = LCHS(hamiltonian_evolution_backend='qsp_block_encoding', duhamel_nodes=2,
        lchs_kernel=ProviderConfig(implementation='cauchy_density'),
        k_quadrature=ProviderConfig(implementation='signed_binary_uniform',
            parameters={'num_qubits':2, 'lsb_position':-1}))
    chosen = plan(LinearDynamics(A=matrix, initial_state=initial, source=source, time=elapsed), method=method)
    data = chosen._native['native_data']
    selected = data.qsp_plan
    # Independent signed-address grid and two-point Gauss rule. Slots4..7
    # are padding; the source kind bit is3, above its one time-address bit.
    nodes = np.array([0., .5, -1., -.5])
    coefficients = .5/(np.pi*(1+nodes**2))
    source_times = elapsed/2*(1+np.array([-1., 1.])/np.sqrt(3))
    l_part, h_part = (matrix+matrix.conj().T)/2, (matrix-matrix.conj().T)/(2j)
    assert np.linalg.norm(l_part@h_part-h_part@l_part)>.01
    radius = np.sqrt(((l_part[0,0].real-l_part[1,1].real)/2)**2+abs(l_part[0,1])**2)
    minimum = (l_part[0,0].real+l_part[1,1].real)/2 - radius
    maximum = (l_part[0,0].real+l_part[1,1].real)/2 + radius
    shift = -minimum + method.psd_tolerance*max(abs(minimum), abs(maximum))
    assert minimum<0
    assert chosen.reconstruction.psd_shift==pytest.approx(shift, abs=2e-16)
    np.testing.assert_allclose(chosen.reconstruction.source_nodes, source_times, rtol=0, atol=1e-16)
    np.testing.assert_allclose(chosen.reconstruction.source_weights, [elapsed/2]*2, rtol=0, atol=1e-16)
    expected_l, expected_h = np.zeros(16), np.zeros(16)
    expected_coefficients = np.zeros(16, dtype=complex)
    expected = np.zeros(2, dtype=complex)
    applications = [(0, elapsed, 1., initial),
        (8, elapsed-source_times[0], elapsed/2, source),
        (12, elapsed-source_times[1], elapsed/2, source)]
    for base, time, weight, vector in applications:
        expected_l[base:base+4] = time/elapsed*nodes
        expected_h[base:base+4] = time/elapsed
        expected_coefficients[base:base+4] = weight*np.linalg.norm(vector)*np.exp(shift*time)*coefficients
        for node, coefficient in zip(nodes, coefficients, strict=True):
            expected += weight*coefficient*np.exp(shift*time)*(expm(
                -1j*time*(node*(l_part+shift*np.eye(2))+h_part))@vector)
    np.testing.assert_allclose(selected['physical_l_diagonal'], expected_l, rtol=0, atol=2e-16)
    np.testing.assert_allclose(selected['physical_h_diagonal'], expected_h, rtol=0, atol=2e-16)
    np.testing.assert_allclose(data.source_layout.coefficients, expected_coefficients, rtol=0, atol=2e-16)
    prepared = selected['prepared_evolution']
    amplitude = .5/prepared.scale
    recovery = 1/(3*amplitude-4*amplitude**3)
    assert prepared.recovery_scale==pytest.approx(recovery, rel=2e-15, abs=0)
    assert chosen.reconstruction.recovery.as_float()==pytest.approx(
        np.sum(abs(expected_coefficients))*recovery, rel=2e-15, abs=0)
    homogeneous = plan(LinearDynamics(A=matrix, initial_state=initial, time=elapsed), method=method)
    np.testing.assert_array_equal(homogeneous._native['native_data'].qsp_plan['physical_h_diagonal'], np.ones(4))
    assert sum(register.width for register in chosen.construction.program.registers)<=13
    from nwqlib.subroutines.qsp import evolution
    monkeypatch.setattr(evolution, 'prepare_qsp_evolution', lambda **kw: pytest.fail('selected phases fitted again'))
    actual = solve(chosen)
    # This is the same finite grid, with QSP polynomial/phase approximation.
    # It is not a claim that this small Cauchy grid solves the exact ODE.
    np.testing.assert_allclose(actual.solution, expected, rtol=0, atol=1e-4)
