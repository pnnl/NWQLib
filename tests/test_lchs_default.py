"""Finite default qualification through the actual public Aer consumer."""

import numpy as np
import pytest

from nwqlib import LinearDynamics, plan, solve
from nwqlib.algorithms import LCHS


def test_two_finite_recipes_reach_aer_and_preserve_their_actual_error_scope(monkeypatch):
    from qiskit.circuit.library import UnitaryGate
    from nwqlib.backends import qiskit_aer

    # Raw UnitaryGate.control reconstructs the full controlled matrix for
    # every branch. The selected circuit owner must synthesize its branch
    # without that repeated wide validation; actual Aer still executes it.
    monkeypatch.setattr(UnitaryGate, "control", lambda *a, **k: pytest.fail("raw dense control"))
    original = qiskit_aer._submit_aer_execution
    widths, arithmetic_sizes = [], []

    def submit(prepared, **kwargs):
        widths.append(prepared.circuit.num_qubits)
        assert prepared.circuit.num_qubits <= 9
        arithmetic_sizes.append(sum(16 * (1 << instruction.operation.num_qubits)
                                    for instruction in prepared.circuit.data
                                    if not getattr(instruction.operation, "_directive", False)))
        return original(prepared, **kwargs)

    monkeypatch.setattr(qiskit_aer, "_submit_aer_execution", submit)
    matrix = np.array([[0.4, 0.15], [0.05, 0.25]])
    initial = np.array([1.0, 0.0])
    elapsed = 0.1
    # Cayley-Hamilton: B=A-tr(A)I/2 has B^2=delta^2 I. Thus the
    # even/odd exponential series sum to cosh(t*delta)I-sinh(t*delta)B/delta.
    # This reference uses neither the LCHS coefficients nor scipy.expm.
    center = np.trace(matrix) / 2
    shifted = matrix - center * np.eye(2)
    delta = np.sqrt(((matrix[0, 0] - matrix[1, 1]) / 2) ** 2 + matrix[0, 1] * matrix[1, 0])
    reference = np.exp(-elapsed * center) * (
        np.cosh(elapsed * delta) * initial - np.sinh(elapsed * delta) / delta * (shifted @ initial)
    )
    problem = LinearDynamics(A=matrix, initial_state=initial, time=elapsed)
    errors, nodes = [], []
    for tolerance in (0.01, 0.001):
        # Execute the approved nine-qubit default once below. Both scalar
        # finite sums also admit an independent two-dimensional reference.
        selected = plan(problem, method=LCHS(approximation_tolerance=tolerance), execution="classical", seed=7)
        result = solve(selected)
        if tolerance == .01:
            classical = result
        errors.append(float(np.linalg.norm(result.solution - reference)))
        nodes.append(selected.reconstruction.physical_branches)
        assert result.assess(absolute_tolerance=tolerance).status == "INCONCLUSIVE"
        assert selected.reconstruction.kernel_approximation_bound <= tolerance/2
        assert selected.reconstruction.quadrature_bound <= tolerance/2
        assert errors[-1] <= selected.reconstruction.kernel_approximation_bound + selected.reconstruction.quadrature_bound
    native = solve(problem, method=LCHS(), seed=7)
    # Independent execution of the same finite sum; numerical approximation
    # accuracy is checked against Cayley-Hamilton above, not inferred here.
    # A k-wire gate multiplies rows with 2**k complex entries. Sixteen real
    # operations per entry allow accumulated native arithmetic roundoff,
    # scaled by the actual physical recovery. This comparison window is
    # separate from the kernel bounds and is not a floating-point certificate.
    window = arithmetic_sizes[0] * np.finfo(float).eps * max(1., native.plan.reconstruction.recovery.as_float())
    assert window < 1e-6  # Keep the native/classical detector scientifically discriminating.
    np.testing.assert_allclose(native.solution, classical.solution, rtol=0, atol=window)
    assert widths == [9]
    assert nodes[0] == 204
    assert nodes[0] < nodes[1]
    assert errors[1] < errors[0]
    # This is a two-point empirical convergence witness, not monotonicity or
    # an epsilon guarantee for arbitrary inputs or all finer parameters.
