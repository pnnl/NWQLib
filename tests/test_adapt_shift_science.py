"""Bounded parameter-shift relations for the default GSD pool."""

from types import SimpleNamespace
import numpy as np
import pytest
from scipy.linalg import expm
from qiskit.quantum_info import SparsePauliOp
from nwqlib.algorithms.gcim import optimization
from nwqlib.execution import ExecutionMode
from nwqlib.subroutines.fermionic_pool import enumerate_spin_adapted_gsd_pool
from _fermionic_references import generator_matrix


@pytest.mark.parametrize("pool_index,force_insertion", ((0, False), (1, False), (1, True)))
def test_measured_product_jacobian_matches_independent_generator_derivative(
    monkeypatch, pool_index, force_insertion
):
    """Compare the optimizer Jacobian with 2 Re< H psi, A psi > and count every shift evaluation."""
    generator = enumerate_spin_adapted_gsd_pool(2)[pool_index]
    reference = np.ones(1, complex)
    for qubit in reversed(range(4)):
        angle = (0.2 + 0.3 * qubit) / 2
        reference = np.kron(reference, [np.cos(angle), np.sin(angle)])
    h = SparsePauliOp.from_list([("ZIII", 1.0), ("IXZX", 0.31)]).to_matrix()
    a = generator_matrix(generator)
    state = expm(0.37 * a) @ reference
    expected = 2 * float(np.vdot(h @ state, a @ state).real)
    rule = optimization._generator_shift_rule(
        generator, max_bytes=100_000_000, max_products=10_000_000
    )
    if pool_index == 1:
        shots = optimization._generator_shift_rule(generator, execution_mode=ExecutionMode.SHOTS)
        assert shots["method"] == "pauli_insertion_shift" and shots["evaluations"] == 16
        assert shots["signed_weight_square_sum"] == pytest.approx(0.25, abs=1e-15, rel=0.0)
        assert shots["uniform_allocation_variance_cost_proxy"] == pytest.approx(
            4.0, abs=1e-14, rel=0.0
        )
        assert rule["evaluations"] == 12
    if force_insertion:
        rule = optimization._generator_shift_rule(generator, execution_mode=ExecutionMode.SHOTS)
    calls = []
    captured = {}

    def evaluate(theta, *, insertion=None, role):
        vector = expm(theta[0] * a) @ reference
        if insertion is not None:
            _, term, sign = insertion
            p = SparsePauliOp.from_list([(generator.pauli_terms[term][0], 1.0)]).to_matrix()
            vector = (vector + sign * 1j * p @ vector) / np.sqrt(2)
        calls.append(role)
        return float(np.vdot(vector, h @ vector).real)

    def inspect(fun, x, *, jac, **kwargs):
        captured["jacobian"] = jac(x)
        return SimpleNamespace(message="independent derivative relation", success=True)

    monkeypatch.setattr(optimization.spo, "minimize", inspect)
    options = SimpleNamespace(
        optimize_every_m=1,
        optimize_rounds_n=1,
        optimize_max_evaluations=32,
    )
    optimization.optimize_product(
        options=options,
        execution="quantum",
        selected=(generator,),
        starting=(0.37,),
        evaluate=evaluate,
        completed_round=lambda: None,
        rules={generator.pool_index: rule},
    )
    assert captured["jacobian"][0] == pytest.approx(expected, abs=2e-11, rel=0.0)
    assert calls.count("objective") == 1 and calls.count("derivative") == rule["evaluations"]


# The gradient reference forms fixed-step Taylor matrices and their
# derivatives by exact rational ascending-power arithmetic, then
# differentiates the final Rayleigh quotient. The unit-scale regression uses
# absolute tolerance 2e-11 with zero relative tolerance. This is a small-case
# regression threshold, not a bound for arbitrary chains or a derivative of
# the step-count selector (docs/ENGINEERING_CONSTANTS.md). For fixed s = 1,
# F = P_m(tA) and F' = A P_(m-1)(tA), because the derivative of P_m(tA) is
# A P_(m-1)(tA) and all polynomials in A commute. Positive per-factor
# normalization cancels from the final Rayleigh quotient.
GRAD_ATOL = 2e-11


def _exact_fixture(pool="xz"):
    import sympy as sp

    x = sp.Matrix([[0, 1], [1, 0]])
    z = sp.diag(1, -1)
    y = sp.Matrix([[0, -sp.I], [sp.I, 0]])
    generators = [sp.I * (x if pool == "xz" else y), sp.I * z]
    hamiltonian = sp.Matrix([[sp.Rational(3, 4), sp.Rational(1, 4)],
                             [sp.Rational(1, 4), sp.Rational(-1, 2)]])
    return sp, generators, hamiltonian, sp.Matrix([1, 0])


def exact_product_reference(degree, angles, pool="xz"):
    """Energy and gradient of the fixed one-step Taylor product, in exact rational arithmetic."""
    sp, generators, hamiltonian, reference = _exact_fixture(pool)
    factors, derivatives = [], []
    for a, t in zip(generators, angles, strict=True):
        factor, derivative = sp.eye(2), sp.zeros(2)
        power = sp.eye(2)
        for r in range(1, degree + 1):
            power = power * a
            factor += power * t**r / sp.factorial(r)
            derivative += power * t**(r - 1) / sp.factorial(r - 1)
        factors.append(factor)
        derivatives.append(derivative)
    z = factors[1] * factors[0] * reference
    dz = [factors[1] * derivatives[0] * reference, derivatives[1] * factors[0] * reference]
    q = sp.expand((z.conjugate().T * z)[0])
    numerator = sp.expand((z.conjugate().T * hamiltonian * z)[0])
    gradient = []
    for direction in dz:
        dq = 2 * sp.re((direction.conjugate().T * z)[0])
        dn = 2 * sp.re((direction.conjugate().T * hamiltonian * z)[0])
        gradient.append(float((dn * q - numerator * dq) / q**2))
    return float(numerator / q), np.asarray(gradient)


def _production(degree, angles, steps):
    from nwqlib._numerics import normalize_state_vector

    sp, generators, hamiltonian, _ = _exact_fixture()
    dense = [np.array(a, dtype=complex) for a in generators]
    h = np.array(hamiltonian, dtype=complex)
    return optimization.normalized_taylor_energy_gradient(
        np.array([1, 0], complex),
        (0, 1)[: len(angles)],
        tuple(float(t) for t in angles),
        action=lambda g, v: dense[g] @ v,
        step_count=steps,
        normalize=normalize_state_vector,
        h0_action=lambda v: h @ v,
        degree=degree,
    )


@pytest.mark.parametrize("degree", (18, 1))
@pytest.mark.parametrize("angles", ((3, -5), (0, -5)))
def test_classical_taylor_product_gradient_matches_exact_rational_reference(degree, angles):
    """The production energy and gradient of iX then iZ against exact rational Taylor matrices.

    Degree 18 is the production degree with noncommuting generators, a
    nontrivial second angle and a zero first angle, whose selected step
    count is zero and whose limiting tangent is A times the input. Degree 1
    has a visibly nonunitary raw factor, so it checks the normalization
    pullback that degree 18 cannot resolve.
    """
    import sympy as sp
    from nwqlib.subroutines.fermionic_pool import _taylor_steps

    exact_angles = tuple(sp.Rational(a, 16) for a in angles)
    energy, gradient = exact_product_reference(degree, exact_angles)
    # One step per nonzero angle, as in the reference; the zero angle takes
    # the owner's zero-step branch.
    value, derivative = _production(
        degree, exact_angles, lambda g, t: 1 if _taylor_steps((1,), t) else 0
    )
    assert value == pytest.approx(energy, abs=GRAD_ATOL, rel=0)
    assert derivative == pytest.approx(gradient, abs=GRAD_ATOL, rel=0)


def test_classical_taylor_product_gradient_empty_chain_and_underflow_branch():
    """The empty chain gives the reference energy; a nonzero angle with zero steps has zero derivative.

    With zero selected steps the first factor is the identity, so the energy
    and the second derivative are those of the second factor alone.
    """
    energy, gradient = _production(18, (), lambda g, t: 1)
    assert energy == 0.75 and gradient.shape == (0,)
    energy, gradient = _production(18, (1e-300, -5 / 16), lambda g, t: 0 if abs(t) < 1e-200 else 1)
    import sympy as sp

    alone, derivative = exact_product_reference(18, (0, sp.Rational(-5, 16)))
    assert gradient[0] == 0.0
    assert energy == pytest.approx(alone, abs=GRAD_ATOL, rel=0)
    assert gradient[1] == pytest.approx(derivative[1], abs=GRAD_ATOL, rel=0)


@pytest.mark.parametrize("pool", ("xz", "yz"))
def test_classical_adapt_energy_query_gradient_matches_exact_rational_reference(pool):
    """One classical ADAPT energy query of iX (or iY) then iZ returns the exact reference energy and gradient.

    The query runs through the Plan's host kernel, which selects each
    factor's step count, orders the chain and names ``gradient_j``. The
    Hamiltonian ``I/8 + X/4 + 5Z/8`` is the reference matrix
    ``[[3/4, 1/4], [1/4, -1/2]]``, the reference state is ``|0>``, and each
    angle takes one Taylor step, as in the reference. With iX and iZ this
    real H and reference make the energy even in theta, so the iY pool also
    detects an objective built with exp(-theta A).
    """
    import sympy as sp
    from test_adapt_primary import plan_for
    from nwqlib.algorithms.gcim import adapt_acquisition
    from nwqlib.operators import ingest_pauli
    from nwqlib.subroutines.fermionic_pool import _taylor_steps
    from nwqlib._prepared_execution import Run, prepare_experiment, submit_experiment

    angles = (sp.Rational(3, 16), sp.Rational(-5, 16))
    right = tuple((index, float(angle)) for index, angle in enumerate(angles))
    assert [_taylor_steps((1j,), theta) for _, theta in right] == [1, 1]
    plan = plan_for(
        execution="classical",
        A=ingest_pauli((("I", 0.125), ("X", 0.25), ("Z", 0.625)), num_qubits=1),
        initial_state=(1, 0),
        pool_rows=(((pool[0].upper(), 1j),), (("Z", 1j),)),
    )
    point = adapt_acquisition._point(plan, "energy", right=right)
    with Run(plan) as run:
        values = submit_experiment(prepare_experiment(point, run=run), run=run).values
    scalars = {value.label: value.value for value in values}
    energy, gradient = exact_product_reference(18, angles, pool)
    assert scalars["energy"] == pytest.approx(energy, abs=GRAD_ATOL, rel=0)
    assert [scalars["gradient_0"], scalars["gradient_1"]] == pytest.approx(gradient, abs=GRAD_ATOL, rel=0)
