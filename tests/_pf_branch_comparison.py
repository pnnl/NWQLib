"""Test-only bounds for the two Qiskit evaluations of one elementary PF circuit."""
from collections import Counter
from fractions import Fraction
import math

import numpy as np
from qiskit.circuit.library import (
    HGate, SXGate, SXdgGate, CXGate, RXGate, RYGate, RZGate,
    RXXGate, RYYGate, RZZGate, RZXGate,
)

_ARITY = {HGate: 1, SXGate: 1, SXdgGate: 1, CXGate: 2,
          RXGate: 1, RYGate: 1, RZGate: 1, RXXGate: 2,
          RYYGate: 2, RZZGate: 2, RZXGate: 2}


def _up_fraction(value):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError("bound is not finite")
    if Fraction(result) < value:
        result = math.nextafter(result, math.inf)
    return result


def _add(a, b):
    if not a:
        return b
    if not b:
        return a
    return math.nextafter(a + b, math.inf)


def _mul(a, b):
    if not a or not b:
        return 0.0
    return math.nextafter(a * b, math.inf)


def _sqrt_up(value):
    value = Fraction(value)
    if not value:
        return 0.0
    exponent = value.numerator.bit_length()-value.denominator.bit_length()
    if Fraction(2)**exponent > value:
        exponent -= 1
    exponent //= 2
    scaled = value / Fraction(2)**(2*exponent)
    result = math.ldexp(math.sqrt(float(scaled)), exponent)
    while Fraction(result) ** 2 < value:
        result = math.nextafter(result, math.inf)
    return result


def _dot_constants(n):
    # g_n = 2*gamma_(2n), eta_n = 8n*2^-1074/(1-2n*u).
    denominator = (1 << 53) - 2*n
    if denominator <= 0:
        raise ValueError("contraction is too long for the rounding model")
    return (_up_fraction(Fraction(4*n, denominator)),
            _up_fraction(Fraction(8*n, 1 << 1074) * Fraction(1 << 53, denominator)))


def _matrix_bounds(matrix):
    """Upper bounds on ||M||_2 and ||abs(M)||_2 from exact represented entries."""
    matrix = np.asarray(matrix, dtype=np.complex128)
    if not np.isfinite(matrix).all():
        raise ValueError("gate matrix must be finite")
    r = len(matrix)
    z = [[(Fraction(float(x.real)), Fraction(float(x.imag))) for x in row]
         for row in matrix]
    # Componentwise l1 absolute values bound complex magnitudes.
    weights = [[abs(a)+abs(b) for a, b in row] for row in z]
    b = max(max(map(sum, weights)), max(sum(weights[i][j] for i in range(r)) for j in range(r)))
    rho = Fraction(0)
    for i in range(r):
        row = Fraction(0)
        for j in range(r):
            real = sum(z[k][i][0]*z[k][j][0] + z[k][i][1]*z[k][j][1] for k in range(r))
            imag = sum(z[k][i][0]*z[k][j][1] - z[k][i][1]*z[k][j][0] for k in range(r))
            row += abs(real - int(i == j)) + abs(imag)
        rho = max(rho, row)
    # ||M||^2 <= 1+rho and sqrt(1+rho) <= 1+rho/2.
    return _up_fraction(1+rho/2), _up_fraction(b)


def selected_pf_comparison_tolerance(circuit, vector):
    """Return a conditional 2-norm bound for Statevector versus Operator @ v.

    Pass the same once-decomposed, bound circuit to both evaluators.
    Only the listed standard gates and one top-level phase are supported.
    Matrices and the phase scalar must be deterministic complex128 values
    shared by both routes. Contractions must use ordinary binary64 complex
    arithmetic, round to nearest with gradual underflow, without overflow,
    with at most 2n rounding factors along any real n-term dot-product path.
    This is a test-only comparison of represented gate matrices, not a
    certificate of transcendental accuracy or of the physical PF error.
    """
    d = 1 << circuit.num_qubits
    v = np.asarray(vector)
    if v.shape != (d,) or v.dtype != np.dtype(np.complex128) or not np.isfinite(v).all():
        raise ValueError("comparison requires a finite complex128 D-vector")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("comparison requires a bound unitary circuit")
    norm_squared = sum(Fraction(float(x.real))**2 + Fraction(float(x.imag))**2 for x in v)
    norm = _sqrt_up(norm_squared)
    sqrt_d = _sqrt_up(d)
    product_a, f, product_plus, h = 1.0, 0.0, 1.0, 0.0
    kinds = Counter()
    for instruction in circuit.data:
        gate = instruction.operation
        arity = _ARITY.get(gate.base_class)
        if arity is None or gate.num_qubits != arity or instruction.clbits:
            raise ValueError("gate needs a comparison law for its actual evaluation")
        matrix = np.asarray(gate.to_matrix(), dtype=np.complex128)
        a, b = _matrix_bounds(matrix)
        g, eta = _dot_constants(1 << arity)
        e = _mul(g, b)
        a_plus_e = _add(a, e)
        f = _add(_mul(a_plus_e, f), _mul(e, product_a))
        h = _add(_mul(a_plus_e, h), eta)
        product_a = _mul(a, product_a)
        product_plus = _mul(a_plus_e, product_plus)
        kinds[gate.name] += 1
    a_phase, e_phase, eta_phase = 1.0, 0.0, 0.0
    if circuit.global_phase != 0:
        z = complex(np.exp(1j * float(circuit.global_phase)))
        a_phase, _ = _matrix_bounds(np.array([[z]], dtype=np.complex128))
        # The Operator phase composes a full D-square scalar operator.
        # A D-term allowance bounds it and the vector's scalar multiply.
        g, eta_phase = _dot_constants(d)
        e_phase = _mul(g, a_phase)
    gate_f = f
    phase_error = _mul(_mul(_mul(_add(1.0, sqrt_d), e_phase), product_plus), norm)
    f = _add(_mul(a_phase, f), _mul(e_phase, product_plus))
    h = _add(h, _mul(eta_phase, product_plus))
    product_plus = _mul(_add(a_phase, e_phase), product_plus)
    g_d, eta_d = _dot_constants(d)
    action_error = _mul(_mul(_add(1.0, sqrt_d), f), norm)
    matvec_error = _mul(_mul(g_d, _mul(sqrt_d, product_plus)), norm)
    underflow = _add(
        _mul(_add(sqrt_d, _mul(_mul(float(d), _add(1.0, g_d)), norm)), h),
        _mul(sqrt_d, eta_d))
    tolerance = _add(_add(action_error, matvec_error), underflow)
    if not math.isfinite(tolerance):
        raise ValueError("comparison bound is not finite")
    return dict(atol_2=tolerance, gate_f=gate_f, E_phase=phase_error,
                underflow=underflow, dimension=d, norm_upper=norm,
                gate_count=len(circuit.data), gate_kinds=dict(kinds))


def assert_represented_vector_distance(left, right, tolerance):
    """Compare the exact squared distance of finite complex128 result arrays."""
    left, right = np.asarray(left), np.asarray(right)
    assert left.shape == right.shape
    assert np.isfinite(left).all() and np.isfinite(right).all()
    squared = sum(
        (Fraction(float(a.real))-Fraction(float(b.real)))**2
        + (Fraction(float(a.imag))-Fraction(float(b.imag)))**2
        for a, b in zip(left.flat, right.flat, strict=True))
    assert squared <= Fraction(float(tolerance))**2


def regression_atol(circuit, vector):
    """Absolute allowance for the named <=4-qubit, <=5000-gate unit-vector fixtures.

    The 2e-12 threshold is an empirical regression contract. It carries no
    error guarantee for other widths, depths, inputs or numerical stacks.
    """
    if circuit.num_qubits > 4 or len(circuit.data) > 5000:
        raise ValueError("fixture exceeds the qualified regression scope")
    if circuit.num_clbits or circuit.parameters:
        raise ValueError("fixture must be a bound unitary circuit")
    if any(i.operation.base_class not in _ARITY or i.clbits for i in circuit.data):
        raise ValueError("fixture contains an unqualified gate kind")
    v = np.asarray(vector, dtype=np.complex128)
    if v.shape != (1 << circuit.num_qubits,) or not np.isfinite(v).all():
        raise ValueError("fixture needs a finite D-vector")
    norm_squared = sum(Fraction(float(x.real))**2 + Fraction(float(x.imag))**2 for x in v)
    if norm_squared > Fraction(1+64*np.finfo(float).eps)**2:
        raise ValueError("fixture norm exceeds one")
    return 2e-12
