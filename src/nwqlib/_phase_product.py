"""Error bound (envelope) of a classical phase correction of a saved statevector, and the error budgets it propagates.

``backends/qiskit_aer.py`` and ``backends/nwqsim.py`` (the factor's envelope
at preparation) and ``PreparedArtifact.saved_state_error`` and
``PreparedArtifact.saved_state_probability_window`` in ``execution.py`` (the
corrected budgets) share the scalar functions of this module.

Aer supplies its stored Python factor. NWQ-Sim supplies the factor reported
by its selected runner during preparation and transmitted in the execution
request. The same finite envelope applies to either represented factor.
Multiplication by ``(1.0, ±0.0)`` has a zero envelope under the assumptions
on classical arithmetic below. The simulator's first-order allowance remains
part of the resulting saved-state estimate.

Finite bound for the complex multiplication as performed. Let z be the binary64
complex vector before correction, with D entries. Let ``c = a + ib`` be the
finite binary64 factor that the code actually multiplies into z. Work with the
exact represented real values a, b, ``u = 2**-53`` and ``eta = 2**-1074``.
Assume IEEE round-to-nearest with gradual underflow, and no overflow in the
component products and sums. This is the assumption on classical (host)
arithmetic, separate from the simulator's own error model. Put::

    gamma_2 = 2u/(1 - 2u),
    kappa = |a| + |b|,
    s = a**2 + b**2,
    t = |s - 1| + gamma_2 kappa,
    e = (3/2 + u) eta ceil(sqrt(2D)).

For one real component of the complex product, two rounded products and one
rounded sum give error at most ``gamma_2 (|ax| + |by|) + (3/2 + u) eta``. The
imaginary component has the corresponding swapped terms. The nonnegative 2x2
coefficient matrix has spectral norm kappa. Combining the 2D component errors
gives ``||fl(cz) - cz||_2 <= gamma_2 kappa ||z||_2 + e``. A fused
implementation is no worse under the same assumptions. If c is nonzero, compare
cz with multiplication by ``c/|c|``. Its modulus error is
``||c| - 1| <= |s - 1|``. If c is zero, the same inequality holds with any unit
phase. Thus ``distance_mod_phase(fl(cz), z) <= t ||z||_2 + e``. This uses the
actual factor's squared modulus. It needs no one-ulp assumption about
``np.exp``, no argument-reduction bound and no estimate of accumulated
phase-angle error. An incorrect phase is still a unit phase for this metric.
It matters for phase-defined outputs, whose budget remains unavailable.

The functions use exact rational arithmetic for the few scalar coefficients
and round each reported nonnegative bound upward. There is no scan or copy of
z. A nonfinite factor or an unrepresentable envelope does not support a finite
claim. The assumption of a finite input vector remains necessary, and the
existing output checks still apply. These scalar functions add no further
validation pass over the state norm.

Source: NWQLib's derivation of the phase convention of the reduction context
and of the classical correction of a saved statevector, stated in full above.
"""

from fractions import Fraction as F
from math import inf, isqrt, nextafter
from sys import float_info

U = F(1, 2**53)
ETA = F(1, 2**1074)


def up_float(value):
    """The smallest binary64 value at or above the exact rational ``value``. It raises ValueError for a nonfinite result."""
    value = F(value)
    if abs(value) > F(float_info.max):
        raise ValueError("phase-product envelope is not finite")
    result = float(value)
    return nextafter(result, inf) if F(result) < value else result


def phase_product_envelope(factor, dimension):
    """Return ``(t, e)`` for multiplying a ``dimension``-entry vector by the binary64 complex ``factor``.

    ``t = |a**2 + b**2 - 1| + gamma_2 (|a| + |b|)`` and
    ``e = (3/2 + u) eta ceil(sqrt(2D))`` (module docstring), each rounded
    upward, so that ``distance_mod_phase(fl(cz), z) <= t ||z||_2 + e``.
    """
    a, b = F(float(factor.real)), F(float(factor.imag))
    gamma2 = 2*U/(1-2*U)
    relative = abs(a*a+b*b-1) + gamma2*(abs(a)+abs(b))
    root = isqrt(2*dimension)
    if root*root < 2*dimension:
        root += 1
    absolute = (F(3, 2)+U)*ETA*root
    return up_float(relative), up_float(absolute)


def corrected_state_error(delta, envelope):
    """Return ``delta_saved = delta + t (1 + delta) + e``, rounded upward, or None when delta is None.

    Given an existing qualified simulator estimate delta and its associated
    assumption ``||z|| <= 1 + delta``, this is the corrected estimate after the
    multiplication whose envelope is ``(t, e)``. The propagation through the
    classical correction is finite and includes higher-order and underflow
    terms. The overall statement is still conditional on the simulator's
    first-order error model. It is not an unconditional bound on the
    simulator's execution.
    """
    if delta is None:
        return None
    t, e = map(F, envelope)
    d = F(delta)
    return up_float(d+t*(1+d)+e)


def corrected_mass_window(window, envelope):
    """Return ``omega_saved``, rounded upward, for the norm window ``window`` after the multiplication.

    With omega the existing nonnegative norm-window convention for z, from
    ``| ||z||**2 - 1 | <= omega`` and the classical-arithmetic inequality,
    ``omega_saved = omega + (2t + t**2)(1 + omega) + 2(1 + t)(1 + omega) e + e**2``.
    Here ``sqrt(1 + omega) <= 1 + omega`` avoids another rounded square root.
    This propagates the existing validation convention through the classical
    correction. It does not turn that convention into a certified accuracy
    bound for the simulator.
    """
    t, e = map(F, envelope)
    w = F(window)
    return up_float(w+(2*t+t*t)*(1+w)+2*(1+t)*(1+w)*e+e*e)
