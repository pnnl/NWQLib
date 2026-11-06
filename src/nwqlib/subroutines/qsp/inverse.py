"""Relative minimax ``1/x`` polynomial for QSVT matrix inversion.

QLS needs an odd polynomial ``P`` with ``|kappa x P(x) - 1| <= epsilon_inv``
on the domain ``D(a) = {x : a <= |x| <= 1}``, ``a = 1/kappa``.

The domain parameter ``kappa`` must exceed 1. QLS passes its
``polynomial_kappa``, the encoding's ``kappa_be`` raised to at least
``QLS_POLYNOMIAL_KAPPA_FLOOR = 1.01`` so that a perfectly conditioned A
still gives a domain with ``a < 1``. ``kappa_be`` is
``max(1, alpha / sigma_min(A))`` for ``kappa="auto"`` and otherwise the
supplied ``kappa``, which must cover that value. With ``kappa="auto"``, the
analytic periodic encoding passes ``alpha / min(sigma_min, encoded gap)``,
rounded outward (QLS periodic encoding). Every ``kappa`` below is
that domain parameter.

The criterion is a relative residual. For each eigenvalue (Hermitian
``A``) or singular value (general ``A``, through its odd singular-value
transform) ``s`` of ``A/alpha`` in ``D``,
``|P(s) - 1/(kappa s)| <= epsilon_inv / (kappa |s|)``. Summing over
components, the transformed vector ``y`` of ``b`` satisfies
``||y - (kappa A/alpha)^{-1} b|| <= epsilon_inv ||(kappa A/alpha)^{-1} b||``.
Childs, Kothari, and Somma [CKS], arXiv:1511.02306v2, and Sunderhauf,
Nemeth, Walayat, Patterson, and Berntson [SNWPB], arXiv:2507.15537v1,
instead bound the absolute error ``|p(x) - 1/x|``.

Construction. An odd ``P`` of degree at most ``2m - 1`` is ``x q(x^2)``
with ``deg q <= m - 1``, so ``R(y) = 1 - kappa y q(y)``, ``y = x^2``, has
degree at most ``m`` and ``R(0) = 1``; conversely every such ``R`` gives
``P(x) = (1 - R(x^2)) / (kappa x)``. The map
``t(y) = (1 + a^2 - 2y) / (1 - a^2)`` sends ``[a^2, 1]`` onto ``[-1, 1]``
and ``y = 0`` to ``cosh(beta)``, ``beta = 2 atanh(a)``. The residual
``R_m(y) = T_m(t(y)) / cosh(m beta)`` alternates ``m + 1`` times with
magnitude ``delta_m = sech(m beta)`` on ``[a^2, 1]``. A degree-``m``
residual with ``R(0) = 1`` and a smaller maximum would differ from ``R_m``
by a polynomial with a zero between each pair of successive extrema and
another at ``y = 0``, which is impossible at degree ``m``. Hence
``delta_m`` is the smallest relative residual of any odd polynomial of
degree ``2m - 1``, and since it decreases strictly with ``m`` the smallest
odd degree meeting ``epsilon_inv`` in exact arithmetic is ``2m - 1`` with
``m = max(1, ceil(acosh(1/epsilon_inv) / beta))``. The ceiling of a
rounded quotient can land on either side of an integer, so the neighbors
of this ``m`` are compared by ``delta_m`` itself. This choice minimizes the
relative residual, not the QSP rescale, the phase synthesis or the
success probability.

Coefficients. ``P`` has degree ``d = 2m - 1``, so its values at the
``d + 1`` Chebyshev zeros determine its Chebyshev coefficients by the
DCT-II divided by ``d + 1`` in exact arithmetic. Odd parity sets every
even coefficient to zero.

Check of the stored table. For the stored coefficients ``p``,
``E(x) = kappa x p(x) - (1 - R_m(x^2))`` is a polynomial of degree
``d + 1`` on all of ``[-1, 1]``. Its values at ``M = 4(d + 2)`` Chebyshev
zeros and the Ehlich-Zeller factor ``1/cos(pi (d + 1)/(2M))``
(doi:10.1007/BF01111276, Satz 2) bound ``max |E|`` in exact arithmetic,
and on ``D(a)`` ``|kappa x p(x) - 1| <= delta_m + max |E|``. Evaluated in
binary64, the same formula gives the formation term ``F``, an estimate of
``max |E|`` that can fall below it (0.79 times a 40-digit evaluation at
``kappa = 1.01`` and degree 13). ``F`` is rounding noise that grows with
``kappa`` and differs from one degree to the next. The check value of a
table is ``delta_m + 2F + R``, computed in binary64 as
``(delta_m + F) + (F + R)``, where ``R = 4 eps b`` is the evaluation
resolution, with ``eps`` the binary64 machine epsilon and ``b`` the bit
length of ``M``. The second ``F`` and ``R`` allow for the error of the
estimate. A table is used only when its check value is at most
``epsilon_inv``. When the check value exceeds ``epsilon_inv``, one
refinement takes the smallest ``m`` above the first with
``delta_m <= epsilon_inv - 8F - R``, which leaves room for a refined
formation term up to four times the first, and checks it the same way. When
that right side is not positive, or the refined table also misses, the
request is refused as unresolved. Raising the degree lowers ``delta_m`` but
not ``F + R``, so the refusal means that this binary64 check cannot decide
the table for ``epsilon_inv``, not that ``epsilon_inv`` is unattainable. At
``kappa = 1.01`` the degree-13 table has check value 6.9e-15, and an exact
rational bound limits its residual to 1.03e-15, so ``epsilon_inv = 3e-15``
is refused although the table meets it. The check value has no arithmetic
enclosure, so it is a numerical check of the table and not a bound on its
residual.

The work and storage are ``O(d log d)`` and ``O(d)``.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.fft import dct

from nwqlib._validation import finite_real
from nwqlib.operators.access import _check_bytes
from nwqlib.subroutines.qsp.phases import _values_at_zeros, chebyshev_grid, chebyshev_norming_sup_bound


# Evaluation resolution of the stored-table check, in machine epsilons per
# bit of the check length M: an untuned allowance for the rounding of the
# DCT-III samples and the filter values, which have no enclosure. Registered
# in docs/ENGINEERING_CONSTANTS.md. Revisit with a different evaluation
# method or an enclosure of the check.
_CHECK_RESOLUTION_EPS_PER_BIT = 4

# The refinement reserves this multiple of the first table's formation term
# for the refined table's. Formation terms are rounding noise: over windows of
# 10% in m they varied by factors up to 3.07 for kappa <= 3000, and refined
# tables reached 2.2 times the first in a 25 x 300 sweep of kappa <= 3000
# and epsilon_inv >= 1e-14. A reserve of 1 lets the refinement at kappa = 100,
# epsilon_inv = 1e-12 miss. Registered in docs/ENGINEERING_CONSTANTS.md.
# Revisit with another evaluation of the table or a sweep that exceeds it.
_REFINEMENT_FORMATION_RESERVE = 4


@dataclass(frozen=True, kw_only=True)
class InverseChebyshevPolynomial:
    """Odd Chebyshev table of the relative minimax approximation to `1/(kappa x)` on `a <= |x| <= 1`, `a = 1/kappa`.

    The QLS Method computes it while it chooses its polynomial. The fields
    are read-only.

    Attributes:
        coefficients: Chebyshev coefficients `c_0, ..., c_d` of `P`. Even
            entries are exactly zero.
        degree: Odd degree `d` of the table.
        ideal_degree: Smallest odd degree whose minimax relative residual
            `sech(m beta)` meets `epsilon_inv`, decided by binary64
            evaluation of that residual. When `epsilon_inv` lies within a few
            rounding units of a residual, this can be one degree pair above
            the exact minimum. It is below `degree` when the numerical check
            of the stored table needed its refinement.
        kappa: Domain parameter, greater than 1.
        epsilon_inv: Requested bound on `|kappa x P(x) - 1|` over the domain.
        minimax_error: `sech(m beta)` with `m = (degree + 1)/2` and
            `beta = 2 atanh(1/kappa)`, the relative residual of the exact
            minimax polynomial of this degree.
        residual_check_value: The value compared with `epsilon_inv`,
            `minimax_error + 2F + R`, computed in binary64 as
            `(minimax_error + F) + (F + R)`. The table is used only when it
            is at most `epsilon_inv`. `F` is the binary64 evaluation of the
            Chebyshev-zero bound on the largest difference between
            `kappa x P(x)` for the stored coefficients and for the exact
            minimax polynomial, and `R` is the evaluation resolution
            `4 eps b` of the module docstring. The second `F` and `R` allow
            for the error of that evaluation. The value has no arithmetic
            enclosure. In the cases checked it exceeds
            `max |kappa x P(x) - 1|` over the domain, but no proof shows that
            it always does.
    """

    coefficients: tuple[float, ...]
    degree: int
    ideal_degree: int
    kappa: float
    epsilon_inv: float
    minimax_error: float
    residual_check_value: float


def _minimax_error(m, beta):
    """``sech(m beta)``, through ``exp(-m beta)`` so that a large argument underflows to 0."""
    e = math.exp(-m * beta)
    return 2 * e / (1 + e * e)


def _filter_complement(x, a, m):
    """Return ``G = 1 - T_m(t(x^2)) / cosh(B)``, ``B = 2m atanh(a)``, at points ``|x| <= 1`` without cancellation.

    With ``t = |x|`` and ``e = exp(-B)``: for ``t >= a``, ``T_m(t(x^2)) =
    cos(m theta)`` with ``theta/2 = atan2(sqrt((t-a)(t+a)), sqrt((1-t)(1+t)))``
    and ``G = ((1-e)^2 + 4e sin^2(m theta/2)) / (1 + e^2)``. For ``t < a``,
    ``T_m(t(x^2)) = cosh(B - h)`` with ``u = sqrt((1-t)(1+t))``,
    ``v = sqrt((a-t)(a+t))`` and
    ``h = 2m log1p(t^2 (1/(1+u) + 1/(a+v)) / (u+v))``, since
    ``exp(h/(2m)) = (1+a)/(u+v)``; then
    ``G = (1 - exp(-h)) (1 - exp(-(2B-h))) / (1 + e^2)``. Near ``x = 0``,
    ``G = (m/a) tanh(B) x^2 + O(x^4)``, which this form keeps to relative
    accuracy.
    """
    t = np.abs(np.asarray(x, float))
    b = 2 * m * math.atanh(a)
    e = math.exp(-b)
    values = np.empty_like(t)
    gap = t < a
    s = t[gap]
    u = np.sqrt((1 - s) * (1 + s))
    v = np.sqrt((a - s) * (a + s))
    h = 2 * m * np.log1p(s * s * (1 / (1 + u) + 1 / (a + v)) / (u + v))
    values[gap] = np.expm1(-h) * np.expm1(h - 2 * b) / (1 + e * e)
    s = t[~gap]
    half = np.arctan2(np.sqrt((s - a) * (s + a)), np.sqrt((1 - s) * (1 + s)))
    values[~gap] = (math.expm1(-b) ** 2 + 4 * e * np.sin(m * half) ** 2) / (1 + e * e)
    return values


def _half_degree(beta, epsilon):
    """Smallest ``m >= 1`` with ``sech(m beta) <= epsilon``, from the ceiling formula and its neighbors."""
    # acosh(1/epsilon) without forming 1/epsilon.
    quotient = (math.log1p(math.sqrt((1 - epsilon) * (1 + epsilon))) - math.log(epsilon)) / beta
    if not math.isfinite(quotient):
        raise ValueError(f"the inverse polynomial degree for epsilon_inv={epsilon:.6g} is not representable")
    m = max(1, math.ceil(quotient))
    if m > 1 and _minimax_error(m - 1, beta) <= epsilon:
        return m - 1
    return m if _minimax_error(m, beta) <= epsilon else m + 1


def _table(kappa, m, max_bytes):
    """Return the DCT coefficients of degree ``2m - 1`` and their formation bound."""
    a, d = 1 / kappa, 2 * m - 1
    checks = 4 * (d + 2)
    _check_bytes(8 * (2 * (d + 1) + 8 * checks), max_bytes, "QLS inverse polynomial")
    x = chebyshev_grid(d + 1)
    coefficients = dct(_filter_complement(x, a, m) / (kappa * x), type=2) / (d + 1)
    # Odd parity: the even coefficients, including c_0 and its half weight,
    # are zero.
    coefficients[::2] = 0.0
    y = chebyshev_grid(checks)
    sampled = np.max(abs(kappa * y * _values_at_zeros(coefficients, checks) - _filter_complement(y, a, m)))
    formation = float(chebyshev_norming_sup_bound(float(sampled), degree=d + 1, num_points=checks))
    resolution = _CHECK_RESOLUTION_EPS_PER_BIT * float(np.finfo(float).eps) * checks.bit_length()
    return coefficients, formation, resolution


def _inverse_chebyshev(kappa, epsilon_inv, *, max_bytes=None) -> InverseChebyshevPolynomial:
    """Return the relative minimax inverse table for ``kappa > 1`` and ``0 < epsilon_inv < 1``.

    See the module docstring for the construction, the numerical check of
    the stored table and its single refinement. ``max_bytes=None`` checks
    the ``O(d)`` arrays against the physical memory. Raises ValueError for
    invalid inputs, an unrepresentable degree, or a check that the
    refinement cannot resolve.
    """
    kappa = finite_real(kappa, "kappa")
    epsilon = finite_real(epsilon_inv, "epsilon_inv")
    if kappa <= 1:
        raise ValueError("kappa must exceed 1")
    if not 0 < epsilon < 1:
        raise ValueError("epsilon_inv must lie in (0, 1)")
    beta = 2 * math.atanh(1 / kappa)
    ideal = m = _half_degree(beta, epsilon)
    for refined in (False, True):
        coefficients, formation, resolution = _table(kappa, m, max_bytes)
        delta = _minimax_error(m, beta)
        value = (delta + formation) + (formation + resolution)
        if value <= epsilon:
            return InverseChebyshevPolynomial(
                coefficients=tuple(float(c) for c in coefficients), degree=2 * m - 1,
                ideal_degree=2 * ideal - 1, kappa=kappa, epsilon_inv=epsilon,
                minimax_error=delta, residual_check_value=value)
        target = epsilon - (2 * _REFINEMENT_FORMATION_RESERVE * formation + resolution)
        if refined or target <= 0:
            break
        m = max(m + 1, _half_degree(beta, target))
    step = ("after one refinement" if refined else
            f"and no refinement fits, since epsilon_inv - {2 * _REFINEMENT_FORMATION_RESERVE} formation - "
            f"resolution = {target!r} is not positive")
    raise ValueError(
        f"the binary64 numerical check of the QLS inverse table of degree {2 * m - 1} for kappa={kappa!r} and "
        f"epsilon_inv={epsilon!r} is unresolved {step}. Its check value (minimax + formation) + (formation + "
        f"resolution) = ({delta!r} + {formation!r}) + ({formation!r} + {resolution!r}) = {value!r} exceeds "
        f"epsilon_inv. This O(d log d) check cannot decide the table, and the refusal does not show that "
        f"epsilon_inv is unattainable.")


__all__ = ["InverseChebyshevPolynomial"]
