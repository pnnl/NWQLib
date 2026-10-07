"""QHD coefficient schedules, their interval integrals and the per-step coefficient rule.

A schedule gives the weights of ``H(t) = a(t) K + b(t) V``, where K is the
kinetic operator ``-Delta/2`` and V the objective. They are ``exp(phi_t)``
and ``exp(chi_t)`` of Eq. (1) in Leng, Hickman, Li and Wu, arXiv:2303.01471v1.
Each record carries only its own parameter, and its ``kind`` names the
formula, so two schedules with equal parameters keep distinct identities.
``step_weights`` turns a record into the per-step weights of the Method's
coefficient rule, from the point values or from the interval integrals.

Each record gives the point values ``a(t)`` and ``b(t)`` and the interval
integrals ``A(t0, t1)`` and ``B(t0, t1)`` of a and b. The closed forms
subtract nearly equal numbers when ``t1 - t0`` is small relative to t0, so
the integrals use factored positive expressions instead (derived at each
function below). The computed value Î of an integral I satisfies
``|Î - I| <= C u I + 2 tau`` to first order in ``u = 2**-53``, with
``tau = 2**-1074`` the smallest subnormal. The term ``2 tau`` covers the
rounding of the positive parts and of their sum when I is subnormal, where
no binary64 result can have a small relative error, and below ``tau/2``
zero is the correctly rounded result. On ``gamma, s`` in ``[1e-8, 1e8]``
and ``0 <= t <= 1e4`` the constant C, derived in the docstring of each
integral, is 20 for quadratic A with ``gamma > 0`` (32 over the whole
finite binary64 range), 1 for quadratic A and B with ``gamma = 0``, 16 for
quadratic B, 48 for cubic A, and 16 for cubic B and both shifted-cubic
integrals. The bounds assume round-to-nearest binary64 with gradual
underflow, correctly rounded basic operations and ``sqrt``, at most one ulp
of error in ``atan``, ``atan2`` and ``cbrt``, and a relative error of at most
2u for ``fsum`` of nonnegative terms. Python's ``math`` module takes these
from the platform C library, so they are platform assumptions.

The endpoints are the given binary64 values, read as exact reals, and must
satisfy ``0 <= t0 < t1``. A request such as ``t0 + 1e-300`` that rounds back
to t0 does not describe a positive interval and is rejected.
"""

from __future__ import annotations

from fractions import Fraction
from math import atan, atan2, cbrt, frexp, fsum, inf, isfinite, isinf, ldexp, log2, sqrt
from typing import Annotated, ClassVar, Literal

from pydantic import Field

from nwqlib._validation import finite_real
from nwqlib.core.records import Nonnegative, Real, Record

from .validation import _normal_quotient, _normal_range

Positive = Annotated[Real, Field(gt=0)]

# atan(z)/z = 1 - z**2/3 + ..., so for z <= 2**-27 using 1 instead has
# relative error at most z**2/3 <= u/6. Dividing a subnormal atan(z) by z
# could also amplify its one-ulp error. Exact bound, not tuned. Registered in
# docs/ENGINEERING_CONSTANTS.md. Revisit only with another floating-point format.
_ATANC_ONE_BELOW = 2.0**-27

# Positive nodes x_i and weights w_i of the 16-point Gauss-Legendre rule on
# [-1, 1]. The rule is symmetric, so -x_i has the same weight, and it is exact
# through degree 31, which the cubic-A error bound (_unit_mean) uses. Each
# value is within 0.46 ulp of the 50-digit value of
# mpmath.gauss_quadrature(16, "legendre") (mpmath 1.3.0), that is, correctly
# rounded to binary64.
_GAUSS_LEGENDRE_16 = (
    (0.09501250983763744, 0.18945061045506850),
    (0.28160355077925891, 0.18260341504492359),
    (0.45801677765722739, 0.16915651939500254),
    (0.61787624440264375, 0.14959598881657673),
    (0.75540440835500303, 0.12462897125553388),
    (0.86563120238783174, 0.09515851168249279),
    (0.94457502307323258, 0.062253523938647894),
    (0.98940093499164993, 0.027152459411754095),
)
# Nodes q and weights w/8 of the same rule on four equal panels of [0, 1], so
# sum(w/8 * g(q)) approximates the mean of g on [0, 1]. Four panels keep each
# panel's half-width at 1/8, which the cubic-A bound (_unit_mean) needs.
# Registered in docs/ENGINEERING_CONSTANTS.md. Revisit if the derived bound or
# the evaluation cost changes.
_UNIT_MEAN_NODES = tuple(
    ((panel + (1.0 + sign * node) * 0.5) * 0.25, weight * 0.125)
    for panel in range(4)
    for node, weight in _GAUSS_LEGENDRE_16
    for sign in (-1.0, 1.0)
)


def _admitted(value, name, remedy, exact=None, *, zero=False):
    """Return a schedule value after ``validation._normal_range`` admits it, or raise its refusal with ``remedy()`` appended.

    ``remedy`` is called only for a refusal. On t >= 0 each kinetic weight
    is nonincreasing and each potential weight is nondecreasing. The
    quadratic weights are constant when gamma = 0. The two cubic kinetic
    weights strictly decrease with s. Their lower-range refusals can
    require reducing both s and total_time, because either denominator
    contribution may already be too large by itself. Increasing s reduces
    an overflowing cubic kinetic weight. For midpoint scheduling, fewer
    steps or more total_time increase the first midpoint and its cubic
    potential weight. Less total_time reduces a large potential weight.
    These directions address the named weight, subject to the other range
    checks on the Plan.
    """
    try:
        return _normal_range(value, name, exact, zero=zero)
    except ValueError as error:
        raise ValueError(f"{error}. {remedy()}") from error


def _time(t):
    """Return a finite nonnegative time, the domain of every schedule."""
    t = finite_real(t, "schedule time")
    if t < 0:
        raise ValueError("schedule time must be nonnegative")
    return t


def _interval(t0, t1):
    """Return the endpoints of a positive interval, checked on their binary64 values."""
    t0, t1 = finite_real(t0, "interval start"), finite_real(t1, "interval end")
    if not 0.0 <= t0 < t1:
        raise ValueError("a schedule interval requires 0 <= t0 < t1 as binary64 values")
    return t0, t1


def _mantissa_exponent(factors):
    """Return ``(m, e)`` with ``prod(factors) = m 2**e`` for positive finite factors.

    ``frexp`` is exact, and each factor multiplies m by a mantissa in
    [0.5, 1), one rounding each, so m cannot leave the normal range.
    """
    mantissa, exponent = 1.0, 0
    for x in factors:
        m, e = frexp(x)
        mantissa *= m
        exponent += e
    return mantissa, exponent


def _scaled_product(numerators, denominators=(), exponent_offset=0):
    """Return ``prod(numerators)/prod(denominators) * 2**exponent_offset`` without intermediate range errors.

    The factors are positive and finite. Each multiplication or division of
    a mantissa is one rounding, and the final power-of-two scaling is exact
    unless the result is subnormal or overflows. A zero numerator gives 0.
    """
    if any(x == 0.0 for x in numerators):
        return 0.0
    mantissa, exponent = _mantissa_exponent(numerators)
    for x in denominators:
        m, e = frexp(x)
        mantissa /= m
        exponent -= e
    try:
        return ldexp(mantissa, exponent + exponent_offset)
    except OverflowError:
        raise ValueError("QHD schedule integral exceeds the binary64 range") from None


def _potential_remedy(value):
    """Return the remedy of a refused potential weight, interval integral or step average (``_admitted``).

    Every schedule's potential weight b is nondecreasing on t >= 0, the
    cubic ``b(t) = 2 t**3`` and the quadratic ``1 + gamma t**2``, so a value
    below the normal range comes from the first step, whose midpoint, integral
    and average fewer steps or a longer ``total_time`` increase, and a value
    above it from late times, where a shorter ``total_time`` lowers b. More
    steps reduce a late step's integral but not its average, which stays
    near ``b(total_time)``, so they are not the remedy above the range.
    ``value`` is the refused weight or step average.
    """
    return "Choose fewer steps or a longer total_time" if value < 1 else "Choose a shorter total_time"


def _cubic_potential_integral(t0, t1):
    """Return ``B = ∫ 2 t**3 dt = (t1**4 - t0**4)/2`` over ``[t0, t1]``.

    With ``h = t1 - t0`` and ``q = t0/t1``,
    ``B = h t1**3 (1 + q)(1 + q**2)/2``, a product of positive factors, so
    no difference of fourth powers is formed. The relative sensitivity of
    ``(1 + q)(1 + q**2)`` to q is at most 3/2 on ``0 <= q <= 1``. With its
    three operations, the endpoint difference and at most six mantissa
    operations the relative error stays below 16u.
    """
    t0, t1 = _interval(t0, t1)
    h, q = t1 - t0, t0 / t1
    # B = h t1**3 (1 + q)(1 + q**2)/2, with t1 repeated so no t1**3 is formed.
    return _scaled_product((h, t1, t1, t1, 1.0 + q, 1.0 + q * q), (2.0,))


def _unit_mean(start, width, numerator_power):
    """Return the Gauss-Legendre mean of ``z**p/(1 + z**3)``, ``z = start + width q``, over q in [0, 1].

    ``p`` is ``numerator_power``, 0 or 1, and both endpoints of z lie in
    [0, 1]. Each of the four panels applies the 16-point Gauss-Legendre rule
    (``_UNIT_MEAN_NODES``).

    For the truncation error, ``f(z) = 1/(1 + z**3)`` has poles ``z_j`` at -1 and
    ``exp(±i pi/3)``, at least ``R = sqrt(3)/2`` from every real point of [0, 1]. A panel has
    midpoint m and half-width ``L <= 1/8``, so ``rho = L/R <= 1/(4 sqrt(3))``.
    Expanding each factor ``1/(z - z_j)`` about m as a geometric series with
    ratio at most ``|z - m|/R`` bounds the degree-k Taylor coefficient of f by
    ``f(m) C(k+2, 2)/R**k``, since ``C(k+2, 2)`` counts the ways to split k
    among three factors. The rule is exact through degree 31 and has positive
    weights, so its panel error is at most ``2 f(m) T_32(rho)`` times the panel
    width, where for a nonnegative integer n,
    ``T_n(rho) = sum_{k >= n} C(k+2, 2) rho**k
    = (rho**n/2)((n+1)(n+2)/(1-rho) + (2n+3) rho/(1-rho)**2 + rho (1+rho)/(1-rho)**3)``.
    On the panel ``f >= f(m)/(1 + rho)**3``, so the relative panel error is at
    most ``2 (1 + rho)**3 T_32(rho)``. For ``z f = (m + (z - m)) f`` the bound
    gains ``(L/m) T_31(rho)``, and ``m >= L`` gives the common bound
    ``2 (1 + rho)**3 (T_32(rho) + T_31(rho)) < 1.9e-23``, below ``1.7e-7 u``,
    with ``u = 2**-53``. Sums of positive panel values keep this relative bound.

    Roundoff dominates. The stored nodes, the mapping and the panel offset
    put q within 3u. With the divisions by which the caller forms start and
    width, the mapped coordinates err by at most ``10u (start + width/2)``
    for the segment below ``s**(1/3)`` and ``12u (start + width/2)`` above
    it (``CubicSchedule.kinetic_integral``). Because ``f >= 1/2`` and
    ``|f'|, |(z f)'| <= 1`` on [0, 1], and the mean of ``z f`` is at least
    ``(start + width/2)/2``, that moves the mean by at most 20u on the
    segment below ``s**(1/3)`` and 24u above it. The cube, denominator and
    quotient add at most 3u and the stored weights, products and ``fsum`` at
    most 4u, 31u in all.
    """
    terms = []
    for q, weight in _UNIT_MEAN_NODES:
        z = start + width * q
        # Weighted integrand z**p/(1 + z**3) at the mapped node, a positive term.
        terms.append(weight * ((z if numerator_power else 1.0) / (1.0 + (z * z) * z)))
    return fsum(terms)


class QuadraticSchedule(Record):
    """Quadratic schedule `a(t) = 1/(1 + gamma t**2)`, `b(t) = 1 + gamma t**2`, the QHD default.

    Build it as `QuadraticSchedule(gamma=0.3)` and pass it as `QHD(schedule=...)`.
    `gamma` is the only argument and is optional. Kushnir, Leng, Peng, Fan and Wu give
    `phi_t = -log(1 + gamma t**2)` and `chi_t = log(1 + gamma t**2)` as an example in
    Sec. 2.1 of QHDOPT, arXiv:2409.03121v1, and report that schedules of this form work
    well for many test problems. For `gamma > 0` the ratio `a/b = (1 + gamma t**2)**-2`
    tends to zero, the condition after Eq. (1) of Leng et al., arXiv:2303.01471v1.
    `gamma = 0` gives the time-independent model `a = b = 1`. A decaying ratio does not
    by itself establish convergence on a finite grid or in a finite time. The guide's
    [schedules](../../algorithms/qhd.md#schedules-and-coefficient-rule) section compares
    the three schedules.

    The methods give the point values, interval integrals and derivative bounds that
    planning and
    [`evolution_bound`][nwqlib.algorithms.qhd.evolution_bounds.evolution_bound] use,
    with their rounding in units of `u = 2**-53`.

    Attributes:
        kind: Default `"quadratic"`, the only accepted value. Formula identifier.
        gamma: Default `0.3`, an illustrative working point and not an accuracy choice.
            Nonnegative finite rate, in inverse squared time units.
    """

    # One closed form for A and one for B per interval, the planning work of
    # the integrated coefficient rule (QHD._admit_symbolic_work).
    interval_work: ClassVar[int] = 2

    kind: Literal["quadratic"] = "quadratic"
    # gamma=0.3 is the illustrative working point of the QHD guide and its
    # optimization notebook, not an accuracy choice. Registered in
    # docs/ENGINEERING_CONSTANTS.md ("QHD Method defaults"). Revisit when a
    # default workload is chosen for a scientific purpose.
    gamma: Nonnegative = 0.3

    def kinetic_weight(self, t):
        """Return ``a(t) = 1/(1 + gamma t**2)`` with relative error at most 4u to first order.

        It is the reciprocal of ``potential_weight``, one more rounding, and must be a
        positive normal binary64 number. That fails once ``b(t)`` exceeds about 4.5e307, and
        the refusal names a shorter ``total_time`` as the remedy.
        """
        b = self.potential_weight(t)
        return _admitted(1.0 / b, f"the kinetic schedule weight a({t!r})", lambda: "Choose a shorter total_time",
                         lambda: 1 / Fraction(b))

    def potential_weight(self, t):
        """Return ``b(t) = 1 + gamma t**2`` with relative error at most 3u to first order, or raise when it overflows.

        ``(gamma t) t`` has two roundings and adding the positive 1 a third, so b is at
        least 1 and fails the range only by overflowing, for which the refusal names a
        shorter ``total_time``.
        """
        t = _time(t)
        return _admitted(1.0 + self.gamma * t * t, f"the potential schedule weight b({t!r})",
                         lambda: "Choose a shorter total_time")

    def kinetic_integral(self, t0, t1):
        """Return ``A = (atan(k t1) - atan(k t0))/k``, ``k = sqrt(gamma)``, or ``t1 - t0`` when gamma is 0.

        The primitive of ``1/(1 + gamma t**2)`` is ``atan(k t)/k``. For
        ``x = k t1 >= y = k t0 >= 0`` the angle difference identity
        ``atan(x) - atan(y) = atan((x - y)/(1 + x y))`` holds with a difference in
        ``[0, pi/2)``, so with ``h = t1 - t0`` and ``d = 1 + gamma t0 t1 > 0`` it is
        ``atan(z)``, ``z = k h/d``. Hence ``A = (h/d) atanc(z)`` with
        ``atanc(z) = atan(z)/z``. For ``z <= 2**-27`` the code replaces atanc(z) by 1, an
        approximation with relative error at most ``z**2/3 <= u/6``. Keeping h as a
        factor preserves a short interval whose two angles round to the same value. d has
        relative error at most 3u and z at most 7u.
        ``d log atanc/d log z = z/((1 + z**2) atan z) - 1`` lies in ``[-1, 0]``, and atan
        and the division add 3u, so atanc contributes 10u, and with h, d and the scaled
        product the value stays below 20u. When d or ``k h`` overflows, the exponent-scaled
        branch carries both as mantissa and exponent and forms the angle as
        ``atan2(k h, d)`` on operands scaled by one common power of two, and it returns
        ``h/d`` when the exponents show ``z <= 2**-27``. Its bound is 32u, which also covers
        a product ``gamma t0`` that underflows before its multiplication by a very large t1
        (the lost term changes d by less than 4u).
        """
        t0, t1 = _interval(t0, t1)
        h, gamma = t1 - t0, self.gamma
        if gamma == 0:
            return h
        # d = 1 + gamma t0 t1 and n = k h, so atan(k t1) - atan(k t0) = atan(n/d).
        d = 1.0 + (gamma * t0) * t1
        n = sqrt(gamma) * h
        if isfinite(d) and isfinite(n):
            z = n / d
            ratio = 1.0 if z <= _ATANC_ONE_BELOW else atan(z) / z
            # A = (h/d) atanc(z)
            return _scaled_product((h, ratio), (d,))
        # When d or k h overflows, carry both as mantissa and exponent, add the
        # one to d at their common exponent and scale the atan2 operands together.
        if t0 == 0:
            dm, de = 1.0, 0
        else:
            dm, de = _mantissa_exponent((gamma, t0, t1))
            if de >= 0:
                dm += ldexp(1.0, -de)
            else:
                dm, de = 1.0 + ldexp(dm, de), 0
        nm, ne = _mantissa_exponent((sqrt(gamma), h))
        # nm < 1 and dm >= 1/8, so z = (nm/dm) 2**(ne - de) < 2**(ne - de + 3). At
        # or below _ATANC_ONE_BELOW, atanc(z) = 1 as in the unscaled branch and A = h/d.
        if ne - de + 3 <= log2(_ATANC_ONE_BELOW):
            return _scaled_product((h,), (dm,), exponent_offset=-de)
        common = max(ne, de)
        # The angle atan(k h/d) from operands scaled by one power of two, then A = angle/k.
        angle = atan2(ldexp(nm, ne - common), ldexp(dm, de - common))
        return _scaled_product((angle,), (sqrt(gamma),))

    def potential_integral(self, t0, t1):
        """Return ``B = (t1 - t0) + gamma (t1**3 - t0**3)/3``.

        With ``h = t1 - t0`` and ``q = t0/t1`` the cubic difference factors as
        ``gamma h t1**2 (1 + q + q**2)/3``, positive like h. The sensitivity of
        ``1 + q + q**2`` to q is at most 2, so the polynomial carries at most 5u, and with
        the mantissa operations and the positive ``fsum`` the relative error stays below
        16u. A q or q**2 that underflows is negligible against the constant 1 of the
        polynomial.
        """
        t0, t1 = _interval(t0, t1)
        h = t1 - t0
        if self.gamma == 0:
            return h
        q = t0 / t1
        polynomial = (1.0 + q) + q * q
        # B = h + gamma h t1**2 (1 + q + q**2)/3, a sum of two positive terms.
        return fsum((h, _scaled_product((self.gamma, h, t1, t1, polynomial), (3.0,))))

    # Relative error constant C of the computed kinetic integral A for
    # gamma > 0 over the whole finite binary64 range, |A_hat - A| <= C u A + 2 tau
    # to first order (derived in kinetic_integral and the module docstring).
    # evolution_bounds._coefficient_term uses it for the first-order
    # coefficient residual. Revisit with kinetic_integral.
    kinetic_integral_roundoff: ClassVar[int] = 32

    def exact_weights(self, t):
        """Return ``(a(t), b(t))`` as exact rationals for an exact rational ``t >= 0``.

        The coefficient residual of ``evolution_bound`` compares the stored midpoint weights
        with these exact values, so under the midpoint rule it needs no roundoff estimate.
        """
        b = 1 + Fraction(self.gamma) * t * t
        return 1 / b, b

    def exact_integrals(self, lower, upper):
        """Return ``(A, B)`` over ``[l, r]`` as exact rationals, with A None where it is not rational.

        ``lower`` and ``upper`` are the exact rational endpoints l and r.
        ``B = (r - l) + gamma (r**3 - l**3)/3`` is rational, and so is ``A = r - l`` for
        gamma = 0. For gamma > 0, A is an arctangent difference and None. The coefficient
        residual of ``evolution_bound`` compares the exact product of the stored step
        duration and a stored step average with its integral exactly where that integral
        is rational. Where A is None it uses the first-order relative error constant
        of ``kinetic_integral``, 32 over the whole finite binary64 range.
        """
        gamma = Fraction(self.gamma)
        potential = (upper - lower) + gamma * (upper**3 - lower**3) / 3
        return (upper - lower if gamma == 0 else None), potential

    def derivative_bounds(self, lower, upper):
        """Return exact bounds ``(a*, b*, A1, A2, B1, B2)`` on ``|a|, |b|`` and their first two derivatives over ``[l, r]``.

        ``lower`` and ``upper`` are exact rationals (``fractions.Fraction``) with
        ``0 <= l < r``, and gamma is read as the exact rational of its binary64 value. With
        ``D = 1 + gamma l**2``, ``a = 1/(1 + gamma t**2)`` decreases and
        ``b = 1 + gamma t**2`` increases on ``t >= 0``, so ``a* = a(l) = 1/D`` and
        ``b* = b(r)``. From ``a' = -2 gamma t/(1 + gamma t**2)**2`` and
        ``a'' = 2 gamma (3 gamma t**2 - 1)/(1 + gamma t**2)**3``, bounding each numerator at
        r and each denominator at l gives ``A1 = 2 gamma r/D**2`` and
        ``|a''| <= 2 gamma (1 + 3 gamma r**2)/D**3``, which
        ``A2 = 2 gamma/D**2 + 8 gamma**2 r**2/D**3`` covers because D >= 1.
        ``b' = 2 gamma t`` and ``b'' = 2 gamma`` give ``B1 = 2 gamma r`` and
        ``B2 = 2 gamma``. These are analytic bounds over the whole interval, not sampled
        maxima, and ``evolution_bound`` uses them in the time-ordering and
        midpoint-quadrature bounds.
        """
        gamma = Fraction(self.gamma)
        d = 1 + gamma * lower * lower
        return (1 / d, 1 + gamma * upper * upper, 2 * gamma * upper / (d * d),
                2 * gamma / (d * d) + 8 * gamma * gamma * upper * upper / (d * d * d),
                2 * gamma * upper, 2 * gamma)


class CubicSchedule(Record):
    """Cubic schedule `a(t) = 2/(s + t**3)`, `b(t) = 2 t**3` of Leng et al., Eq. (C.4).

    Build it as `CubicSchedule(s=...)` and pass it as `QHD(schedule=...)`. `s` is
    required and has no default, because it is a model parameter that the caller
    chooses. This is Eq. (C.4) of Leng et al., arXiv:2303.01471v1, p. 32, whose two
    weights follow the ODE model of Nesterov's accelerated gradient method, with s
    removing the singularity of `2/t**3` at t = 0. Liu et al., arXiv:2607.16996v1, Eq.
    (92), write it with `s = 1` and call it QHD-C, but their code runs the shifted form
    ([`ShiftedCubicSchedule`][nwqlib.algorithms.qhd.schedules.ShiftedCubicSchedule])
    under that name, and with `s = 1` only that form reproduces the Rz counts of their
    Table II ([circuit synthesis](../../algorithms/qhd.md#circuit-synthesis)). s is a
    model parameter. Leng et al. set it to their time step, but here it stays fixed when
    the step count changes, because changing s changes the Hamiltonian. For the
    classical Hamiltonian `a |p|**2/2 + b f(x)`, Hamilton's equations `x' = a p` and
    `p' = -b grad f` give `x'' = (a'/a) x' - a b grad f`, so the damping is `-a'/a` and
    the gradient factor `a b`. Here they are `3 t**2/(s + t**3)` and
    `4 t**3/(s + t**3)`, which tends to 4, and they differ from those of
    `ShiftedCubicSchedule`.

    The methods give the point values, interval integrals and derivative bounds that
    planning and
    [`evolution_bound`][nwqlib.algorithms.qhd.evolution_bounds.evolution_bound] use,
    with their rounding in units of `u = 2**-53`.

    Attributes:
        kind: Default `"cubic"`, the only accepted value. Formula identifier.
        s: Required. Positive finite regularization, in cubed time units.
    """

    # At most 128 integrand evaluations for A (64 Gauss nodes on each side of
    # t = s**(1/3)) and one closed form for B per interval, the planning work
    # of the integrated coefficient rule (QHD._admit_symbolic_work).
    interval_work: ClassVar[int] = 129

    kind: Literal["cubic"] = "cubic"
    s: Positive

    def kinetic_weight(self, t):
        """Return ``a(t) = 2/(s + t**3)`` with relative error at most 4u to first order, or raise outside the normal range.

        ``t**3`` as ``(t t) t`` has two roundings, the positive sum one and the division
        one. a is positive for every t, so a result that is zero or subnormal, such as
        ``a(1e103)`` with s = 1 whose exact value is about 2e-309, lies outside the normal
        binary64 range and raises. A lower-range refusal recommends reducing s and/or
        total_time, since both may need to change. An upper-range refusal recommends a
        larger s.
        """
        t = _time(t)
        value = 2.0 / (self.s + t * t * t)
        return _admitted(value, f"the kinetic schedule weight a({t!r})", lambda: (
            "Choose a larger s" if not value < 1 else
            "Reduce s and/or total_time"))

    def potential_weight(self, t):
        """Return ``b(t) = 2 t**3`` with relative error at most 2u to first order, or raise outside the normal range.

        Doubling is exact, and ``(2 t t) t`` has two roundings. b is zero only at t = 0, so
        another zero or a subnormal value raises. The refusal names fewer steps or a longer
        ``total_time`` below the range, and a shorter ``total_time`` above it.
        """
        t = _time(t)
        value = 2.0 * t * t * t
        return _admitted(value, f"the potential schedule weight b({t!r})", lambda: _potential_remedy(value),
                         zero=t == 0)

    def kinetic_integral(self, t0, t1):
        """Return ``A = ∫ 2/(s + t**3) dt`` over ``[t0, t1]`` by positive Gauss-Legendre quadrature.

        The partial-fraction primitive in ``y = t/s**(1/3)``,
        ``log(1 + y)/3 - log(y**2 - y + 1)/6 + atan((2 y - 1)/sqrt(3))/sqrt(3)``, cancels
        catastrophically for large y, where its terms are much larger than the integral, of
        order ``h/y**3``. The quadrature therefore works on positive integrands only.
        ``r = cbrt(s)`` defines the nearby exact ``s' = r**3`` within 6u of s, and
        ``|d log a/d log s| <= 1`` turns that into at most 6u in A. The interval is split at
        r. On a segment below r of length h starting at t0, ``y = t/r`` gives
        ``2/(s' + t**3) dt = (2/r**2) dy/(1 + y**3)``, and with ``y = t0/r + (h/r) q`` this
        is ``A_low = (2 h/r**3) mean_q 1/(1 + y**3)``. On a segment ``[l, v]`` above r of
        length h, ``x = r/t`` maps it to ``[r/v, r/l]``, whose length ``(r/l)(h/v)`` is
        formed without subtracting ``r/l - r/v``, and ``dt = -r dx/x**2`` gives
        ``2/(s' + t**3) dt = -(2/r**2) x dx/(1 + x**3)``, so
        ``A_high = (2 h/(r l v)) mean_q x/(1 + x**3)`` with ``x = r/v + (r/l)(h/v) q``. Both
        means are over q in [0, 1], each by the 16-point Gauss-Legendre rule on four equal
        panels, and carry at most 31u. The mantissa operations, the segment subtraction, the
        6u change of s and the sum of the two positive parts add at most 14u, so the
        first-order relative error is at most 45u, which the bound of 48u covers. Each
        interval costs 64 or 128 integrand evaluations, however close its endpoints or small
        s. Outside s in [1e-8, 1e8] and t <= 1e4 a transformed coordinate can underflow only
        where the tail it describes is itself below the smallest subnormal 2**-1074 or
        negligible in a bounded mean. For example ``r/l < 2**-1022`` needs ``l > 2**663``
        even for the smallest s, and the whole remaining tail is then at most
        ``∫_l^∞ 2/t**3 dt = 1/l**2 < 2**-1326``.
        """
        t0, t1 = _interval(t0, t1)
        r = cbrt(self.s)
        parts = []
        if t0 < r:
            right = min(t1, r)
            h = right - t0
            # A_low = (2 h/r**3) mean 1/(1 + y**3), y from t0/r over a width h/r.
            parts.append(_scaled_product((2.0, h, _unit_mean(t0 / r, h / r, 0)), (r, r, r)))
        if t1 > r:
            left = max(t0, r)
            h = t1 - left
            # A_high = (2 h/(r l v)) mean x/(1 + x**3), x from r/v over a width (r/l)(h/v).
            mean = _unit_mean(r / t1, (r / left) * (h / t1), 1)
            parts.append(_scaled_product((2.0, h, mean), (r, left, t1)))
        return fsum(parts)

    def potential_integral(self, t0, t1):
        """Return ``B = (t1**4 - t0**4)/2``, with C = 16 in the error relation of the module docstring.

        With ``h = t1 - t0`` and ``q = t0/t1`` it is formed as
        ``B = h t1**3 (1 + q)(1 + q**2)/2``, a product of positive factors, so no difference
        of fourth powers is formed. The relative error stays below 16u.
        """
        return _cubic_potential_integral(t0, t1)

    # Relative error constant C of the computed kinetic integral A, valid on
    # s in [1e-8, 1e8] and 0 <= t <= 1e4 (kinetic_integral_range), to first
    # order (derived in kinetic_integral and the module docstring). Used by
    # evolution_bounds._coefficient_term. Revisit with kinetic_integral.
    kinetic_integral_roundoff: ClassVar[int] = 48
    kinetic_integral_range: ClassVar[tuple[float, float, float]] = (1e-8, 1e8, 1e4)

    def exact_weights(self, t):
        """Return ``(a(t), b(t))`` as exact rationals for an exact rational ``t >= 0``.

        The coefficient residual of ``evolution_bound`` compares the stored midpoint weights
        with these exact values.
        """
        cube = t**3
        return 2 / (Fraction(self.s) + cube), 2 * cube

    def exact_integrals(self, lower, upper):
        """Return ``(A, B)`` over ``[l, r]`` as exact rationals, with A None because it is not rational.

        ``lower`` and ``upper`` are the exact rational endpoints l and r.
        ``B = (r**4 - l**4)/2``. A mixes a logarithm and an arctangent. The coefficient
        residual of ``evolution_bound`` compares the exact product of the stored step
        duration and the stored potential step average with the rational integral B
        exactly, and for A it uses the first-order relative error constant of
        ``kinetic_integral``, 48 on s in [1e-8, 1e8] and 0 <= t <= 1e4.
        """
        return None, (upper**4 - lower**4) / 2

    def derivative_bounds(self, lower, upper):
        """Return exact bounds ``(a*, b*, A1, A2, B1, B2)`` on ``|a|, |b|`` and their first two derivatives over ``[l, r]``.

        ``lower`` and ``upper`` are exact rationals with ``0 <= l < r``, and s is the exact
        rational of its binary64 value. With ``D = s + l**3``, ``a = 2/(s + t**3)``
        decreases and ``b = 2 t**3`` increases, so ``a* = 2/D`` and ``b* = 2 r**3``.
        ``a' = -6 t**2/(s + t**3)**2`` gives ``A1 = 6 r**2/D**2``.
        ``a'' = 12 t (2 t**3 - s)/(s + t**3)**3`` has
        ``|a''| <= 12 t/(s + t**3)**2 + 12 t**4/(s + t**3)**3``, since
        ``|2 t**3 - s| <= (s + t**3) + t**3``, which ``A2 = 12 r/D**2 + 36 r**4/D**3``
        covers. ``b' = 6 t**2`` and ``b'' = 12 t`` give ``B1 = 6 r**2`` and ``B2 = 12 r``.
        Numerators are bounded at r and denominators at l. ``evolution_bound`` uses these
        analytic bounds in the time-ordering and midpoint-quadrature bounds.
        """
        s = Fraction(self.s)
        d = s + lower ** 3
        return (2 / d, 2 * upper ** 3, 6 * upper * upper / (d * d),
                12 * upper / (d * d) + 36 * upper ** 4 / (d * d * d), 6 * upper * upper, 12 * upper)


class ShiftedCubicSchedule(Record):
    """Shifted cubic schedule `a(t) = (2/(s + t))**3`, `b(t) = 2 t**3` of Wu et al., Eq. (15).

    Build it as `ShiftedCubicSchedule(s=...)` and pass it as `QHD(schedule=...)`. `s` is
    required and has no default, because it is a model parameter that the caller
    chooses. This is Eq. (15) of Wu et al., arXiv:2605.12066v1, Sec. VI, the schedule of
    their published results, offered for replication. The paper calls s a small
    regularization parameter, and the code behind its results sets `s = T/N_t = 2e-4`
    for `T = 10` and the paper's `N_t = 50,000` time steps. That code uses N_t as the
    number of time points, so its step `T/(N_t - 1)` is close to s but not equal. The
    paper presents the schedule as the QHD-C schedule of Leng et al.,
    arXiv:2303.01471v1, but its dynamics differ from Leng's Eq. (C.4), which is
    [`CubicSchedule`][nwqlib.algorithms.qhd.schedules.CubicSchedule]. The damping
    `-a'/a` and gradient factor `a b` (`CubicSchedule` derives them) are `3/(s + t)` and
    `16 t**3/(s + t)**3`, which tends to 16 instead of 4. Liu et al.,
    arXiv:2607.16996v1, run this form under the name QHD-C in their code, and with
    `s = 1` only this form reproduces the Rz counts of their Table II
    ([circuit synthesis](../../algorithms/qhd.md#circuit-synthesis)). As for the cubic
    form, s stays fixed when the step count changes.

    The methods give the point values, interval integrals and derivative bounds that
    planning and
    [`evolution_bound`][nwqlib.algorithms.qhd.evolution_bounds.evolution_bound] use,
    with their rounding in units of `u = 2**-53`.

    Attributes:
        kind: Default `"shifted_cubic"`, the only accepted value. Formula identifier.
        s: Required. Positive finite time shift, in time units.
    """

    # One closed form each for A and B per interval, the planning work of the
    # integrated coefficient rule (QHD._admit_symbolic_work).
    interval_work: ClassVar[int] = 2

    kind: Literal["shifted_cubic"] = "shifted_cubic"
    s: Positive

    def kinetic_weight(self, t):
        """Return ``a(t) = (2/(s + t))**3`` with relative error at most 8u to first order, or raise outside the normal range.

        ``x = 2/(s + t)`` has two roundings, which cubing triples, and the two products add
        two more. a is positive for every t, so a result that is zero or subnormal raises. A
        lower-range refusal recommends reducing s and/or total_time, since both may need to
        change. An upper-range refusal recommends a larger s.
        """
        t = _time(t)
        x = 2.0 / (self.s + t)
        return _admitted(x * x * x, f"the kinetic schedule weight a({t!r})", lambda: (
            "Choose a larger s" if not x < 1 else
            "Reduce s and/or total_time"))

    def potential_weight(self, t):
        """Return ``b(t) = 2 t**3`` with relative error at most 2u to first order, or raise outside the normal range.

        Doubling is exact, and ``(2 t t) t`` has two roundings. b is zero only at t = 0, so
        another zero or a subnormal value raises. The refusal names fewer steps or a longer
        ``total_time`` below the range, and a shorter ``total_time`` above it.
        """
        t = _time(t)
        value = 2.0 * t * t * t
        return _admitted(value, f"the potential schedule weight b({t!r})", lambda: _potential_remedy(value),
                         zero=t == 0)

    def kinetic_integral(self, t0, t1):
        """Return ``A = 4 ((s + t0)**-2 - (s + t1)**-2)``.

        With ``x0 = s + t0`` and ``x1 = s + t1``, whose exact difference is ``h = t1 - t0``
        even when the rounded sums are equal, ``A = 4 h (1 + x0/x1)/(x0**2 x1)``, with no
        difference of inverse squares. The logarithmic sensitivities to x0 and x1 are at
        most 2 and 3/2, so the two rounded sums contribute 3.5u, and the relative error
        stays below 16u. On ``[0, s]``, A is exactly ``3/s**2``. When ``s + t1`` overflows,
        s exceeds ``2**969`` and the whole integral is at most ``4/s**2 < 2**-1936``, below
        the subnormal range, so the result is zero.
        """
        t0, t1 = _interval(t0, t1)
        h, x0, x1 = t1 - t0, self.s + t0, self.s + t1
        if isinf(x1):
            return 0.0
        # A = 4 h (1 + x0/x1)/(x0**2 x1), which keeps h exact when x0 and x1 round equal.
        return _scaled_product((4.0, h, 1.0 + x0 / x1), (x0, x0, x1))

    def potential_integral(self, t0, t1):
        """Return ``B = (t1**4 - t0**4)/2``, with C = 16 in the error relation of the module docstring.

        With ``h = t1 - t0`` and ``q = t0/t1`` it is formed as
        ``B = h t1**3 (1 + q)(1 + q**2)/2``, a product of positive factors, so no difference
        of fourth powers is formed. The relative error stays below 16u.
        """
        return _cubic_potential_integral(t0, t1)

    def exact_weights(self, t):
        """Return ``(a(t), b(t))`` as exact rationals for an exact rational ``t >= 0``.

        The coefficient residual of ``evolution_bound`` compares the stored midpoint weights
        with these exact values.
        """
        return 8 / (Fraction(self.s) + t) ** 3, 2 * t**3

    def exact_integrals(self, lower, upper):
        """Return ``(A, B)`` over ``[l, r]`` as exact rationals.

        ``lower`` and ``upper`` are the exact rational endpoints l and r.
        ``A = 4 ((s + l)**-2 - (s + r)**-2)`` and ``B = (r**4 - l**4)/2``, both rational, so
        the exact subtraction has no cancellation error. The coefficient residual of
        ``evolution_bound`` compares the exact product of the stored step duration and each
        stored step average with these integrals exactly.
        """
        s = Fraction(self.s)
        return 4 * (1 / (s + lower) ** 2 - 1 / (s + upper) ** 2), (upper**4 - lower**4) / 2

    def derivative_bounds(self, lower, upper):
        """Return exact bounds ``(a*, b*, A1, A2, B1, B2)`` on ``|a|, |b|`` and their first two derivatives over ``[l, r]``.

        ``lower`` and ``upper`` are exact rationals with ``0 <= l < r``, and s is the exact
        rational of its binary64 value. ``a = 8/(s + t)**3`` decreases with
        ``a' = -24/(s + t)**4`` and ``a'' = 96/(s + t)**5``, so ``a* = 8/(s + l)**3``,
        ``A1 = 24/(s + l)**4`` and ``A2 = 96/(s + l)**5``. ``b = 2 t**3`` gives
        ``b* = 2 r**3``, ``B1 = 6 r**2`` and ``B2 = 12 r``. ``evolution_bound`` uses these
        analytic bounds in the time-ordering and midpoint-quadrature bounds.
        """
        shifted = Fraction(self.s) + lower
        return (8 / shifted ** 3, 2 * upper ** 3, 24 / shifted ** 4, 96 / shifted ** 5,
                6 * upper * upper, 12 * upper)


# The type of QHD.schedule. The kind field selects the record when a saved
# Method is loaded, so each formula keeps its own name and identity.
Schedule = Annotated[
    QuadraticSchedule | CubicSchedule | ShiftedCubicSchedule, Field(discriminator="kind")
]


def step_weights(schedule, rule, total_time, num_steps):
    """Return one ``(t, kinetic_weight, potential_weight)`` row per step of the selected evolution.

    Step k covers ``[k dt, (k + 1) dt]`` with ``dt = total_time/num_steps``,
    and t is its midpoint ``(k + 1/2) dt`` under both rules. The
    ``"midpoint"`` rule gives the point values ``a(t)`` and ``b(t)``. The
    ``"integrated"`` rule gives the step averages ``A/dt`` and ``B/dt`` of
    the interval integrals, so each step's kinetic and potential exponents
    ``dt * weight`` equal A and B. The step propagator of
    ``H(t) = a(t) K + b(t) V`` is ``exp(Omega_1 + Omega_2 + ...)`` with the
    Magnus terms ``Omega_1 = -i (A K + B V)`` and
    ``Omega_2 = -(1/2) [K, V] ∫∫_{v<u} (a(u) b(v) - b(u) a(v)) du dv``, which
    expands about the midpoint to ``(dt**3/12)(a b' - a' b) [K, V]``. The
    integrated rule therefore removes the quadrature error of the
    coefficients but keeps the time-ordering error of ``Omega_2`` and higher
    terms, which vanishes when ``[K, V] = 0`` or when ``a/b`` is constant in
    time. The ``split_step`` module docstring separates these time-error
    terms, ``evolution_bounds.evolution_bound`` bounds them for a compiled
    product, ``method._error_model`` declares them, and
    ``compiler.QHDCompiler.build_step_pauli_ir`` splits the exponent
    ``-i dt (a_k K + b_k V)`` of step k, with the weights ``a_k`` and ``b_k``
    of its row, into product-formula factors. Under the integrated rule that
    exponent is ``Omega_1``.
    Adjacent steps share their binary64 endpoint ``(k + 1) dt``, so the
    intervals tile ``[0, num_steps dt]``.

    Range admission (``validation._normal_range``). With ``N = num_steps``
    and fl denoting binary64 round-to-nearest,
    ``dt = fl(total_time/N)`` is positive and normal, and so is ``dt/2``,
    which is both the first midpoint and the second-order half duration
    and is then exact. Every step's binary64
    endpoints ``k dt < (k + 1) dt`` are finite and increasing, and under the
    midpoint rule its midpoint lies strictly between them. Every weight is
    positive and normal. The point values raise otherwise, and under the
    integrated rule each interval integral and its division by dt are
    checked. These conditions protect the endpoint-rounding argument and the
    exact half durations of the error ledger, whose reference remains the
    exact products of the stored dt (``circuit_errors``, ``evolution_bounds``).
    A refused duration, partition or point weight names the input to change
    (``_admitted``). The two interval integrals are admitted separately
    (``_step_average``). A refused potential integral or step average names
    its remedy too (``_potential_remedy``): fewer steps or a longer
    total_time below the range, and a shorter total_time above it, since
    more steps reduce a step's integral but not its average. A refused
    kinetic integral or step average names none, because a short first
    step, late times and the schedule's s can each cause it.

    Raises:
        ValueError: A duration, endpoint, midpoint or weight fails these
            conditions.
    """
    try:
        dt = _normal_quotient(total_time, num_steps, "the step duration total_time/num_steps")
        # dt/2, exact and normal when dt >= 2**-1021.
        _normal_quotient(dt, 2.0, "half the step duration total_time/(2 num_steps)")
    except ValueError as error:
        # dt = T/N is at most T, so only the lower end of the range can fail.
        raise ValueError(f"{error}. Choose fewer steps or a longer total_time") from error
    rows = []
    for step in range(num_steps):
        start, time, end = step * dt, (step + 0.5) * dt, (step + 1) * dt
        if not (isfinite(end) and start < end and (rule != "midpoint" or start < time < end)):
            # An end beyond the range needs T near the largest binary64 number, and equal endpoints or
            # midpoints need step indices near 2**52, where k dt and (k + 1) dt round together.
            remedy = "Choose a shorter total_time" if not isfinite(end) else "Choose fewer steps"
            raise ValueError(f"step {step} of the time partition has the binary64 start {start!r}, midpoint "
                             f"{time!r} and end {end!r}, which are not finite and strictly increasing. {remedy}")
        if rule == "midpoint":
            kinetic, potential = schedule.kinetic_weight(time), schedule.potential_weight(time)
        else:
            # Whole-step averages A/dt and B/dt, so dt times each weight is the step's integral.
            kinetic = _step_average(schedule, "kinetic", step, start, end, dt)
            potential = _step_average(schedule, "potential", step, start, end, dt)
        rows.append((time, kinetic, potential))
    return tuple(rows)


def _step_average(schedule, name, step, start, end, dt):
    """Return the step average ``I/dt`` of the kinetic or potential interval integral I of one step, both admitted.

    Each integral is computed and admitted on its own, so a refusal names
    the schedule, the quantity and the step. An integral whose value leaves
    the binary64 range is refused by the arithmetic helper
    (``_scaled_product``), and one below the range and the average by
    ``validation._normal_range``. A refused potential integral or average
    names the direction of ``_potential_remedy``, read from the average,
    infinite for an overflow. A refused kinetic one names none, because a
    short first step, late times and a large s can each put it below the
    range, and a small s above it.
    """
    integral = schedule.kinetic_integral if name == "kinetic" else schedule.potential_integral
    quantity = f"the {name} integral over step {step} of the {schedule.kind} schedule"
    try:
        value = integral(start, end)
    except ValueError as error:
        # The arithmetic helper refuses an integral above the range without naming it.
        value, refusal = inf, f"{quantity} is refused: {error}"
    else:
        try:
            return _normal_quotient(_normal_range(value, quantity), dt, f"the {name} step average of step {step}")
        except ValueError as error:
            refusal = str(error)
    raise ValueError(refusal + (f". {_potential_remedy(value / dt)}" if name == "potential" else ""))
