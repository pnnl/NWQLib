"""Proposal laws and the compact pair-row table of a sampled LCHS observable.

The selected LCHS vector is ``v = integral a(x) |chi_x> dmu(x)`` over
x = (application, kernel node). An application is the initial atom at
elapsed time T, a Gauss-Legendre source application at elapsed time t_a, or
a point t of the continuous source interval. With the signed scalar
compensation s, the actual simulation children L_c = L - c_L I and
H_c = H - c_H I, and the stored kernel coefficient c(k),

    a(x) = m(application) * c(k) * exp(i*phi(t, k)),
    phi(t, k) = -t*(k*(c_L + s) + c_H),
    |chi_x> = exp(-i*t*(k*L_c + H_c)) |psi_kind>,

where m is the initial mass nu_0*exp(s*T), a discrete source mass
w_a*nu_b*exp(s*t_a), or the continuous source density nu_b*exp(s*t) on the
active elapsed-time interval [a, b] (An, Childs and Lin, arXiv:2312.03916v2,
Eqs. (2), (4), (7); the scalar phase follows from exact commutation of the
identity parts). The masses are the PhysicalScale values of
``source_selection.application_facts``. The physical relative phase of the
initial and source states stays in psi_kind and is not part of a.

The proposal is a product q(x) = p(application) * p(k), and the importance
weight is w = a/q. A row holds two independent draws x, y and an
observable-term draw, all fixed during the row's m shots, so the row is the
outer cluster of `sampled_moments`. The weight always divides by the
probability the sampler actually implements, so the mean of w*chi is
sum_x a(x)*chi_x exactly for every finite law, whatever its rounding:

* A finite law (discrete applications, discrete kernel nodes, the
  initial/source choice of a continuous source, and observable terms) is an
  integer table n_j with sum 2**52 drawn by one uniform 52-bit integer, so
  its probabilities are exactly pi_j = n_j/2**52. Every nonzero magnitude
  receives n_j >= 1, so no nonzero coefficient loses its mass. The rounding
  of the exact-magnitude law |a_j|/sum|a| only changes the weight bound and
  the second moments, not the mean. The kernel law is built from the
  lifted magnitudes 2**k*|c_j| (`kernel_law`).
* A continuous inverse-distribution draw uses the midpoint
  u = (j + 1/2)/2**52 of a uniform 52-bit integer j, which is exactly
  representable and never an endpoint. The weight uses the ideal density,
  so the floating transform of the midpoint is a numerical approximation of
  that law without an arithmetic enclosure here.

Continuous source time uses the exponentially tilted density
``p(t) = exp(s*t)/integral_a^b exp(s*v) dv`` (`source_times`). Its weight
nu_b*exp(s*t)/(pi_src*p(t)) is the constant M_src/pi_src for the source
mass M_src that the application facts supply, so exp(s*t) is not applied
again after the draw.

The row table holds each branch weight as a separate magnitude and phase. The magnitude
of branch x is |w_x|/g in one binary frame g = ``LCHSSampledRows.scale``, a
power of two chosen from the cap W = max_i (M_i/pi_i) * W_kernel on every
branch weight of the whole support, so that W/g lies in [1/2, 1). W is not
the mean mass sum_i M_i, because the integer probabilities pi_i are not
exactly proportional to M_i. For a discrete kernel W_kernel is
max_j |c_j|/p_j = 2**-k*max_j K_j*2**52/n_j with K_j = 2**k*|c_j|, and the
integer -k enters each category exponent once, so g, W and every scaled
weight are physical; omitting it would scale physical Q and N by 2**(2k)
and their second moments by 2**(4k). Integer exponents of the masses are
subtracted from the frame exponent as Python integers before any binary64
conversion, so a mass far outside binary64 still gives a representable
scaled weight. The phase is theta_x = arg c(k_x) + phi(t_x, k_x) reduced to
[-pi, pi]: arg c is atan2 of the stored discrete coefficient or the
analytic phase of a continuous kernel, phi enters through the unit complex
number of `solution_error_budget.extracted_phase`, and the two bounded
angles are added and reduced by ``math.remainder``. Reading a phase from a
scaled weight would lose it when the weight's Cartesian parts are tiny,
and adding an unreduced large angle would lose the coefficient's angle to
rounding, so neither is done. A drawn nonzero magnitude below the normal
binary64 range of the frame is refused, because the binary64 scores could
not keep it.
"""

from dataclasses import dataclass
import json
from math import atan2, exp, expm1, frexp, fsum, inf, ldexp, log, pi, remainder, tau
from typing import Literal

import numpy as np
from pydantic import model_validator

from nwqlib._validation import integer
from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.core.records import FrozenArray, Nonnegative, PositiveInt, Record, Text
from nwqlib.problems.inputs import PhysicalScale

UNIFORM_LAW = "midpoint-52bit/1"
CATEGORICAL_LAW = "integer-2**52/1"
DRAW_ORDER = "application_x,source_time_x,application_y,source_time_y,kernel_x,kernel_y,term/2"
INITIAL, SOURCE = 0, 1
_TOTAL = 1 << 52
_LOG2 = log(2.0)
_TINY = float(np.finfo(float).tiny)
_ZERO = PhysicalScale(mantissa=0.0, exponent=0)


@dataclass(frozen=True)
class IntegerCategorical:
    """A finite law with exact probabilities counts[j]/2**52, drawn by one uniform 52-bit integer.

    The cumulative counts are formed once, so each later draw is a binary
    search of O(log n) work rather than a new O(n) prefix sum; the PTSC
    label tables are drawn once per correction label (`sampled_recipes`).
    """

    counts: np.ndarray

    def __post_init__(self):
        object.__setattr__(self, "cumulative", np.cumsum(self.counts))

    def draw(self, rng, size):
        draws = rng.integers(0, _TOTAL, size=size, dtype=np.int64)
        return np.searchsorted(self.cumulative, draws, side="right")


def integer_categorical(log_magnitudes):
    """Return an integer law near the exact-magnitude law of ``log_magnitudes`` that keeps every nonzero entry.

    The exact law is proportional to exp(log_magnitudes); -inf marks an exact
    zero, which is never drawn. A finite log whose relative exponential
    underflows is still nonzero. Counts are floor(2**52*p_j), raised to 1 for
    a nonzero p_j that rounds to zero, and a positive remaining difference to
    2**52 is assigned to the largest fractional parts. An excess
    e = sum_j n_j - 2**52 > 0 is removed from the largest counts in turn,
    each down to one count, so no nonzero entry loses its mass. The spare
    counts n_j - 1 of the r nonzero entries sum to 2**52 + e - r >= e,
    because r <= 2**51, so the removal always completes. The returned
    probabilities n_j/2**52 are exact by construction; unbiased weights
    divide by them, so their distance from the exact-magnitude law changes
    only the weight bound and second moments.
    """
    values = np.asarray(log_magnitudes, dtype=float)
    nonzero = values > -inf
    if values.ndim != 1 or not nonzero.any() or np.isnan(values).any() or (values == inf).any():
        raise ValueError("a finite law needs finite log magnitudes and at least one nonzero entry")
    if int(nonzero.sum()) > _TOTAL // 2:
        raise ValueError("a 52-bit integer law holds at most 2**51 nonzero entries")
    relative = np.zeros_like(values)
    relative[nonzero] = np.exp(values[nonzero] - values[nonzero].max())
    target = relative / fsum(relative) * float(_TOTAL)
    counts = np.floor(target).astype(np.int64)
    counts[nonzero & (counts == 0)] = 1
    difference = _TOTAL - int(counts.sum(dtype=np.int64))
    if difference > 0:
        order = np.argsort(np.floor(target) - target, kind="stable")
        order = order[nonzero[order]]
        step, extra = divmod(difference, order.size)
        counts[order] += step
        counts[order[:extra]] += 1
    else:
        # One pass in descending count order: entry j gives
        # min(spare_j, excess not yet removed), where spare_j = n_j - 1 on
        # nonzero entries and 0 on exact zeros.
        order = np.argsort(-counts, kind="stable")
        spare = counts[order] - nonzero[order]
        before = np.cumsum(spare) - spare
        counts[order] -= np.minimum(spare, np.maximum(-difference - before, 0))
    if counts.sum(dtype=np.int64) != _TOTAL or np.any(counts[nonzero] < 1):
        raise ValueError("the integer law cannot keep every nonzero entry")
    return IntegerCategorical(counts=counts)


def kernel_law(coefficients):
    """Return the integer law of a finite kernel table, its lifted magnitudes K and the lift k.

    K_j = 2**k*|c_j| comes from ``providers.lifted_magnitudes``, whose exact
    power-of-two lift brings every nonzero Cartesian part of the table into
    the normal binary64 range before the magnitude is taken, so a subnormal
    coefficient keeps its relative accuracy: |(1+i)*2**-1074| keeps its
    factor sqrt(2). The law is ``integer_categorical(log K)``; a common lift
    changes its counts n_j only through the rounding of log K. With
    p_j = n_j/2**52 the importance ratio is
    |c_j|/p_j = 2**-k*(K_j*2**52/n_j), and the caller keeps 2**-k as an
    integer exponent. Both consumers of the law, `pair_laws` and
    ``sampled_plan._pair_bias``, call this function, so their counts agree.
    The raw coefficient phases are not part of the law. The input must be an
    admitted provider table (``providers.coefficient_lift``), which bounds
    K below binary64 overflow.
    """
    from .providers import lifted_magnitudes
    magnitudes, lift = lifted_magnitudes(coefficients)
    with np.errstate(divide="ignore"):
        return integer_categorical(np.log(magnitudes)), magnitudes, lift


def midpoint_uniforms(rng, size):
    """Return midpoints (2*j + 1)/2**53 of uniform 52-bit integers j, exact binary64 values in (0, 1)."""
    return (2 * rng.integers(0, _TOTAL, size=size, dtype=np.int64) + 1).astype(float) * 2.0**-53


@dataclass(frozen=True, kw_only=True)
class DiscreteKernel:
    """Finite kernel nodes k_j and processed coefficients c_j = w_j*g(k_j), without PSD growth or source factors."""

    nodes: np.ndarray
    coefficients: np.ndarray
    formula_id: str


def source_times(decay, start, stop, uniforms):
    """Return the tilted elapsed times on [start, stop] of midpoint ``uniforms``.

    The density is p(t) = exp(s*t)/integral_a^b exp(s*v) dv with s = decay
    and a = start < b = stop, for a nonzero source mass. With
    Delta = b - a and x = |s|*Delta, its inverse distribution function t(u)
    is evaluated in three branches:

    - s = 0, or x = 0 because the product underflowed: t = a + u*Delta.
    - 0 < x <= 1: t = a + Delta*u*phi*psi. For s > 0, E = expm1(x),
      phi = E/x and psi = log1p(z)/z with z = u*E, which equals
      a + log1p(u*expm1(s*Delta))/s. For s < 0, E = -expm1(-x), phi = E/x
      and psi = -log1p(-z)/z, which equals a + log1p(-u*E)/s. phi lies in
      [1, e - 1] or [1 - 1/e, 1] and psi tends to 1 as z tends to 0, so a
      positive tilt whose product u*E underflows keeps its displacement
      u*Delta instead of collapsing to an endpoint.
    - x > 1: the time is measured from the endpoint nearer to it. Let
      p = u for s > 0 and p = 1 - u for s < 0 be the probability between the
      low-density end (a for s > 0, b for s < 0) and the draw, and
      v = 1 - p; both subtractions are exact for the dyadic midpoints. The
      distance from the low-density end is d_L = log1p(p*expm1(x))/|s| and
      the distance from the high-density end is
      d_H = -log(p + v*exp(-x))/|s|. Multiplying p + v*exp(-x) by exp(x)
      gives 1 + p*expm1(x), so d_L + d_H = Delta, and d_L <= Delta/2 exactly
      when p <= 1/(1 + exp(x/2)) = exp(-x/2)/(1 + exp(-x/2)), which selects
      the branch. There p >= 2**-53 gives x <= 2*log(2**53 - 1) < 74, so
      expm1(x) is finite. It is evaluated only when some draw takes that
      branch, so a larger or overflowed x never reaches it. On the other
      branch the logarithm is
      log1p(-z) with z = v*(-expm1(-x)) for z <= 1/2, and otherwise
      log(p + v*exp(-x)), whose argument is at least 2**-53. Neither
      subtracts nearly equal numbers, and the sum keeps v*exp(-x) where
      -expm1(-x) rounds to 1. For s < 0 that sum is (1 - u) + u*exp(-x).
      Each distance is at most Delta/2, so a short displacement from a huge
      endpoint is never formed by subtraction from the far endpoint.

    Each form follows from exp(s*(t - a)) = 1 + u*expm1(s*Delta), the
    inverse of the distribution function
    (exp(s*(t - a)) - 1)/expm1(s*Delta). The result is projected onto [a, b], which
    keeps every time in the support and cannot move it away from the exact
    t(u) in [a, b]. The draw is the floating transform of the midpoint, not
    the ideal density; no enclosure of expm1, log1p, log, exp or the
    division is established. Each draw takes one logarithm, and a call at
    most the four scalar exponentials exp(-x/2), expm1(x), expm1(-x) and
    exp(-x).

    Raises:
        ValueError: If an evaluated time is not finite.
    """
    s, a, b = float(decay), float(start), float(stop)
    u = np.asarray(uniforms, dtype=float)
    delta = b - a
    x = abs(s) * delta
    if s == 0.0 or x == 0.0:
        times = a + u * delta
    elif x <= 1.0:
        tilt = expm1(x) if s > 0.0 else -expm1(-x)
        z = u * tilt
        safe = np.where(z == 0.0, 0.5, z)
        relative = (np.log1p(safe) if s > 0.0 else -np.log1p(-safe)) / safe
        times = a + delta * (u * (tilt / x) * np.where(z == 0.0, 1.0, relative))
    else:
        sigma, p = abs(s), u if s > 0.0 else 1.0 - u
        half = exp(-0.5 * x)
        near = p <= half / (1.0 + half)
        distance = np.empty_like(u)
        if near.any():
            distance[near] = np.log1p(p[near] * expm1(x)) / sigma
        far = ~near
        if far.any():
            q, v = p[far], 1.0 - p[far]
            z = v * -expm1(-x)
            small = z <= 0.5
            logarithm = np.empty_like(z)
            logarithm[small] = np.log1p(-z[small])
            logarithm[~small] = np.log(q[~small] + v[~small] * exp(-x))
            distance[far] = -logarithm / sigma
        if s > 0.0:
            times = np.where(near, a + distance, b - distance)
        else:
            times = np.where(near, b - distance, a + distance)
    if not np.all(np.isfinite(times)):
        raise ValueError("an evaluated continuous source time is not finite")
    return np.clip(times, a, b)


def _category_law(masses):
    """Return the integer law over category masses and each category's ratio M_i/pi_i as ``(mantissa, exponent)``.

    The exact-magnitude law is proportional to the masses. Its logarithms are
    taken relative to the largest exponent E, log(mantissa) + (e - E)*log 2,
    with the integer difference e - E formed first and clipped at -2**20,
    far below the relative mass 2**-1075 that already gets the minimum count.
    Category i then has M_i/pi_i = (mantissa_i*2**52/n_i)*2**e_i, where the
    first factor is one rounded binary64 quotient in [1/2, 2**52] and e_i
    stays a Python integer.
    """
    nonzero = [mass.mantissa > 0.0 for mass in masses]
    if not any(nonzero):
        raise ValueError("a row table needs at least one nonzero application mass")
    top = max(mass.exponent for mass, kept in zip(masses, nonzero) if kept)
    logs = np.array([log(mass.mantissa) + max(mass.exponent - top, -(1 << 20)) * _LOG2 if kept else -inf
                     for mass, kept in zip(masses, nonzero)])
    law = integer_categorical(logs)
    ratios = [(mass.mantissa * (float(_TOTAL) / int(count)), mass.exponent) if kept else (0.0, 0)
              for mass, count, kept in zip(masses, law.counts, nonzero)]
    return law, ratios


def _continuous_kernel_phase(kernel, nodes):
    """Return the phase of a continuous kernel at ``nodes``, reduced to [-pi, pi].

    The phase is that of the kernel definition, not of an evaluated weight,
    whose Cartesian parts can underflow: Apel-Berntson
    -(Delta/pi)*(log1p(k/R) - log1p(-k/R)) on |k| < R
    (`providers.apel_berntson_kernel`) and Low-Somma -c*k
    (`providers.low_somma_fhat_2`). The provider's positive weight factors
    do not change it. The reduction goes through the unit complex number
    exp(i*psi).
    """
    k = np.asarray(nodes, dtype=float)
    if kernel.resolved_lchs_kernel.implementation == "apel_berntson":
        radius, delta = float(kernel.radius), float(kernel.derived_parameters["Delta"])
        psi = -(delta / pi) * (np.log1p(k / radius) - np.log1p(-k / radius))
    else:
        psi = -float(kernel.resolved_lchs_kernel.parameters["c"]) * k
    return np.angle(np.exp(1j * psi))


class LCHSSampledRows(Record):
    """The B independent pair rows of a sampled LCHS observable, each measured m times.

    Planning draws the rows once from the Method's random stream and saves
    them here. Analysis, loading and the continuation of a partial run read
    these rows and never draw again. Row r holds the two branches x and y:
    their kind (``INITIAL`` = 0 or ``SOURCE`` = 1), elapsed time t, kernel
    node k, scaled weight magnitude |w|/g and phase theta in [-pi, pi], and
    the observable term l with its signed range o_l/p(l). The pair phase of
    the row is ``remainder(theta_y - theta_x, 2*pi)`` and its score
    magnitude is |w_x|*|w_y|/g**2.

    Attributes:
        clusters: Number of rows B.
        repeats: Shots m per row. m = 1 gives fresh pairs.
        kernel_law: ``"discrete"`` (integer law over the provider's nodes) or
            ``"continuous"`` (truncated Cauchy draw of the continuous kernel).
        kernel_formula_id: Versioned formula of the kernel.
        application_law: ``"discrete"`` (integer law over the initial and
            Gauss-Legendre source applications) or ``"continuous_source"``
            (initial atom and tilted source time).
        uniform_law, categorical_law, draw_order: Versioned sampler laws and
            the order of the draws from the random stream. A source time is
            drawn right after its application, only for rows that chose the
            continuous source.
        draw_state: PCG64 state of the Method's random stream before the
            draws, the random identity of the rows.
        scale: Weight frame g, a power of two.
        weight_bound: W/g for the all-support cap W on every branch weight.
        score_bound_q, score_bound_n: Bounds on the exact-product scores in
            the scaled frame, max_l|o_l/p(l)|*(W/g)**2 and (W/g)**2.
        active_time: T_e, the supremum of the active elapsed-time support,
            fixed before drawing and never the largest drawn time.
        kind_x, kind_y, time_x, time_y, node_x, node_y: Branch draws.
        magnitude_x, magnitude_y: Scaled branch weight magnitudes |w|/g.
        phase_x, phase_y: Branch phases arg c(k) + phi(t, k), in [-pi, pi].
        term: Index of the drawn observable term in the Plan's term table,
            or -1 when the output needs no observable.
        signed_range: o_l/p(l) of the drawn term, or 0 without a term.
        qualification: Numerical status of the sampler and weights.
    """

    clusters: PositiveInt
    repeats: PositiveInt
    kernel_law: Literal["discrete", "continuous"]
    kernel_formula_id: Text
    application_law: Literal["discrete", "continuous_source"]
    uniform_law: Literal["midpoint-52bit/1"] = UNIFORM_LAW
    categorical_law: Literal["integer-2**52/1"] = CATEGORICAL_LAW
    draw_order: Literal["application_x,source_time_x,application_y,source_time_y,kernel_x,kernel_y,term/2"] = \
        DRAW_ORDER
    draw_state: Text
    scale: PhysicalScale
    weight_bound: Nonnegative
    score_bound_q: Nonnegative
    score_bound_n: Nonnegative
    active_time: Nonnegative
    kind_x: FrozenArray
    kind_y: FrozenArray
    time_x: FrozenArray
    time_y: FrozenArray
    node_x: FrozenArray
    node_y: FrozenArray
    magnitude_x: FrozenArray
    magnitude_y: FrozenArray
    phase_x: FrozenArray
    phase_y: FrozenArray
    term: FrozenArray
    signed_range: FrozenArray
    qualification: Text

    @model_validator(mode="after")
    def _shape(self):
        for name in ("kind_x", "kind_y", "time_x", "time_y", "node_x", "node_y", "magnitude_x", "magnitude_y",
                     "phase_x", "phase_y", "term", "signed_range"):
            if getattr(self, name).array.shape != (self.clusters,):
                raise ValueError(f"row field {name} must hold one entry per row")
        return self

    def weights(self):
        """Return the complex scaled weights ``(w_x/g, w_y/g)``, a display form of the stored magnitudes and phases."""
        return (self.magnitude_x.array * np.exp(1j * self.phase_x.array),
                self.magnitude_y.array * np.exp(1j * self.phase_y.array))

    def branch_phases(self):
        """Return the pair phase remainder(theta_y - theta_x, 2*pi) of every row, in [-pi, pi]."""
        difference = self.phase_y.array - self.phase_x.array
        return difference - tau * np.rint(difference / tau)

    def score_magnitudes(self):
        """Return |w_x|*|w_y|/g**2 of every row."""
        return self.magnitude_x.array * self.magnitude_y.array


@dataclass(frozen=True, kw_only=True)
class PairLaws:
    """The deterministic proposal laws and weight frame of one row table, formed before any draw.

    `pair_laws` builds it from the application facts, the kernel and the
    observable terms alone, so the frame g = 2**frame, the all-support cap
    ``weight_bound`` = W/g and the term range H_O = ``range_bound`` are known
    before the row count B is chosen; `draw_pair_rows` then draws from these
    laws. Nothing here depends on a drawn value.
    """

    final_time: float
    active_time: float
    signed_shift: float
    continuous_source: bool
    interval: tuple | None
    category_law: IntegerCategorical
    kinds: np.ndarray
    elapsed: np.ndarray
    category_mantissa: np.ndarray
    category_exponent: np.ndarray
    kernel: object
    node_law: IntegerCategorical | None
    node_ratio: np.ndarray | None
    node_phase: np.ndarray | None
    kernel_law_name: str
    formula: str
    frame: int
    weight_bound: float
    term_law: IntegerCategorical | None
    term_ranges: np.ndarray
    range_bound: float
    notes: tuple


def pair_laws(*, kernel, applications, terms):
    """Return the `PairLaws` of a row table: category, kernel and term laws and the common weight frame.

    Args:
        kernel: A `DiscreteKernel`, or the provider's continuous kernel plan
            with ``node(u)``, ``weight(k)``, ``importance_weight_bound`` and
            ``lchs_kernel_formula_id``. Its weight must divide by the
            proposal it actually samples, including any endpoint guard.
        applications: ``source_selection.LCHSApplicationFacts``. A
            ``"continuous_source"`` draws the initial atom or the tilted
            source time on ``source_interval``, the other kinds one of
            ``applications``.
        terms (tuple): Real Pauli terms ``(label, coefficient)`` of the
            observable, or ``()`` when only the norm is needed.

    Raises:
        ValueError: For no nonzero application or kernel coefficient, or a
            continuous kernel without a finite positive weight bound.
    """
    final_time = float(applications.final_time)
    # Applications: kind and elapsed time of each category.
    continuous_source = applications.kind == "continuous_source"
    interval = None
    if continuous_source:
        initial = next((item.mass for item in applications.applications if item.kind == "initial"), _ZERO)
        masses = (initial, _ZERO if applications.source_mass is None else applications.source_mass)
        kinds, elapsed = np.array([INITIAL, SOURCE]), np.array([final_time, np.nan])
        interval = tuple(float(value) for value in applications.source_interval)
    else:
        items = applications.applications
        masses = tuple(item.mass for item in items)
        kinds = np.array([INITIAL if item.kind == "initial" else SOURCE for item in items], dtype=np.int64)
        elapsed = np.array([float(item.elapsed_time) for item in items])
    law, ratios = _category_law(masses)

    # Kernel: node law, ratio magnitudes |c|/p and phases arg c. A discrete
    # kernel's ratios are 2**lift*|c|/p (kernel_law).
    notes = ["finite laws have exact integer probabilities n/2**52",
             "branch magnitudes and phases are stored separately"]
    node_law = ratio = phase = None
    lift = 0
    if isinstance(kernel, DiscreteKernel):
        node_law, lifted, lift = kernel_law(kernel.coefficients)
        kept = node_law.counts > 0
        ratio = np.zeros(len(lifted))
        ratio[kept] = lifted[kept] * (float(_TOTAL) / node_law.counts[kept])
        coefficients = np.asarray(kernel.coefficients, dtype=complex)
        phase = np.arctan2(coefficients.imag, coefficients.real)
        kernel_bound = float(ratio[kept].max())
        kernel_law_name, formula = "discrete", kernel.formula_id
    else:
        kernel_bound = float(kernel.importance_weight_bound)
        if not (np.isfinite(kernel_bound) and kernel_bound > 0):
            raise ValueError("a continuous kernel needs a finite positive importance-weight bound")
        kernel_law_name, formula = "continuous", kernel.lchs_kernel_formula_id
        notes.append("continuous kernel draws use midpoint-52bit uniforms and the provider's guarded weight")
    if continuous_source:
        notes.append("continuous source times use midpoint-52bit uniforms and the stable tilted inverse "
                     "distribution function")
    if kernel_law_name == "continuous" or continuous_source:
        notes.append("the floating transforms of the midpoint draws have no arithmetic enclosure")
    notes.append("score, reduction and radius arithmetic have no enclosure")

    # One binary frame g = 2**E_g with W/g in [1/2, 1). A kept category's
    # cap (M/pi)*W_kernel is the mantissa pm in [1/2, 1) of its two mantissa
    # factors times 2**E_i, with E_i a Python integer, so E_g = max E_i needs
    # no logarithm and no int64 conversion of an absolute exponent. Each
    # category exponent carries -lift once, which the cap, the frame and the
    # branch weights all share, so they are physical.
    kernel_mantissa, kernel_exponent = frexp(kernel_bound)
    caps, factors = [], []
    for ratio_value, exponent in ratios:
        if ratio_value > 0.0:
            ratio_mantissa, ratio_exponent = frexp(ratio_value)
            exponent += ratio_exponent - lift
            pm, pe = frexp(ratio_mantissa * kernel_mantissa)
            caps.append((pm, exponent + kernel_exponent + pe))
            factors.append((ratio_mantissa, exponent))
        else:
            factors.append((0.0, None))
    frame = max(exponent for _, exponent in caps)
    weight_bound = max(ldexp(pm, exponent - frame) for pm, exponent in caps)
    # Relative exponents are bounded integers once the frame is subtracted;
    # anything below -4000 gives a zero magnitude, which the draw refuses.
    category_mantissa = np.array([mantissa for mantissa, _ in factors])
    category_exponent = np.array([0 if exponent is None else max(exponent - frame, -4000)
                                  for _, exponent in factors], dtype=np.int64)

    # Observable term: fixed for the row, with signed range o_l/p(l).
    values = np.array([float(coefficient) for _, coefficient in terms], dtype=float)
    term_law = None
    ranges = np.zeros_like(values)
    if values.size and np.any(values != 0):
        with np.errstate(divide="ignore"):
            term_law = integer_categorical(np.log(np.abs(values)))
        kept = term_law.counts > 0
        ranges[kept] = values[kept] * (float(_TOTAL) / term_law.counts[kept])
    return PairLaws(final_time=final_time, active_time=float(applications.active_time),
                    signed_shift=applications.signed_shift, continuous_source=continuous_source, interval=interval,
                    category_law=law, kinds=kinds, elapsed=elapsed, category_mantissa=category_mantissa,
                    category_exponent=category_exponent, kernel=kernel, node_law=node_law, node_ratio=ratio,
                    node_phase=phase, kernel_law_name=kernel_law_name, formula=formula, frame=frame,
                    weight_bound=weight_bound, term_law=term_law, term_ranges=ranges,
                    range_bound=float(np.abs(ranges).max()) if term_law is not None else 0.0, notes=tuple(notes))


def draw_pair_rows(*, rng, clusters, repeats, laws, scalar_phase):
    """Draw and save the pair rows of a sampled observable from the Method's random stream.

    The caller admits the known bytes of the laws and rows before this call
    (``sampled_plan._row_bytes``).

    Args:
        rng: The Plan's `RandomStreams`. The rows use ``rng.method``.
        clusters (int): Rows B.
        repeats (int): Shots m per row.
        laws (PairLaws): From `pair_laws`.
        scalar_phase (tuple): ``(c_L + s, c_H)`` of phi(t, k).

    Returns:
        rows (LCHSSampledRows): The saved rows.

    Raises:
        ValueError: For nonpositive counts or a nonfinite source time.
        ApplicabilityError: For a drawn nonzero branch magnitude below the
            normal binary64 range of the common frame, a range limit of these
            binary64 rows, or a nonfinite scalar phase argument.
    """
    from .solution_error_budget import extracted_phase
    clusters, repeats = integer(clusters, "row table clusters", 1), integer(repeats, "row table repeats", 1)
    draw_state = rng.method.bit_generator.state
    branches = []
    for _ in range(2):
        category = laws.category_law.draw(rng.method, clusters)
        times = laws.elapsed[category]
        if laws.continuous_source:
            chosen = category == SOURCE
            times[chosen] = source_times(laws.signed_shift, *laws.interval,
                                         midpoint_uniforms(rng.method, int(chosen.sum())))
        branches.append((laws.kinds[category], times, category))
    nodes, kernel_ratios, kernel_phases = [], [], []
    kernel = laws.kernel
    if laws.node_law is not None:
        for _ in range(2):
            index = laws.node_law.draw(rng.method, clusters)
            nodes.append(np.asarray(kernel.nodes, dtype=float)[index])
            kernel_ratios.append(laws.node_ratio[index])
            kernel_phases.append(laws.node_phase[index])
    else:
        for _ in range(2):
            k = np.array([kernel.node(float(u)) for u in midpoint_uniforms(rng.method, clusters)])
            magnitude = np.array([abs(kernel.weight(float(value))) for value in k])
            if np.any(magnitude < _TINY):
                raise ApplicabilityError("a continuous kernel weight magnitude is below the normal binary64 range at "
                                         "a drawn node, which these binary64 pair rows cannot keep")
            nodes.append(k)
            kernel_ratios.append(magnitude)
            kernel_phases.append(_continuous_kernel_phase(kernel, k))
    magnitudes_out, phases_out = [], []
    for (_, times, category), node, kernel_ratio, kernel_phase in zip(branches, nodes, kernel_ratios,
                                                                       kernel_phases, strict=True):
        mantissa, exponent = np.frexp(kernel_ratio)
        magnitude = np.ldexp(laws.category_mantissa[category] * mantissa, laws.category_exponent[category] + exponent)
        if np.any(magnitude < _TINY):
            raise ApplicabilityError("a drawn nonzero branch weight is below the normal binary64 range of the common "
                                     "weight frame; the application and kernel magnitudes span too wide a range "
                                     "for these binary64 pair rows")
        phase = np.empty(clusters)
        for row in range(clusters):
            unit = extracted_phase(float(times[row]), float(node[row]), scalar_phase)
            phase[row] = remainder(float(kernel_phase[row]) + atan2(unit.imag, unit.real), tau)
        magnitudes_out.append(magnitude)
        phases_out.append(phase)
    if laws.term_law is not None:
        term = laws.term_law.draw(rng.method, clusters).astype(np.int64)
        signed_range = laws.term_ranges[term]
    else:
        term = np.full(clusters, -1, dtype=np.int64)
        signed_range = np.zeros(clusters)
    (kind_x, time_x, _), (kind_y, time_y, _) = branches
    weight_bound = laws.weight_bound
    return LCHSSampledRows(
        clusters=clusters, repeats=repeats, kernel_law=laws.kernel_law_name, kernel_formula_id=laws.formula,
        application_law="continuous_source" if laws.continuous_source else "discrete",
        draw_state=json_state(draw_state), scale=PhysicalScale(mantissa=0.5, exponent=laws.frame + 1),
        weight_bound=weight_bound, score_bound_q=laws.range_bound * weight_bound**2, score_bound_n=weight_bound**2,
        active_time=laws.active_time, kind_x=FrozenArray(kind_x), kind_y=FrozenArray(kind_y),
        time_x=FrozenArray(time_x), time_y=FrozenArray(time_y), node_x=FrozenArray(nodes[0]),
        node_y=FrozenArray(nodes[1]), magnitude_x=FrozenArray(magnitudes_out[0]),
        magnitude_y=FrozenArray(magnitudes_out[1]), phase_x=FrozenArray(phases_out[0]),
        phase_y=FrozenArray(phases_out[1]), term=FrozenArray(term), signed_range=FrozenArray(signed_range),
        qualification="; ".join(laws.notes))


def json_state(state):
    """Return a PCG64 state dictionary as compact sorted JSON text, as `RandomStreams.snapshot` writes it."""
    return json.dumps(state, sort_keys=True, separators=(",", ":"))
