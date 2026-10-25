"""Pilot/main shot allocation for the sampled pencil of a fixed GCiM basis.

The sampled pencil ``H = H0 + alpha S`` of a fixed ordered basis is measured
in two stages. A pilot gives every setting the same number of shots, up to
one, and its solve gives the lowest Ritz value E with coefficient vector c.
The main stage then takes fresh shots in proportions chosen from the pilot,
and only the main counts form the final H and S. These functions hold the
arithmetic of that decision on Python numbers. They read the settings
``(name, i, j, quadrature, group)`` of ``fixed_basis.sampled_settings`` in
their given order, the pooled pilot moments of each setting that
``fixed_basis.pooled_statistics`` returns in ``pools``, and the offset-free
Ritz values ``ProjectedPencil.offset_free_eigenvalues`` and the coefficient
vector of the pilot ``ProjectedPencil``. The offset-free values are those
the solve computed from ``(H0, S)`` before alpha is added. They make no
measurement, I/O or random draw.

Energy sensitivity. Let T be the fixed coordinate map that the overlap
cutoff selects, ``H_r = T^dagger H T`` and ``S_r = T^dagger S T`` with S_r
positive definite, and E a simple Ritz value with ``H_r z = E S_r z``,
``z^dagger S_r z = 1`` and ``c = T z``. Differentiating at fixed T and
multiplying by z^dagger from the left removes dz through the reduced
relation ``z^dagger H_r = E z^dagger S_r``, so

    dE = c^dagger (dH - E dS) c = c^dagger (dH0 - E0 dS) c,   E0 = E - alpha.

The proof does not need ``Hc = ESc`` in the full coordinates, which fails
when the cutoff drops directions, and the ``-E dS`` term remains when
``S_r = I``. The formula is local: a rotation of the cutoff eigenvectors, a
change of kept rank or of basis, or a degenerate E leaves it without
support. For i < j and ``w = conj(c_i) c_j``, the entry and its conjugate
together contribute ``2 Re(w dM_ij) = 2 Re(w) dM_ij,R - 2 Im(w) dM_ij,I``,
so the sensitivity weights are ``kappa_R = 2 Re w``, ``kappa_I = -2 Im w`` and
``kappa_ii = |c_i|**2``. The coefficient of ``Im w`` is -2 while the joint
state measures ``Im <phi_i|P|phi_j>`` with a positive ancilla-Y sign
(``fixed_basis._sampled_construction``). The two signs are separate facts,
and ``kappa_I`` itself can have either sign.

Influence of one shot. With E0, c and the overlap weights omega frozen, one
shot of setting s changes E by ``l_s = kappa_s (y_g - E0 omega_g x)``, where
y_g is the group's coefficient-weighted parity and x the ancilla sign. The
overlap comes from group zero only (``fixed_basis.pooled_statistics``), so
``omega_0 = 1`` and ``omega_g = 0`` otherwise, and a diagonal setting has no x
term because ``S_ii = 1``. Its per-shot variance is
``v_s = kappa_s**2 (Var y + E0**2 omega_g**2 Var x - 2 E0 omega_g Cov(y, x))``.
For independent settings with N_s shots the local energy variance is
``V_local = sum_s v_s/N_s``. Adding ``delta I`` to the operator moves E and
alpha together and leaves E0, c and the moments unchanged. The functions
take E0 as the lowest offset-free Ritz value e0 that the solve computed,
not as ``E - alpha`` from the physical E, which is rounded once after alpha
is added: for ``alpha = 2**53`` the offset-free values 1/4 and 1/2 both
round to the physical value ``2**53``, and subtracting alpha cannot recover
them. With e0 from the computed offset-free solve, the influence variances,
the exact-tie decision, the rounded gap ``e1 - e0`` of the two lowest
offset-free values and therefore the allocation depend on the represented
H0 and S, the computed offset-free solve, the coefficients, the pilot
moments, the setting order and the budgets. Changing only the identity
coefficient changes none of them, provided the physical report stays valid.
A finite identity coefficient does not ensure a finite physical E.

Allocation with unit shot cost. Minimizing ``sum_s v_s/N_s`` subject to
``sum_s N_s = C`` gives ``N_s = C sqrt(v_s)/sum_t sqrt(v_t)`` and
``V_min = (sum_s sqrt(v_s))**2/C`` from ``-v_s/N_s**2 + lambda = 0``. With the
two-shot floor the continuous solution is ``N_s = max(2, sqrt(v_s/lambda))``
with ``sum_s N_s = C``. Two shots make the sample variance of the main stage
defined. They do not make it precise. ``floored_allocation`` realizes this
solution in integer arithmetic by a deterministic rounding that spends the
budget exactly. Its counts need not be the integer optimum.

Nakamura et al., arXiv:2311.07893v2, Sec. II.2, Eqs. (7)-(8), update the
measurement strategy with the subspace solution. The weights, the pair and
group covariance and the integer rules here are NWQLib's own derivation.
The rules differ from the Lanczos float split and largest-remainder
allocation (``lanczos.numerical._allocate``), so the two are not shared.

None of these quantities is a confidence bound. The pilot plug-in, the
cutoff, near-degeneracy and hardware bias are outside the local model, and
unbiased main matrix entries do not make the lowest Ritz value unbiased.
"""

from __future__ import annotations

from fractions import Fraction
from math import fsum, isfinite, sqrt


def sensitivity_weights(settings, coefficients):
    """Return the weight ``kappa_s`` of each setting in ``dE = sum_s kappa_s dM_s``.

    ``coefficients`` is the S-normalized lowest Ritz vector c, as
    ``ProjectedPencil.coefficients`` gives it, of Python complex numbers or
    ``Complex128`` records. With ``w = conj(c_i) c_j``, the weight is
    ``2 Re w`` for a real setting and ``-2 Im w`` for an imaginary one of
    ``i < j``, and ``Re w = |c_i|**2`` for a diagonal setting, whose
    perturbation ``dM_ii`` is real. ``M`` is H0 or S, and an S entry enters
    E with the further factor ``-E0`` (module docstring). w does not depend
    on the global phase of c.
    """
    c = [complex(z.real, z.imag) for z in coefficients]
    weights = []
    for _, i, j, quadrature, _ in settings:
        w = c[i].conjugate() * c[j]
        weights.append(w.real if i == j else 2 * w.real if quadrature == "real" else -2 * w.imag)
    return tuple(weights)


def influence_variances(settings, pools, weights, e0):
    """Return the per-shot energy influence variance ``v_s`` of each setting from its pooled pilot moments.

    ``e0`` is the lowest offset-free Ritz value of the pilot solve
    (``ProjectedPencil.offset_free_eigenvalues[0]``), so the identity
    coefficient never enters. ``weights`` are the ``sensitivity_weights`` of
    ``settings``. ``pools`` maps the key ``(i, j, quadrature, group)`` of
    each setting to its pooled pilot ``(N, V_Y, V_X, C_XY)``, as
    ``fixed_basis.pooled_statistics`` fills it: N the setting's total pilot
    shots and the variances and covariance those of its sample means, with
    repeated measurements pooled by squared shot weights. Multiplying by N
    restores per-shot values once:

        v_s = N kappa_s**2 (V_Y + E0**2 omega_g**2 V_X - 2 E0 omega_g C_XY)

    with ``omega_0 = 1``, ``omega_g = 0`` for the other groups, and only the
    Y term on a diagonal setting. A later stage with n shots then has
    variance ``v_s/n`` without a second division by N. The bracket is
    summed with ``fsum``, so the exact cancellation of ``y = x`` with
    ``E0 = 1`` gives zero.

    Each setting reads only what its role needs: ``V_Y`` on a diagonal or
    a group other than zero, and ``V_Y``, ``V_X`` and ``C_XY`` on group zero
    of an off-diagonal pair. Variances and covariance have different sign
    domains. A finite ``C_XY`` of either sign is a legal centered cross
    product and enters with its sign. The value is None when there is no
    usable estimate: a needed field is None (a one-shot measurement) or not
    finite, or a pooled ``V_Y`` or ``V_X`` estimate is negative. On group
    zero it is None as well when a term of the bracket is not finite or
    ``fsum`` overflows, which needs moments or an e0 of order 1e154; the
    other roles reach the same fallback through a nonfinite product. None
    is never replaced by zero, and nothing is clipped. The raw values stay
    in the ``GroupMoment`` records. A negative ``v_s`` from the rounding of
    legal moments is returned as computed. ``pilot_fallback_reason`` treats
    None and a negative or nonfinite ``v_s`` as unusable. No positivity or
    correlation consistency of the moment triple is tested.

    Raises:
        ValueError: If a setting has no pilot moment. A missing pilot
            measurement is an acquisition failure, not a fallback case.
    """
    variances = []
    for (name, i, j, quadrature, g), kappa in zip(settings, weights, strict=True):
        if (i, j, quadrature, g) not in pools:
            raise ValueError(f"pencil sampling has no pilot moment for setting {name}")
        n, vy, vx, cov = pools[i, j, quadrature, g]
        overlap = i != j and g == 0
        needed = (vy, vx, cov) if overlap else (vy,)
        if any(v is None or not isfinite(v) for v in needed) or vy < 0 or (overlap and vx < 0):
            variances.append(None)
        elif overlap:
            terms = (vy, e0 * e0 * vx, -2 * e0 * cov)
            try:
                bracket = fsum(terms) if all(isfinite(t) for t in terms) else None
            except OverflowError:
                bracket = None
            variances.append(None if bracket is None else n * kappa * kappa * bracket)
        else:
            variances.append(n * kappa * kappa * vy)
    return tuple(variances)


def pilot_split(total_shots, pilot_fraction, active):
    """Return ``(pilot, main)`` shots of an integer budget B for m active settings.

    m = 0 needs no measurement and returns ``(0, 0)``, and the fraction does
    not apply. Otherwise
    ``B_p = min(B - 2m, max(2m, floor(val(f) B)))`` and ``B_main = B - B_p``,
    where ``val(f)`` is the exact rational ``p/q = f.as_integer_ratio()`` of
    the stored binary64 f, and the floor is the integer ``p*B // q``. Forming
    ``f*B`` in floating point first could round across an integer, as
    ``0.3*100 == 30.0`` while ``floor(val(0.3)*100) == 29``. Both stages then
    have at least 2m shots. ``total_shots`` is a Python int, and f is a
    finite binary64 in (0, 1), which the calling option checks.

    Raises:
        ValueError: If m > 0 and B < 4m, before any measurement.
    """
    if active == 0:
        return 0, 0
    if total_shots < 4 * active:
        raise ValueError(f"PencilSampling.total_shots must be at least 4m = {4 * active} shots for m = {active} "
                         f"active settings, two pilot and two main shots each, but is {total_shots}")
    p, q = pilot_fraction.as_integer_ratio()
    pilot = min(total_shots - 2 * active, max(2 * active, p * total_shots // q))
    return pilot, total_shots - pilot


def uniform_allocation(shots, active):
    """Return the counts ``q + 1`` for the first r settings and q for the rest, ``q, r = divmod(shots, m)``.

    The pilot and a fallback main stage use it in the order of the
    settings. It spends ``shots`` exactly, and every count is at least 2 when
    ``shots >= 2m``, which ``pilot_split`` guarantees for both stages.
    """
    if active == 0:
        return ()
    q, r = divmod(shots, active)
    return (q + 1,) * r + (q,) * (active - r)


def floored_allocation(shots, variances):
    """Return integer main counts that round the continuous minimizer of ``sum_s v_s/N_s`` with the two-shot floor.

    ``variances`` are the pilot ``v_s`` in setting order, finite and
    nonnegative with at least one positive. ``pilot_fallback_reason`` has
    excluded the other cases. ``shots`` is the main budget ``B_main >= 2m``
    as a Python int, as in ``pilot_split``. A NumPy integer would overflow
    in the integer products below.

    The weights ``u_s = sqrt(v_s)`` are binary64 values. Each is the exact
    rational ``as_integer_ratio()``, whose denominator is a power of two, so
    scaling by the largest denominator D gives integers ``W_s`` with
    ``W_s/D = u_s`` exactly. All later steps are integer or exact rational,
    so budgets beyond 2**53 cannot be overspent.

    Settings with ``v_s = 0`` start on the floor. Over the remaining active
    set A with ``C_A = B_main - 2|F|`` for the floor set F, the candidate is
    ``C_A W_s/sum_A W``. In ascending u, a setting whose candidate is below
    2 (``C_A W_s < 2 sum_A W``) joins F and C_A and the sum are updated.
    Removing one lowers the remaining candidates, and once the smallest
    reaches 2 all others do, so the sweep stops there and each setting
    leaves at most once. ``B_main >= 2m`` keeps the last active setting.
    This is the continuous solution with the floor. Each active count is
    then ``(C_A W_s) // sum_A W``, at least 2, and the remainder
    ``R = B_main - sum_s n_s`` satisfies ``0 <= R < |A|``. The R active
    settings with the largest decrease ``v_s/(n_s(n_s + 1))`` of one extra
    shot receive one shot each, ties in setting order. The decreases are
    compared as exact rationals of the stored v. Every count is at least 2
    and the counts sum to B_main. The work is O(m log m) and does not loop
    over shots.

    Raises:
        ValueError: If ``shots < 2m`` or every variance is zero, where this
            allocation is not defined.
    """
    m = len(variances)
    if shots < 2 * m or not any(variances):
        raise ValueError("the floored allocation needs B_main >= 2m and a positive pilot variance")
    ratios = [sqrt(v).as_integer_ratio() for v in variances]
    scale = max(q for _, q in ratios)
    weights = [p * (scale // q) for p, q in ratios]
    active = sorted((s for s in range(m) if weights[s]), key=weights.__getitem__)
    rest = shots - 2 * (m - len(active))
    total = sum(weights[s] for s in active)
    first = 0
    while rest * weights[active[first]] < 2 * total:
        rest -= 2
        total -= weights[active[first]]
        first += 1
    counts = [2] * m
    for s in active[first:]:
        counts[s] = rest * weights[s] // total
    extra = sorted(active[first:], key=lambda s: (-Fraction(variances[s]) / (counts[s] * (counts[s] + 1)), s))
    for s in extra[:shots - sum(counts)]:
        counts[s] += 1
    return tuple(counts)


def pilot_fallback_reason(failure_reason, eigenvalues, coefficients, variances):
    """Return why the main stage takes the uniform allocation, or None when the pilot allocation applies.

    The arguments are the pilot pencil's ``failure_reason``, its ascending
    ``offset_free_eigenvalues`` (one or two values, before the identity
    coefficient is added) and lowest ``coefficients``, and the
    ``influence_variances`` of its m >= 1 settings. The checks, in order:

    - ``pilot_solve_failed``: the solver rejected the pilot pencil or found
      no Ritz pair. Its own code stays in the pencil's ``failure_reason``.
      The other arguments are not read. A sampled Ritz value outside the
      operator's enclosure is a diagnostic of the pencil, not a solver
      failure, and gives no reason.
    - ``pilot_solution_nonfinite``: a Ritz value or coefficient is not finite.
    - ``pilot_lowest_ritz_degenerate``: the two offset-free values are equal.
      Physical values that a large identity coefficient rounds together are
      therefore no tie. A rank-one pencil has no second value. A small
      nonzero gap is not a reason, and no gap tolerance is applied.
    - ``pilot_variance_unusable``: a ``v_s`` is None (a required moment
      field is None or not finite, or a pooled variance estimate is
      negative), not finite, or negative. A negative covariance alone is
      not a reason. A missing moment row raises in ``influence_variances``
      instead.
    - ``pilot_variances_all_zero``: every ``v_s`` is zero. Zero estimates do
      not mean zero uncertainty, and the floored allocation has no positive
      weight to follow.

    With a reason, the main stage measures every setting afresh with
    ``uniform_allocation``. A fallback does not stand in for a missing
    measurement.
    """
    if failure_reason is not None:
        return "pilot_solve_failed"
    if not all(isfinite(x) for x in eigenvalues) or not all(
            isfinite(z.real) and isfinite(z.imag) for z in coefficients):
        return "pilot_solution_nonfinite"
    if len(eigenvalues) > 1 and eigenvalues[0] == eigenvalues[1]:
        return "pilot_lowest_ritz_degenerate"
    if not all(_usable(v) for v in variances):
        return "pilot_variance_unusable"
    if not any(variances):
        return "pilot_variances_all_zero"
    return None


def local_variance(variances, counts):
    """Return ``V_local = sum_s v_s/N_s`` of the frozen pilot model for counts N_s, or None for an unusable v_s.

    A ``v_s`` that is None, not finite or negative makes the value None, by
    the same test that makes ``pilot_fallback_reason`` reject it. It is the
    local model's prediction for independent main settings, not a confidence
    bound on the final Ritz value. All-zero pilot estimates give zero, which
    is not a claim of zero uncertainty.
    """
    if not all(_usable(v) for v in variances):
        return None
    return fsum(v / n for v, n in zip(variances, counts, strict=True))


def _usable(v):
    """Whether a per-shot influence variance is an estimate: not None, finite and nonnegative."""
    return v is not None and isfinite(v) and v >= 0
