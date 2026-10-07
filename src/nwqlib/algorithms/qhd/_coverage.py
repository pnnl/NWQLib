"""Simultaneous population-mass bounds for QHD's selected old-grid regions.

NWQLib derives this coverage bound in Proposition 49 of docs/mathematics.md
from Hoeffding and one-sided Clopper–Pearson bounds. The refinement
procedure follows Wu et al., arXiv:2605.12066v1, Sec. V.
Given earlier history and validity indicators, the S valid
positions of a level are iid from its conditional population. Its raw shot
count, grid, seam and distribution were fixed before seeing those positions.
The total failure probability alpha and maximum number H of levels are also
fixed in advance. For AL, H is max_iterations * max_levels. Dimensions may
depend on earlier rounds and include slack coordinates.

There are I = K (K + 1)/2 nonempty nonwrapping intervals per axis, hence
J = d I marginal events and N = I**d joint boxes. These families cover the
data-selected ordinary box and either stall-split side including its valley.
Half of alpha goes to each method. Their conditional failure probabilities
per level are at most alpha/(2H). Conditional averaging and a union bound
over H levels give total failure at most alpha. No independence between
the two methods or different levels is required. Taking the larger lower
bound is valid on their common simultaneous event. Each bound concerns its
own level population. Multiplying them does not give final-box coverage.

All numerical formulas use ordinary binary64 evaluation, not directed
rounding. Reporting reads the integer sufficient statistics already stored
by refinement._LevelReadout and never revisits observations.
"""

from math import exp, isfinite, log, sqrt
from sys import float_info


# Default total run failure probability of the coverage report, the
# conventional 0.05, a chosen value and not a derived one. It is fixed in
# advance, not fitted to observations. Both result types use it and split it
# equally between Hoeffding and Clopper-Pearson. failure_probability is the
# user's parameter. report(failure_probability=alpha) uses another value, and
# None omits the statistical report. See docs/ENGINEERING_CONSTANTS.md.
DEFAULT_FAILURE_PROBABILITY = 0.05


def validate_alpha(value):
    """Return a numeric failure probability in (0, 1), or the explicit None opt-out."""
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value < 1:
        raise ValueError("failure_probability must be a number strictly between 0 and 1, or None")
    return float(value)


def selected_intervals(level, k):
    """Return the kept old-grid event, including the valley of a selected split side."""
    if level.split is None:
        return level.intervals
    split = level.split
    intervals = [(0, k - 1)] * len(level.box)
    intervals[split.axis] = (0, split.valley) if split.chosen == 0 else (split.valley, k - 1)
    return tuple(intervals)


def interval_radius(d, k, horizon, alpha, n):
    """Evaluate sqrt(log(4 H J / alpha)/(2 S)), J = d K (K + 1)/2, S = n.

    For a fixed marginal interval Hoeffding gives probability at most
    2 exp(-2 S e**2) of an absolute empirical error greater than e. A union
    over J intervals makes this at most alpha/(2H) at the stated radius.
    Source: Hoeffding, JASA 58 (1963), Theorem 1 and Eq. (1.4),
    doi:10.1080/01621459.1963.10500830. The module docstring explains the
    adaptive-level union and the joint use with cp_lower.
    """
    # Keep alpha in logarithmic form: alpha/2 may underflow for an allowed
    # positive binary64 alpha. log(A/B) = log(A) - log(B) for positive A, B.
    log_family = log(alpha) - log(2) - log(horizon)
    j = d * (k * (k + 1) // 2)
    return sqrt((log(2 * j) - log_family) / (2 * n))


def cp_lower(m, n, log_beta):
    """Evaluate the one-sided Clopper-Pearson lower endpoint for M=m successes in S=n draws.

    For a fixed candidate region, M is Binomial(S, p). Its right tail is
    I_p(M, S-M+1), which increases with p. Inverting its level-beta test
    gives Pr(L_CP(M,S;beta) > p) <= beta, with L_CP=0 when M=0 and
    I_beta**(-1)(M, S-M+1) otherwise. For M=S, the tail is p**S, so
    L_CP=exp(log_beta/S). This is frequentist binomial-test inversion,
    without a prior. Source: Clopper and Pearson, Biometrika 26 (1934),
    doi:10.1093/biomet/26.4.404. SciPy's scipy.special.betaincinv implements
    the inverse regularized incomplete beta function.

    bounds_from_counts allocates beta=alpha/(2 H I**d) to every candidate
    joint box. The module docstring supplies the selection and horizon
    union bounds. Exact integer shapes and a normal target beta delimit
    the nonanalytic evaluation. They do not establish an inverse error
    bound. Large shapes or extreme tails can lose relative tail accuracy.
    A finite output in [0,1) is an ordinary binary64 approximation and may
    be subnormal or zero. An output rounded to one is unavailable because
    finite S and 0<beta<1 give a strictly positive deficit for M>0.
    """
    if m == 0:
        return 0.0, None
    if m == n:
        # Avoid forming beta, which can underflow even when its S-th root
        # is representable. This branch needs no special-function inverse.
        value = exp(log_beta / n)
    else:
        # S+1 <= 2**53 preserves both integer Beta shapes in binary64.
        # This is a representation condition, not an inverse error bound.
        if n + 1 > 2**53:
            return None, "beta_shapes_not_exactly_representable"
        if log_beta < log(float_info.min):
            return None, "beta_tail_below_normal_range"
        from scipy.special import betaincinv

        value = float(betaincinv(m, n - m + 1, exp(log_beta)))
    if not isfinite(value) or not 0 <= value <= 1:
        return None, "beta_inverse_unavailable"
    if value == 1:
        return None, "beta_lower_rounded_to_one"
    return value, None


def bounds_from_counts(*, n, axis_counts, joint_count, intervals, k, horizon, alpha):
    """Bound the mass of the selected region from its sufficient integer counts.

    The producer supplies C_j and M for the same S = n valid draws.
    Whole axes have population mass one exactly. For q other axes,
    the complement union bound and interval_radius give
    max(0, 1 - sum_j(1 - C_j/S) - q epsilon). Form the misses as integers
    before dividing, so rounded marginal masses never enter this formula.
    CP protects every candidate joint box with beta = alpha/(2 H I**d).
    Both corrections use d even when q is selected after observing counts.
    """
    radius = interval_radius(len(intervals), k, horizon, alpha, n)
    active = [j for j, interval in enumerate(intervals) if interval != (0, k - 1)]
    if not active:
        # This is the entire conditional support, independently of sampling.
        return dict(radius=radius, hoeffding_lower=1.0, cp_lower=1.0, lower_bound=1.0,
                    selected_method="full_support", cp_unavailable=None)
    misses = sum(n - axis_counts[j] for j in active)
    empirical_union_lower = max(0, n - misses) / n
    h_lower = max(0.0, empirical_union_lower - len(active) * radius)
    # N = I**d can be enormous. log(N) = d log(I) avoids forming it.
    intervals_per_axis = k * (k + 1) // 2
    log_beta = log(alpha) - log(2) - log(horizon) - len(intervals) * log(intervals_per_axis)
    c_lower, reason = cp_lower(joint_count, n, log_beta)
    use_cp = c_lower is not None and c_lower > h_lower
    return dict(radius=radius, hoeffding_lower=h_lower, cp_lower=c_lower,
                lower_bound=c_lower if use_cp else h_lower,
                selected_method="clopper_pearson" if use_cp else "hoeffding", cp_unavailable=reason)


def coverage(entries, *, k, horizon, failure_probability=DEFAULT_FAILURE_PROBABILITY):
    """Evaluate one simultaneous report from (round-or-None, completed level) entries.

    The callers supply the configured horizon, never the observed number
    of levels. A completed counts level has positive valid_count by
    RefinementLevel._domain. Exact levels supply no sampling statement.
    CP unavailability keeps the original alpha/2 Hoeffding allocation.
    Reallocating it after inspecting the counts would change the event.
    """
    alpha = validate_alpha(failure_probability)
    if alpha is None:
        return None
    rows = []
    for round_number, level in entries:
        if level.valid_count is None:
            continue
        intervals = selected_intervals(level, k)
        n, axes, joint = level.valid_count, level.region_axis_counts, level.region_count
        bounds = bounds_from_counts(n=n, axis_counts=axes, joint_count=joint, intervals=intervals,
                                    k=k, horizon=horizon, alpha=alpha)
        rows.append(dict(round=round_number, level=level.level, dimension=len(intervals),
                         intervals=[list(item) for item in intervals],
                         event="split_region" if level.split is not None else "interval_box",
                         result_id=level.result_id, marginal_events=len(intervals) * (k * (k + 1) // 2),
                         valid_count=n, axis_counts=list(axes), joint_count=joint,
                         empirical_mass=joint / n, **bounds))
    if not rows:
        return None
    return dict(failure_probability=alpha, horizon=horizon,
                budget_fractions=dict(hoeffding=0.5, clopper_pearson=0.5),
                numerical_evaluation="binary64", levels=rows,
                meaning="The selected region mass bounds concern each level's backend-sampled distribution "
                        "conditioned on valid decoding, including all slack coordinates. They hold simultaneously "
                        "over the configured "
                        "run horizon under Proposition 49's conditional iid sampling assumptions, with the "
                        "failure probability fixed before inspecting the counts. The two methods each "
                        "use half the failure budget. Numerical values are binary64 approximations, not "
                        "directed-rounding enclosures. They do not give an optimizer-retention or convergence "
                        "probability, and they cannot be multiplied into final-box coverage.")


def coverage_lines(confidence):
    """Format the same evaluated bounds for standalone and AL summaries."""
    if confidence is None:
        return []
    alpha = confidence["failure_probability"]
    label = "95%" if alpha == DEFAULT_FAILURE_PROBABILITY else f"1 - {alpha!r}"
    lines = [f"Selected region mass in each level's backend-sampled distribution conditioned on valid decoding: "
             f"simultaneous confidence {label} "
             f"(horizon {confidence['horizon']}, failure probability {alpha!r}), "
             "conditional on the sampling premises of Proposition 49."]
    for row in confidence["levels"]:
        where = (f"Level {row['level']}" if row["round"] is None
                 else f"Round {row['round']}, level {row['level']}")
        if row["selected_method"] == "full_support":
            bound = "1 (full valid-grid support)"
        else:
            # repr preserves a binary64 value below one instead of rounding
            # it to a displayed one. A computed endpoint remains approximate.
            bound = f"approximately {row['lower_bound']!r}"
            if row["lower_bound"] == 1.0:
                bound += " (rounded)"
        lines.append(f"{where} selected region mass lower bound: {bound}. {row['valid_count']} valid draws.")
    return lines
