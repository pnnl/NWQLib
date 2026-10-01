"""QPE numerical estimators in their stated radians/energy conventions.

These functions consume admitted scalar data only. They do not prepare circuits,
submit observations, redraw schedules, or obtain a reference spectrum.

Conventions shared by every kernel. A Hamiltonian H is queried through
U = exp(-i tau H), so an eigenvalue E contributes the signal exp(-i p tau E)
at evolution multiplier p and the unitary eigenphase -tau E. Unitary input reports
the principal eigenphase phi of U v = exp(i phi) v. The Hadamard-test signal
is z_p = <psi|U^p|psi>, and an ancilla phase shift s turns the measured
ancilla mean into Re(exp(i s) z_p).

Equation, theorem and page numbers below refer to these paper versions:

- QCELS: Ding and Lin, arXiv:2211.11973v2.
- SPE: Wan, Berta and Campbell, arXiv:2110.12071v2.
- RFE: Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3.
- RWPE: Granade and Wiebe, arXiv:2208.04526v1.

Each kernel docstring states which step follows its paper, where NWQLib
departs from it and what the departure means for the paper's guarantee. The
family guide docs/algorithms/qpe.md collects these locations in its source
map. Kernels work with phases in radians. Their returned value is the
energy in operator units for a Hamiltonian or the principal eigenphase in
radians for a unitary. records.public_estimate leaves that value unchanged,
derives the phase in turns and, for Eigenphase output, rescales the interval
to turns.
"""

from dataclasses import dataclass
from math import asin, e, floor, gcd, isfinite, log, pi, sqrt

import numpy as np
from scipy.linalg import schur
from scipy.linalg.lapack import dsyevd, zheevd
from .records import QCELSFit


# Nominal level of the RWPE Gaussian model interval, registered in
# docs/ENGINEERING_CONSTANTS.md. QPEInterval.level stores the same value in
# saved Results. The level belongs to the moment-matched Gaussian model, so it
# gives no frequentist coverage and does not certify the single-phase
# approximation for mixed spectra. Revisit together with QPEInterval.level.
QPE_INTERVAL_CONFIDENCE = .95

# Objective evaluations spent refining each discrete local minimum of the
# QCELS grid by golden-section search. Two go to the initial interior points
# and one to each later iteration. The 62 iterations shrink the two-cell
# bracket by the factor ((sqrt(5)-1)/2)**62, about 1.1e-13. At the default
# 4096-point grid the final width is about 3.4e-16, below the binary64
# spacing 4.4e-16 near pi. Because the count
# is fixed and has no optimizer-dependent stopping rule, the number of
# evaluations is known before the search starts, which lets the analysis
# work law (method._analysis_size) and the saved-fit check
# (records.QPEAnalysis.validate_plan) bound it.
# Registered in docs/ENGINEERING_CONSTANTS.md (QCELS finite complex search).
# Revisit if the grid default or the precision changes.
QCELS_BRACKET_EVALUATIONS = 64


@dataclass(frozen=True)
class NumericalEstimate:
    """Scalar output of one estimator kernel, before conversion to a Result.

    A kernel returns ``value=None`` when the data do not identify an estimate,
    for example a flat objective, and names the reason in ``stop_reason``.
    records.public_estimate derives the phase in turns and the output-frame
    interval from these fields. No field carries a coverage or ground-state
    identification claim.

    Attributes:
        value: Energy in operator units for Hamiltonian input, or principal
            eigenphase in radians for unitary input. None when unidentified.
        phase: Principal eigenphase of U in radians. For a Hamiltonian this is -tau*value.
        interval_low: Lower endpoint in the frame of ``value``, unwrapped.
        interval_high: Upper endpoint in the frame of ``value``, unwrapped.
        interval_method: Name of the estimator-specific interval construction.
        interpretation: Scope statement that travels with the interval.
        fit: QCELS finite-search record. None for the other estimators.
        stop_reason: Reason an estimate is unavailable or aliased, when one applies.
    """

    value: float | None
    phase: float | None
    interval_low: float | None
    interval_high: float | None
    interval_method: str | None
    interpretation: str | None
    fit: QCELSFit | None = None
    stop_reason: str | None = None


def angle_principal(angle):
    """Map an angle in radians to the principal branch [-pi, pi)."""
    return float((angle + pi) % (2*pi) - pi)


def _signal_vanishes(total, terms, magnitude, mean_window, weight):
    """Return whether a summed complex signal is zero at the resolution of its data.

    ``total`` is the computed sum of ``terms`` complex values w_k*z_k, with
    positive integer weights w_k whose sum is ``weight`` and with
    ``magnitude`` the computed sum of w_k*|z_k|. QCELS and RFE call this to
    decide which powers carry a signal.

    ``mean_window=None`` marks count data. A count mean is an exact rational
    rounded once, and it is zero only when the zero and one outcomes are
    equally frequent, so only an exact binary64 zero counts as no signal. A
    power repeated in the input sums several count means, and an exact
    cancellation among them can leave a roundoff residue that then counts as
    a signal. RFE's own plan pools repeated draws, so it passes one count
    mean per power.

    Otherwise the samples are exact means, each within ``mean_window`` of its
    exact-arithmetic value (the window that admitted it, see
    records.bounded_mean). Their real and imaginary parts then put each z_k
    within sqrt(2)*mean_window of its exact value, and the exact sum within
    sqrt(2)*mean_window*weight of the sum of the computed samples. Forming
    that sum rounds each weighted term once and adds the terms in
    ``terms - 1`` complex additions. Complex addition has relative error at
    most u (Higham, Accuracy and Stability of Numerical Algorithms, 2nd ed.,
    SIAM 2002, doi:10.1137/1.9780898718027, Lemma 3.5), and so has
    multiplication by a real weight, by the standard model (2.4) applied to
    each part. The rounding is therefore at most gamma_terms times the exact
    sum of w_k*|z_k| (Lemma 3.1, as in Eq. (3.5)). gamma_(2*terms) also
    covers the rounding of the computed moduli and of their sum. A total
    inside the sum of the two bounds is consistent with an exact zero, and
    the signal counts as absent.
    Revisit with the admission windows (docs/ENGINEERING_CONSTANTS.md).
    """
    if mean_window is None:
        return total == 0
    from nwqlib._validation import UNIT_ROUNDOFF

    rounding = 2*terms*UNIT_ROUNDOFF
    return abs(total) <= sqrt(2)*mean_window*weight + rounding/(1 - rounding)*magnitude


def qcels_schedule_size(options, tau):
    """Return (number of positive powers, maximum admitted power) for a QCELS schedule.

    With a single positive power the drawn schedule uses power 1, and the
    caller _schedule_size records that maximum.

    Powers are dimensionless. ``max_time=None`` selects maximum power 10,
    independent of units. A supplied ``max_time`` selects floor(max_time/tau)
    for a Hamiltonian and floor(max_time) for unitary input. Planning and
    admission call this same preview, so the admitted size bounds the
    schedule that planned_power_schedule later draws.

    Departure from Ding and Lin (arXiv:2211.11973v2). Their data set, Eq. (6)
    (p. 8), uses the consecutive times t_n = n*tau for n = 0, ..., N-1, and
    their Theorem 1 is stated for that data set. When the maximum power P
    exceeds num_times, planned_power_schedule instead takes num_times integer
    powers spread over 1..P, so the theorem does not describe that schedule.
    The schedule always contains powers 0 and 1, so the spacing alone cannot
    make the objective exactly periodic. Exact aliasing needs a vanishing
    signal at some powers (see qcels).

    The spread schedule can make the objective nearly periodic. Its positive
    powers are p_j = floor(1 + j*gbar) with gbar = (P - 1)/(num_times - 1).
    At theta + 2*pi/gbar each term exp(i*p_j*theta) turns by
    2*pi*(1 - f_j)/gbar, where f_j is the fractional part of 1 + j*gbar, and
    the power-0 term does not turn. Every turn lies in [0, 2*pi/gbar], about
    one radian for gbar near 6, so the rotated terms stay nearly aligned and
    the objective there keeps most of the depth of the minimum at theta.
    With a mixed spectrum, finite shots or evolution error, such a replica
    minimum can win the fit and move the energy by 2*pi/(gbar*tau). Nothing
    in the Result reports this. On the H4 problem of the eigenvalue example
    notebook (Hartree-Fock reference, tau = 0.2/Ha, num_times = 32),
    max_time = 38/Ha gives P = 190, gbar = 6.10 and an energy 5.156 Ha above
    the ground energy, next to the replica spacing 2*pi/(gbar*tau) =
    5.153 Ha, while 36 and 40/Ha stay within 7 mHa. Consecutive powers 0..P,
    which planned_power_schedule uses when num_times >= P, have no such near
    period.
    """
    if options.max_time is None:
        maximum = 10
    else:
        ratio = options.max_time / (1. if tau is None else tau)
        if not isfinite(ratio):
            raise ValueError("QCELS time schedule exceeds finite scalar range")
        maximum = floor(ratio)
    if maximum < 1:
        raise ValueError("QCELS max_time must admit at least one positive integer power")
    if maximum > options.max_power:
        raise ValueError(f"selected QPE power {maximum} exceeds max_power={options.max_power}")
    return min(maximum, options.num_times), maximum


def qcels_grid_size(requested, span):
    """Return the effective QCELS grid size G = max(requested, 8*span + 1).

    ``span`` is the largest minus the smallest selected power. The reduced
    objective contains |sum_p z_p*exp(i*p*theta)|**2, whose fastest term is
    exp(i*span*theta) with period 2*pi/span, so 8*span + 1 points on
    [-pi, pi) put at least eight points in each period. The factor eight is a
    search choice registered in docs/ENGINEERING_CONSTANTS.md (QCELS finite
    complex search), not a theorem. Ding and Lin, arXiv:2211.11973v2,
    Sec. II.2 (p. 10), suggest a uniform grid of up to ceil(T_max) points,
    with T_max = N*tau their maximal time, which is about one point per
    period, followed by gradient ascent from each point.
    """
    return max(requested, 8 * span + 1)


def planned_power_schedule(options, tau, rng):
    """Return the planned (power, ancilla phase) settings in acquisition order.

    Each entry is one logical quadrature setting. For QCELS, SPE and RFE the
    power is an integer multiplier p of U, and each power appears twice, with
    quadrature phase 0 (Re z_p) and -pi/2 (Im z_p), read as the ancilla X and
    Y of one trajectory point or measured by a sampled Hadamard test, see
    method._quantum_program. Random draws use the Method RNG stream once at
    planning, so execution and reanalysis use the same schedule.

    - QCELS: power 0 and the positive powers up to the maximum P of
      qcels_schedule_size, all of 1..P when num_times >= P, otherwise the
      num_times values of np.linspace(1, P, num_times) truncated to
      integers.
    - RFE: num_samples powers drawn uniformly from {0, ..., K-1}, as in
      Kshirsagar, Katabarwa and Johnson arXiv:2209.11322v3, Algorithm 1
      (p. 5).
    - SPE: num_samples odd frequencies 2j+1 drawn with probability
      |F_(2j+1)|/S, S = sum_j |F_(2j+1)| (spe_fourier_weights). spe explains
      how this differs from the paper's sampling over all of S1.
    - RWPE: the relative evolution times 1/sigma_k for
      k = 0, ..., max(1, max_steps)-1, with sigma_k = rwpe_scale(prior_std, k).
      The phase entry 0 is a placeholder. The actual feedback depends on
      earlier outcomes and is chosen by rwpe_choose during execution.

    The caller admits the schedule size (method._schedule_size) before this
    function materializes it.
    """
    if options.estimator == "qcels":
        count, max_power = qcels_schedule_size(options, tau)
        nonzero = (range(1, max_power+1) if count >= max_power else
                   np.unique(np.linspace(1, max_power, count, dtype=int)))
        return tuple((int(power), phase) for power in (0, *nonzero) for phase in (0., -pi/2))
    if options.estimator == "rfe":
        powers = rng.integers(0, options.num_frequencies, size=options.num_samples)
        return tuple((int(power), phase) for power in powers
                     for phase in (0., -pi/2))
    if options.estimator == "spe":
        coefficients = spe_fourier_weights(options.fourier_degree, options.filter_beta)
        powers = 2*rng.choice(len(coefficients), options.num_samples,
                              p=coefficients/np.sum(coefficients)) + 1
        return tuple((int(power), phase) for power in powers for phase in (0., -pi/2))
    return tuple((1/rwpe_scale(options.prior_std, step), 0.)
                 for step in range(max(1, options.max_steps)))


def qcels(powers, samples, *, tau, grid_size, mean_window=None):
    """Fit one complex exponential to the Hadamard signal and return theta/tau.

    Model. The data are z_p = <psi|U^p|psi> at integer powers p, with
    U = exp(-i*tau*H). QCELS fits z_p ~ a*exp(-i*p*theta) by minimizing
    L(a, theta) = mean_p |z_p - a*exp(-i*p*theta)|**2 over complex a and real
    theta. For an eigenstate with energy E the minimum is at theta = E*tau.

    Paper locations (Ding and Lin, arXiv:2211.11973v2). Eq. (2) (p. 4) is the
    objective L(r, theta). Eq. (9) (p. 9) writes it for the times
    t_n = n*tau, so their energy variable theta equals this theta/tau. For
    fixed theta the optimal amplitude is a(theta) = mean_p z_p*exp(i*p*theta),
    Eq. (11), and the reduced objective is Eq. (12) (both p. 9). Unitary input
    fits theta = -phi for eigenphase phi. Shifting every power by the same
    integer multiplies a by a phase and leaves theta unchanged, so the powers
    are centered at their minimum before evaluation.

    Search. Sec. II.2 (p. 10) proposes a grid of up to ceil(T_max) points
    with gradient ascent from each. NWQLib evaluates the qcels_grid_size grid
    on [-pi, pi) and refines every discrete local minimum by golden-section
    search inside its two neighboring cells, with a fixed
    QCELS_BRACKET_EVALUATIONS objective evaluations per minimum. The finite
    grid and bracket search is neither a global-optimum nor a coverage bound.

    Validity. The accuracy results, Theorems 1 and 2 (p. 14), assume
    p0 > 0.71, where p0 = |<psi|psi_0>|**2 is the squared overlap of the
    prepared state with the target eigenvector (defined on p. 4).
    Sec. IV (p. 15) treats smaller overlaps by first applying a Fourier
    filter to the data, which is not implemented here. NWQLib does not
    check p0. This kernel is one level of QCELS on one schedule. Algorithm 1
    (p. 13), the multi-level variant covered by Theorem 2, doubles the time
    step between levels as in Eq. (35) (p. 14) and is not implemented.

    Aliasing. When the powers carrying a nonzero signal all differ by
    multiples of g > 1, the objective has period 2*pi/g in theta. The result
    then reports the representative that the grid search selects and names
    the aliasing in its stop reason. A power carries a signal when the sum of
    its samples is nonzero at the data's resolution (_signal_vanishes). Exact
    data that vanish in exact arithmetic arrive as roundoff, and an exact
    comparison with zero would treat that roundoff as signal. A caller avoids
    that by passing ``mean_window``, as ``method._static_estimate`` does.
    A nearly periodic objective of a spread schedule is not detected
    (qcels_schedule_size).

    Args:
        powers: Integer powers p. Repeated powers keep their multiplicity.
        samples: Complex signal z_p, one per power.
        tau: Positive Hamiltonian time step, or None for unitary input.
        grid_size: Requested minimum number of grid cells.
        mean_window: None for count data. For exact data, the largest error
            of one real ancilla mean, the window that admitted it.

    Returns:
        NumericalEstimate with value theta/tau (Hamiltonian) or the principal
        phi (unitary), no interval, and the QCELSFit search record. A flat
        objective returns no estimate.
    """
    powers = tuple(powers)
    if not powers or any(isinstance(p, (bool, np.bool_)) or int(p) != p for p in powers):
        raise ValueError("QCELS requires nonempty integer powers")
    if tau is not None and (not isfinite(tau) or tau <= 0):
        raise ValueError("QCELS tau must be positive and finite")
    samples = np.asarray(samples, dtype=complex)
    if samples.shape != (len(powers),) or not np.isfinite(samples).all():
        raise ValueError("QCELS requires one finite complex sample per power")
    minimum = min(powers)
    centered = np.asarray([int(p - minimum) for p in powers], dtype=float)
    size = qcels_grid_size(grid_size, int(max(powers) - minimum))
    evaluations = 0

    def objective(theta):
        """Return the reduced objective of Eq. (12) at theta and count the call.

        With w_p = z_p*exp(i*p*theta) and centered p,
        |z_p - a*exp(-i*p*theta)| = |w_p - a|, so the minimum over a is the
        variance mean_p |w_p - mean(w)|**2. Forming the residuals directly
        avoids the cancellation in mean|z|**2 - |a|**2 near a perfect fit.
        """
        nonlocal evaluations
        evaluations += 1
        rotating = samples * np.exp(1j * centered * theta)
        residual = rotating - np.mean(rotating)
        return float(np.mean(residual.real**2 + residual.imag**2))

    # Repeated powers keep their actual multiplicity. If their aggregate
    # Fourier signal has only one nonzero coefficient, the objective is flat.
    coefficients = {}
    for power, sample in zip(powers, samples, strict=True):
        total, terms, magnitude = coefficients.get(power, (0j, 0, 0.))
        coefficients[power] = (total + sample, terms + 1, magnitude + abs(sample))
    support = tuple(power for power, (total, terms, magnitude) in coefficients.items()
                    if not _signal_vanishes(total, terms, magnitude, mean_window, terms))
    if len(support) <= 1:
        return NumericalEstimate(None, None, None, None, None, None,
            QCELSFit(effective_grid=size, evaluations=1, residual=objective(0.)),
            "flat complex-fit objective; eigenvalue/phase unavailable")

    grid = np.linspace(-pi, pi, size, endpoint=False)
    scores = np.empty(size)
    for index, theta in enumerate(grid):
        scores[index] = objective(theta)
    best = int(np.argmin(scores))
    theta_hat, residual_hat = float(grid[best]), float(scores[best])
    if np.all(scores == scores[0]):
        return NumericalEstimate(None, None, None, None, None, None,
            QCELSFit(effective_grid=size, evaluations=evaluations, residual=residual_hat),
            "flat complex-fit objective on the finite grid; eigenvalue/phase unavailable")
    # Golden-section refinement. ratio = (sqrt(5)-1)/2 is the inverse golden
    # ratio. Each bracket [left, right] holds two interior points x < y at
    # distance ratio*(right-left) from the opposite ends. Dropping the side
    # beyond the larger of f(x), f(y) leaves one old point in the correct
    # interior position, so each iteration costs one new evaluation. Every
    # evaluated point competes with the best grid point for the minimum.
    cell, ratio = 2*pi/size, (sqrt(5.) - 1.)/2.
    for index in range(size):
        if scores[index] > scores[index-1] or scores[index] > scores[(index+1) % size]:
            continue
        # Index arithmetic wraps, so a minimum at the first or last grid point
        # is refined across +/-pi. angle_principal maps the result back.
        # The fixed evaluation count has no optimizer-dependent stopping floor.
        left, right = float(grid[index]-cell), float(grid[index]+cell)
        x, y = right-ratio*(right-left), left+ratio*(right-left)
        fx, fy = objective(x), objective(y)
        if fx < residual_hat:
            theta_hat, residual_hat = x, fx
        if fy < residual_hat:
            theta_hat, residual_hat = y, fy
        for _ in range(QCELS_BRACKET_EVALUATIONS - 2):
            if fx <= fy:
                right, y, fy = y, x, fx
                x = right-ratio*(right-left)
                fx = objective(x)
                candidate, residual = x, fx
            else:
                left, x, fx = x, y, fy
                y = left+ratio*(right-left)
                fy = objective(y)
                candidate, residual = y, fy
            if residual < residual_hat:
                theta_hat, residual_hat = candidate, residual
    theta_hat = angle_principal(theta_hat)
    value = angle_principal(-theta_hat) if tau is None else theta_hat/tau
    phase = angle_principal(-theta_hat)
    # g = gcd of the differences between powers with a nonzero summed signal.
    # The objective is then 2*pi/g periodic in theta, and g > 1 means the fit
    # cannot distinguish theta from theta + 2*pi*k/g.
    alias_order = 0
    for power in support:
        alias_order = gcd(alias_order, int(power-support[0]))
    reason = ("aliased integer-power schedule; finite fit selects one periodic representative"
              if alias_order > 1 else None)
    return NumericalEstimate(value, phase, None, None, None, None,
        QCELSFit(effective_grid=size, evaluations=evaluations, residual=residual_hat), reason)


def spe_filter_domain(degree, beta, eta):
    """Return (delta, degree_ok) for the SPE Fourier filter from WBC Theorem 3.

    Wan, Berta and Campbell, arXiv:2110.12071v2, Theorem 3 (p. 10) bounds
    |Theta(x) - F(x)| by (epsilon1 + epsilon2 + epsilon3)/2 on
    [-pi+delta, -delta] U [delta, pi-delta], PDF Eq. (A12), HTML Eq. (27),
    when beta >= max(W(2/(pi*epsilon3**2))/(4*sin(delta)**2), 1) and the
    degree d satisfies PDF Eqs. (A6)-(A7), HTML Eqs. (21)-(22). W is the
    principal Lambert W function. NWQLib chooses epsilon1 = epsilon2 = eta/8
    and epsilon3 = eta/2. The bound is then 3*eta/8, which leaves eta/8
    between the filter error and the eta/2 decision threshold of Algorithm 1
    for sampling error.

    - delta: the smallest transition half-width the selected beta supports,
      asin(sqrt(W3/(4*beta))) with W3 = W(8/(pi*eta**2)). None when beta < 1
      or W3/(4*beta) >= 1, where no delta in (0, pi/2) meets the premise.
    - degree_ok: whether d satisfies Eq. (A6), d**2 >= t*w1 with
      w1 = W(512/(pi*eta**2)), for an integer t that also satisfies Eq. (A7).
      With epsilon' = sqrt(2*pi*w1)*epsilon2, Eq. (A7) requires t >= beta
      when epsilon' >= 1, and otherwise t >= f(beta, epsilon') of PDF
      Eq. (A5), HTML Eq. (20). For t > beta that is the same as
      (e*beta/t)**t*exp(-beta) <= epsilon' (Proposition 9, PDF Eq. (A15),
      HTML Eq. (30)), whose logarithm the code compares. That expression
      decreases for t > beta, so checking the largest candidate
      t = floor(d**2/w1) suffices.

    Only scalar work is needed. The bound concerns the filter alone. It is
    not a guarantee for a finite sample count or for the grid scan in spe.
    A missing delta or a false degree premise leaves filter support
    unverified, which identification_note reports.
    """
    def positive_w(log_argument):
        """Return W(a) for a > 0 from log(a), by bisection on w + log(w) = log(a).

        Working with log(a) avoids overflow of a = 512/(pi*eta**2) at small
        eta. The bracket [0, max(1, log a)] contains the root because
        w + log(w) increases from -inf, equals 1 at w = 1 and exceeds log(a)
        at w = log(a) when log(a) > 1. Bisection stops when the midpoint
        repeats an endpoint in binary64 and returns the upper endpoint.
        """
        lower, upper = 0., max(1., log_argument)
        while True:
            middle = lower+(upper-lower)/2
            if middle == lower or middle == upper:
                return upper
            if middle+log(middle) >= log_argument:
                upper = middle
            else:
                lower = middle

    if beta < 1:
        return None, False
    log_eta = log(eta)
    w3 = positive_w(log(8/pi)-2*log_eta)
    ratio = (w3/beta)/4
    if ratio >= 1:
        return None, False
    delta = asin(sqrt(ratio))
    w1 = positive_w(log(512/pi)-2*log_eta)
    t = floor(degree**2/w1)
    log_scaled_epsilon2 = log_eta-log(8)+.5*log(2*pi*w1)
    degree_ok = t >= beta and (log_scaled_epsilon2 >= 0 or (
        t > beta and t*(1+log(beta)-log(t))-beta <= log_scaled_epsilon2))
    return delta, degree_ok


def spe_phase_radius(degree, beta, eta):
    """Return the largest scaled spectral radius tau*R for the automatic SPE time.

    The scan in spe evaluates F(x - y) for x in [-pi/2, pi/2) and scaled
    eigenvalues y in [-tau*R, tau*R]. Theorem 3 of Wan--Berta--Campbell
    arXiv:2110.12071v2 bounds |Theta - F| on delta <= |x - y| <= pi - delta.
    The inner limit is the intended jump at an eigenvalue, while the outer
    limit excludes the wrap transition of the 2*pi-periodic F. Every
    evaluated |x - y| is at most pi - delta when tau*R + delta <= pi/2. The
    value pi/3 is used when delta is unavailable and caps the radius
    otherwise.
    method._safe_time multiplies the returned radius by the common 0.9
    margin. Both values are registered in docs/ENGINEERING_CONSTANTS.md
    (QPE automatic time).
    """
    delta, _ = spe_filter_domain(degree, beta, eta)
    return pi/3 if delta is None else min(pi/3, pi/2-delta)


def spe_fourier_weights(degree, beta):
    """Return a_j = |F_(2j+1)| for j = 0, ..., d, the positive-frequency filter weights.

    Wan--Berta--Campbell arXiv:2110.12071v2, PDF Eq. (A2), HTML Eq. (17):
    F_(2j+1) = -i*sqrt(beta/(2*pi))*exp(-beta)*(I_j(beta) + I_(j+1)(beta))/(2j+1)
    for 0 <= j <= d-1, while the terminal F_(2d+1) keeps I_d(beta) alone.
    I_j is the modified Bessel function of the first kind. The returned
    float array has shape (d+1,), and entry j belongs to frequency 2j+1.
    Every F_(2j+1) is -i times a nonnegative number, so a_j is that number.

    F_0 = 1/2 is known exactly and the coefficient of exp(-i*j*x) is
    conj(F_j) (PDF Eq. (A1), HTML Eq. (16)), so sampling positive
    frequencies suffices when both quadratures are acquired.
    scipy.special.ive(j, beta) = exp(-beta)*I_j(beta) evaluates the product
    without forming exp(beta), which overflows for large beta.
    """
    from scipy.special import ive
    from nwqlib._validation import integer, finite_real

    degree = integer(degree, "fourier_degree", 1)
    beta = finite_real(beta, "filter_beta")
    if beta <= 0:
        raise ValueError("filter_beta must be positive")
    values = ive(np.arange(degree+1), beta)
    weights = values.copy()
    weights[:-1] += values[1:]
    weights *= sqrt(beta/(2*pi))/(2*np.arange(degree+1)+1)
    if not np.isfinite(weights).all() or not np.sum(weights) > 0:
        raise ValueError("SPE Fourier weights are outside the finite numerical domain")
    return weights


def _draw_multiplicities(size, supplied):
    """Return (counts, total) for the random Fourier draws behind the observations.

    For SPE and RFE, method.plan makes one query per distinct
    (power, phase) setting and records how many random draws selected it.
    Exact readout evaluates the setting once. Counts acquire shots times
    that multiplicity fresh shots and enter as the received-shot mean
    (records.QPEQuery states the premises of that pooling). Weighting
    each observation by count/total reproduces the empirical distribution of
    the original draws. ``supplied=None`` means one draw per observation.
    """
    from nwqlib._validation import integer

    counts = (1,)*size if supplied is None else tuple(integer(n, "draw multiplicity", 1) for n in supplied)
    if len(counts) != size or not size:
        raise ValueError("Fourier draw multiplicities must match the nonempty observed population")
    return counts, sum(counts)


def spe_cdf(powers, samples, coordinates, *, tau, fourier_degree, filter_beta, multiplicities=None):
    """Evaluate the sampled approximate CDF (ACDF) at the given coordinates.

    Wan--Berta--Campbell arXiv:2110.12071v2, Eq. (6) (p. 3), define
    C~(x) = sum_j F_j*exp(i*j*x)*tr[rho*exp(i*Hhat*t_j)] with
    t_j = -j*tau*lambda and Hhat = H/lambda, so the trace is the Hadamard
    signal z_j = <psi|U^j|psi> of U = exp(-i*tau*H). NWQLib's time step tau
    takes the place of the paper's normalization tau = pi/(2*lambda + Delta)
    of Eq. (2), where lambda is the Pauli l1 norm of Eq. (1) and Delta the
    target precision. The automatic tau uses the radius of spe_phase_radius.
    With F_0 = 1/2, F_j = -i*a_j and F_(-j) = i*a_j for positive odd j
    (PDF Eqs. (A1)-(A2), HTML Eqs. (16)-(17)), and z_(-j) = conj(z_j), each
    +-j pair contributes 2*a_j*Im(exp(i*j*x)*z_j). Drawing M positive
    frequencies with probability a_j/S, S = sum_j a_j, therefore gives the
    unbiased estimate

        C~(x) ~ 1/2 + (2*S/M)*sum over draws of Im(exp(i*j*x)*z_j).

    Repeated draws enter as one exact or pooled observation with their draw
    multiplicity (_draw_multiplicities). For a
    Hamiltonian the coordinate x is tau*E. A unitary with eigenphase phi has
    z_j = exp(i*j*phi), the conjugate of the Hamiltonian form
    exp(-i*j*tau*E), so unitary samples are conjugated and x is then phi.

    Returns:
        Float array with the shape of ``coordinates``. Workspace is
        O(len(coordinates)).
    """
    from nwqlib._validation import integer

    weights = spe_fourier_weights(fourier_degree, filter_beta)
    powers = tuple(integer(power, "SPE Fourier power", 1) for power in powers)
    if not powers or any(power % 2 != 1 or power > 2*fourier_degree+1 for power in powers):
        raise ValueError("SPE powers must be positive odd frequencies of the selected filter")
    samples = np.asarray(samples, dtype=complex)
    if samples.shape != (len(powers),) or not np.isfinite(samples).all():
        raise ValueError("SPE requires one finite complex observation per Fourier draw")
    coordinates = np.asarray(coordinates, dtype=float)
    cdf = np.full_like(coordinates, .5)
    counts, total = _draw_multiplicities(len(powers), multiplicities)
    factor = 2*float(np.sum(weights))/total
    for power, sample, count in zip(powers, samples, counts, strict=True):
        if tau is None:
            sample = sample.conjugate()
        cdf += (factor*count)*np.imag(np.exp(1j*power*coordinates)*sample)
    return cdf


def spe(powers, samples, *, tau, grid_size, fourier_degree, filter_beta, overlap_lower_bound,
        multiplicities=None):
    """Return the first grid point where the sampled ACDF reaches eta/2.

    The target is the lowest eigenvalue in the prepared state's spectral
    support, energy for a Hamiltonian or principal eigenphase for a unitary.
    The caller declares eta = overlap_lower_bound as a lower bound on that
    eigenvalue's weight, so the CDF C(x) of Eq. (2) (p. 2) jumps by at
    least eta there. Wan--Berta--Campbell arXiv:2110.12071v2, Algorithm 1
    (p. 4), decides at one point x whether C(x - delta) < eta or
    C(x + delta) > 0 by comparing the sampled C~(x) with eta/2. This kernel
    makes that comparison at every point of a grid of grid_size points on
    [-pi/2, pi/2) and returns the first point at or above eta/2. A crossing
    at either end of the grid is reported as unidentified.

    Departures from the paper and their consequences:

    - Search. Sec. III (p. 4) locates the jump with s = O(log(1/delta))
      decisions chosen as in binary search. It reuses one sample set for all
      of them and sets the failure probability of each decision to xi/s, so
      a union bound gives total failure probability at most xi (Theorem 1,
      p. 4). The grid scan also reuses one sample set, which lets
      ``analyze(grid_size=...)`` recompute the estimate without acquisition,
      but it makes grid_size decisions. The same union bound would need
      failure probability xi/grid_size per decision in Eq. (11).
    - Sample count. Algorithm 1 sets C_sample by Eq. (11). num_samples is a
      caller choice that is not derived from Eq. (11), so the default implies
      no failure probability and Theorem 1 does not transfer to the scan.
    - Sampling. The paper samples j over all of S1 = {0, +-1, +-3, ...} with
      probability proportional to |F_j|. NWQLib keeps F_0 = 1/2 exactly and
      pairs each positive draw with its conjugate (spe_cdf). This is also an
      unbiased estimate of C~(x).
    - Evolution. The paper compiles exp(i*Hhat*t_j) with the randomized LCU
      of Lemma 2 (p. 3). NWQLib uses the Method's controlled-power backend,
      whose own error model applies, so the paper's gate-count claims do not.

    The periodic filter needs the scaled spectrum inside (-pi/2, pi/2) with
    the margin of spe_phase_radius. The declared eta, the filter
    approximation, finite sampling and evolution error are separate premises,
    and the crossing supplies no confidence interval.
    """
    if not 0 < overlap_lower_bound <= 1:
        raise ValueError("SPE overlap_lower_bound must lie in (0,1]")
    if tau is not None and (not isfinite(tau) or tau <= 0):
        raise ValueError("SPE tau must be positive and finite")
    grid = np.linspace(-pi/2, pi/2, grid_size, endpoint=False)
    cdf = spe_cdf(powers, samples, grid, tau=tau,
                  fourier_degree=fourier_degree, filter_beta=filter_beta, multiplicities=multiplicities)
    crossing = np.flatnonzero(cdf >= overlap_lower_bound/2)
    if not len(crossing):
        return NumericalEstimate(None, None, None, None, None, None,
                                 stop_reason="sampled ACDF has no selected overlap-threshold crossing")
    if crossing[0] in (0, grid_size-1):
        return NumericalEstimate(None, None, None, None, None, None,
            stop_reason="sampled ACDF crossing is at the scan boundary, so the target is unidentified")
    coordinate = float(grid[crossing[0]])
    phase = angle_principal(coordinate if tau is None else -coordinate)
    # For a unitary the value is the eigenphase itself, so it is the same
    # binary64 number as phase. angle_principal rounds x + pi, which moves a
    # grid point x by up to half a unit in the last place of x + pi, many
    # units in the last place of x when x is near zero. QPEAnalysis.validate_plan
    # derives the phase in turns from the value, and public_estimate derives
    # it from the phase, so the two fields must not differ by that rounding.
    value = phase if tau is None else coordinate/tau
    return NumericalEstimate(value, phase, None, None, None,
        "First Fourier ACDF crossing of the declared ground-overlap threshold eta/2. "
        "Ground overlap, half-period spectral support, filter approximation, finite "
        "sampling and controlled-evolution error are not certified by the crossing.")


def rfe(powers, samples, *, tau, num_frequencies, multiplicities=None, mean_window=None):
    """Return the phase 2*pi*j/K of the largest sampled Fourier coefficient.

    Kshirsagar, Katabarwa and Johnson, arXiv:2209.11322v3, Sec. 2. For an
    eigenstate the Hadamard signal is g(k) = exp(i*k*theta), Eqs. (5)-(6)
    (p. 4). Algorithm 1 (p. 5) draws M powers k_i uniformly from
    {0, ..., K-1} and forms f_j = (1/M)*sum_i z_i*exp(-2*pi*i*k_i*j/K), whose
    expectation is the discrete Fourier transform of g, Eq. (7) (p. 5). Its
    magnitude peaks near j = K*theta/(2*pi). Here z_i = <psi|U^k_i|psi>
    equals exp(i*k_i*phi) for an eigenphase phi of U, so the peak phase is
    +2*pi*j/K, reported on the principal branch. For U = exp(-i*tau*H) the
    energy is -phase/tau. Repeated powers are separate random draws, passed
    one by one or grouped as one observation with their multiplicity.

    Sign convention. The arXiv:2209.11322v3 note on p. 1 records that
    Eq. (4) and the output line of Algorithm 1 were corrected for a missing
    sign. With the paper's S gate before the final Hadamard, outcome 0 has
    probability (1 - sin(k*theta))/2, so the raw imaginary outcome has mean
    -sin(k*theta) and the printed output is 2*pi - 2*pi*j/K. NWQLib's
    imaginary quadrature uses ancilla phase -pi/2 (method._quantum_program)
    and measures +Im z_k directly, so the peak phase is +2*pi*j/K.

    Guarantee. Theorem 2.1 (p. 6) gives an epsilon-accurate estimate with
    probability above 1 - delta for an eigenstate when K >= ceil(2*pi/epsilon)
    and M >= ceil((81*pi**2/2)*ln(8*pi/(delta*epsilon))) single-shot draws.
    For any epsilon < pi/2 and delta < 1 that bound exceeds 1100 draws, so
    the default M = 97 is a finite workload without the guarantee. Each
    NWQLib draw also averages ``shots`` outcomes, pooled over the repeated
    draws of one power, or uses exact readout, which the theorem does
    not model. In a mixed spectrum the
    largest peak need not belong to the ground energy.

    Flat spectrum. When at most one power carries a signal, every f_j is
    one term c*exp(-2*pi*i*k*j/K) of the same modulus, no peak identifies a
    phase and the estimate is unavailable. The per-power sums decide this
    (_signal_vanishes), because the computed |f_j| of such a spectrum differ
    by roundoff. ``mean_window`` is None for count data and otherwise the
    largest error of one real ancilla mean, the window that admitted it. An
    exactly equal computed spectrum also counts as flat.
    """
    from nwqlib._validation import integer

    num_frequencies = integer(num_frequencies, "num_frequencies", 2)
    powers = tuple(integer(power, "sampled power", 0) for power in powers)
    if not powers or any(power >= num_frequencies for power in powers):
        raise ValueError("RFE requires sampled powers in [0, num_frequencies)")
    if tau is not None and (not isfinite(tau) or tau <= 0):
        raise ValueError("RFE tau must be positive and finite")
    samples = np.asarray(samples, dtype=complex)
    if samples.shape != (len(powers),) or not np.isfinite(samples).all():
        raise ValueError("RFE requires one finite complex sample per drawn power")
    grid = 2*pi*np.arange(num_frequencies)/num_frequencies
    coefficients = np.zeros(num_frequencies, dtype=complex)
    counts, total = _draw_multiplicities(len(powers), multiplicities)
    signals = {}
    for power, sample, count in zip(powers, samples, counts, strict=True):
        coefficients += (sample*(count/total)) * np.exp(-1j*power*grid)
        summed, terms, magnitude, weight = signals.get(power, (0j, 0, 0., 0))
        signals[power] = (summed + count*sample, terms + 1, magnitude + count*abs(sample),
                          weight + count)
    support = sum(not _signal_vanishes(summed, terms, magnitude, mean_window, weight)
                  for summed, terms, magnitude, weight in signals.values())
    scores = np.abs(coefficients)
    if support <= 1 or np.all(scores == scores[0]):
        return NumericalEstimate(None, None, None, None, None, None,
                                 stop_reason="flat sampled Fourier spectrum; phase unavailable")
    phase = angle_principal(float(grid[int(np.argmax(scores))]))
    value = phase if tau is None else -phase/tau
    return NumericalEstimate(value, phase, None, None, None,
        "Largest sampled Fourier coefficient. The precision theorem of Kshirsagar, Katabarwa "
        "and Johnson arXiv:2209.11322v3 requires an eigenstate and a sufficient "
        "independent-sample budget. Mixed-spectrum ground identification and "
        "circuit-synthesis error have separate obligations.")


def rwpe_scale(prior_std, step):
    """Return sigma_k = prior_std*((e-1)/e)**(k/2), the Gaussian width after k updates.

    By Granade and Wiebe arXiv:2208.04526v1, Eq. (7b) (p. 5), each one-bit
    update multiplies the width by sqrt((e-1)/e) whatever the outcome, so the
    width after k updates is known before any data arrive. The next relative
    evolution time is 1/sigma_k (rwpe_choose). Raises ValueError when the
    width or that time leaves the finite positive binary64 range.
    """
    from nwqlib._validation import integer, finite_real

    step = integer(step, "RWPE step", 0)
    prior_std = finite_real(prior_std, "prior_std")
    width = prior_std*sqrt((e-1)/e)**step
    if prior_std <= 0 or not isfinite(width) or width <= 0 or not isfinite(1/width):
        raise ValueError("RWPE Gaussian width and reciprocal time must be finite and positive")
    return width


def rwpe_choose(mean, std):
    """Return (time, feedback) for the next RWPE experiment.

    Granade and Wiebe, arXiv:2208.04526v1, Algorithm 1 (p. 3), choose the
    relative evolution time t = 1/sigma, with sigma = std the current
    Gaussian width, and acquire one Bernoulli datum.
    NWQLib's phase coordinate is phi = -tau*E, and the circuit H, controlled
    exp(-i*tau*H*t), RZ(feedback), H on the ancilla gives

        P(0 | phi) = (1 + cos(t*phi + feedback))/2.

    With feedback = -t*mean - pi/2 this is (1 + sin((phi - mean)/sigma))/2.
    For a Gaussian prior N(mean, sigma**2) and x = (phi - mean)/sigma,
    E[sin(x)] = 0 and E[x*sin(x)] = exp(-1/2), so outcome 0 has probability
    1/2 and moves the posterior mean to mean + sigma/sqrt(e). This is the
    positive d = 0 shift of Eq. (7a) (p. 5).

    Sign inconsistency in Algorithm 1. The likelihood Eq. (2) (p. 2),
    P(d | omega) = cos(t*(omega - omega_inv)/2 + d*pi/2)**2, corresponds to
    feedback = -t*omega_inv, so NWQLib's choice is
    omega_inv = mean + pi*sigma/2. Algorithm 1 (and Algorithm 2, p. 13)
    prints omega_inv = mean - pi*sigma/2, which gives
    P(0) = (1 - sin((omega - mean)/sigma))/2 and would lower the mean after
    d = 0, opposite to the update that Algorithm 1 applies. The paper's own
    exact update, Eq. (6a) (p. 4), agrees with NWQLib. At t = 1/sigma and
    omega_inv = mean + pi*sigma/2 it gives mean + (-1)**d*sigma/sqrt(e),
    which is Eq. (7a), while omega_inv = mean - pi*sigma/2 gives the
    opposite sign. The sentence that introduces Eq. (7) on p. 5 states
    omega_inv = mean. For the rescaled prior of Eq. (6a), with mean 0, that
    is omega_inv = 0, so the numerator t*sin(t*omega_inv) vanishes and the
    mean does not move. The test
    test_rwpe_one_bit_update_matches_independent_gaussian_integration in
    tests/test_qpe_workflow.py checks the sign by numerical integration of
    this likelihood against the Gaussian prior.

    Returns:
        The time 1/sigma in units of tau and the feedback angle in radians
        on the principal branch.
    """
    if not isfinite(mean) or not isfinite(std) or std <= 0:
        raise ValueError("RWPE requires a finite Gaussian mean and positive width")
    time = 1/std
    # Sign derived above and in docs/algorithms/qpe.md (RWPE). In the paper's
    # notation this is omega_inv = mean + pi*std/2, as Eq. (6a) requires, not
    # Algorithm 1's mean - pi*std/2.
    feedback = -time*mean-pi/2
    if not isfinite(time) or not isfinite(feedback):
        raise ValueError("RWPE experiment is outside the finite scalar domain")
    return time, angle_principal(feedback)


def rwpe_update(mean, std, datum):
    """Return the Gaussian mean after one datum d, mean + (-1)**d*std/sqrt(e).

    Granade and Wiebe arXiv:2208.04526v1, Eq. (7a) (p. 5), for the
    experiment chosen by rwpe_choose. ``std`` is rwpe_scale(prior_std, k),
    the width after the k updates already processed. The width after this
    update, Eq. (7b), is rwpe_scale(prior_std, k + 1), so no width is
    stored.
    """
    from nwqlib._validation import integer

    datum = integer(datum, "RWPE datum", 0)
    if datum not in (0, 1) or not isfinite(mean) or not isfinite(std) or std <= 0:
        raise ValueError("RWPE requires one binary datum and a finite Gaussian prior")
    updated = mean+(1-2*datum)*std/sqrt(e)
    if not isfinite(updated):
        raise ValueError("RWPE mean is outside the finite scalar domain")
    return updated


def rwpe_estimate(mean, std, *, tau):
    """Return the energy -mean/tau with its nominal Gaussian model interval.

    The posterior model for phi = -tau*E is N(mean, std**2). The interval is
    mean +- q*std, with q the standard normal quantile at
    QPE_INTERVAL_CONFIDENCE, mapped to energy by E = -phi/tau, which
    reverses the endpoints. The phase field is the principal branch of mean.
    Granade and Wiebe, arXiv:2208.04526v1, use the mean as the estimate
    (Algorithm 1). The interval is a property of the Gaussian model, not a
    confidence interval.

    Departure. Granade and Wiebe arXiv:2208.04526v1, Algorithm 2 (p. 13),
    add a consistency check at time tau_check/sigma after each update and,
    when it fails, unwind earlier updates and widen sigma. Unwinding past the
    initial prior is what lets their walk leave the finite range of Eq. (8)
    (pp. 8 and 14). NWQLib implements only Algorithm 1 and never tests its
    data for a failed Gaussian approximation or for a target outside that
    range.
    records.identification_note reports the range condition from the stored
    spectral bound instead.
    """
    from statistics import NormalDist

    if tau is None or not isfinite(tau) or tau <= 0:
        raise ValueError("RWPE continuous evolution requires a positive Hamiltonian time unit")
    radius = NormalDist().inv_cdf((1+QPE_INTERVAL_CONFIDENCE)/2)*std
    return NumericalEstimate(-mean/tau, angle_principal(mean),
        -(mean+radius)/tau, -(mean-radius)/tau, "rwpe_gaussian_credible",
        "Nominal 95-percent interval of the moment-matched Gaussian model for one eigenphase. "
        "The basic random walk has finite exploration range and omits consistency-check unwinding. "
        "Gaussian approximation, mixed spectra and evolution bias have no coverage guarantee.")


def diagonalize_hermitian(matrix):
    """Return (eigenvalues, eigenvectors) of a Hermitian matrix from LAPACK.

    Eigenvalues are float64 in ascending order with shape (D,). Eigenvectors
    are the complex128 columns of a (D, D) array. A real symmetric input uses
    the real driver dsyevd, which is cheaper and gives real eigenvectors, and
    complex input uses zheevd. Both read only the lower triangle, so the
    input must already be Hermitian. Callers admit the O(D**3) work first.
    """
    if np.count_nonzero(np.imag(matrix)) == 0:
        values, vectors, info = dsyevd(np.array(np.real(matrix), dtype=float, order="F"),
                                      compute_v=1, lower=1, overwrite_a=0)
        vectors = np.asarray(vectors, dtype=complex)
    else:
        values, vectors, info = zheevd(np.array(matrix, dtype=complex, order="F"),
                                      compute_v=1, lower=1, overwrite_a=0)
    if info != 0:
        raise np.linalg.LinAlgError(f"Hermitian eigensystem computation failed with LAPACK info={info}")
    return values, vectors


def nominal_eigensystem(matrix, *, kind):
    """Return (eigenvalues, eigenvectors) of a Hermitian or unitary matrix.

    ``kind="hamiltonian"`` calls diagonalize_hermitian. For
    ``kind="unitary"`` the eigenvalues are the complex diagonal of the Schur
    form. A unitary matrix is normal, so its complex Schur form is diagonal up
    to roundoff and the Schur vectors are orthonormal eigenvectors, including
    within degenerate eigenspaces. No reference state is projected here.
    """
    if kind == "unitary":
        triangular, vectors = schur(np.asarray(matrix, dtype=complex), output="complex")
        values = triangular.diagonal().copy()
    else:
        values, vectors = diagonalize_hermitian(matrix)
    return values, vectors


def spectral_weights(vectors, state, *, occupation=None):
    """Return the spectral weights w_j = |<v_j|psi>|**2, shape (D,).

    v_j are the eigenvector columns. An occupation-number reference |m>
    reads row m of the eigenvector matrix instead of forming the state
    vector. The weights sum to |psi|**2 up to roundoff. The classical route
    computes them once per Run context (method._spectrum).
    """
    coefficients = vectors[occupation, :].conj() if occupation is not None else vectors.conj().T@state
    return np.abs(coefficients)**2


def overlap(values, weights, power, tau):
    """Ideal Hadamard-test signal z_p = <psi|U^p|psi> from the spectral weights.

    For a Hamiltonian z_p = sum_j w_j exp(-i*E_j*tau*p). For unitary input
    z_p = sum_j w_j lambda_j**p with eigenvalues lambda_j. Power 0 is exactly 1.
    """
    if power == 0:
        return 1.+0j
    phases = values**power if tau is None else np.exp(-1j*values*tau*power)
    return complex(np.dot(weights, phases))
