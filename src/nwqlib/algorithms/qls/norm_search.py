"""Classical models of Dalzell's norm-estimation searches for the shortcut ``t``.

Dalzell, arXiv:2406.12086, Sec. 5.1-5.3. Numbers refer to
arXiv:2406.12086v2. Each model evaluates the predicted success
probabilities of the paper's detectors from an encoded reference norm
instead of sampling them, and records the trials and queries that a
quantum run of the same search would spend. Those counts are planned
work, never executed jobs. The numeric literals are the paper's: threshold
``0.725`` and ``eta = 0.025`` (Sec. 5.1, p. 6), the trial counts of
Eqs. (24) and (28), ``eta = sqrt(1/8)`` and ``ceil(log_{3/2} |T|)`` rounds
(Sec. 5.2, p. 7).
"""

from __future__ import annotations
from math import ceil, exp, frexp, log
from typing import Any
import numpy as np

_ENCODED_NORM_ERROR = "encoded_solution_norm_estimate must lie in [1, kappa_be]"


def _success_center(encoded_norm: float, t_value: float) -> float:
    """Return ``sin^2(2 theta_t) = 4 t^2 ||x||^2 / (t^2 + ||x||^2)^2``.

    ``theta_t = arctan(||x||/t)`` is Dalzell arXiv:2406.12086v2, Eq. (7),
    and this value is the center of the Eq. (17) success window (p. 5).
    """

    numerator = 4.0 * t_value**2 * encoded_norm**2
    denominator = (t_value**2 + encoded_norm**2) ** 2
    return float(numerator / denominator)


def _success_window(center: float, eta: float) -> tuple[float, float]:
    """Return the Dalzell arXiv:2406.12086v2 Eq. 17 lower/upper success-probability bounds.

    ``center (1-eta)^2/(1+eta)^2 <= p_succ <= center + 4 eta^2/(1+eta)^2``
    for Algorithm 1 with KR parameter ``eta``.
    """

    return (
        float(center * ((1.0 - eta) / (1.0 + eta)) ** 2),
        float(center + 4.0 * eta**2 / (1.0 + eta) ** 2),
    )


def _ceil_log2(value: float) -> int:
    """Return ``ceil(log2(value))`` exactly for a positive finite binary64 value.

    ``frexp`` writes ``value = m * 2**e`` with ``0.5 <= m < 1`` without
    rounding. The value is the power of two ``2**(e-1)`` when ``m == 0.5``
    and lies strictly between ``2**(e-1)`` and ``2**e`` otherwise. A rounded
    logarithm can land on the wrong side of the integer in either direction.
    ``log2(nextafter(16, inf))`` returns 4.0, and ``log(2**29, 2)`` returns
    29.000000000000004. The norm-search ladders here and the planning
    envelope in ``host_planning.search_envelope`` call this function, so
    planning admits exactly the rows, trials and queries that a model
    records.
    """

    mantissa, exponent = frexp(value)
    return exponent - 1 if mantissa == 0.5 else exponent


def _norm_search_candidate_ladder(kappa_be: float) -> list[float]:
    """Return the Dalzell arXiv:2406.12086v2 Eq. 23 log-grid ladder clamped to ``kappa_be``.

    The rungs are the powers of two up to ``ceil(log2(kappa_be))``. The top
    rung can exceed ``kappa_be`` and is clamped to it, so every recorded t
    replays through the user path's ``[1, kappa_be]`` bound. The Sec. 5.1
    grid and the Sec. 5.2 noisy search both search this ladder.
    """

    max_step = _ceil_log2(kappa_be) if kappa_be > 1.0 else 0
    return [min(2.0**step, kappa_be) for step in range(max_step + 1)]


def _noisy_binary_search_norm(
    *,
    kappa_be: float,
    encoded_norm: float,
) -> tuple[float, str, dict[str, Any]]:
    """Run the Dalzell arXiv:2406.12086v2 Sec. 5.2 noisy binary search on the log ladder.

    Validation-scale probability evaluations stand in for the sampled KP
    estimates (Eqs. 26-27: success probability ``t^2/(t^2 + ||x||^2)``,
    equal to 1/2 exactly at ``t = ||x||``); the recorded repetition counts
    are planned accounting, not executed batches. The elimination step
    keeps the half of the active set containing the KP threshold crossing,
    which is the kept-approximation invariant the reference's
    correctness argument relies on. Because ``cos^2(theta_t)`` increases
    with ``t`` (Eq. (44), p. 10), a probe with success probability above 1/2
    lies above ``||x||``, so candidates above the probe are eliminated. The
    Sec. 5.2 text (p. 7) states the opposite direction ("eliminate all
    elements of S less than tau"). The implemented direction is the one
    consistent with Eq. (27) and Eq. (44). Ladder medians of even-size active
    sets round to the closest element with ties resolved downward, where the
    paper breaks ties arbitrarily. A two-element final set selects the
    pair's geometric midpoint (``e^(tau + ln(2)/2)`` on unclamped rungs).
    """

    candidates = _norm_search_candidate_ladder(kappa_be)
    ladder = [log(candidate) for candidate in candidates]
    ladder_size = len(ladder)
    search_rounds = int(ceil(log(ladder_size) / log(1.5)))
    repetitions = int(ceil(72.0 * log(40.0 * search_rounds)))
    search_eta = (1.0 / 8.0) ** 0.5
    queries_per_trial = int(ceil(kappa_be * log(2.0 / search_eta) / 2.0))
    active = list(range(ladder_size))
    rounds = []
    for round_index in range(search_rounds):
        size = len(active)
        if size % 2 == 1:
            probe = active[size // 2]
        else:
            left = active[size // 2 - 1]
            right = active[size // 2]
            midpoint = 0.5 * (ladder[left] + ladder[right])
            probe = left if midpoint - ladder[left] <= ladder[right] - midpoint else right
        t_probe = float(candidates[probe])
        kp_probability = float(t_probe**2 / (t_probe**2 + encoded_norm**2))
        if kp_probability > 0.5:
            decision = "eliminate_above"
            active = [index for index in active if index <= probe]
        else:
            decision = "eliminate_below"
            active = [index for index in active if index >= probe]
        rounds.append(
            {
                "round": round_index + 1,
                "active_size_before": size,
                "probe_t": t_probe,
                "kp_success_probability": kp_probability,
                "decision": decision,
            }
        )
    if len(active) == 1:
        chosen = float(candidates[active[0]])
        chosen_rule = "single_element"
    else:
        chosen = float(exp(0.5 * (ladder[active[0]] + ladder[active[-1]])))
        chosen_rule = "active_span_geometric_midpoint"
    return (
        chosen,
        "noisy_binary_search",
        {
            "strategy": "noisy_binary_search",
            "paper_section": "Dalzell arXiv:2406.12086v2 Sec. 5.2 noisy binary search in log space",
            "candidate_ladder": candidates,
            "search_eta": search_eta,
            "search_eta_source": "eta = sqrt(1/8) [Dalzell arXiv:2406.12086v2 Sec. 5.2]",
            "kp_success_model": (
                "cos^2(theta_t) = t^2/(t^2 + ||x||^2) [Dalzell arXiv:2406.12086v2 Eqs. 26-27, 44]"
            ),
            "search_rounds": search_rounds,
            "search_rounds_formula": "ceil(log_1.5(|T|)) [Dalzell arXiv:2406.12086v2 Sec. 5.2]",
            "repetitions_per_estimate": repetitions,
            "repetitions_formula": (
                "ceil(72*ln(40*ceil(log_1.5(|T|)))) [Dalzell arXiv:2406.12086v2 Eq. 28]"
            ),
            "queries_per_trial": queries_per_trial,
            "rounds": rounds,
            "final_active_t": [float(candidates[index]) for index in active],
            "chosen_rule": chosen_rule,
            "chosen_t": chosen,
            "planned_trials_total": repetitions * search_rounds,
            "planned_queries_total": (2 * repetitions * search_rounds * queries_per_trial),
            "planned_queries_formula": (
                "2*k*ceil(log_1.5(|T|))*ceil(kappa_be*ln(2/eta)/2) "
                "[Dalzell arXiv:2406.12086v2 Eq. 29]"
            ),
        },
    )


def _linear_kappa_sequence_norm(
    *,
    kappa_be: float,
    left: np.ndarray,
    values: np.ndarray,
    alpha: float,
    rhs: np.ndarray,
) -> tuple[float, str, dict[str, Any]]:
    """Run the Dalzell arXiv:2406.12086v2 Sec. 5.3 linear-in-kappa sequence search.

    The sigma-parameterized systems of Eqs. (31)-(36) are evaluated in
    closed form from the original factors: ``left`` is ``U`` of the
    original SVD and ``values`` its singular values, or the eigenvectors
    and signed eigenvalues of a Hermitian original, and ``s_j =
    values_j/alpha`` are the singular values of ``A/alpha`` (``s_j**2`` is
    the same for a signed eigenvalue). The weights ``|U† b_hat|**2`` come
    from ``numerical.linear_model_weights``. Each step
    runs the Sec. 5.1 detector over the property-3 window
    ``[t/2, 4t]`` around the previous estimate (four ladder candidates,
    first threshold crossing accepted), and the recorded per-step trial
    counts carry the Sec. 5.3 log-log overhead ``1 + log2(kappa) - j``.
    Dalzell states that overhead as ``O(1 + log2(kappa) - j)`` (p. 9). The
    model takes its constant as one. The per-trial query count uses the
    step's condition number ``2^j`` (property 1, p. 9) in place of
    ``kappa``. Recorded counts are planned accounting, not executed batches;
    the sequence terminates at the first sigma with ``f(sigma) = 0``
    (``sigma = 1/kappa_rounded``, the exact-norm system).
    """

    sequence_length = _ceil_log2(kappa_be) if kappa_be > 1.0 else 1
    kappa_rounded = 2.0**sequence_length
    from .numerical import linear_model_weights

    # |w_j|^2 with w_j = <u_j, b_hat>, the left-singular weights of Eq. (36).
    weights = linear_model_weights(left, np.asarray(rhs, dtype=complex))
    squared_singular = (values / alpha) ** 2

    # Sec. 5.1 detector (p. 6): threshold 0.725 at eta = 0.025, and Eq. (24)
    # trials k = ceil(100 ln(20 |T|)) with |T| = 4 candidates per step.
    threshold = 0.725
    detector_eta = 0.025
    detector_trials_base = int(ceil(100.0 * log(20.0 * 4.0)))
    per_trial_query_factor = log(2.0 / detector_eta) / 2.0

    steps = []
    visited_sigmas = [1.0]
    t_current = 1.0
    planned_trials_total = 0
    planned_queries_total = 0
    for step in range(1, sequence_length + 1):
        sigma = 2.0**-step
        # Eq. (31) squared: f(sigma)^2 = (sigma^2 kappa^2 - 1)/(kappa^2 - 1),
        # with kappa rounded up to a power of two as on p. 9, so f is zero
        # at the last step, sigma = 1/kappa_rounded.
        f_squared = (sigma**2 * kappa_rounded**2 - 1.0) / (kappa_rounded**2 - 1.0)
        # Eq. (36): ||x_bar_sigma||^2 = sum_j |w_j|^2 / (f^2 + (1 - f^2) s_j^2)
        # with singular values s_j of A/alpha.
        step_norm = float(
            np.sqrt(np.sum(weights / (f_squared + (1.0 - f_squared) * squared_singular)))
        )
        # When t is a 2-approximation of the previous norm, property 3 (p. 9)
        # puts the new norm in [t/2, 4t]. The candidates are t/2, t, 2t and
        # 4t, clamped to kappa_be.
        window_low = t_current / 2.0
        window_high = 4.0 * t_current
        candidates = [min(window_low * 2.0**offset, kappa_be) for offset in range(4)]
        # Median-amplification overhead O(1 + log2(kappa) - j) of p. 9, with
        # its constant taken as one.
        overhead = 1 + sequence_length - step
        trials_per_candidate = detector_trials_base * overhead
        # Eq. (6) half-degree ell, the calls to each of U_A and U_A^dagger per
        # trial, with kappa replaced by the step condition number 2^j
        # (property 1).
        queries_per_trial = int(ceil(2.0**step * per_trial_query_factor))
        rows = []
        first_detected = None
        best = candidates[0]
        best_probability = -1.0
        for candidate in candidates:
            center = _success_center(step_norm, candidate)
            rows.append(
                {
                    "t": float(candidate),
                    "predicted_success_probability": center,
                    "detected": center >= threshold,
                }
            )
            if center > best_probability:
                best, best_probability = candidate, center
            if first_detected is None and center >= threshold:
                first_detected = candidate
        chosen_step = float(first_detected if first_detected is not None else best)
        step_trials = trials_per_candidate * len(candidates)
        step_queries = 2 * step_trials * queries_per_trial
        planned_trials_total += step_trials
        planned_queries_total += step_queries
        steps.append(
            {
                "step": step,
                "sigma": sigma,
                "f_squared": f_squared,
                "step_norm": step_norm,
                "window": [window_low, window_high],
                "candidates": [float(candidate) for candidate in candidates],
                "rows": rows,
                "chosen_t": chosen_step,
                "overhead_factor": overhead,
                "trials_per_candidate": trials_per_candidate,
                "planned_trials": step_trials,
                "queries_per_trial": queries_per_trial,
                "planned_queries": step_queries,
            }
        )
        visited_sigmas.append(sigma)
        t_current = chosen_step
        if f_squared <= 0.0:
            break
    final_t = float(min(max(t_current, 1.0), kappa_be))
    return (
        final_t,
        "linear_kappa_sequence",
        {
            "strategy": "linear_kappa_sequence",
            "paper_section": "Dalzell arXiv:2406.12086v2 Sec. 5.3 linear-in-kappa sequence",
            "kappa_rounded": kappa_rounded,
            "sequence_length": sequence_length,
            "visited_sigmas": visited_sigmas,
            "detector_threshold": threshold,
            "detector_eta": detector_eta,
            "detector_eta_source": "eta = 0.025 [Dalzell arXiv:2406.12086v2 Sec. 5.1]",
            "trials_per_candidate_formula": (
                "ceil(100*ln(20*|T_step|))*(1 + log2(kappa_rounded) - j) "
                "[Dalzell arXiv:2406.12086v2 Eq. 24 with |T_step| = 4; Sec. 5.3 log-log overhead]"
            ),
            "planned_queries_formula": (
                "sum_j 2*k_j*|T_step|*ceil(2^j*ln(2/eta)/2) "
                "[Dalzell arXiv:2406.12086v2 Eq. 25 with kappa -> 2^j; Sec. 5.3, Eqs. 41-42]"
            ),
            "early_stop_rule": (
                "stop at the first sigma with f(sigma) = 0 (sigma = "
                "1/kappa_rounded, the exact-norm system); within each step "
                "the detector accepts the first threshold crossing"
            ),
            "steps": steps,
            "chosen_t": final_t,
            "planned_trials_total": planned_trials_total,
            "planned_queries_total": planned_queries_total,
        },
    )


def _resolve_shortcut_norm(
    requested,
    *,
    kappa_be: float,
    encoded_norm: float | None,
    eta: float,
    factors=None,
    alpha: float | None = None,
    rhs: np.ndarray | None = None,
) -> tuple[float, str, dict[str, Any]]:
    """Resolve the Dalzell ``t`` value and norm-search metadata.

    A numeric ``t`` is the caller's premise and must lie in
    ``[1, kappa_be]`` (Dalzell arXiv:2406.12086v2, Sec. 4, takes ``t`` in
    ``[1, kappa]``). The named models run only in explicit classical
    execution. ``"grid"`` is the Sec. 5.1 exhaustive search on the
    Eq. (23) ladder: it accepts the first candidate whose predicted success
    probability reaches ``0.725``, records ``ceil(100 ln(20 |T|))`` trials
    per candidate (Eq. (24)), and falls back to the most probable candidate
    when none reaches the threshold. That fallback is NWQLib's choice. The
    paper's analysis expects a crossing with high probability.

    Args:
        requested: Numeric ``t`` or one of ``"grid"``,
            ``"noisy_binary_search"`` and ``"linear_kappa_sequence"``.
        kappa_be: Polynomial-domain condition parameter bounding ``t``.
        encoded_norm: ``||(A/alpha)^{-1} b_hat||`` from the model's one
            reference solve, needed by the grid and noisy-search models.
        eta: Kernel-reflection parameter of the selected polynomial, used
            for the Eq. (17) window rows of the grid model.
        factors: Original ``OriginalSVD`` or ``OriginalEigensystem`` of
            ``A``, needed by the linear sequence.
        alpha: Selected encoding normalization, needed by the linear
            sequence.
        rhs: Unit RHS direction, needed by the linear sequence.

    Returns:
        ``(t, source, metadata)`` with the selected ``t``, its source label
        and the model's rows and planned trial and query accounting.

    Raises:
        ValueError: When no value is supplied, a numeric ``t`` lies outside
            ``[1, kappa_be]``, or a named model lacks the classical data it
            evaluates.
    """

    if requested is None:
        raise ValueError(
            "shortcut execution requires a supplied numeric encoded_solution_norm_estimate t"
        )
    if requested == "noisy_binary_search":
        if encoded_norm is None:
            raise ValueError(
                "encoded_solution_norm_estimate='noisy_binary_search' needs the encoded "
                "reference norm from its selected classical solve; quantum shortcut "
                "execution requires a numeric t in [1, kappa_be]"
            )
        return _noisy_binary_search_norm(kappa_be=kappa_be, encoded_norm=encoded_norm)
    if requested == "linear_kappa_sequence":
        if factors is None or alpha is None or rhs is None:
            raise ValueError(
                "encoded_solution_norm_estimate='linear_kappa_sequence' needs the original "
                "factors, alpha and RHS of classical execution; quantum shortcut "
                "execution requires a numeric t in [1, kappa_be]"
            )
        left, values, _ = factors.frames()
        return _linear_kappa_sequence_norm(kappa_be=kappa_be, left=left, values=values, alpha=alpha, rhs=rhs)
    if requested != "grid":
        t_value = float(requested)
        if t_value < 1.0 or t_value > kappa_be:
            raise ValueError(_ENCODED_NORM_ERROR)
        return t_value, "user", {}

    if encoded_norm is None:
        raise ValueError(
            "encoded_solution_norm_estimate='grid' needs the encoded reference "
            "norm from its selected classical solve; quantum shortcut execution "
            "requires a numeric t in [1, kappa_be]"
        )
    candidates = _norm_search_candidate_ladder(kappa_be)
    trials_per_candidate = int(ceil(100.0 * log(20.0 * len(candidates))))
    rows = []
    best = candidates[0]
    best_probability = -1.0
    threshold = 0.725
    first_detected = None
    for candidate in candidates:
        center = _success_center(encoded_norm, candidate)
        lower, upper = _success_window(center, eta)
        rows.append(
            {
                "t": float(candidate),
                "predicted_success_probability": center,
                "eq17_lower": lower,
                "eq17_upper": upper,
                "detected": center >= threshold,
            }
        )
        if center > best_probability:
            best, best_probability = candidate, center
        if first_detected is None and center >= threshold:
            first_detected = candidate
    chosen = float(first_detected if first_detected is not None else best)
    return (
        chosen,
        "grid",
        {
            "grid": candidates,
            "trials_per_candidate": trials_per_candidate,
            "total_trials": trials_per_candidate * len(candidates),
            "threshold": threshold,
            "rows": rows,
            "chosen_t": chosen,
            "best_t": float(best),
            "paper_section": "Dalzell arXiv:2406.12086v2 Sec. 5.1 log-grid exhaustive search",
        },
    )
