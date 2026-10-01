"""Chebyshev moments to projected pencil, overlap cutoff and shot allocation.

The Gram and projected matrices follow from the moments by the product
identity T_i T_j=(T_(i+j)+T_|i-j|)/2, as in Kirby, Motta and Mezzacapo,
arXiv 2208.00567v4, Eqs. (16), (17) and (19), p. 5. Oumarou et al., arXiv
2603.15552v1, Eqs. (12), (14) and (15), pp. 10-11, state the same assembly.
The shared projected eigensolver thresholds the overlap spectrum as in
Epperly, Lin and Nakatsukasa, arXiv 2110.07492v2, Algorithm 1.1, p. 5. Shot
allocation follows the moment-sensitivity idea of Oumarou et al., Section
3.3.2, Eqs. (59)-(60), p. 29. The sampled cutoff, the allocation weights,
the pilot floor and the analytic derivative are NWQLib's own choices,
documented at their functions.

Every matrix here is m by m in the raw Chebyshev coordinates
T_0(K)|psi>, ..., T_(m-1)(K)|psi>, and every moment mu_k=<psi|T_k(K)|psi>
is dimensionless in the normalized frame K=(H-center*I)/alpha.
"""

from __future__ import annotations

from nwqlib._limits import DEFAULT_MAX_BYTES

import math

import numpy as np

from nwqlib._projected_eigensolver import (
    DEFAULT_OVERLAP_EIGENVALUE_CUTOFF,
    _solve_projected_pencil,
)


def _moment_indices(m: int) -> tuple[np.ndarray, ...]:
    """Return the six m-by-m integer arrays of moment degrees used by S and K.

    Entry (i, j) of the arrays (a, b) holds the degrees i+j and |i-j| of the
    two moments in S_ij. Entry (i, j) of (c, d, e, f) holds i+j+1, |i+j-1|,
    |i-j+1| and |i-j-1|, the degrees of the four moments in K_ij. The
    absolute values use T_(-n)=T_n. The largest degree, 2m-1, occurs in c at
    i=j=m-1.
    """
    i, j = np.indices((m, m))
    return i + j, abs(i - j), i + j + 1, abs(i + j - 1), abs(i - j + 1), abs(i - j - 1)


def _projected_matrices(moments: np.ndarray, m: int):
    """Return the projected operator K and the Gram matrix S built from moments.

    S_ij=(mu_(i+j)+mu_|i-j|)/2 and K_ij=(mu_(i+j+1)+mu_|i+j-1|+mu_|i-j+1|
    +mu_|i-j-1|)/4 (Kirby et al., arXiv 2208.00567v4, Eqs. (17) and (19),
    p. 5. Oumarou et al., arXiv 2603.15552v1, Eqs. (14) and (15), p. 11).
    S_ii=(mu_(2i)+1)/2, so the raw Chebyshev columns are not normalized. Both
    matrices are in the normalized K frame. reconstruct forms the physical
    matrix center*S+alpha*K.

    Args:
        moments: Real array of the 2m moments mu_0, ..., mu_(2m-1).
        m: Trial dimension.

    Returns:
        (K, S), two real symmetric m-by-m float64 arrays.
    """
    a, b, c, d, e, f = _moment_indices(m)
    # S_ij = <T_i T_j> = (mu_(i+j) + mu_|i-j|)/2 by T_i T_j = (T_(i+j) + T_|i-j|)/2.
    overlap = (moments[a] + moments[b]) / 2
    # c,d are symmetric; e,f exchange under transpose. Pair the latter so
    # cancellation cannot make opposite entries follow different sum orders.
    shifted = moments[c] + moments[d]
    paired = moments[e]  # Advanced indexing owns this temporary.
    paired += moments[f]
    shifted += paired
    del paired
    # K_ij = <T_i T_1 T_j>. The product identity applied twice gives
    # T_1 T_j = (T_(j+1) + T_|j-1|)/2 and then four moments, each with weight 1/4.
    shifted = shifted / 4
    return shifted, overlap


def _allocate(total: int, weights: np.ndarray, *, minimum: int = 2) -> np.ndarray:
    """Allocate exactly total shots, with stable-index largest-remainder ties.

    Shares are proportional to weights, as in Oumarou et al., arXiv
    2603.15552v1, Eq. (60). Where the paper floors each share and sets a
    one-shot minimum, this function first reserves the minimum per setting,
    splits the rest in proportion to the weights and spends the total
    exactly. The shots left after flooring go one each to the largest
    fractional shares, lower index first on ties. Two shots make the
    empirical variance denominator shots-1 positive (ENGINEERING_CONSTANTS
    "Per-setting stage minimum"). Revisit the minimum only if the
    observation model or variance estimator changes.

    Returns:
        Integer array of shots per setting, summing to total.
    """
    count = len(weights)
    if count == 0:
        return np.zeros(0, dtype=int)
    if total < minimum * count:
        raise ValueError(f"total shots requires at least {minimum * count} for {count} settings")
    scaled = np.asarray(weights, dtype=float)
    if np.any(~np.isfinite(scaled)) or np.any(scaled < 0) or not np.any(scaled > 0):
        raise ValueError("allocation weights must be finite, nonnegative, and not all zero")
    # Normalize by the maximum before summing to keep finite weights finite.
    scaled = scaled / np.max(scaled)
    shares = (total - minimum * count) * scaled / math.fsum(scaled)
    result = np.floor(shares).astype(int) + minimum
    remainder = total - int(np.sum(result))
    order = np.argsort(-(shares - np.floor(shares)), kind="stable")
    result[order[:remainder]] += 1
    return result


def _apply_pilot_floor(allocation: np.ndarray, pilot_shots: np.ndarray):
    """Reserve feasible pilot information, preserving nonbinding allocations.

    A pilot sensitivity near zero, possibly from sampling noise alone, would
    otherwise leave that moment with almost no main-stage shots. Each
    setting's floor is min(its returned pilot shots, uniform share), which is always
    feasible. If no setting falls below its floor the allocation is
    unchanged. Otherwise every setting receives its floor and the remaining
    shots follow each setting's surplus above its floor. This is a starvation
    heuristic (ENGINEERING_CONSTANTS "Sensitivity main floor"), not a
    confidence statement.

    Args:
        allocation: Integer main-stage shots per setting from _allocate.
        pilot_shots: Integer shots each setting actually returned in the pilot.

    Returns:
        (allocation, floor, binding): the final main-stage shots per setting,
        which keep the same total, the per-setting floor, and the number of
        settings whose proposed allocation was below its floor.
    """
    total = int(np.sum(allocation))
    uniform = _allocate(total, np.ones(len(allocation)))
    floor = np.minimum(pilot_shots, uniform)
    binding = int(np.count_nonzero(allocation < floor))
    if binding:
        remaining = total - int(np.sum(floor))
        # A deficient setting implies positive donor surplus whenever any
        # budget remains. Preserve donor proportions and stable-index ties.
        allocation = (
            floor
            + _allocate(
                remaining,
                np.maximum(allocation - floor, 0),
                minimum=0,
            )
            if remaining
            else floor.copy()
        )
    return allocation, floor, binding


def _assemble_moments(reconstruction, statistics) -> tuple[np.ndarray, np.ndarray]:
    """Place known and acquired moments and their sample-mean variances by degree.

    Args:
        reconstruction: LanczosReconstruction with the known moments and the
            acquired degrees.
        statistics: MomentStatistics aligned with reconstruction.moment_indices.

    Returns:
        (moments, variances), two float64 arrays of length 2m indexed by
        degree. A variance is (second_moment - mean**2)/(shots - 1), the
        unbiased estimate of the sample-mean variance. It is zero for a known
        moment or an exact probability, and NaN for a one-shot population.
    """
    moments = np.zeros(2 * reconstruction.krylov_dimension)
    variances = np.zeros_like(moments)
    for degree, value in reconstruction.known_moments.items():
        moments[degree] = value
    for degree, stat in zip(reconstruction.moment_indices, statistics, strict=True):
        moments[degree] += stat.mean
        if stat.shots is not None:
            # This is an empirical estimate, not an uncertainty certificate.
            variance = (
                (stat.second_moment - stat.mean**2) / (stat.shots - 1) if stat.shots > 1 else np.nan
            )
            variances[degree] += variance
    return moments, variances


def _overlap_regularization_record(
    *, requested, cutoff, source, radius, bound, kept_rank, tolerance, tolerance_source
):
    """Return the dict that records how the overlap cutoff was chosen.

    The sensitivity allocation stores it as allocation evidence, and
    reconstruct copies the source, bound and RMS into LanczosResult. Its
    gram_input_tolerance entry is the input part of the solver's allowance
    for a slightly negative deterministic Gram eigenvalue.
    """
    return {
        "requested_cutoff": requested,
        "resolved_cutoff": cutoff,
        "source": source,
        "empirical_gram_noise_frobenius_rms": radius,
        "gram_sampling_bound": bound,
        "kept_rank": kept_rank,
        "gram_input_tolerance": tolerance,
        "gram_input_tolerance_source": tolerance_source,
    }


def _resolve_overlap_cutoff(requested, m, *, sampled, variances=None,
                            moment_shots=None, failure_probability=0.05,
                            policy="empirical", noise_multiplier=1.0):
    """Choose the Gram-eigenvalue cutoff and compute both sampling noise scales.

    The cutoff, in priority order, is the caller's fixed value, the numerical
    floor DEFAULT_OVERLAP_EIGENVALUE_CUTOFF for deterministic moments,
    noise_multiplier times the empirical Gram RMS (default policy) or twice
    the Hoeffding Gram bound (confidence policy). Both sampled policies are
    raised to at least the numerical floor. The empirical RMS is the
    adjustable default because the strict bound can remove useful trial
    directions, or all of them, at modest shot counts (Lanczos guide,
    "Continue from the same observations").

    Empirical RMS. Independent moment estimates with sample-mean variances
    sigma_k^2 (the per-shot variance divided by the shot count) give
    Var(S_ij)=(sigma_a^2+sigma_b^2+2[a=b]sigma_a^2)/4 for a=i+j and b=|i-j|.
    Summing entry variances gives E||Delta S||_F^2 without assuming entries
    of S are independent. Plugging in sample variances gives the empirical
    noise scale. The default multiplier 1 filters at that scale, without
    claiming coverage or energy accuracy. It is a tunable working choice.

    Hoeffding bound. For r sampled moments entering S and independent
    observations in [-1,1], Hoeffding (1963),
    doi:10.1080/01621459.1963.10500830, Theorem 2, Eq. (2.6), p. 16,
    with b_i-a_i=2 gives P(mu_hat-mu >= e) <= exp(-n e^2/2). Adding the
    lower tail as in Eq. (1.4), p. 13, gives P(|mu_hat-mu| >= e) <=
    2 exp(-n e^2/2) per moment. A union bound over the r moments at
    delta/r each gives the simultaneous error
    e=sqrt(2*log(2*r/delta)/n_min), where n_min is the smallest population.
    Since S_ij=(mu_(i+j)+mu_|i-j|)/2,
    ||Delta S||_2 <= ||Delta S||_F <= m*e. The confidence policy cuts at twice
    this bound, so Weyl places each kept population eigenvalue above m*e.
    This does not bound Hamiltonian bias, subspace error or the Ritz energy.
    Only scalar shot counts are needed, independent of Hilbert-space width.

    Args:
        requested: Caller's fixed positive cutoff, or None.
        m: Trial dimension.
        sampled: Whether the moments are finite-shot estimates.
        variances: Sample-mean variance per degree (length 2m), or None.
        moment_shots: Population per degree, 0 for a known moment and None
            when unavailable.
        failure_probability: delta of the Hoeffding bound.
        policy: ``empirical`` or ``confidence``.
        noise_multiplier: Positive scale applied to the empirical RMS.

    Returns:
        (cutoff, record). cutoff is None when the selected policy lacks its
        data. record is the _overlap_regularization_record dict.
    """
    if not 0 < failure_probability < 1:
        raise ValueError("overlap_failure_probability must lie strictly between zero and one")
    if policy not in {"empirical", "confidence"}:
        raise ValueError("overlap_cutoff_policy must be empirical or confidence")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("overlap_noise_multiplier must be finite and positive")
    radius = None
    gram_variances = None if variances is None else variances[:2*m-1]
    if (gram_variances is not None and np.all(np.isfinite(gram_variances))
            and np.all(gram_variances >= 0)):
        # sum_ij Var(S_ij) collected by moment degree k: sigma_k^2 appears
        # m-|k-(m-1)| times as i+j (Hankel part), m times for k=0 and 2(m-k)
        # times for 0<k<m as |i-j| (Toeplitz part), and once more with weight 2
        # in each entry where i+j=|i-j|=k, which is (0,0) for k=0 and (0,k),
        # (k,0) for 0<k<m. The result equals the m*m entry sum in O(m) scalar
        # work, which archive validation also repeats.
        def weight(k):
            hankel = m-abs(k-(m-1))
            toeplitz = m if k == 0 else 2*(m-k) if k < m else 0
            coincident = 2 if k == 0 else 4 if k < m else 0
            return (hankel+toeplitz+coincident)/4
        squared = math.fsum(weight(k)*float(variances[k]) for k in range(2*m-1))
        if math.isfinite(squared):
            radius = math.sqrt(squared)
    # Deterministic moments are declared known to eta=1e-12 relative to
    # mu_0=1, the solver's numerical-null resolution. Each S_ij averages two
    # moments, so |Delta S_ij| <= eta and ||Delta S||_2 <= ||Delta S||_F <= m*eta.
    # That m*eta is the input part of the solver's allowance for a slightly
    # negative Gram eigenvalue (ENGINEERING_CONSTANTS "Gram input allowance"),
    # to which the solver adds its own m*eps*max|s|. It is a declared
    # resolution, not a proved arithmetic bound, and it is separate from the
    # positive rank cutoff.
    numerical_resolution = DEFAULT_OVERLAP_EIGENVALUE_CUTOFF
    bound = None
    if sampled and moment_shots is not None:
        # mu_(2m-1) enters H but not S. Zero names an algebraically known
        # moment, while None means the sampling population is unavailable.
        populations = tuple(moment_shots[:2*m-1])
        if len(populations) == 2*m-1 and all(n is not None and n >= 0 for n in populations):
            counts = tuple(n for n in populations if n)
            # m*e with e=sqrt(2*log(2r/delta)/n_min) and r=len(counts).
            # log(2r)-log(delta) stays finite where 2r/delta overflows for a tiny delta.
            bound = (m*math.sqrt(2*(math.log(2*len(counts))-math.log(failure_probability))
                                 / min(counts))) if counts else 0.
    if requested is not None:
        cutoff, source = requested, "user_fixed"
    elif not sampled:
        cutoff, source = numerical_resolution, "deterministic_default"
    elif policy == "empirical":
        cutoff, source = ((max(numerical_resolution, noise_multiplier*radius), "empirical_gram_rms")
                          if radius is not None else (None, "unavailable_empirical_variance"))
    elif bound is not None:
        cutoff, source = max(numerical_resolution, 2*bound), "hoeffding_gram_bound"
    else:
        cutoff, source = None, "unavailable_sampling_population"
    if cutoff is not None and not math.isfinite(cutoff):
        raise ValueError("resolved Gram cutoff exceeds the finite scalar range")
    return cutoff, _overlap_regularization_record(
        requested=requested,
        cutoff=cutoff,
        source=source,
        radius=radius,
        bound=bound,
        kept_rank=None,
        tolerance=0.0 if sampled else m * numerical_resolution,
        tolerance_source=(
            "not_applicable_sampled" if sampled else "declared_normalized_moment_resolution"
        ),
    )


def _sensitivity_weights(reconstruction, statistics, center, alpha, options):
    """Return main-stage allocation weights from the pilot's regularized solve.

    Variance formula and its premises. Let the main stage estimate each
    acquired moment mu_k from n_k fresh shots with per-shot variance v_k, and
    let g_k=dE/dmu_k. The pilot outcomes fix n_k before any main shot is
    taken. Conditional on the pilot, n_k is therefore a fixed sample size,
    and if the main shots are independent of the pilot and of each other, the
    first-order (delta-method) variance of the main-stage Ritz value is
    sum_k g_k**2*v_k/n_k. At fixed total N=sum n_k, Cauchy-Schwarz gives
    (sum_k |g_k|*sqrt(v_k))**2 <= N*sum_k g_k**2*v_k/n_k, with equality at
    n_k proportional to |g_k|*sqrt(v_k). These are the returned weights, with
    g_k evaluated at the pilot moments and v_k the pilot's unbiased per-shot
    sample variance. Oumarou et al., arXiv:2603.15552v1, Section 3.3.2,
    Eq. (60), p. 29, instead allocate by |g_k|, which gives the same shares
    when every v_k is equal. Integer floors, changing Gram rank and nonlinear
    Ritz error are outside that optimality statement.
    Only main-stage moments enter the final estimate (Lanczos.analyze).

    For a fixed cutoff, require a one-RMS margin to that boundary. For an
    automatic cutoff, require kept modes above twice the Gram RMS and a
    kept/discarded gap above twice the RMS, since two independently moving
    endpoints can close that gap. Numerical null modes need no distance from
    the automatic cutoff. This empirical derivative-stability heuristic is
    separate from the policy that selects the cutoff.
    Unresolved setting variances or spectral derivatives use uniform weights.

    Args:
        reconstruction: LanczosReconstruction of the sensitivity Plan.
        statistics: Pilot MomentStatistics aligned with its moment_indices.
        center: Identity shift of the physical frame, in operator units.
        alpha: Positive scale of the physical frame, in operator units.
        options: The Lanczos Method, read for its overlap cutoff settings.

    Returns:
        (weights, evidence). weights has one nonnegative entry per acquired
        moment, uniform ones after any fallback. evidence is a JSON-ready dict
        with the fallback reason, the cutoff record, the pilot Ritz value and
        the derivatives when available.
    """
    count = len(reconstruction.moment_indices)
    uniform = np.ones(count)
    evidence = {"policy": "regularized_solve_empirical_sensitivity", "fallback_reason": None}
    moments, moment_variances = _assemble_moments(reconstruction, statistics)
    populations = {degree: stat.shots for degree, stat in zip(reconstruction.moment_indices, statistics, strict=True)}
    populations.update({degree: 0 for degree in reconstruction.known_moments})
    cutoff, regularization = _resolve_overlap_cutoff(
        options.overlap_cutoff,
        reconstruction.krylov_dimension,
        sampled=True,
        variances=moment_variances,
        moment_shots=[populations.get(k) for k in range(2*reconstruction.krylov_dimension)],
        failure_probability=options.overlap_failure_probability,
        policy=options.overlap_cutoff_policy,
        noise_multiplier=options.overlap_noise_multiplier,
    )
    evidence["overlap_regularization"] = regularization
    if cutoff is None:
        evidence["fallback_reason"] = "unavailable_sampling_population"
        return uniform, evidence
    h, s = _projected_matrices(moments, reconstruction.krylov_dimension)
    result, spectrum, failure, _ = _solve_projected_pencil(
        h,
        s,
        overlap_eigenvalue_cutoff=cutoff,
        _sampled_overlap=True,
        _keep_overlap_eigenvectors=True,
    )
    regularization["kept_rank"] = None if spectrum is None else int(np.sum(spectrum > cutoff))
    if result is None:
        evidence["fallback_reason"] = failure
        return uniform, evidence
    # In this normalized operator frame, unity is the declared spectral scale.
    # Resolve the Ritz gap above the m*eps eigensolver arithmetic floor, including
    # ill-conditioned sampled pencils whose Ritz spectrum exceeds that scale.
    spectral_roundoff = reconstruction.krylov_dimension * np.finfo(float).eps
    ritz_resolution = spectral_roundoff * max(1.0, float(np.max(np.abs(result.eigenvalues))))
    if (
        len(result.eigenvalues) > 1
        and result.eigenvalues[1] - result.eigenvalues[0] <= ritz_resolution
    ):
        evidence["fallback_reason"] = "unresolved_lowest_pilot_ritz_gap"
        return uniform, evidence
    # One moment per setting: Var(single observation)=shots*Var(sample mean).
    # Reuse the assembled variance and reject unresolved evidence before the
    # projector derivative divides by a kept/discarded spectral gap.
    variances = np.array(
        [
            moment_variances[degree] * stat.shots if stat.shots is not None else np.nan
            for degree, stat in zip(reconstruction.moment_indices, statistics, strict=True)
        ]
    )
    if np.any(~np.isfinite(variances)) or np.any(variances <= 0):
        evidence["fallback_reason"] = "unavailable_or_nonpositive_empirical_setting_variance"
        return uniform, evidence
    radius = regularization["empirical_gram_noise_frobenius_rms"]
    if radius is None:
        evidence["fallback_reason"] = "unavailable_empirical_gram_variance"
        return uniform, evidence
    evidence["empirical_gram_noise_frobenius_rms"] = radius
    kept = spectrum > cutoff
    gap = float(np.min(spectrum[kept]) - np.max(spectrum[~kept])) if np.any(~kept) else math.inf
    overlap_resolution = spectral_roundoff * float(np.max(np.abs(spectrum)))
    if options.overlap_cutoff is None:
        evidence["rank_stability_rule"] = "kept_modes_and_gap_exceed_two_empirical_frobenius_rms"
        unstable = np.min(spectrum[kept]) <= 2 * radius or gap <= max(
            2 * radius, overlap_resolution
        )
    else:
        evidence["rank_stability_rule"] = (
            "distance_to_fixed_cutoff_exceeds_one_empirical_frobenius_rms"
        )
        unstable = np.any(np.abs(spectrum - cutoff) <= radius) or gap <= overlap_resolution
    if unstable:
        evidence["fallback_reason"] = "pilot_overlap_rank_unstable"
        return uniform, evidence
    energy = float(result.eigenvalues[0])
    # Differentiate normalized K/S first; restoring center in the pencil or
    # subtracting E from center loses small offsets under large identity shifts.
    derivatives = alpha * _moment_energy_derivatives(
        result,
        h,
        s,
        0.0,
        1.0,
        cutoff,
    )
    evidence["pilot_energy"] = center + alpha * energy
    evidence["pilot_normalized_ritz_value"] = energy
    evidence["pilot_kept_rank"] = result.kept_overlap_rank
    if np.any(~np.isfinite(derivatives)):
        evidence["fallback_reason"] = "nonfinite_energy_sensitivity"
        return uniform, evidence
    evidence["moment_derivatives"] = derivatives.tolist()
    weights = np.abs(derivatives[list(reconstruction.moment_indices)]) * np.sqrt(variances)
    if np.any(~np.isfinite(weights)) or not np.any(weights > 0):
        evidence["fallback_reason"] = "nonfinite_or_zero_allocation_weights"
        return uniform, evidence
    evidence["setting_weights"] = weights.tolist()
    return weights, evidence


def _moment_energy_derivatives(result, hamiltonian, overlap, center, alpha, cutoff):
    """Differentiate the entire fixed-rank regularized Ritz solve in O(m^3).

    The thresholded solve keeps the eigenvectors V_R of S with eigenvalues
    s_R above the cutoff and discards V_D with s_D. It returns E and the
    length-m vector c=V_R y, where (E, y) is the lowest eigenpair of the
    reduced pencil (V_R^dag H V_R, V_R^dag S V_R), scaled so c^dag S c=1.
    H is the supplied projected matrix, the normalized K in this module's
    calls. With y=V_R^dag c, first-order perturbation gives
    dE=c^dag(dH-E dS)c+2Re[(dV_R y)^dag(H-E S)c]. The kept block of
    (H-E S)c vanishes, and V_D^dag dV_R has entries
    (V_D^dag dS V_R)_dr/(s_r-s_d), so the kept projector rotates with S.
    With r_D=V_D^dag(H-E S)c the rotation term is Tr((W+W^dag)dS),
    W=V_D [(r_D conj(y))/(s_R-s_D)] V_R^dag. The chain rule through the
    degree arrays a, ..., f of _moment_indices then gives dE/dmu_k, using
    dS_ij/dmu_k=([a_ij=k]+[b_ij=k])/2 and
    dK_ij/dmu_k=([c_ij=k]+[d_ij=k]+[e_ij=k]+[f_ij=k])/4. This analytic derivative
    replaces the automatic differentiation of Oumarou et al., arXiv
    2603.15552v1, Section 3.3.2. Only kept/discarded
    spectral gaps occur; degeneracies within either block require no division.
    The caller's pilot stability rule excludes a cutoff crossing. This is
    classical postprocessing sensitivity, not an error certificate.

    Args:
        result: Thresholded solve of (hamiltonian, overlap) that kept its
            overlap eigenvectors.
        hamiltonian: Projected matrix center*S+alpha*K whose pencil result
            solved. This module passes K with center 0 and alpha 1.
        overlap: Gram matrix S.
        center: Coefficient of S in hamiltonian.
        alpha: Coefficient of K in hamiltonian.
        cutoff: Overlap-eigenvalue cutoff that result used.

    Returns:
        Real array of length 2m with dE/dmu_k for k=0, ..., 2m-1, in the
        units of E.
    """
    eigenbasis = result.overlap_eigenvectors
    if eigenbasis is None:
        raise ValueError("regularized-solve sensitivity requires the kept pilot overlap eigenbasis")
    coefficients = result.ground_state_coefficients
    energy = float(result.eigenvalues[0])
    density = np.outer(coefficients, coefficients.conj())
    # With H=center*S+alpha*K and D=c c^dag, c^dag(dH-E dS)c equals
    # Tr(alpha*D dK)+Tr((center-E)*D dS). metric_gradient collects every dS term.
    metric_gradient = (center - energy) * density
    keep = result.overlap_eigenvalues > cutoff
    if np.any(~keep):
        kept, discarded = eigenbasis[:, keep], eigenbasis[:, ~keep]
        residual = hamiltonian @ coefficients - energy * overlap @ coefficients
        residual_d = discarded.conj().T @ residual
        y = kept.conj().T @ coefficients
        gaps = (
            result.overlap_eigenvalues[keep][None, :] - result.overlap_eigenvalues[~keep][:, None]
        )
        rotation = discarded @ (residual_d[:, None] * y.conj()[None, :] / gaps) @ kept.conj().T
        metric_gradient += rotation + rotation.conj().T
    a, b, c, d, e, f = _moment_indices(len(coefficients))
    derivatives = np.zeros(2 * len(coefficients))
    # Tr(G dM) = sum_ij G_ji dM_ij; all moment derivatives are real.
    for index in (a, b):
        np.add.at(derivatives, index.ravel(), (metric_gradient.T.real / 2).ravel())
    for index in (c, d, e, f):
        np.add.at(derivatives, index.ravel(), (alpha * density.T.real / 4).ravel())
    return derivatives


def restore_ritz_values(values, center, alpha):
    """Restore c+alpha*x; bounded scalar cancellation recovery never repeats a solve.

    A binary64 product alpha*x can overflow although c+alpha*x is finite, for
    example under a large identity shift. Such entries are recomputed once in
    exact rational arithmetic. Entries that remain unrepresentable stay
    nonfinite for the caller to report.
    """
    from fractions import Fraction

    with np.errstate(over="ignore", invalid="ignore"):
        restored = center + alpha * values
    for index in np.flatnonzero(~np.isfinite(restored)):
        try:
            restored[index] = float(
                Fraction(center) + Fraction(alpha) * Fraction(float(values[index]))
            )
        except OverflowError:
            pass
    return restored


def _moment_payload(reconstruction, statistics):
    """Return the stored moment and variance tuples, each of length 2m, by degree.

    Known moments carry variance 0. An acquired degree carries its pooled
    mean and MomentStatistics.variance, which is None for exact data or a
    one-shot population. A degree without data stays None in both tuples.
    No pencil is built, so archive validation can call this cheaply.
    """
    moments = [None] * (2 * reconstruction.krylov_dimension)
    variances = [None] * len(moments)
    admitted = set(reconstruction.moment_indices)
    for degree, value in reconstruction.known:
        moments[degree], variances[degree] = value, 0.0
    for degree, stat in statistics.items():
        if degree not in admitted:
            raise ValueError("statistic is not an acquired degree of this reconstruction")
        moments[degree], variances[degree] = stat.mean, stat.variance
    return tuple(moments), tuple(variances)


def admit_projected_analysis(
    m, *, max_bytes, max_work, solves=1, operation="Lanczos projected analysis"
):
    """Raise ValueError if ``solves`` projected analyses of trial dimension m exceed the limits.

    One analysis is what reconstruct does after the moments are assembled.
    Its envelope depends on m alone, so Lanczos.plan calls this before any
    acquisition, and reconstruct calls it again for the limits it receives.
    For the sensitivity pilot, Lanczos.plan passes ``solves=2``. The pilot
    solves the same pencil and then differentiates the solve
    (_moment_energy_derivatives), and each of the two costs one analysis's
    work. The derivative keeps (K, S) and the solve's Ritz and overlap
    eigenvector matrices besides its own m-by-m temporaries, so one byte
    envelope covers both.

    512*m*m bytes, the size of 32 complex128 m-by-m arrays, bound the
    projected arrays alive at once. They include the six index maps and
    moment gathers, K and S with their complex and Hermitian copies, the
    overlap eigenvectors, the Lowdin orthogonalizer, the effective
    Hamiltonian and its eigenvectors, and the stored S and physical H.
    128*m bytes cover the length-2m moment, variance, spectrum and
    coefficient vectors. In the work of one analysis, 16*m**3 covers the two
    Hermitian eigensolves and the congruence X^dagger H X, and 32*m*m the
    gathers and Hermitian checks. docs/ENGINEERING_CONSTANTS.md registers
    this envelope, and docs/CODE_TOUR.md, "Byte and work budgets", explains
    the units.
    """
    from nwqlib.operators.access import _check_bytes

    _check_bytes(512 * m * m + 128 * m, max_bytes, operation)
    work = solves * (32 * m * m + 16 * m * m * m)
    if work > max_work:
        raise ValueError(
            f"{operation} exceeds max_analysis_work: work {work} = solves*(32*m*m + 16*m**3) "
            f"with solves={solves}, m={m}; raise Lanczos(max_analysis_work=...) to at least {work} "
            f"(now {max_work})"
        )


def reconstruct(
    reconstruction,
    statistics,
    *,
    cutoff,
    sampled,
    max_bytes=DEFAULT_MAX_BYTES,
    max_work=1_000_000_000,
    overlap_failure_probability=0.05,
    overlap_cutoff_policy="empirical",
    overlap_noise_multiplier=1.0,
):
    """Turn moment data into the lowest physical Ritz value and its diagnostics.

    Quantum analysis, classical analysis and reanalysis all call this one
    function. Known algebraic moments and the supplied statistics fill
    mu_0..mu_(2m-1).
    A scalar operator (alpha=0) returns its constant without a solve. A
    missing moment leaves the result partial, with no Ritz value. Otherwise
    the cutoff is resolved, the normalized pencil (K, S) is solved by
    thresholding and the Ritz values are restored as center+alpha*x. The
    physical pencil center*S+alpha*K is stored only when representable, and
    the normalized solve does not depend on it.

    This numerical kernel does not mint observations. The caller preserves the
    provenance of the supplied statistics.

    Args:
        reconstruction: LanczosReconstruction of the Plan.
        statistics: Dict from acquired degree to pooled MomentStatistics.
        cutoff: Caller's fixed overlap cutoff, or None for the policy.
        sampled: Whether the moments are finite-shot estimates.
        max_bytes: Byte limit for the projected arrays.
        max_work: Work limit for the projected solve.
        overlap_failure_probability: delta of the Hoeffding Gram bound.
        overlap_cutoff_policy: ``empirical`` or ``confidence``.
        overlap_noise_multiplier: Scale of the empirical Gram RMS cutoff.

    Returns:
        Dict of the LanczosResult analysis fields, with Ritz values in the
        physical units of the operator.
    """
    m = reconstruction.krylov_dimension
    # Lanczos.plan admitted this envelope under the planning Method's limits.
    # The limits passed here come from the Method that runs the analysis, or
    # from a direct caller, and can be smaller.
    admit_projected_analysis(m, max_bytes=max_bytes, max_work=max_work)
    moments, variances = _moment_payload(reconstruction, statistics)
    missing = tuple(i for i, value in enumerate(moments) if value is None)
    fields = dict(
        eigenvalue=None,
        eigenvalues=None,
        coefficients=None,
        moments=tuple(moments),
        moment_variances=tuple(variances),
        hamiltonian=None,
        overlap=None,
        kept_rank=None,
        overlap_spectrum=None,
        cutoff=cutoff,
        cutoff_source="not_evaluated",
        failure=None,
        missing=missing,
        projected_backward_error=None,
        overlap_normalization_error=None,
        gram_sampling_bound=None,
        empirical_gram_noise_frobenius_rms=None,
    )
    if reconstruction.alpha == 0:
        # All states have energy c. No pencil or solve is necessary.
        fields.update(
            eigenvalue=reconstruction.center,
            eigenvalues=(reconstruction.center,),
            coefficients=(1.0,),
            kept_rank=1,
            cutoff_source="scalar_operator",
        )
        return fields
    if missing:
        return fields
    values = np.asarray(moments)
    variance = np.array([np.nan if v is None else v for v in variances]) if sampled else None
    known = reconstruction.known_moments
    moment_shots = [0 if k in known else
                    statistics[k].shots if k in statistics else None for k in range(2*m)]
    cutoff, regularization = _resolve_overlap_cutoff(cutoff, m, sampled=sampled, variances=variance,
        moment_shots=moment_shots, failure_probability=overlap_failure_probability,
        policy=overlap_cutoff_policy, noise_multiplier=overlap_noise_multiplier)
    fields.update(cutoff=cutoff, cutoff_source=regularization["source"],
        gram_sampling_bound=regularization["gram_sampling_bound"],
        empirical_gram_noise_frobenius_rms=regularization["empirical_gram_noise_frobenius_rms"])
    if cutoff is None:
        fields["failure"] = "unavailable sampling information for the selected Gram cutoff policy"
        return fields
    h, s = _projected_matrices(values, m)
    result, spectrum, failure, _ = _solve_projected_pencil(
        h,
        s,
        overlap_eigenvalue_cutoff=cutoff,
        _sampled_overlap=sampled,
        overlap_input_tolerance=regularization["gram_input_tolerance"],
    )
    # Keep physical pencil only if representable; normalized solve remains valid.
    with np.errstate(over="ignore", invalid="ignore"):
        physical_h = reconstruction.center * s + reconstruction.alpha * h
    fields.update(
        overlap=tuple(map(tuple, s.tolist())),
        hamiltonian=tuple(map(tuple, physical_h.tolist()))
        if np.all(np.isfinite(physical_h))
        else None,
        overlap_spectrum=None if spectrum is None else tuple(spectrum),
        failure=failure,
    )
    if result is not None:
        physical = restore_ritz_values(
            result.eigenvalues, reconstruction.center, reconstruction.alpha
        )
        fields.update(
            eigenvalue=float(physical[0]) if np.isfinite(physical[0]) else None,
            eigenvalues=tuple(physical) if np.all(np.isfinite(physical)) else None,
            coefficients=tuple(float(x.real) for x in result.ground_state_coefficients),
            kept_rank=result.kept_overlap_rank,
            projected_backward_error=result.generalized_eigenpair_backward_error,
            overlap_normalization_error=result.overlap_normalization_error,
        )
        if not np.all(np.isfinite(physical)):
            fields["failure"] = "physical Ritz value outside binary64 range"
    return fields


def recurrence_requirements(operator, m, *, affine=False):
    """Return (bytes, work) that chebyshev_moments will need, before any vector exists.

    Callers check both against their limits before the reference state is
    materialized. bytes counts the peak live arrays (docs/CODE_TOUR.md, "Byte
    and work budgets"). work counts the m operator applications and the
    vector passes of the recurrence. affine selects the matrix path, which
    first forms (A - center*I)/alpha as a copy of the stored matrix.
    """
    from nwqlib.operators.inputs import _matvec_requirements

    d, action_bytes, action_work = _matvec_requirements(operator)
    # m operator applications. Each recurrence step adds an allowance of 8
    # operations per coordinate for scaling by 2, subtracting v_(k-1) and the
    # two complex inner products. 4*d covers the initial finiteness,
    # normalization and first-moment scans.
    work = m * action_work + (8 * m + 4) * d
    scaled_bytes = 0
    if affine:
        if operator.reference.representation in ("csr", "csc"):
            nnz = operator._data.nnz
            # Forming A - center*I holds the d-entry identity and its multiple
            # center*I beside the difference. SciPy allocates the difference
            # at nnz + d entries, and its prune step may copy the result once
            # more, so at most 2*nnz + 4*d entries are live. 32 bytes per
            # entry exceed one complex128 value plus a 64-bit index, and
            # 32*(d+1) exceed the pointer arrays of the three matrices.
            scaled_bytes = 32 * (2 * nnz + 4 * d) + 32 * (d + 1)
            # Forming the identity and its multiple visits 2d entries, the
            # difference reads both operands and writes and may copy at most
            # nnz + d entries each, and the division visits each stored entry
            # once, at most 4*(nnz + 2*d) visits. Each of the m applications
            # visits up to d more entries than an action of A.
            work += 4 * (nnz + 2 * d) + m * d
        else:
            # The dense copy of A - center*I: 32 bytes per entry exceed one
            # complex128 value.
            scaled_bytes = 32 * d * d
            # d diagonal subtractions, then one division per entry.
            work += d + d * d
    # 64*d: four complex128 length-d vectors, enough for the supplied
    # reference and v_(k-1), v_k and v_(k+1). 16*m: the 2m float64 moments.
    # action_bytes: the shared matvec envelope of _matvec_requirements.
    return 64 * d + 16 * m + action_bytes + scaled_bytes, work


def chebyshev_moments(
    operator,
    reference,
    *,
    m,
    center=0.0,
    alpha=1.0,
    max_bytes=DEFAULT_MAX_BYTES,
    max_products=1_000_000_000,
):
    """Evaluate mu_0, ..., mu_(2m-1) classically with the Chebyshev recurrence.

    Only explicit classical execution calls this. Quantum analysis never
    does. The caller chooses this O(m * matvec_work), O(dimension)
    computation and supplies its work cap. The logical byte envelope is the
    kept recurrence frontier plus one complete shared action envelope
    (including Pauli conversion/chunk space), not a cumulative sum of
    allocations over all m calls or a peak RSS bound. No quantum access is
    inferred from the classical action.

    The recurrence v_(k+1)=2Kv_k-v_(k-1) gives v_k=T_k(K)|psi>. The product
    identity (Kirby et al., arXiv 2208.00567v4, Eq. (16)) then gives
    mu_2k=2<v_k|v_k>-mu_0 and mu_(2k+1)=2<v_k|v_(k+1)>-mu_1, so m operator
    applications yield all 2m moments.

    Args:
        operator: Matvec access. Pauli access must already be K, and its
            action ignores center and alpha. Dense, CSR or CSC access is the
            original A, and the kernel applies K=(A-center*I)/alpha.
        reference: Normalized float64 or complex128 state of the operator
            dimension.
        m: Trial dimension.
        center: Identity shift of the matrix frame.
        alpha: Positive scale of the matrix frame.
        max_bytes: Byte limit for recurrence_requirements.
        max_products: Work limit for recurrence_requirements.

    Returns:
        Float64 array of the 2m moments, with mu_0=1.
    """
    if type(m) is not int or m < 1:
        raise ValueError("m must be positive")
    dimension = operator.manifest.basis.dimension
    from nwqlib.operators.access import _check_bytes

    affine = "pauli_terms" not in operator.manifest.access
    required_bytes, work = recurrence_requirements(operator, m, affine=affine)
    _check_bytes(required_bytes, max_bytes, "classical Chebyshev moments")
    if work > max_products:
        raise ValueError("classical Chebyshev moments exceed max_products")
    if alpha <= 0 or not math.isfinite(alpha) or not math.isfinite(center):
        raise ValueError("recurrence needs a finite positive affine scale")
    if (
        type(reference) is not np.ndarray
        or reference.shape != (dimension,)
        or reference.dtype not in (np.dtype("float64"), np.dtype("complex128"))
    ):
        raise ValueError(
            "reference must be a native float64/complex128 vector of the admitted dimension"
        )
    previous = np.asarray(reference, dtype=complex)
    if not np.all(np.isfinite(previous)):
        raise ValueError("reference must be a finite vector matching operator dimension")
    norm = float(np.vdot(previous, previous).real)
    # 1e-12 admits a supplied normalized vector up to roundoff. It is an input
    # window, not an eigenvalue accuracy (ENGINEERING_CONSTANTS "Explicit
    # recurrence normalization admission").
    if not np.isfinite(norm) or abs(norm - 1.0) > 1e-12:
        raise ValueError("explicit recurrence requires a normalized reference")
    moments = np.zeros(2 * m)
    moments[0] = 1.0
    if affine:
        # Remove center from the stored diagonal before scaling. The product
        # (A/alpha) v rounds its diagonal term at the size |center/alpha|*|v_i|,
        # and subtracting (center/alpha)*v afterwards cannot remove that
        # rounding, so the moments would carry an error that grows with
        # |center|/alpha. In the Gershgorin frame of the caller A_ii - center
        # lies within alpha of zero and rounds once. Scaling once before multiplying
        # also avoids overflow of A@v when K@v is finite. The admitted copy
        # lives only here, and the sparse difference keeps CSR/CSC storage.
        representation = operator.reference.representation
        if representation in ("csr", "csc") and center:
            from scipy.sparse import identity

            scaled = operator._data - center * identity(dimension, format=representation)
        else:
            scaled = operator._data.copy()
            if center:
                scaled[np.diag_indices(dimension)] -= center
        entries = scaled.data if representation in ("csr", "csc") else scaled
        # Sparse scalar division may form 1/alpha first, which overflows for
        # subnormal alpha even though every elementwise quotient is finite.
        entries.real /= alpha
        if np.iscomplexobj(entries):
            entries.imag /= alpha

        def action(vector):
            return scaled @ vector
    else:

        def action(vector):
            return operator.matvec(vector, max_bytes=max_bytes, max_products=max_products)

    current = action(previous)
    moments[1] = float(np.vdot(previous, current).real)
    # At the top of each pass previous=v_(degree-1) and current=v_degree.
    # T_d T_d=(T_2d+T_0)/2 and T_d T_(d+1)=(T_(2d+1)+T_1)/2 give the two moments.
    for degree in range(1, m):
        moments[2 * degree] = 2 * float(np.vdot(current, current).real) - 1.0
        following = action(current)
        following *= 2
        following -= previous
        moments[2 * degree + 1] = 2 * float(np.vdot(current, following).real) - moments[1]
        previous, current = current, following
    return moments
