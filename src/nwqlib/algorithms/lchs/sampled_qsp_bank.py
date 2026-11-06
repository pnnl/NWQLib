"""Finite Laurent phase banks for independently sampled branch polynomials.

A bucket of polynomial time tau approximates ``f(z) = exp(-i tau x)``,
``x = (z + 1/z)/2``, on the unit circle by the Jacobi–Anger polynomial
written in the paired-shell basis ``B_0 = 1``, ``B_n = (z**n + z**-n)/2``,
whose elements have circle norm at most 1:
``p_d = sum_{n<=d} c_n B_n`` with ``delta >= ||p_d - f||`` from
`jacobi_anger_expansion` (tail plus stored-coefficient discrepancy, counted
once). The deterministic bank stores p_d. The stochastic bank keeps the
shells through the cutoff s and replaces the rest by one drawn shell: with
``W = sum_{s<n<=d} |c_n|`` and ``gamma_n = W c_n/|c_n|``, member n is
``R_n = p_s + gamma_n B_n``, drawn with probability proportional to |c_n|.
A zero W leaves the deterministic polynomial. The labels below are cited by
`sampled_qsp`.

B5 (signal degree). A target with lowest power ``-l`` and highest power h
uses ``max(0, l) + max(0, h)`` walk signals, which is 2n for a member of
highest shell n. Exact-zero end coefficients are trimmed, internal zeros
are kept.

B6 (common recovery). The triangle inequality gives
``||R_n - f|| <= 2W + delta``, so every member divided by
``zeta >= (1 + 2W + delta)/(1 - eta)`` has circle norm at most ``1 - eta``
for the margin eta. The code takes the larger of this numerator and the
stored members' coefficient one-norm, so the cap also covers the stored
coefficients. Every member of a bucket uses the same recovery zeta.

B7 (actual law). The draw uses the integer probabilities
``pi_n = count_n/2**52`` of `sampled_rows.integer_categorical`. Because
``sum pi_n = 1``, the mean member is ``p_s + sum pi_n gamma_n B_n``, and its
distance from p_d is at most ``e_law = sum_{s<n<=d} |pi_n gamma_n - c_n|``.

B8 (individual and mean errors). With kappa_n the normalized residual of
the fitted phases of member n, the recovered error of member n is at most
``delta + sum_k |r_{n,k} - c_k| + zeta kappa_n``, with ``r_{n,k}`` its
stored shell coefficients (at most ``2W + delta + zeta kappa_n``), and that
of the mean member at most ``delta + e_law + zeta sum pi_n kappa_n``. Each
adds the rounding of the normalized coefficients once. The phase fit of
every member receives the uniform residual allowance
``(allowance - delta - e_law - normalization)/zeta``, and a bucket whose
allowance is used up before the fit is refused.

B9 (memory). The coefficient and angle arrays retained by every member of
the bank are checked against the memory limit before any member is built.
The first FFT attempt of a phase fit depends on the member's degree alone
and grows with it, so the first length and arrays of each bucket's
highest-degree member are checked once the bank is planned, before any fit,
and each fit checks its later doublings.

B22 (physical inner allowance). For an inner-bias allowance h of the
squared physical weight cap, a common mean amplitude error beta needs
``2 beta + beta**2 <= h``, so ``beta_max = h/(sqrt(1 + h) + 1)``, the
positive root ``sqrt(1 + h) - 1`` written without its subtraction.

The actual positive integer law, common recovery and actual normalized
coefficients determine every error and resource quantity below. All
polynomial error estimates are numerical; Bessel, coefficient construction
and phase arithmetic have no complete enclosure.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import math

import numpy as np

from nwqlib._limits import ResourceLimitError
from nwqlib._validation import finite_real, integer
from nwqlib.operators.access import _check_bytes
from nwqlib.algorithms.lchs.sampled_rows import integer_categorical
from nwqlib.subroutines.qsp.evolution import (
    JacobiAngerExpansion,
    QubitizedTarget,
    QubitizedTargetPlan,
    _check_target_start,
    jacobi_anger_expansion,
    plan_jacobi_anger_laurent,
    plan_laurent_target,
    prepare_qubitized_target,
    with_phase_allowance,
)


@dataclass(frozen=True, kw_only=True)
class SampledQSPMemberPlan:
    """One actual polynomial; shell None denotes the deterministic endpoint."""

    shell: int | None
    tail_coefficient: complex | None
    target: QubitizedTargetPlan
    normalization_error: float


@dataclass(frozen=True, kw_only=True)
class SampledQSPBucketPlan:
    """A finite bucket with a common recovery and its exact 52-bit count law.

    Errors and allowances are in recovered amplitude units. The reference is
    the expansion's numerical polynomial. ``law_error`` includes the actual
    fixed-W tail coefficient's construction discrepancy through B7;
    ``mean_normalization_error`` adds division/recovery discrepancy once.
    The separate ``arithmetic_error_bound`` remains unknown.
    """

    polynomial_time: float
    expansion: JacobiAngerExpansion
    reference_coefficients: tuple[complex, ...]
    cutoff: int
    tail_mass: float
    counts: tuple[int, ...]
    members: tuple[SampledQSPMemberPlan, ...]
    recovery: float
    mean_amplitude_allowance: float
    law_error: float
    mean_normalization_error: float
    mean_approximation_error: float
    epsilon_phase: float
    expected_signals: float
    maximum_signals: int
    expected_independent_pair_signals: float
    expected_maximum_pair_signals: float
    arithmetic_error_bound: None = None

    @property
    def shells(self):
        return tuple(member.shell for member in self.members)

    @property
    def probabilities(self):
        total = sum(self.counts)
        return tuple(count / total for count in self.counts)


@dataclass(frozen=True, kw_only=True)
class SampledQSPBankPlan:
    """The complete finite population fixed before any phase solve.

    ``retained_array_bytes`` estimates the coefficient and angle arrays its
    members retain; ``max_bytes`` None is the physical memory.
    """

    buckets: tuple[SampledQSPBucketPlan, ...]
    policy: str
    allowance_frame: str
    retained_array_bytes: int
    max_bytes: int | None
    branch_policy: str = "independent"
    arithmetic_error_bound: None = None

    @property
    def target_count(self):
        return sum(len(bucket.members) for bucket in self.buckets)


@dataclass(frozen=True, kw_only=True)
class PreparedQSPBucket:
    """Actual-angle targets and numerical B8 individual/mean amplitude errors."""

    plan: SampledQSPBucketPlan
    members: tuple[QubitizedTarget, ...]
    normalized_individual_error: float | None
    normalized_mean_error: float | None
    recovered_mean_error: float | None
    arithmetic_error_bound: None = None


@dataclass(frozen=True, kw_only=True)
class PreparedSampledQSPBank:
    """The selected bank, prepared once, with no phase fitting at row binding."""

    plan: SampledQSPBankPlan
    buckets: tuple[PreparedQSPBucket, ...]
    solver_calls: int


class SampledQSPBankError(ValueError):
    """A refusal or failure with its stage, member and cause; completed phase targets remain available for diagnosis.

    ``shell`` is the failing member's shell (None for a deterministic
    endpoint), and ``evidence`` the phase solver's FFT lengths, residuals
    and stopping reason when the cause carries them.
    """

    def __init__(self, message, *, stage, cause=None, bucket_index=None, member_index=None, shell=None,
                 completed_targets=()):
        super().__init__(message)
        self.stage = stage
        self.cause = cause
        self.bucket_index = bucket_index
        self.member_index = member_index
        self.shell = shell
        self.evidence = getattr(cause, "evidence", None)
        self.completed_targets = tuple(completed_targets)


class _SampledQSPBankLimitError(SampledQSPBankError, ResourceLimitError):
    """A bank refusal whose cause is a capacity limit, with the same stage, member and evidence."""


def _bank_failure(message, *, cause, **context):
    """Return the ``SampledQSPBankError`` of ``cause``, a capacity one when ``cause`` is a limit refusal."""
    kind = _SampledQSPBankLimitError if isinstance(cause, ResourceLimitError) else SampledQSPBankError
    return kind(message, cause=cause, **context)


# A larger allowance lowers the expansion degree and raises the phase allowance.
# A first transform refused at a limit also starts under a larger byte or FFT
# limit. After a failed fit the cause states what those limits do, so its
# refusal names the allowance alone.
_PHASE_ALLOWANCE = "the mean-amplitude allowance, which LCHS sets from approximation_tolerance"
_PHASE_DIRECTION = f"raise max_bytes (or an explicit max_fft) or {_PHASE_ALLOWANCE}"


def _shell_coefficients(expansion):
    return tuple(complex(
        expansion.cos_coefficients[n] if n < len(expansion.cos_coefficients) else 0,
        -(expansion.sin_coefficients[n] if n < len(expansion.sin_coefficients) else 0))
        for n in range(expansion.degree + 1))


def _laurent(coefficients):
    # T_n((z+z^-1)/2) = (z^n+z^-n)/2, with the constant counted once.
    positive = tuple(value / 2 for value in coefficients[1:])
    return positive[::-1] + (coefficients[0],) + positive


def _normalization_error(physical, normalized, scale):
    return math.fsum(abs(scale * a - b) for a, b in zip(normalized, physical))


def _signal_statistics(members, counts):
    total = sum(counts)
    ordered = sorted(zip((member.target.signals for member in members), counts))
    expected = sum(degree * count for degree, count in ordered) / total
    # Bounded finite CDF factorization: P(max(D_x,D_y)<=d) = F(d)^2.
    cumulative, numerator = 0, 0
    for degree, count in ordered:
        following = cumulative + count
        numerator += degree * (following * following - cumulative * cumulative)
        cumulative = following
    return expected, max(degree for degree, _ in ordered), numerator / (total * total)


def plan_sampled_qsp_bank(bucket_times: tuple[float, ...], *,
                          mean_amplitude_allowances: float | tuple[float, ...],
                          policy: str, cutoffs: tuple[int, ...] | None = None,
                          allowance_frame: str = "algorithmic", margin: float = 1 / 16,
                          max_bytes: int | None = None, max_fft: int | None = None) -> SampledQSPBankPlan:
    """Plan supplied finite bucket times and their recovered mean-amplitude limits.

    ``policy`` is explicitly ``"deterministic"`` or ``"stochastic"``. Times
    are nonnegative and strictly increasing; this function does not establish
    physical row coverage. A scalar allowance applies to every bucket, or a
    tuple supplies one positive allowance per bucket. The frame is
    ``"algorithmic"`` or ``"physical_inner"``; the latter requires the caller's
    B22 allowance after other applicable physical errors have been accounted.
    Neither frame supplies a total physical guarantee here.

    Stochastic cutoffs are supplied once per bucket or floor(d/2). Exact-zero
    tail mass collapses to the same deterministic target. No alternative is
    selected after a refusal. Half the allowance controls reference truncation;
    the actual law and normalization discrepancies leave the frozen uniform
    phase allowance (allowance-beta_0)/zeta. ``max_bytes`` (default None, the
    physical memory) bounds the retained member arrays and each phase fit,
    and an explicit ``max_fft`` caps every fit's FFT length. The first FFT
    attempt of each bucket's highest-degree member, and with it that of
    every member, is checked against both before this function returns
    (B9). Errors keep their stage and cause in ``SampledQSPBankError``.
    """
    if policy not in ("deterministic", "stochastic"):
        raise ValueError("policy must be deterministic or stochastic")
    if allowance_frame not in ("algorithmic", "physical_inner"):
        raise ValueError("allowance_frame must be algorithmic or physical_inner")
    times = tuple(finite_real(t, "bucket time") for t in bucket_times)
    if not times or times[0] < 0 or any(a >= b for a, b in zip(times, times[1:])):
        raise ValueError("bucket times must be nonnegative and strictly increasing")
    allowances = (tuple(mean_amplitude_allowances) if isinstance(mean_amplitude_allowances, tuple)
                  else (mean_amplitude_allowances,) * len(times))
    if len(allowances) != len(times) or (cutoffs is not None and len(cutoffs) != len(times)):
        raise ValueError("one allowance and cutoff are required per bucket")
    allowances = tuple(finite_real(a, "mean amplitude allowance") for a in allowances)
    if any(a <= 0 for a in allowances):
        raise ValueError("mean amplitude allowances must be positive")
    margin = finite_real(margin, "margin")
    if not 0 < margin < 1:
        raise ValueError("margin must lie in (0,1)")
    max_bytes = None if max_bytes is None else integer(max_bytes, "max_bytes", 1)
    descriptors = []
    for index, (time, allowance) in enumerate(zip(times, allowances)):
        try:
            expansion = jacobi_anger_expansion(time, min(allowance / 2, .5), max_bytes=max_bytes)
        except ValueError as cause:
            raise _bank_failure(str(cause), stage="reference selection", cause=cause, bucket_index=index) from cause
        coefficients = _shell_coefficients(expansion)
        cutoff = (expansion.degree if policy == "deterministic" else
                  expansion.degree // 2 if cutoffs is None else integer(cutoffs[index], "cutoff", 0))
        if cutoff > expansion.degree:
            raise SampledQSPBankError("cutoff exceeds selected expansion degree", stage="cutoff",
                                     bucket_index=index)
        shells = tuple(n for n in range(cutoff + 1, len(coefficients)) if coefficients[n] != 0)
        mass = math.fsum(abs(coefficients[n]) for n in shells)
        descriptors.append((expansion, coefficients, cutoff, shells, mass, allowance))

    # The member population grows quadratically with the degree: a target
    # retains two complex coefficient vectors and three real angles per
    # signal slot, 56 bytes/slot. Check it before constructing any member.
    retained = 0
    for expansion, coefficients, cutoff, shells, mass, _ in descriptors:
        orders = shells if mass else (max((n for n, c in enumerate(coefficients) if c != 0), default=0),)
        retained += 32 * len(coefficients) + 56 * sum(2 * n + 1 for n in orders)
    try:
        _check_bytes(retained, max_bytes, "sampled QSP member arrays")
    except ValueError as cause:
        raise _bank_failure(str(cause), stage="retained member arrays", cause=cause) from cause
    buckets = []
    for bucket_index, (expansion, coefficients, cutoff, shells, mass, allowance) in enumerate(descriptors):
        try:
            # A positive tail with no ordinary binary64 value retains its log;
            # the already established selection tolerance is a conservative proxy.
            delta = expansion.epsilon if expansion.tail_bound is None else expansion.tail_bound
            if not mass:
                counts = (2**52,)
                if expansion.tau:
                    deterministic = plan_jacobi_anger_laurent(expansion, margin=margin)
                    physical = _laurent(coefficients)
                    normalization = _normalization_error(physical, deterministic.coefficients, deterministic.scale)
                    deterministic = replace(deterministic, approximation_error_bound=delta + normalization,
                                            full_ja=(deterministic.scale, delta + normalization))
                else:
                    recovery = float(np.nextafter(1 / (1 - margin), np.inf))
                    normalized = (1 / recovery,)
                    normalization = abs(recovery * normalized[0] - 1)
                    deterministic = plan_laurent_target(
                        normalized, minimum_power=0, polynomial_time=0, scale=recovery, margin=margin,
                        norm_bound=float(np.nextafter(abs(normalized[0]), np.inf)),
                        norm_cap_qualification="constant coefficient; numerical division without enclosure",
                        approximation_error_bound=normalization, tail_bound=0)
                members = [SampledQSPMemberPlan(shell=None, tail_coefficient=None, target=deterministic,
                                                normalization_error=normalization)]
                recovery = deterministic.scale
                law_error = 0.0
                mean_normalization = normalization
            else:
                law = integer_categorical(tuple(math.log(abs(coefficients[n])) for n in shells))
                counts = tuple(int(count) for count in law.counts)
                gammas = tuple(mass * (coefficients[n] / abs(coefficients[n])) for n in shells)
                prefix_l1 = math.fsum(abs(c) for c in coefficients[:cutoff + 1])
                # B6 plus a direct one-norm cap for the actual stored members.
                # The direct cap includes each stored gamma; no ideal phase/mass
                # assumption is used to certify its construction arithmetic.
                cap = max(1 + 2 * mass + delta, prefix_l1 + max(abs(gamma) for gamma in gammas))
                recovery = float(np.nextafter(cap / (1 - margin), np.inf))
                members = []
                for n, gamma in zip(shells, gammas):
                    raw_shells = coefficients[:cutoff + 1] + (0j,) * (n - cutoff - 1) + (gamma,)
                    physical = _laurent(raw_shells)
                    normalized = tuple(c / recovery for c in physical)
                    normalization = _normalization_error(physical, normalized, recovery)
                    # Direct B8 form: subtract the stored reference in the shell
                    # basis, whose elements have circle norm <=1. This includes
                    # the actual gamma discrepancy once, before normalization.
                    individual = delta + math.fsum(abs(
                        (raw_shells[k] if k < len(raw_shells) else 0j) - c)
                        for k, c in enumerate(coefficients)) + normalization
                    target = plan_laurent_target(
                        normalized, minimum_power=-n, polynomial_time=expansion.tau, scale=recovery,
                        margin=margin, norm_bound=float(np.nextafter(math.fsum(abs(c) for c in normalized), np.inf)),
                        norm_cap_qualification="stored normalized coefficient one-norm; numerical without enclosure",
                        approximation_error_bound=individual, tail_bound=expansion.tail_bound,
                        tail_log_bound=expansion.tail_log_bound)
                    members.append(SampledQSPMemberPlan(shell=n, tail_coefficient=gamma, target=target,
                                                         normalization_error=normalization))
                total = sum(counts)
                # B7 uses the actual integer law and actual fixed-W gamma.
                law_error = math.fsum(abs(count / total * gamma - coefficients[n])
                                      for count, gamma, n in zip(counts, gammas, shells))
                mean_normalization = math.fsum(count / total * member.normalization_error
                                              for count, member in zip(counts, members))
            mean_error = delta + law_error + mean_normalization
            epsilon_phase = (allowance - mean_error) / recovery
            if not epsilon_phase > 0:
                raise SampledQSPBankError("mean-amplitude allowance exhausted before phase synthesis",
                                         stage="phase allowance", bucket_index=bucket_index)
        except SampledQSPBankError:
            raise
        except (ValueError, ArithmeticError, MemoryError) as cause:
            raise _bank_failure(str(cause), stage="member construction", cause=cause,
                                bucket_index=bucket_index) from cause
        scheduled = [replace(member, target=with_phase_allowance(member.target, epsilon_phase=epsilon_phase,
                                                                 max_bytes=max_bytes, max_fft=max_fft))
                     for member in members]
        expected, maximum, expected_maximum = _signal_statistics(scheduled, counts)
        buckets.append(SampledQSPBucketPlan(
            polynomial_time=expansion.tau, expansion=expansion, reference_coefficients=coefficients,
            cutoff=cutoff, tail_mass=mass, counts=counts, members=tuple(scheduled), recovery=recovery,
            mean_amplitude_allowance=allowance, law_error=law_error, mean_normalization_error=mean_normalization,
            mean_approximation_error=mean_error, epsilon_phase=epsilon_phase, expected_signals=expected,
            maximum_signals=maximum, expected_independent_pair_signals=2 * expected,
            expected_maximum_pair_signals=expected_maximum))
    # B9: the first FFT attempt grows with the degree (the target's signal
    # count), so each bucket's highest-degree member admits that of all its
    # members, before any fit and any draw.
    for bucket_index, bucket in enumerate(buckets):
        member_index = max(range(len(bucket.members)), key=lambda index: bucket.members[index].target.signals)
        member = bucket.members[member_index]
        try:
            _check_target_start(member.target)
        except ValueError as cause:
            raise _bank_failure(
                f"the phase fit of bucket {bucket_index} member {member_index} (shell {member.shell}) cannot "
                f"start: {cause}; {_PHASE_DIRECTION}", stage="phase start", cause=cause,
                bucket_index=bucket_index, member_index=member_index, shell=member.shell) from cause
    return SampledQSPBankPlan(buckets=tuple(buckets), policy=policy, allowance_frame=allowance_frame,
                              retained_array_bytes=retained, max_bytes=max_bytes)


def prepare_sampled_qsp_bank(plan: SampledQSPBankPlan) -> PreparedSampledQSPBank:
    """Prepare every admitted target once, retaining actual-angle residuals.

    The first failure stops this bank, with its member, shell, the solver's
    evidence, the completed targets and the original cause. No member is
    replaced and no other target is tried.
    """
    completed = []
    buckets = []
    for bucket_index, bucket in enumerate(plan.buckets):
        targets = []
        for member_index, member in enumerate(bucket.members):
            try:
                target = prepare_qubitized_target(member.target)
            except ValueError as cause:
                raise _bank_failure(
                    f"the phase fit of bucket {bucket_index} member {member_index} (shell {member.shell}) failed: "
                    f"{cause}; raise {_PHASE_ALLOWANCE}", stage="phase preparation", cause=cause,
                    bucket_index=bucket_index, member_index=member_index, shell=member.shell,
                    completed_targets=completed) from cause
            completed.append(target)
            targets.append(target)
        residuals = tuple(target.synthesis_residual for target in targets)
        if any(residual is None for residual in residuals):
            individual = mean = recovered = None
        else:
            individual = max(member.target.approximation_error_bound / bucket.recovery + residual
                             for member, residual in zip(bucket.members, residuals))
            mean = bucket.mean_approximation_error / bucket.recovery + math.fsum(
                probability * residual for probability, residual in zip(bucket.probabilities, residuals))
            recovered = bucket.recovery * mean
        buckets.append(PreparedQSPBucket(plan=bucket, members=tuple(targets),
                                        normalized_individual_error=individual, normalized_mean_error=mean,
                                        recovered_mean_error=recovered))
    return PreparedSampledQSPBank(plan=plan, buckets=tuple(buckets), solver_calls=len(completed))
