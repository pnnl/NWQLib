"""Finite product-formula recipes of the sampled-pair LCHS observable: Richardson, PTSC and two-member MPF.

``LCHS(estimator=SampledPairs(product_formula=...))``
replaces the plain branch product formula of every pair row by one of three
finite recipes. The pair rows, their proposal law, phase-aware readout and
signed Q/N reducer are those of `sampled_rows`, `sampled_native` and
`sampled_moments`; a recipe adds per-row random choices, a recovery factor
and its own deterministic bias relation. Planning selects every recipe
parameter from support-wide bounds before any row is drawn, draws the rows
and then the recipe choices once from the Method's random stream, and saves
them here. Loading, analysis, resource estimates and continuation read the
saved choices and never draw again.

Notation. A pair row has branches x, y with elapsed times t_b, kernel nodes
k_b and importance weights w_b = a/q of the original outer proposal q
(`sampled_rows`). B >= E|w| is the mean weight bound and W >= sup|w| the
weight cap (``LCHSPairBias.mean_weight_bound``, ``LCHSSampledRows``). The
centered Pauli tables are H_c = sum_l h_l P_l and L_c = sum_l l_l P_l with
one-norms lambda_H, lambda_L; K_* bounds |k| on the whole kernel support and
T_e is the supremum of the elapsed-time support. O = sum_l o_l P_l with
C_O = sum_l |o_l| >= ||O||.

Signed recipe factors (P1). A row samples member J with the actual integer
probability pi_j and multiplies both ordinary scores by gamma_j = b_j/pi_j.
At fixed x, y and term, E_J[gamma_J F_J] = sum_j b_j F_j, so the mean is the
signed combination. Both Z_Q and Z_N carry gamma_J, and their same-shot
product carries gamma_J**2. The row keeps the nonnegative magnitude
|w_x w_y|*|gamma_J| and implements a negative gamma_J by adding pi to the
readout phase, which reverses s_a and so Q and N together and leaves the
system parity of QN unchanged. The recipe sign is saved in the member table,
apart from the physical phases of the row table. With the stored multiplier
gamma_hat_j the exact-product score ranges are H_N = Gamma_num*W**2 and
H_Q = Gamma_num*H_O*W**2 with Gamma_num = max_j |gamma_hat_j| over the whole
member table, and the single-shot second moment factor is
J2_num = sum_j pi_j*gamma_hat_j**2. A maximum over the drawn rows would not
bound unseen rows.

Richardson (P2, P3; Watson and Watkins, arXiv:2408.14385v2, Sec. 5,
Lemma 7, Eq. (84)). Every level uses the same Strang word over the sorted
union P_1, ..., P_L of the centered labels: half factors of P_1..P_{L-1},
the full factor of P_L, then the half factors in reverse, which is the
order-2 occurrence template of `sampled_native.pair_select_plan`. At a pair
the time-one joint generator terms are
G_j = diag(t_x c_xj P_j, t_y c_yj P_j) with c_bj = h_j + k_b l_j, and
G_j = ((t_x c_xj + t_y c_yj)/2) I (x) P_j + ((t_x c_xj - t_y c_yj)/2) Z_a (x) P_j,
two commuting Pauli rotations whose norm sum is max(|t_x c_xj|, |t_y c_yj|).
Summing over j gives sum_j ||G_j|| <= L_* = T_e*(lambda_H + K_* lambda_L)
for every finite or continuous pair of the support, including unequal
source times. Reading the Strang word as 2L - 1 stages with one G_j each and
coefficient magnitude at most 1 gives the conservative XS9 constants
lambda = 4 L_*, T = 1 and A = (2L - 1)*lambda. The levels are r = (s, 2s);
with x_l = r_l**-2 the Lagrange weights at x = 0 are
w_l = r_l**(2(m-1))/prod_{h != l}(r_l**2 - r_h**2), here (-1/3, 4/3) for
any s, with Lambda = sum |w_l| = 5/3, sum w = 1 and sum w_l r_l**-2 = 0. For
r_min = s > 2A,

    B_ext <= 4*Lambda*max(1, A)**2*(A/s)**4

bounds |sum_l w_l f_l - f_exact| for every unit-norm pair observable f.
L <= 1 or A = 0 is exact evolution and B_ext = 0. The member law is the
integer table of |w_l|/Lambda, gamma_hat_l is the correctly rounded
w_l/pi_l, and the complete coefficient discrepancy is the exact rational
E_eff = sum_l |pi_l*gamma_hat_l - w_l|. Since |f_l| <= 1,
|sum_l pi_l gamma_hat_l f_l - f_exact| <= B_ext + E_eff, and integrating
|w_x w_y| over independent x, y (whose mean is (E|w|)**2 <= B**2) gives

    |E Z_N - N_v| <= B**2*(B_ext + E_eff),   |E Z_Q - Q_v| <= C_O*B**2*(B_ext + E_eff).

With a Richardson.pf_allowance epsilon on N (and C_O*epsilon on Q), the selection
needs B_ext(s) <= epsilon/B**2 - E_eff; B_ext decreases as s**-4 once
s > 2A. A set TrotterErrorBudgeted.max_steps gives the cap s_max = floor(max_steps/2),
which is tested first, and otherwise the upper end doubles until it meets
the allowance. An integer bisection then finds the least sufficient s, all
in exact rationals. A fixed
``Trotter.steps`` publishes the bound at s when s > 2A and, in the general
case of L >= 2 centered labels with A > 0, otherwise leaves it unknown. With
L <= 1 or L_* = 0, B_ext = 0 needs no s > 2A, and the published operator
bound B_ext + E_eff is E_eff.

PTSC (XS12-XS14, XS28; Zeng, Sun, Jiang and Zhao, arXiv:2212.04566v2,
Sec. IV.2, Proposition 4). One branch evolves for t under
H_c + k L_c = sum over the fixed unmerged H list then L list, with L
coefficients k*l_l; the Lie step S_1 applies the list in that order and
lambda(k) = lambda_H + |k| lambda_L. With x = t/r, a = 2 lambda x < 1,
eta_s = a**s/s!, h = eta_2 + eta_3 and A_0 = sqrt(1 + h**2),

    mu_step = A_0 + sum_{s=4}^{s_c} eta_s,   log mu = r*log mu_step,
    log mu_step = log1p(h**2)/2 + log1p(sum_{s>=4} eta_s/A_0),
    p_2 = (A_0/mu_step)/(1 + a/3),  p_3 = (A_0/mu_step)(a/3)/(1 + a/3),
    p_s = eta_s/mu_step (4 <= s <= s_c).

A segment draws s, b ~ Binomial(s, 1/2) as s fair bits, and s labels, each
choosing the H or L list with weights lambda_H/lambda(k) and
|k| lambda_L/lambda(k) and then a list entry by coefficient magnitude. The
first b labels keep their drawn order and the other s - b are sorted in the
Lie order, and

    Q = (-i)**b i**(s - b) Pbar_{l_1}...Pbar_{l_b} Pbar_{j(1)}...Pbar_{j(s-b)}

with Pbar = sign(coefficient)*P and the full four-valued Pauli phase. For
s >= 4 the correction is C = Q; for s = 2, 3 it is C = exp(i atan(h) P')
with P' = Q for Hermitian Q and -iQ otherwise, so P' = +-I gives a phase
that is relative between the two branches. Each segment applies S_1 and
then C, and mu_step*E[C] = I + F_2 + F_3 + sum_{s=4}^{s_c} F_s, the Taylor
truncation of V = exp(-ixH)S_1^dagger (XS28e). The two branches draw
independent sequences, fixed for every shot of the row, and the row
magnitude is multiplied by mu_x*mu_y under the original outer proposal.

Finite law (P4). Every finite law is an integer table with probabilities
n_i/2**52. For an ideal law of k positive entries the count part of its L1
change is at most 2k/D, D = 2**52, which gives d_mix <= 4/D,
d_H <= 2 L_H/D, d_L <= 2 L_L/D, d_label <= d_mix + max(d_H, d_L),
d_order <= 2(s_c - 1)/D and d_seg <= min(2, d_order + s_c d_label), an
absent component contributing nothing. Each C is unitary and each recovered
segment has norm at most mu_step, so replacing the segments one at a time
gives delta_law <= r mu d_seg. With the Taylor tail
delta_Taylor <= r mu [2e lambda t/(r(s_c + 1))]**(s_c + 1) the branch error
is delta <= delta_Taylor + delta_law, and mu <= mu_cap with lambda t <= L_time
= T_e(lambda_H + K_* lambda_L) makes it a uniform delta_*. The recovered
branch depends on its own x only, so ||v_comp - v|| <= E|w| delta_* <= B delta_*
and

    |E Z_N - N_v| <= B**2*(2 delta_* + delta_***2),  Q: C_O times this (P6).

Selection: the least r with r > 2 L_time and
r**3 log mu_cap >= (e + 2/9)(2 L_time)**4 (XS13), at most a set
TrotterErrorBudgeted.max_steps, found by bisection after testing that cap or after
doubling to a sufficient r, then the first s_c = 3, 4, ... at
most PTSC.taylor_cutoff with
r mu_cap {[2e L_time/(r(s_c + 1))]**(s_c + 1) + d_seg(s_c)} <= delta_allow,
delta_allow = min(1, epsilon/(3 B**2)), because 2 delta + delta**2 <= 3 delta
for delta <= 1. The sum is not monotone in s_c, so it is scanned, not
bisected. The arithmetic forming the integer targets (e_target of P4), the
low-order angles and the stored normalizations is not enclosed: the bound
covers its named Taylor and count components only. The score range uses
the mathematical cap mu_cap**2, and a drawn row whose stored normalization
exceeds it is refused.

Two-member MPF (P7, P8; arXiv:2607.12225v1, Sec. IV and Methods Sec. VIII).
For K >= 2 ordered joint terms G_1..G_K, beta = 1 - alpha and h = 1/r in the
time-one convention,

    S_alpha(h) = F_alpha exp(-ihG_K) R_beta F_beta exp(-ihG_K) R_alpha,
    F_a = prod_{j=1}^{K-1} exp(-i a h G_j),  R_a = prod_{j=K-1}^{1} exp(-i a h G_j),

a palindrome of 4K - 3 factors after merging the two central G_1 factors,
applied r/2 times for even r (physical time 2h per macrostep). The members
alpha_1 = (3 + sqrt 3)/9 and alpha_2 = (1 + sqrt 3)/3 with c = (3/2, -1/2)
satisfy sum c = 1, sum c alpha = 1/3 and sum c alpha**2 = 0, which cancels
every cubic word defect (f_1 = alpha**2 - 3 alpha + 1, f_2 = alpha - 1/3)
and hence the global h**2 term of the pair observable. The member law is
exactly (3/4, 1/4), so gamma = (2, -2), Gamma_num = 2 and J2_num = 4. No
uniform finite-h remainder is admitted, so the deterministic bias stays
unknown for K >= 2: raw signed estimates and sampling radii are published,
no covered radius or ratio. The binary64 alpha differs from the exact value
by Delta alpha; for exact exponentials it changes the pair observable by at
most 4 sum_{j<K} ||G_j|| sum_l |c_l||Delta alpha_l| (P8), which is part of
that unknown remainder because no enclosure of the stored angles is given.
K <= 1 is exact evolution for both members.
"""

from nwqlib._limits import ResourceLimitError, exceeds
from dataclasses import dataclass
from fractions import Fraction
from math import atan, e as euler, exp, floor, fsum, lgamma, log, log1p, pi, sqrt
from typing import Literal

import numpy as np

from nwqlib.algorithms.protocol import ApplicabilityError
from nwqlib.core.records import FrozenArray, Nonnegative, PositiveInt, Real, Record, Text
from .sampled_rows import IntegerCategorical, integer_categorical

RICHARDSON = "richardson-m2-dyadic-strang/1"
MPF = "mpf-two-member-p7/1"
PTSC = "ptsc-first-order-p4/1"
MEMBER_DRAW_ORDER = "pair rows, then one member per row/1"
PTSC_DRAW_ORDER = ("pair rows, then per row branch x then y: segment orders, fair bits, list choices when both lists "
                   "are present, label integers/1")
_TOTAL = 1 << 52
# Richardson m = 2 at levels (s, 2s): w = (-1/3, 4/3), Lambda = 5/3.
_RICHARDSON_WEIGHTS = (Fraction(-1, 3), Fraction(4, 3))
# MPF members: binary64 values of alpha_1 = (3 + sqrt 3)/9, alpha_2 = (1 + sqrt 3)/3.
_MPF_ALPHAS = ((3.0 + sqrt(3.0)) / 9.0, (1.0 + sqrt(3.0)) / 3.0)
_MPF_WEIGHTS = (Fraction(3, 2), Fraction(-1, 2))
_XS13 = euler + 2.0 / 9.0
# Bytes per saved eight-byte element: the value, its base64 Record JSON
# (4/3 of the bytes) and that text's encoded copy, as in sampled_plan._row_bytes.
_SAVED = 40


class LCHSPairMembers(Record):
    """Richardson or two-member MPF choices of a sampled-pair Plan: one finite member per row, shared by both branches.

    `LCHSReconstruction.pair_recipe` holds it, chosen and drawn once at
    planning (module docstring). Member j evolves both branches of a row
    with its own product formula; the row's score magnitude is multiplied by
    |gamma_hat_j| and a negative gamma_hat_j adds pi to the readout phase.

    Attributes:
        recipe: ``"richardson"`` or ``"mpf"``.
        version: Versioned recipe definition.
        draw_order: Versioned order of the member draws after the pair rows.
        member_steps: Step divisor r_j of each member's angles: Richardson
            (s, 2s), MPF (r, r).
        member_repetitions: Repetitions of the member's one-step word:
            Richardson r_j Strang steps, MPF r/2 macrosteps.
        member_alphas: Binary64 alpha of each MPF member, empty for
            Richardson.
        member_weights: Exact rational weights b_j = numerator/denominator.
        member_counts: Integer law counts n_j, summing to 2**52.
        member_multipliers: Stored multipliers gamma_hat_j.
        multiplier_bound: Gamma_num = max_j |gamma_hat_j|.
        multiplier_second_moment: J2_num = sum_j (n_j/2**52) gamma_hat_j**2,
            rounded upward.
        coefficient_discrepancy: E_eff = sum_j |pi_j gamma_hat_j - b_j|,
            rounded upward; zero for MPF, whose factors are exact.
        remainder: Richardson B_ext at the selected s, rounded upward, zero
            for exact evolution, or None when the XS9 premise s > 2A fails
            or (MPF with K >= 2) no finite remainder is admitted.
        allowance: The ``Richardson.pf_allowance`` the selection met, or None for
            fixed steps.
        member: Member index of every row, int64.
        qualification: Scope of the bound and the unenclosed arithmetic.
    """

    recipe: Literal["richardson", "mpf"]
    version: Text
    draw_order: Text = MEMBER_DRAW_ORDER
    member_steps: tuple[PositiveInt, ...]
    member_repetitions: tuple[PositiveInt, ...]
    member_alphas: tuple[Real, ...] = ()
    member_weights: tuple[tuple[int, PositiveInt], ...]
    member_counts: tuple[PositiveInt, ...]
    member_multipliers: tuple[Real, ...]
    multiplier_bound: Nonnegative
    multiplier_second_moment: Nonnegative
    coefficient_discrepancy: Nonnegative
    remainder: Nonnegative | None
    allowance: Nonnegative | None = None
    member: FrozenArray
    qualification: Text

    def row_factors(self):
        """Return |gamma_hat| of every row's member, the factor of its score magnitude."""
        return np.abs(np.asarray(self.member_multipliers))[self.member.array]

    def row_phases(self):
        """Return the recipe phase of every row: pi for a negative member multiplier, else 0."""
        return np.where(np.asarray(self.member_multipliers)[self.member.array] < 0, pi, 0.0)

    @property
    def score_factor(self):
        """Gamma_num, the factor of the exact-product score ranges."""
        return self.multiplier_bound


class LCHSPairCompensation(Record):
    """First-order PTSC choices of a sampled-pair Plan: two independent correction sequences per row.

    `LCHSReconstruction.pair_recipe` holds it (module docstring). Arrays
    indexed (row, branch) use branch 0 for x and 1 for y; segment arrays add
    the segment index j = 0..r-1 in chronological order. A segment's
    correction acts after its Lie step. A Pauli word is stored as int64
    masks over the system qubits, bit j for qubit j (the last label
    character), X on bits of ``segment_x`` and Z on bits of ``segment_z``,
    both for Y; Q = i**phase times that word.

    Attributes:
        recipe: ``"ptsc"``.
        version, draw_order: Versioned recipe and draw order.
        steps: Common segment count r of every branch.
        cutoff: Taylor cutoff s_c.
        mu_cap: The ``PTSC.mu_cap`` that bounds every branch normalization.
        allowance: The ``PTSC.pf_allowance`` the selection met.
        taylor_bound: r mu_cap [2e L_time/(r(s_c + 1))]**(s_c + 1).
        law_bound: r mu_cap d_seg(s_c), the count part of the finite law.
        branch_bound: delta_* = taylor_bound + law_bound.
        h_counts, l_counts: Integer counts of the H and L list laws, empty
            for an absent list.
        log_mu: Stored log mu of every branch, shape (B, 2).
        rotation_angle: atan(h) of every branch, shape (B, 2).
        order_counts: Counts of the orders 2..s_c of every branch, shape
            (B, 2, s_c - 1), zero for a branch with a = 0.
        mixture_counts: Counts of the (H, L) list choice, shape (B, 2, 2).
        segment_order: Drawn order s of every segment, 0 for an identity
            correction (a = 0), shape (B, 2, r).
        segment_phase: Power of i of Q, shape (B, 2, r).
        segment_x, segment_z: Pauli masks of Q, shape (B, 2, r).
        qualification: Scope of the bound and the unenclosed arithmetic.
    """

    recipe: Literal["ptsc"]
    version: Text = PTSC
    draw_order: Text = PTSC_DRAW_ORDER
    steps: PositiveInt
    cutoff: PositiveInt
    mu_cap: Real
    allowance: Nonnegative
    taylor_bound: Nonnegative
    law_bound: Nonnegative
    branch_bound: Nonnegative
    h_counts: FrozenArray
    l_counts: FrozenArray
    log_mu: FrozenArray
    rotation_angle: FrozenArray
    order_counts: FrozenArray
    mixture_counts: FrozenArray
    segment_order: FrozenArray
    segment_phase: FrozenArray
    segment_x: FrozenArray
    segment_z: FrozenArray
    qualification: Text

    def row_factors(self):
        """Return mu_x*mu_y of every row, the factor of its score magnitude."""
        return np.exp(self.log_mu.array.sum(axis=1))

    def row_phases(self):
        """Return zeros: PTSC recovery factors are positive."""
        return np.zeros(self.log_mu.array.shape[0])

    @property
    def score_factor(self):
        """mu_cap**2, the factor of the exact-product score ranges."""
        return self.mu_cap * self.mu_cap

    def corrections(self, row):
        """Return the r segments of ``row`` as ((x descriptor), (y descriptor)) pairs.

        A descriptor is ``(order, phase, x_mask, z_mask, angle)`` with the
        branch's low-order angle atan(h).
        """
        order, phase = self.segment_order.array[row], self.segment_phase.array[row]
        xs, zs, angle = self.segment_x.array[row], self.segment_z.array[row], self.rotation_angle.array[row]
        return tuple(tuple((int(order[b, j]), int(phase[b, j]), int(xs[b, j]), int(zs[b, j]), float(angle[b]))
                           for b in (0, 1)) for j in range(self.steps))


@dataclass(frozen=True)
class Support:
    """Support-wide facts that every recipe selection needs before a row is drawn.

    Attributes:
        active_time: T_e.
        node_bound: K_*, a bound on |k| over the whole kernel support.
        h_norm, l_norm: lambda_H and lambda_L as exact rationals.
        labels: Number L of distinct nonidentity labels of H_c and L_c.
        mean_weight: B as a PhysicalScale.
        num_qubits: System qubits q.
    """

    active_time: float
    node_bound: float
    h_norm: Fraction
    l_norm: Fraction
    labels: int
    mean_weight: object
    num_qubits: int

    @property
    def time_norm(self):
        """L_time = L_* = T_e*(lambda_H + K_* lambda_L), exact for the stored binary64 inputs."""
        return Fraction(self.active_time) * (self.h_norm + Fraction(self.node_bound) * self.l_norm)


def _operator_allowance(allowance, mean_weight, factor=1):
    """Return epsilon/(factor*B**2) exactly, for B = mantissa*2**exponent."""
    square = Fraction(mean_weight.mantissa) ** 2 * Fraction(2) ** (2 * mean_weight.exponent)
    return Fraction(allowance) / (factor * square)


def _upward(value):
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    return _upward_float(value)


def _richardson_remainder(a, s):
    """B_ext = 4*Lambda*max(1, A)**2*(A/s)**4 (XS9 with m = 2, p = 2), exact."""
    if a == 0:
        return Fraction(0)
    return 4 * Fraction(5, 3) * max(Fraction(1), a) ** 2 * (a / s) ** 4


def _least_step(start, admitted, cap):
    """Return the least step count s >= ``start`` with ``admitted(s)``, for a test that stays true once it holds.

    With ``cap`` the caller has checked ``admitted(cap)``, and the search
    bisects [start, cap]. Without one the upper end doubles from ``start``
    until it is admitted, which ends because both callers' tests hold for
    every large enough s.
    """
    high = cap
    if high is None:
        high = start
        while not admitted(high):
            high *= 2
    low = start
    while low < high:
        middle = (low + high) // 2
        if admitted(middle):
            high = middle
        else:
            low = middle + 1
    return low


def _member_table(weights, counts):
    """Return the stored multipliers, Gamma_num, J2_num and E_eff of an integer member law."""
    probabilities = [Fraction(int(count), _TOTAL) for count in counts]
    multipliers = tuple(float(weight / probability) for weight, probability in zip(weights, probabilities))
    discrepancy = sum((abs(p * Fraction(g) - w) for p, g, w in zip(probabilities, multipliers, weights)), Fraction(0))
    second = sum((p * Fraction(g) ** 2 for p, g in zip(probabilities, multipliers)), Fraction(0))
    return multipliers, max(abs(g) for g in multipliers), _upward(second), discrepancy


def select_richardson(method, support):
    """Select the m = 2 Richardson levels before any row is drawn (module docstring).

    Returns:
        fields (dict): `LCHSPairMembers` fields without ``member``.
        operator (Fraction | None): B_ext + E_eff, or None when B_ext is unknown.
        rule (str): ``"fixed"`` or ``"weighted"``.
        reasons (tuple): Why the remainder is unknown.

    Raises:
        ResourceLimitError: When a set TrotterErrorBudgeted.max_steps leaves no s up to
            floor(max_steps/2) that meets the allowance.
        ApplicabilityError: When E_eff alone exceeds the allowance.
    """
    count = support.labels
    a = Fraction(0) if count <= 1 else 4 * (2 * count - 1) * support.time_norm
    with np.errstate(divide="ignore"):
        counts = integer_categorical(np.log([abs(float(w)) for w in _RICHARDSON_WEIGHTS])).counts
    multipliers, bound, second, discrepancy = _member_table(_RICHARDSON_WEIGHTS, counts)
    reasons = ()
    formula, recipe = method.ham_evo, method.estimator.product_formula
    if formula.kind == "trotter":
        rule, s = "fixed", formula.steps
        remainder = _richardson_remainder(a, s) if a == 0 or s > 2 * a else None
        if remainder is None:
            reasons = (f"the Richardson remainder needs s > 2A for the XS9 bound; s = {s} and "
                       f"A = {float(a)!r}",)
    else:
        rule = "weighted"
        target = _operator_allowance(recipe.pf_allowance, support.mean_weight) - discrepancy
        if target <= 0:
            raise ApplicabilityError(
                f"the Richardson coefficient discrepancy E_eff = {float(discrepancy)!r} times B**2 already "
                f"reaches Richardson.pf_allowance={recipe.pf_allowance!r}; no level s reduces it. Raise "
                "pf_allowance")
        cap = None if formula.max_steps is None else formula.max_steps // 2
        if cap is not None and cap < 1:
            raise ResourceLimitError("Richardson levels (s, 2s) need TrotterErrorBudgeted.max_steps >= 2")
        if a == 0:
            s = 1
        else:
            if cap is not None and not (cap > 2 * a and _richardson_remainder(a, cap) <= target):
                reached = (f"{_upward(_richardson_remainder(a, cap) + discrepancy)!r}" if cap > 2 * a
                           else f"unavailable, because s = {cap} is not above 2A = {float(2 * a)!r}")
                raise ResourceLimitError(
                    f"Richardson levels (s, 2s) need 2s <= TrotterErrorBudgeted.max_steps={formula.max_steps}; at "
                    f"s = {cap} the bound B_ext + E_eff on each pair observable is {reached}, above the "
                    f"allowance Richardson.pf_allowance/B**2 = {float(target + discrepancy)!r}. Raise "
                    "pf_allowance or max_steps")
            # B_ext decreases as s**-4 above 2A and the target is positive.
            s = _least_step(floor(2 * a) + 1, lambda steps: _richardson_remainder(a, steps) <= target, cap)
        remainder = _richardson_remainder(a, s)
    qualification = ("Richardson m=2 levels (s, 2s) of one Strang word; XS9 remainder with A = (2L - 1)*4*L_* "
                     "for exact elementary exponentials; E_eff is exact for the stored multipliers; rounded "
                     "angles, native gates and the score, reduction and radius arithmetic have no enclosure; "
                     "observable-term and phase arithmetic perturbations stay unknown and would scale by "
                     "sum_j |pi_j gamma_hat_j|")
    fields = dict(recipe="richardson", version=RICHARDSON, member_steps=(s, 2 * s), member_repetitions=(s, 2 * s),
                  member_weights=tuple((w.numerator, w.denominator) for w in _RICHARDSON_WEIGHTS),
                  member_counts=tuple(int(c) for c in counts), member_multipliers=multipliers,
                  multiplier_bound=bound, multiplier_second_moment=second,
                  coefficient_discrepancy=_upward(discrepancy),
                  remainder=None if remainder is None else _upward(remainder),
                  allowance=None if rule == "fixed" else float(recipe.pf_allowance), qualification=qualification)
    return fields, None if remainder is None else remainder + discrepancy, rule, reasons


def select_mpf(method, support):
    """Return the two fixed MPF members at the Method's even ``Trotter.steps`` (module docstring).

    Returns the same tuple as `select_richardson`. The operator bound is zero
    for K <= 1 (exact evolution) and None otherwise, because no uniform
    finite-step remainder is admitted.
    """
    r = method.ham_evo.steps
    counts = (3 * _TOTAL // 4, _TOTAL // 4)
    multipliers, bound, second, discrepancy = _member_table(_MPF_WEIGHTS, counts)
    exact = support.labels <= 1
    reasons = () if exact else ("the two-member MPF has no admitted uniform finite-step remainder; its leading h**2 "
                                "cancellation is not an accuracy bound at this step size",)
    fields = dict(recipe="mpf", version=MPF, member_steps=(r, r), member_repetitions=(r // 2, r // 2),
                  member_alphas=_MPF_ALPHAS,
                  member_weights=tuple((w.numerator, w.denominator) for w in _MPF_WEIGHTS),
                  member_counts=counts, member_multipliers=multipliers, multiplier_bound=bound,
                  multiplier_second_moment=second, coefficient_discrepancy=_upward(discrepancy),
                  remainder=0.0 if exact else None,
                  qualification=("Two-member MPF with exact (3/4, 1/4) member law and gamma = (2, -2). At most one "
                                 "distinct nonidentity Pauli label occurs, so both members implement the same "
                                 "evolution and the exact-exponential remainder is zero. A covering radius also "
                                 "requires the complete planned row set and known model bias. Gate, score, "
                                 "reduction and radius arithmetic have no complete error enclosure." if exact else
                                 "two-member MPF on the P7 skeleton with exact (3/4, 1/4) member law and gamma = "
                                 "(2, -2); the finite-step remainder and the binary64 alpha perturbation are "
                                 "unknown; raw signed estimates carry sampling radii only"))
    return fields, Fraction(0) if exact else None, "fixed", reasons


def _segment_law_bound(cutoff, h_count, l_count, mixed):
    """Return d_seg(s_c) = min(2, d_order + s_c d_label), the count part of P4 (module docstring)."""
    lists = [2 * n / _TOTAL for n in (h_count, l_count) if n]
    label = (4 / _TOTAL if mixed else 0.0) + max(lists, default=0.0)
    return min(2.0, 2 * (cutoff - 1) / _TOTAL + cutoff * label)


def select_ptsc(method, support, *, h_count, l_count, clusters):
    """Select the common segment count r and cutoff s_c before any row is drawn (module docstring).

    Returns:
        fields (dict): `LCHSPairCompensation` fields without the drawn arrays.
        operator (float): 2 delta_* + delta_***2, the factor of B**2 in the N bias.
        work (int): Nominal generation work units, admitted against
            ``max_select_work`` before the draws.
        size (int): Known bytes of the saved arrays and generation scratch.

    Raises:
        ResourceLimitError: For an r above a set ``TrotterErrorBudgeted.max_steps`` or
            generation work above ``max_select_work``.
        ApplicabilityError: For more than 63 system qubits, or no cutoff up
            to ``PTSC.taylor_cutoff`` within the allowance.
    """
    q = support.num_qubits
    if q > 63:
        raise ApplicabilityError("PTSC saves each correction as int64 Pauli masks and supports at most 63 system "
                                 f"qubits; got {q}")
    recipe = method.estimator.product_formula
    mu_cap, ceiling, cap = float(recipe.mu_cap), recipe.taylor_cutoff, method.ham_evo.max_steps
    time_norm = support.time_norm
    log_cap = log(mu_cap)

    def admitted(r):
        return r > 2 * time_norm and r ** 3 * log_cap >= _XS13 * float(2 * time_norm) ** 4

    if time_norm == 0:
        r = 1
    else:
        if cap is not None and not admitted(cap):
            raise ResourceLimitError(
                f"PTSC needs r > 2*L_time and r**3*log(mu_cap) >= (e + 2/9)*(2*L_time)**4 with "
                f"L_time = {float(time_norm)!r}; r = TrotterErrorBudgeted.max_steps={cap} does not meet them. Raise "
                "PTSC.mu_cap or max_steps")
        # mu_cap > 1, so both conditions hold for every large enough r.
        r = _least_step(1, admitted, cap)
    delta = float(min(Fraction(1), _operator_allowance(recipe.pf_allowance, support.mean_weight, 3)))
    if not delta > 0:
        raise ApplicabilityError("PTSC.pf_allowance/(3*B**2) is not a positive binary64 number; raise "
                                 "pf_allowance")
    mixed = bool(h_count and l_count and support.node_bound > 0)
    for cutoff in range(3, ceiling + 1):
        if time_norm == 0:
            taylor = law = 0.0
        else:
            taylor = r * mu_cap * exp((cutoff + 1) * log(2 * euler * float(time_norm) / (r * (cutoff + 1))))
            law = r * mu_cap * _segment_law_bound(cutoff, h_count, l_count, mixed)
        if taylor + law <= delta:
            break
    else:
        raise ApplicabilityError(
            f"no PTSC cutoff s_c <= PTSC.taylor_cutoff={ceiling} brings r*mu_cap*(Taylor tail + finite-law term) "
            f"at r = {r} to the branch allowance {delta!r} = min(1, PTSC.pf_allowance/(3*B**2)); at s_c = "
            f"{ceiling} it is {taylor + law!r}. Raise taylor_cutoff or pf_allowance")
    branch = taylor + law
    lists = h_count + l_count
    work = (h_count * max(1, h_count.bit_length()) + l_count * max(1, l_count.bit_length())
            + 2 * clusters * cutoff * max(1, cutoff.bit_length())
            + 2 * clusters * r * (lists + cutoff * (max(1, cutoff.bit_length()) + max(1, lists.bit_length()) + q)))
    if exceeds(work, method.max_select_work):
        raise ResourceLimitError(f"PTSC correction generation needs {work} work units, above "
                                 f"LCHS.max_select_work={method.max_select_work}")
    size = (2 * clusters * r * 4 * _SAVED + 2 * clusters * (cutoff + 3) * _SAVED + 32 * r * cutoff
            + 256 * lists)
    fields = dict(recipe="ptsc", steps=r, cutoff=cutoff, mu_cap=mu_cap, allowance=float(recipe.pf_allowance),
                  taylor_bound=taylor, law_bound=law, branch_bound=branch,
                  qualification=("first-order PTSC with common r and s_c; the bound covers the XS12 Taylor tail and "
                                 "the P4 count quantization only; the arithmetic forming the integer targets, the "
                                 "low-order angles and the stored normalizations has no enclosure, and the score "
                                 "range uses the mathematical cap mu_cap; observable-term and phase arithmetic "
                                 "perturbations stay unknown and would scale by mu_x*mu_y"))
    return fields, 2 * branch + branch * branch, work, size


def member_bytes(clusters):
    """Known bytes of the member index array of a member recipe."""
    return clusters * _SAVED + 4096


def draw_members(fields, rng, clusters):
    """Draw one member per row from the Method's random stream and return the `LCHSPairMembers` record."""
    law = IntegerCategorical(counts=np.asarray(fields["member_counts"], dtype=np.int64))
    member = law.draw(rng.method, clusters).astype(np.int64)
    return LCHSPairMembers(**fields, member=FrozenArray(member))


def pauli_masks(label):
    """Return the (x, z) int masks of a Pauli label, bit j for qubit j (the last label character)."""
    x = z = 0
    for position, axis in enumerate(reversed(label)):
        if axis in "XY":
            x |= 1 << position
        if axis in "ZY":
            z |= 1 << position
    return x, z


def mask_label(x, z, width):
    """Return the Pauli label of masks (x, z) over ``width`` qubits."""
    return "".join("IXZY"[((x >> j) & 1) | (((z >> j) & 1) << 1)] for j in reversed(range(width)))


def correction_word(b, labels, words):
    """Return (phase, x, z) of Q = (-i)**b i**(s-b) Pbar_1...Pbar_s for one segment.

    ``labels`` are the s list indices with the first ``b`` in drawn order and
    the rest sorted, and ``words[g]`` is the signed operator Pbar_g as
    (e, x, z) for i**e X**x Z**z (a label word is i**popcount(x&z) X**x Z**z,
    because Y = iXZ, and a negative sign adds 2 to e). Multiplying
    i**e1 X**x1 Z**z1 by i**e2 X**x2 Z**z2 moves Z**z1 past X**x2, which
    gives i**(e1 + e2 + 2 popcount(z1 & x2)) X**(x1^x2) Z**(z1^z2). The
    returned phase is that of Q in front of its label word. The prefactor
    (-i)**b i**(s-b) is i**(s + 2b).
    """
    exponent, x, z = len(labels) + 2 * b, 0, 0
    for g in labels:
        ge, gx, gz = words[g]
        exponent += ge + 2 * (z & gx).bit_count()
        x ^= gx
        z ^= gz
    return (exponent - (x & z).bit_count()) % 4, x, z


def signed_words(h_terms, l_terms, node):
    """Return Pbar of every list entry as (e, x, z): H entries signed by h_l, L entries by k*l_l (sign(k) absorbed)."""
    words = []
    for index, (label, value) in enumerate((*h_terms, *l_terms)):
        x, z = pauli_masks(label)
        negative = value < 0 if index < len(h_terms) else (value < 0) != (node < 0)
        words.append(((x & z).bit_count() + 2 * negative, x, z))
    return words


def mixture_law(h_norm, l_norm, node):
    """Return the integer law of the (H, L) list choice, ideal weights lambda_H and |k| lambda_L; an absent list has zero count."""
    with np.errstate(divide="ignore"):
        return integer_categorical(np.log([h_norm, abs(node) * l_norm]))


def segment_correction(bits, labels, words):
    """Return (phase, x, z) of one segment from its s fair bits and its s drawn list indices.

    b = sum(bits) is Binomial(s, 1/2); the first b labels keep their drawn
    order and the other s - b are sorted in the Lie order (`correction_word`).
    """
    b = int(sum(bits))
    labels = list(labels)
    return correction_word(b, labels[:b] + sorted(labels[b:]), words)


def _branch_laws(a, cutoff):
    """Return (log mu_step, atan(h), order law) of one branch with a = 2 lambda x > 0 (module docstring)."""
    log_a = log(a)
    eta2, eta3 = exp(2 * log_a - lgamma(3)), exp(3 * log_a - lgamma(4))
    h = eta2 + eta3
    a0 = sqrt(1.0 + h * h)
    tail = fsum(exp(s * log_a - lgamma(s + 1)) for s in range(4, cutoff + 1))
    log_step = 0.5 * log1p(h * h) + log1p(tail / a0)
    low = 0.5 * log1p(h * h) - log1p(a / 3.0)
    logs = [low, low + log_a - log(3.0)] + [s * log_a - lgamma(s + 1) for s in range(4, cutoff + 1)]
    return log_step, atan(h), integer_categorical(np.asarray(logs))


def draw_compensation(fields, rng, rows, l_terms, h_terms, num_qubits):
    """Draw the two correction sequences of every row and return the `LCHSPairCompensation` record.

    The laws are the fixed H and L list tables, built once with their
    cumulative counts, and each branch's order and list-choice tables. The
    signed words depend on the node only through k < 0 (`signed_words`), so
    one table for k >= 0, including -0.0, and one for k < 0 serve every
    branch. The draws follow ``PTSC_DRAW_ORDER``; a branch with a = 0 draws
    nothing and keeps identity corrections.

    Raises:
        ApplicabilityError: When a drawn row's stored normalization
            mu_x*mu_y exceeds mu_cap**2, so that the declared score range
            would not hold for it.
    """
    r, cutoff = fields["steps"], fields["cutoff"]
    lists = []
    for terms in (h_terms, l_terms):
        values = np.array([abs(float(value)) for _, value in terms])
        law = integer_categorical(np.log(values)) if values.size else None
        lists.append((terms, law, fsum(values)))
    (_, h_law, h_norm), (_, l_law, l_norm) = lists
    # Indexed by k < 0.
    signed = (signed_words(h_terms, l_terms, 0.0), signed_words(h_terms, l_terms, -1.0))
    h_size = len(h_terms)
    clusters = rows.clusters
    log_mu = np.zeros((clusters, 2))
    angles = np.zeros((clusters, 2))
    order_counts = np.zeros((clusters, 2, cutoff - 1), dtype=np.int64)
    mixture_counts = np.zeros((clusters, 2, 2), dtype=np.int64)
    segments = np.zeros((4, clusters, 2, r), dtype=np.int64)
    generator = rng.method
    for row in range(clusters):
        for side in (0, 1):
            t = float((rows.time_x, rows.time_y)[side].array[row])
            k = float((rows.node_x, rows.node_y)[side].array[row])
            weight = h_norm + abs(k) * l_norm
            a = 2.0 * weight * t / r
            if a == 0.0:
                continue
            log_step, angles[row, side], order_law = _branch_laws(a, cutoff)
            log_mu[row, side] = r * log_step
            order_counts[row, side] = order_law.counts
            mixture = mixture_law(h_norm, l_norm, k)
            mixture_counts[row, side] = mixture.counts
            words = signed[k < 0]
            orders = order_law.draw(generator, r) + 2
            total = int(orders.sum())
            bits = generator.integers(0, 2, size=total, dtype=np.int64)
            if mixture.counts.min() > 0:
                choice = mixture.draw(generator, total)
            else:
                choice = np.full(total, int(mixture.counts[1] > 0), dtype=np.int64)
            uniforms = generator.integers(0, _TOTAL, size=total, dtype=np.int64)
            labels = np.empty(total, dtype=np.int64)
            for component, law, offset in ((0, h_law, 0), (1, l_law, h_size)):
                chosen = choice == component
                if chosen.any():
                    labels[chosen] = offset + np.searchsorted(law.cumulative, uniforms[chosen], side="right")
            start = 0
            for j, s in enumerate(orders.tolist()):
                phase, x, z = segment_correction(bits[start:start + s].tolist(), labels[start:start + s].tolist(),
                                                 words)
                segments[:, row, side, j] = (s, phase, x, z)
                start += s
    if np.any(log_mu.sum(axis=1) > 2.0 * log(fields["mu_cap"])):
        raise ApplicabilityError("a drawn PTSC row has stored normalization mu_x*mu_y above PTSC.mu_cap**2, so the "
                                 "declared score range does not hold; raise mu_cap")
    return LCHSPairCompensation(
        **fields, h_counts=FrozenArray(np.zeros(0, np.int64) if h_law is None else h_law.counts),
        l_counts=FrozenArray(np.zeros(0, np.int64) if l_law is None else l_law.counts),
        log_mu=FrozenArray(log_mu), rotation_angle=FrozenArray(angles), order_counts=FrozenArray(order_counts),
        mixture_counts=FrozenArray(mixture_counts), segment_order=FrozenArray(segments[0]),
        segment_phase=FrozenArray(segments[1]), segment_x=FrozenArray(segments[2]),
        segment_z=FrozenArray(segments[3]))


def correction_matrix(descriptor, width):
    """Return the dense matrix of one correction descriptor on ``width`` qubits (development and oracle use)."""
    order, phase, x, z, angle = descriptor
    dimension = 1 << width
    if order == 0:
        return np.eye(dimension, dtype=complex)
    pauli = {"I": np.eye(2), "X": np.array([[0, 1], [1, 0]]), "Y": np.array([[0, -1j], [1j, 0]]),
             "Z": np.diag([1, -1])}
    word = np.ones((1, 1), dtype=complex)
    for axis in mask_label(x, z, width):
        word = np.kron(word, pauli[axis])
    if order >= 4:
        return 1j ** phase * word
    return np.cos(angle) * np.eye(dimension) + 1j * low_order_sign(phase) * np.sin(angle) * word


def low_order_sign(phase):
    """Return the sign of P' = Q (even phase) or -iQ (odd phase) in front of the label word of Q."""
    return 1.0 if phase in (0, 1) else -1.0


def mpf_template(labels, alpha):
    """Return the one-macrostep P7 word of the sorted ``labels`` as (label, multiplier) occurrences.

    The multiplier is twice the exponent coefficient in units of h, so a
    factor exp(-i c h G_j) becomes the rotation angle multiplier*t*c_P/r of
    `sampled_native.pair_select_plan` with ``steps`` = r. In chronological
    order: alpha for G_1..G_{K-1}, 1 for G_K, beta for G_{K-1}..G_2, the
    merged 2 beta for G_1, beta for G_2..G_{K-1}, 1 for G_K and alpha for
    G_{K-1}..G_1, 4K - 3 factors whose coefficients sum to 2 for every j.
    One label is exact evolution, exp(-2ihG_1); no label is the identity.
    """
    labels = tuple(labels)
    count = len(labels)
    if count <= 1:
        return tuple((label, 4.0) for label in labels)
    beta = 1.0 - alpha
    inner = labels[:-1]
    word = [(label, 2.0 * alpha) for label in inner] + [(labels[-1], 2.0)]
    word += [(label, 2.0 * beta) for label in reversed(inner[1:])] + [(inner[0], 4.0 * beta)]
    word += [(label, 2.0 * beta) for label in inner[1:]] + [(labels[-1], 2.0)]
    word += [(label, 2.0 * alpha) for label in reversed(inner)]
    return tuple(word)


def correction_cx(descriptors):
    """Return the CX-basis count of correction descriptors: w per s >= 4 controlled Pauli word, 2w per low-order rotation, 0 for phases."""
    total = 0
    for order, _, x, z, _ in descriptors:
        support = (x | z).bit_count()
        if order and support:
            total += support if order >= 4 else 2 * support
    return total


__all__ = [
    "LCHSPairCompensation",
    "LCHSPairMembers",
]
