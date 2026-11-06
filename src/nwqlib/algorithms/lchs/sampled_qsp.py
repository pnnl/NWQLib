"""Sampled LCHS pairs whose branches evolve by a finite bank of QSP polynomials.

``LCHS(observable_estimator="sampled_pairs",
hamiltonian_evolution_backend="qsp_block_encoding")`` keeps the pair rows of
`sampled_rows` and replaces the product formula of each branch by a
qubitized Laurent polynomial from a finite phase bank (`sampled_qsp_bank`).
The labels B1–B18 below name the finite-bank relations of the sampled pairs,
derived here; `sampled_qsp_bank` derives the bank's own construction, and the
other owners cite these labels.

Children. The scaled branch queries call two children fixed for the whole
Plan, the encodings of L_c and H_c (`_children`). A native Pauli or band
part X is its centered child ``X_c = X - c_X I``, its identity coefficient
c_X moved into the row phase. A supplied encoding of a Hermitian part is queried as it is,
from the operator input that the Plan keeps, so ``c_X = 0``: the PSD shift
s stays in the row magnitude ``exp(s t)`` and the phase
``-t (k (c_L + s) + c_H)``, which is ``-t k s`` for a supplied L, and is
never inserted into the supplied query. A general ``A = L + i H`` given as
one supplied encoding with ``||alpha_A B_A - A|| <= delta_A`` has no
separate parts. The single-A construction of `joint` (module docstring,
Single A) with the one coefficient c = 1 or c = -i and radius 1 gives
``Herm(A) = L`` and ``Herm(-i A) = H``, ``Herm X = (X + X^dagger)/2``, each
with normalization alpha_A, a full-unitary involution and error
``delta_A + delta_c (alpha_A + delta_A)``, delta_c the coefficient table's
stored-value residual. So ``alpha_L = alpha_H = alpha_A``,
``a_x = (|k_x| + 1) alpha_A`` and ``b_max = T_e alpha_A (K_* + 1)``. One
single-A query of ``(k - i) A`` would have the smaller normalization
``sqrt(k**2 + 1) alpha_A``, but its coefficient table would change with k
in every row, which the row templates do not bind; with the two children, k
enters only the scale, combine and sign choices that they bind. A supplied
part's assertion that its target is Hermitian is neither an
encoded-Hermitian nor a full-involution fact, so a scaled query that is not
a recorded full involution takes the generic lift of
`plan_qubitized_evolution`, which acts on the Hermitian part of the encoded
block within the same recovered error ``t_x delta_x + t_b |r_hat - r|``
(B3). A supplied L or general A without an error bound lacks the norm
bound that the PSD shift and the kernel need (an asserted lower bound alone
is not enough), and `selection.compact_common` refuses it; a supplied H
without one leaves its child term, and with it the inner bias, unknown.

Coverage and buckets (B1). Branch x evolves ``U_x = exp(-i t_x G_x)``,
``G_x = k_x L_c + H_c``, through the separate-parts query of normalization
``a_x = |k_x| alpha_L + alpha_H`` (`joint.ScaledBranchAccess`; an absent
part contributes zero). Before any draw, ``b_max = T_e (K_* alpha_L +
alpha_H)`` covers the whole support: T_e is the supremum of the active
elapsed times (``application_facts.active_time``, never a drawn time) and
K_* the largest node of positive probability or the continuous kernel's
guarded endpoint. ``b_max`` is the larger of its upward exact value and the
binary64 value of the same operations at the support maximum; binary64
addition and multiplication of nonnegative numbers are monotone, so the
second covers every row value ``s_x = fl(t_x fl(|k_x| alpha_L + alpha_H))``
formed below. The bank holds the K edges ``t_b = b_max 2**(b - K + 1)``,
exact powers of two of b_max. A row takes the smallest edge with
``t_b >= max(s_x, s_y)``; both branches use that bucket and its recovery
zeta_b, and a both-zero row (``s_x = s_y = 0``) bypasses QSP with recovery 1.

Scaled branch (B2–B4). With ``r_x = s_x / t_b`` in [0, 1], the query
``D(r_x) (x) V_x`` has normalization one and target ``t_x G_x / t_b``, so the
bank's target of polynomial time t_b evolves exactly ``U_x``: its selected
block approximates ``U_x / zeta_b``. A zero branch is ``D(0) (x) I`` in the
same bucket polynomial, whose value at zero is ``1/zeta_b`` up to the
polynomial error, so the common recovery stays zeta_b. The two branches
are separate controlled qubitized sequences (``ctrl_state`` 0 and 1) on a
common clean workspace, which composes to the block-diagonal pair unitary
even for unequal degrees; its cost is the sum of the two sequences.

Rounded weights and stored angles (B23). With ``a_x = |k_x| alpha_L +
alpha_H`` exact, the planner forms ``w_l = fl(|k_x| alpha_L)``, ``a_hat =
fl(w_l + alpha_H)``, ``s_x = fl(t_x a_hat)`` and ``r_x = fl(s_x / t_b)``. The
query's block is ``r_hat (c**2 B_L' + (1 - c**2) B_H)``, with B_L' the L
block signed by k, ``r_hat = cos(theta_s/2)`` for ``theta_s = 2
fl(arccos r_x)``, and for two parts ``c**2 = cos**2(theta_c/2)`` for
``theta_c = 2 fl(atan2(fl(sqrt alpha_H), fl(sqrt w_l)))`` (one part takes
c = 1, and a zero branch has block ``r_hat I``). Writing ``alpha_L B_L``
and ``alpha_H B_H`` as the centered parts within delta_L and delta_H,
``||t_b X_hat - t_x G_x|| <= t_x (|k_x| delta_L + delta_H)
+ |t_b r_hat - t_x a_x| + 2 t_x |a_x c**2 - |k_x| alpha_L|``, where a zero
branch has ``t_b r_hat + t_x a_x`` in the second term, which the chain
below also bounds because its r_x is zero. Round to
nearest gives ``|fl(z) - z| <= u|z| + eta/2`` for a product or quotient
and ``u|z|`` for a sum, u = 2**-53, eta = 2**-1074, and libm arccos and
atan2 are assumed within 2 ulp (relative error 4u) of a normal result and
within eta of a subnormal one. Then ``|t_b r_hat - t_x a_x|`` chains the
arccos term ``2 pi u t_b``, the division ``u s_x + t_b eta/2``, the
product ``u t_x a_hat + eta/2`` and the weights ``t_x |a_hat - a_x| <=
(2u + u**2) t_x a_x + (1 + u) t_x eta/2``. With ``c0**2 = w_l/(w_l +
alpha_H)``, ``|c**2 - c0**2| <= (1.01 + 2 pi) u + eta`` and ``a_x c0**2 -
|k_x| alpha_L = alpha_H (w_l - |k_x| alpha_L)/(w_l + alpha_H)``. Since
``t_x a_x``, ``s_x`` and ``t_b`` are at most b_max and ``t_x`` at most
T_e, the last two terms of every branch of the support are at most
``b_max rho + eta (3 b_max + 2 T_e + 1)``, ``rho = 32 u``
(``BRANCH_ROUNDING``), a numerical estimate under these hypotheses. The
eta term is needed: alpha_L = eta, k = 1/2 and t = 2**1023 round w_l and
s_x to zero, and the whole generator time ``t_x a_x = 2**-52 = b_max/2``
of that zero branch is rounding.

Members (B7, B12). Each bucket's actual integer law has counts n_j and
probabilities n_j/2**52 (`sampled_rows.IntegerCategorical`). After the
outer draws, the Method's stream draws, bucket by bucket in ascending
order, the branch-x members of that bucket's rows and then, separately,
their branch-y members (``MEMBER_DRAW_ORDER``), even when a row's two
physical labels coincide. Independence gives
``E[P_x^dagger O P_y] = Pbar_x^dagger O Pbar_y`` (B12), so the measured cross
term carries only the mean amplitudes. Repeated shots reuse the row's
members.

Scores (B16–B18). One shot with all workspace flags zero (I_g = 1) scores
``Z_N = A zeta_b**2 I_g s_a`` and ``Z_Q = h_l A zeta_b**2 I_g s_a s_P``;
every bad shot is a zero observation and the row mean divides by all
returned shots. The score ranges use ``F = max(1, max_b zeta_b**2)`` over
the whole bank, never the buckets of one draw.

Bias (B13, B15). With recovered mean amplitude errors ``beta_x``, ``beta_y``
of the two branches, ``|E[cross] - U_x^dagger O U_y| <= ||O|| (beta_x +
beta_y + beta_x beta_y)``, so with one support-wide ``beta`` the N bias is
``(G_cap/g)**2 (2 beta + beta**2)`` with the mean-weight cap G_cap and
the Q bias multiplies by the observable bound; a deterministic bank is the
single-member case (B15). ``beta`` adds to the bank's largest recovered
mean error the support-wide child term ``T_e (K_* delta_L + delta_H)``,
once, and the support-wide rounding term of B23 (`generator_error`), so
it does not depend on the drawn rows. The shared model bias is added once
(``sampled_plan._model_bias``).
"""

from dataclasses import dataclass
from fractions import Fraction
from math import fsum, remainder, tau

import numpy as np

from nwqlib.core.records import FrozenArray, Nonnegative, Real, Record, Text
from .sampled_rows import DiscreteKernel, IntegerCategorical, json_state

MEMBER_DRAW_ORDER = "bucket_ascending,member_x,member_y/1"
BANK_POLICIES = ("deterministic", "stochastic")
BANK_PARAMETERS = {"deterministic": ("buckets", "margin"), "stochastic": ("buckets", "margin", "cutoff")}
# B23: rho = 32 u exceeds (6 pi + 8.02) u + 3 u**2 + u**3, u = 2**-53, and eta = 2**-1074.
BRANCH_ROUNDING = Fraction(32, 2**53)
SMALLEST_SUBNORMAL = Fraction(1, 2**1074)


class LCHSPairQSP(Record):
    """The finite QSP bank summary and the per-row bucket and member choices of a sampled-pair Plan.

    `LCHSReconstruction.pair_qsp` holds it, chosen once at planning; the
    prepared phases are in the Plan's native payload. Arrays are indexed by
    row of `LCHSSampledRows`.

    Attributes:
        policy: ``"deterministic"`` or ``"stochastic"`` (`LCHS.pair_qsp_bank`).
        branch_policy: ``"independent"``: separate x and y member draws.
        allowance_frame: ``"algorithmic"``; the allowance is
            ``LCHS_QSP_EPSILON_FRACTION * approximation_tolerance``.
        mean_amplitude_allowance: Recovered amplitude allowance of every
            bucket: the mean of a stochastic bank, the individual
            polynomial of a deterministic one.
        margin: Normalized circle cap ``1 - margin``.
        cutoffs: Stochastic cutoff of each bucket, empty for a
            deterministic bank.
        coverage: b_max, the support bound of ``t a``.
        node_bound: K_*.
        active_time: T_e.
        alpha_l, alpha_h: Normalizations of the two branch children (module
            docstring, Children), alpha_A each for a general supplied A, 0
            for an absent part.
        child_error_l, child_error_h: Their error bounds, None when unknown.
        bucket_times: Edges t_b.
        recoveries: zeta_b of each bucket.
        counts: Integer law of each bucket, summing to 2**52.
        shells: Shell n of each member, None for the deterministic endpoint.
        member_signals: Signal degree of each member of each bucket.
        individual_errors: Largest recovered individual polynomial error of
            each bucket's members, including the actual phase residual, or
            None when unknown.
        mean_errors: Recovered mean polynomial error of each bucket (B8), or
            None when unknown.
        score_factor: F = max(1, max_b zeta_b**2), the bound of every row
            factor.
        flag_qubits: Plan-wide success workspace width; flags occupy the
            classical bits after the branch bit and the system bits.
        bucket: Bucket of each row, -1 for a both-zero row.
        member_x, member_y: Member of each branch, -1 for a both-zero row.
        member_draw_state: PCG64 state before the member draws.
        member_draw_order: Order of those draws.
        solver_calls: Phase targets fitted.
        generator_error: The recovered generator term of beta on the
            whole support, the child term plus the rounding term of the
            branch weights and stored angles (module docstring, B23), or
            None when a child's error bound is unknown.
        expected_pair_signals: E[d_x + d_y] over the outer and member laws
            (`expected_pair_signals`), None for a continuous outer law.
        qualification: Numerical status of the bank and its errors.
    """

    policy: Text
    branch_policy: Text = "independent"
    allowance_frame: Text = "algorithmic"
    mean_amplitude_allowance: Nonnegative
    margin: Real
    cutoffs: tuple[int, ...] = ()
    coverage: Nonnegative
    node_bound: Nonnegative
    active_time: Nonnegative
    alpha_l: Nonnegative
    alpha_h: Nonnegative
    child_error_l: Nonnegative | None
    child_error_h: Nonnegative | None
    bucket_times: tuple[Nonnegative, ...]
    recoveries: tuple[Real, ...]
    counts: tuple[tuple[int, ...], ...]
    shells: tuple[tuple[int | None, ...], ...]
    member_signals: tuple[tuple[int, ...], ...]
    individual_errors: tuple[Nonnegative | None, ...]
    mean_errors: tuple[Nonnegative | None, ...]
    score_factor: Real
    flag_qubits: int
    bucket: FrozenArray
    member_x: FrozenArray
    member_y: FrozenArray
    member_draw_state: Text
    member_draw_order: Text = MEMBER_DRAW_ORDER
    solver_calls: int
    generator_error: Nonnegative | None
    expected_pair_signals: Nonnegative | None
    qualification: Text

    def row_factors(self):
        """Return zeta_b**2 of each row's bucket, 1 for a both-zero row."""
        recovery = np.asarray(self.recoveries + (1.0,), dtype=float)
        return recovery[self.bucket.array] ** 2


@dataclass(frozen=True)
class QSPForecast:
    """The B-independent QSP facts of one planning pass: children, coverage, phase-free bank plan and F."""

    l_child: object
    h_child: object
    alpha_l: float
    alpha_h: float
    node_bound: float
    active_time: float
    coverage: float
    bank: object
    factor: Fraction
    flag_qubits: int
    query_work: int | None = 0
    query_unresolved: str | None = None


def bank_options(method):
    """Return the validated ``(policy, buckets, margin, cutoff)`` of ``LCHS.pair_qsp_bank``."""
    from nwqlib._validation import finite_real, integer
    config = method.pair_qsp_bank
    policy, parameters = config.implementation, dict(config.parameters)
    if policy not in BANK_POLICIES:
        raise ValueError(f"pair_qsp_bank implementation must be 'deterministic' or 'stochastic', not {policy!r}")
    unknown = sorted(set(parameters) - set(BANK_PARAMETERS[policy]))
    if unknown:
        raise ValueError(f"pair_qsp_bank {policy!r} takes the parameters {', '.join(BANK_PARAMETERS[policy])}, "
                         f"not {unknown}")
    buckets = integer(parameters.get("buckets", 1), "pair_qsp_bank buckets", 1)
    if buckets > 64:
        raise ValueError("pair_qsp_bank buckets above 64 would place edges below the binary64 range of b_max")
    margin = finite_real(parameters.get("margin", 1 / 16), "pair_qsp_bank margin")
    if not 0 < margin < 1:
        raise ValueError("pair_qsp_bank margin must lie strictly between 0 and 1")
    cutoff = parameters.get("cutoff")
    cutoff = None if cutoff is None else integer(cutoff, "pair_qsp_bank cutoff", 0)
    return policy, buckets, margin, cutoff


def _coverage(alpha_l, alpha_h, node_bound, active_time):
    """Return b_max >= every s_x = fl(t_x fl(|k_x| alpha_L + alpha_H)) and >= the exact support value (module docstring)."""
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    exact = _upward_float(Fraction(active_time) * (Fraction(node_bound) * Fraction(alpha_l) + Fraction(alpha_h)))
    return max(exact, active_time * (node_bound * alpha_l + alpha_h))


def _children(method, common):
    """Return the ``(l_child, h_child)`` that every scaled branch query of the Plan calls, None for a zero part.

    The module docstring (Children) gives the three forms. A supplied part
    is hidden from the native planner
    (`compiled_selection._compiled_qsp_compact_part_plans`, shared with the
    coherent route), so that planner's work ledger and byte count cover the
    native parts only, and the supplied operator input is used as the child.
    """
    from dataclasses import replace
    from nwqlib.subroutines.block_encoding.joint import plan_single_a_generator
    from .compiled_selection import _compiled_qsp_compact_part_plans
    generator = common.generator
    if generator.single_a is not None:
        return tuple(plan_single_a_generator(a_child=generator.single_a, node_coefficients=(coefficient,),
                                             radius=1.0) for coefficient in (1.0, -1j))
    parts = (generator.dissipative, generator.hamiltonian)
    supplied = tuple(part if part is not None and part.manifest.reference.representation == "block_encoding"
                     else None for part in parts)
    native = replace(generator, dissipative=None if supplied[0] is not None else parts[0],
                     hamiltonian=None if supplied[1] is not None else parts[1])
    unitary = common.l_norm == 0
    first, second, _, _ = _compiled_qsp_compact_part_plans(native, unitary=unitary, max_bytes=method.max_bytes,
                                                           max_select_work=method.max_select_work)
    planned = (second, first) if unitary else (first, second)
    return tuple(given if given is not None else None if plan.implementation == "exact_zero" else plan
                 for given, plan in zip(supplied, planned))


def child_units(child):
    """Return P, the size of one call of a branch child in the pair block's construction-size rule.

    A native child counts its stored Pauli terms or bands. A supplied
    encoding counts the instructions of its circuit, which each query copies
    and controls gate by gate. A single-A child calls its encoding forward
    and adjoint and adds its one-entry coefficient table.
    """
    from nwqlib.subroutines.block_encoding.core import BlockEncodingPlan
    from nwqlib.subroutines.block_encoding.joint import JointEncodingPlan, _supplied
    if isinstance(child, JointEncodingPlan):
        return 2 * child_units(child.components["a_child"]) + 1
    if isinstance(child, BlockEncodingPlan):
        return len(child.decomposition.terms) if child.decomposition is not None else len(child.source.offsets)
    return len(_supplied(child).circuit.data)


def forecast_qsp(method, common, kernel, facts):
    """Plan the children, coverage and phase-free bank of a sampled QSP pass, before any row exists.

    No phase is synthesized: the bank plan fixes every admitted target and its
    phase allowance (`plan_sampled_qsp_bank`), whose retained member arrays
    are checked against ``LCHS.max_bytes``. The queries of the widest branch
    are checked against ``LCHS.max_bytes`` by their builders' laws, at the
    controls that `build_qsp_pair` builds them with, and their work in one
    pair circuit against ``LCHS.max_select_work`` (``query_work``). A
    supplied gate whose definition is not built leaves that work unresolved
    (``query_work`` None, ``query_unresolved`` the reason), which an explicit
    ``max_select_work`` refuses.

    Raises:
        ApplicabilityError: For a bucket count that puts the first bank time
            below the normal binary64 range of b_max.
        ValueError: For a native child whose planning exceeds
            ``max_select_work`` or ``max_bytes``, a branch query that
            exceeds ``max_bytes``, branch query work above
            ``max_select_work``, or an explicit ``max_select_work`` with
            unresolved branch query work.
        SampledQSPBankError: For a bank refusal, with its stage.
    """
    from nwqlib.algorithms.protocol import ApplicabilityError
    from nwqlib.subroutines.block_encoding.joint import (ScaledBranchAccess, _component_query_requirements,
                                                         access_summary)
    from nwqlib.subroutines.qsp.evolution import _qubitized_census
    from nwqlib._limits import ResourceLimitError, exceeds
    from .native import LCHS_QSP_EPSILON_FRACTION
    from .quantum import _refuse_uncounted_select
    from .sampled_plan import _kernel_support
    from .sampled_qsp_bank import plan_sampled_qsp_bank
    generator = common.generator
    l_child, h_child = _children(method, common)
    alpha_l = 0.0 if l_child is None else access_summary(l_child).alpha
    alpha_h = 0.0 if h_child is None else access_summary(h_child).alpha
    end = _kernel_support(kernel)[0] if l_child is not None else 0.0
    active = float(facts.active_time)
    coverage = _coverage(alpha_l, alpha_h, end, active)
    policy, buckets, margin, cutoff = bank_options(method)
    bank, factor, flags, query_work, query_unresolved = None, Fraction(1), 0, 0, None
    if coverage > 0:
        # The Plan-wide workspace is the largest branch sequence of the support: every present part (the L part
        # needs a nonzero node), and the lift helper unless the scaled query is a recorded full involution.
        node = end if end > 0 else 0.0
        widest = ScaledBranchAccess(l_child=l_child, h_child=h_child, node=node, scale=1.0,
                                    system_qubits=generator.num_qubits)
        summary = access_summary(widest)
        flags = 1 + int(summary.full_unitary_involution is not True) + summary.num_ancillas
        # build_qsp_pair calls each branch sequence under the branch qubit. A branch at node 0 calls the H child
        # alone, without the combine control, and its sequence is lifted only if the widest one is, so the
        # widest branch's queries, at the controls of _qubitized_census, are the largest that any row builds.
        # The builders' laws refuse their bytes here with the message that preparation would give. A pair
        # circuit builds them for each of its two branch sequences, which is the work that max_select_work
        # bounds. A supplied gate whose definition is not built leaves that work unresolved.
        for _, controls, adjoint, _ in _qubitized_census(1, summary.full_unitary_involution is not True, 1):
            _, work, missing = _component_query_requirements(widest, num_controls=controls, adjoint=adjoint,
                                                             max_bytes=method.max_bytes)
            query_work, query_unresolved = query_work + 2 * work, query_unresolved or missing
        if query_unresolved is not None:
            _refuse_uncounted_select(method, query_unresolved)
            query_work = None
        elif exceeds(query_work, method.max_select_work):
            raise ResourceLimitError(f"the branch queries of one sampled QSP pair circuit need {query_work} work units, "
                                     f"above LCHS.max_select_work={method.max_select_work}")
        edges = tuple(coverage * 2.0 ** (index - buckets + 1) for index in range(buckets))
        if edges[0] < np.finfo(float).tiny:
            raise ApplicabilityError(f"pair_qsp_bank buckets={buckets} put the first bank time below the normal "
                                     f"binary64 range of b_max={coverage!r}; use fewer buckets")
        bank = plan_sampled_qsp_bank(
            edges, mean_amplitude_allowances=LCHS_QSP_EPSILON_FRACTION * float(method.approximation_tolerance),
            policy=policy, cutoffs=None if cutoff is None else (cutoff,) * buckets, margin=margin,
            max_bytes=method.max_bytes)
        factor = max(Fraction(1), max(Fraction(bucket.recovery) ** 2 for bucket in bank.buckets))
    return QSPForecast(l_child=l_child, h_child=h_child, alpha_l=alpha_l, alpha_h=alpha_h, node_bound=end,
                       active_time=active, coverage=coverage, bank=bank, factor=factor, flag_qubits=flags,
                       query_work=query_work, query_unresolved=query_unresolved)


def branch_products(nodes, times, alpha_l, alpha_h):
    """Return ``s = fl(t fl(|k| alpha_L + alpha_H))`` of each branch, the binary64 value that both bucket and scale use."""
    nodes, times = np.asarray(nodes, dtype=float), np.asarray(times, dtype=float)
    return times * (np.abs(nodes) * alpha_l + alpha_h)


def bind_rows(forecast, rows, prepared, rng):
    """Return ``(bucket, member_x, member_y, draw_state)`` of every row; members are drawn here once.

    Rows of one bucket draw the members of branch x, then of branch y, in
    ascending bucket order, from the saved integer law of that bucket.
    """
    clusters = rows.clusters
    state = json_state(rng.method.bit_generator.state)
    bucket = np.full(clusters, -1, dtype=np.int64)
    member_x = np.full(clusters, -1, dtype=np.int64)
    member_y = np.full(clusters, -1, dtype=np.int64)
    if prepared is None:
        return bucket, member_x, member_y, state
    s_x = branch_products(rows.node_x.array, rows.time_x.array, forecast.alpha_l, forecast.alpha_h)
    s_y = branch_products(rows.node_y.array, rows.time_y.array, forecast.alpha_l, forecast.alpha_h)
    largest = np.maximum(s_x, s_y)
    edges = np.asarray([plan.polynomial_time for plan in prepared.plan.buckets])
    if np.any(largest > edges[-1]):
        raise ValueError("a drawn branch lies beyond the bank coverage b_max; the support bound does not cover it")
    nonzero = largest > 0
    bucket[nonzero] = np.searchsorted(edges, largest[nonzero], side="left")
    for index, plan in enumerate(prepared.plan.buckets):
        chosen = np.flatnonzero(bucket == index)
        if chosen.size:
            law = IntegerCategorical(counts=np.asarray(plan.counts, dtype=np.int64))
            member_x[chosen] = law.draw(rng.method, chosen.size)
            member_y[chosen] = law.draw(rng.method, chosen.size)
    return bucket, member_x, member_y, state


def row_branches(payload, rows, record, row):
    """Return the two branch descriptions ``(access, target, time)`` of ``row``, or None for a both-zero row.

    ``access`` is the `joint.ScaledBranchAccess` with the saved scale
    ``r = s/t_b``, ``target`` the prepared member and ``time`` t_b.
    """
    from nwqlib.subroutines.block_encoding.joint import ScaledBranchAccess
    bucket = int(record.bucket.array[row])
    if bucket < 0:
        return None
    prepared = payload.qsp_bank.buckets[bucket]
    time = prepared.plan.polynomial_time
    result = []
    for node, elapsed, member in ((rows.node_x.array[row], rows.time_x.array[row], record.member_x.array[row]),
                                  (rows.node_y.array[row], rows.time_y.array[row], record.member_y.array[row])):
        value = float(branch_products(node, elapsed, record.alpha_l, record.alpha_h))
        result.append((ScaledBranchAccess(l_child=payload.l_child, h_child=payload.h_child, node=float(node),
                                          scale=value / time, system_qubits=payload.num_system_qubits),
                       prepared.members[int(member)], time))
    return tuple(result)


def row_evolutions(payload, rows, record):
    """Yield, row by row, the ``(access, evolution)`` of each branch, () for a both-zero row, for row counts.

    The evolution plan (`plan_qubitized_evolution`) of a branch is formed
    once per branch query choice (`_branch_key`), children, bank member and
    bucket time, and its degree, signal access and query census are reused
    for the other branches with the same choices. Those three depend on the
    member's phases and on the parts the query holds, not on the row's node
    or scale, which change only the child error term that row counts do not
    read. The memo holds one entry per distinct choice, with the objects it
    keys by identity, and each row's accesses are released once the caller
    moves to the next row.
    """
    from nwqlib.subroutines.qsp.evolution import plan_qubitized_evolution
    memo = {}
    for row in range(rows.clusters):
        branches = row_branches(payload, rows, record, row)
        pairs = []
        for access, target, time in () if branches is None else branches:
            key = (_branch_key(access), id(access.l_child), id(access.h_child), access.system_qubits,
                   id(target), time)
            if key not in memo:
                memo[key] = access, target, plan_qubitized_evolution(access, target, evolution_time=time)
            pairs.append((access, memo[key][2]))
        yield tuple(pairs)


def _branch_key(access):
    """Return the discrete choices of one branch query: zero block, part presence and the sign of k."""
    if access.scale == 0.0:
        return "zero"
    parts = ("L" if access.l_child is not None and access.node != 0.0 else "") + (
        "H" if access.h_child is not None else "")
    return parts + ("-" if parts.startswith("L") and access.node < 0 else "")


def _continuous_angles(access):
    """Return the scale and combine angles of one branch query, None for an absent combine qubit."""
    from nwqlib.subroutines.block_encoding.joint import _branch_parts
    from nwqlib.subroutines.block_encoding.nodes import _rotation_table
    scale = float(_rotation_table(np.array([access.scale]), 1.0)[0][0])
    _, theta, _ = _branch_parts(access)
    return scale, theta


def build_qsp_pair(payload, rec, problem, row, *, parameters=None):
    """Build the pair circuit of ``row`` on [branch, system, flags], unmeasured.

    The gate order is H on the branch qubit, the input preparations (one
    uncontrolled for equal kinds, each under its branch value for mixed
    kinds, `sampled_native`), the x sequence under branch value 0 and the y
    sequence under value 1 (`build_qubitized_evolution_query` on the scaled
    branch queries with the saved member targets and ``evolution_time =
    t_b``), the readout phase ``P(theta)``, H on the branch qubit and the
    basis change of the row's term. A both-zero row applies no evolution.

    ``parameters`` (a row template) is the list of `qsp_row_skeleton`'s five
    slots, the two scale angles, the two combine angles and theta, with a
    Qiskit parameter in every slot the row uses; the gates are otherwise
    those of the row.
    """
    from qiskit import QuantumCircuit, QuantumRegister
    from nwqlib.subroutines.qsp.evolution import build_qubitized_evolution_query
    from .sampled_native import _input_gate
    from .sampled_plan import _input_preparations
    rows, record = rec.pair_rows, rec.pair_qsp
    q, flags = payload.num_system_qubits, record.flag_qubits
    branch, system = QuantumRegister(1, "branch"), QuantumRegister(q, "system")
    registers = [branch, system] + ([QuantumRegister(flags, "flags")] if flags else [])
    circuit = QuantumCircuit(*registers, name="lchs_sampled_qsp_pair")
    root, workspace = branch[0], list(circuit.qubits[1 + q:])
    kinds = (int(rows.kind_x.array[row]), int(rows.kind_y.array[row]))
    preparations = _input_preparations(payload, problem)
    circuit.h(root)
    if preparations is not None and kinds[0] == kinds[1]:
        circuit.append(_input_gate(preparations, kinds[0]), list(system))
    elif preparations is not None:
        for slot, kind in enumerate(kinds):
            circuit.append(_input_gate(preparations, kind, slot), [root, *system])
    branches = row_branches(payload, rows, record, row)
    if branches is not None:
        for slot, (access, target, time) in enumerate(branches):
            if parameters is not None:
                from dataclasses import replace
                access = replace(access, angles=(parameters[slot], parameters[2 + slot]))
            gate = build_qubitized_evolution_query(access, target, evolution_time=time, num_controls=1,
                                                   ctrl_state=slot, max_bytes=payload.max_bytes)
            circuit.append(gate, [root, *workspace[:gate.num_qubits - 1 - q], *system])
    theta = remainder(float(rows.phase_y.array[row]) - float(rows.phase_x.array[row]), tau)
    circuit.p(theta if parameters is None else parameters[4], root)
    circuit.h(root)
    term = int(rows.term.array[row])
    label = payload.term_labels[term] if term >= 0 else "I" * q
    for qubit, axis in enumerate(reversed(label)):
        if axis == "Y":
            circuit.sdg(system[qubit])
        if axis in "XY":
            circuit.h(system[qubit])
    return circuit


def qsp_row_skeleton(payload, rec, row):
    """Return ``(key, values)``: the discrete gate choices of ``row`` and its five continuous angles.

    The key holds the bucket, both members, both input kinds, the term and
    each branch's zero block, present parts and sign of k, so rows with equal
    keys emit the same gates; the values are the five slots of
    `build_qsp_pair`, None for a slot the row does not use (a both-zero row
    uses theta only, a branch with one part has no combine angle). Equal keys
    use the same slots.
    """
    rows, record = rec.pair_rows, rec.pair_qsp
    theta = remainder(float(rows.phase_y.array[row]) - float(rows.phase_x.array[row]), tau)
    kinds = (int(rows.kind_x.array[row]), int(rows.kind_y.array[row]))
    branches = row_branches(payload, rows, record, row)
    if branches is None:
        return ("both_zero", kinds, int(rows.term.array[row])), (None, None, None, None, theta)
    angles = [_continuous_angles(access) for access, _, _ in branches]
    key = (int(record.bucket.array[row]), int(record.member_x.array[row]), int(record.member_y.array[row]), kinds,
           tuple(_branch_key(access) for access, _, _ in branches), int(rows.term.array[row]))
    return key, (angles[0][0], angles[1][0], angles[0][1], angles[1][1], theta)


def qsp_row_templates(plan, points, block, payload, rec, problem):
    """Group sampled QSP row points by `qsp_row_skeleton` and build one parameterized pair circuit per group."""
    from hashlib import sha256
    from qiskit.circuit import ParameterVector
    from nwqlib.algorithms.protocol import RowTemplate
    from .sampled_plan import point_row
    groups = {}
    for point in points:
        row = point_row(point)
        key, values = qsp_row_skeleton(payload, rec, row)
        groups.setdefault(key, (row, values, []))[2].append((point.key, tuple(v for v in values if v is not None)))
    templates = []
    for key, (row, first, members) in groups.items():
        vector = iter(ParameterVector("lchs_qsp_pair", sum(value is not None for value in first)))
        slots = [None if value is None else next(vector) for value in first]
        circuit = build_qsp_pair(payload, rec, problem, row, parameters=slots)
        templates.append(RowTemplate(key="sha256:" + sha256(repr(key).encode()).hexdigest(),
                                     block_id=block.record.content_id, circuit=circuit,
                                     parameters=tuple(slot for slot in slots if slot is not None),
                                     rows=tuple(members)))
    return tuple(templates)


def generator_error(forecast, prepared):
    """Return the recovered generator term of beta on the whole support (module docstring, B3 and B23), or None.

    It is the child term ``T_e (K_* delta_L + delta_H)`` plus the rounding
    term ``b_max rho + eta (3 b_max + 2 T_e + 1)`` of the branch weights and
    stored angles, evaluated exactly. It does not depend on the drawn rows,
    and it is None when a child's error bound is unknown.
    """
    from nwqlib.subroutines.block_encoding.joint import access_summary
    if prepared is None:
        return Fraction(0)
    deltas = [0.0 if child is None else access_summary(child).error_bound
              for child in (forecast.l_child, forecast.h_child)]
    if None in deltas:
        return None
    active, coverage = Fraction(forecast.active_time), Fraction(forecast.coverage)
    child = active * (Fraction(forecast.node_bound) * Fraction(deltas[0]) + Fraction(deltas[1]))
    return child + coverage * BRANCH_ROUNDING + SMALLEST_SUBNORMAL * (3 * coverage + 2 * active + 1)


def qsp_pair_bias(method, problem, common, data, kernel, facts, frame, terms, forecast, prepared, generator):
    """Return the `LCHSPairBias` of a sampled QSP Plan (module docstring, B13/B15).

    ``beta`` is the largest recovered mean polynomial error of the bank
    (`PreparedQSPBucket.recovered_mean_error`, B8 with the actual-angle
    residuals) plus ``generator`` (`generator_error`). The N bias in the row
    frame is ``(G_cap/g)**2 (2 beta + beta**2)`` with ``G_cap =
    LCHSPairBias.mean_weight_bound``; the shared model bias is added once.
    """
    from nwqlib.problems.inputs import compose_recovery
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    from .primary_records import LCHSPairBias
    from .sampled_plan import _application_masses, _kernel_support, _model_bias, _observable_bound, _row_frame
    from .solution_error_budget import scaled_sum
    reasons = []
    total = scaled_sum([mass for mass, _ in _application_masses(facts)])
    end, cap, _ = _kernel_support(kernel)
    mean = compose_recovery(total, cap)
    beta = Fraction(0)
    if prepared is not None:
        means = [bucket.recovered_mean_error for bucket in prepared.buckets]
        if None in means:
            reasons.append("a bank member has no actual-angle phase residual, so its mean polynomial error is unknown")
        if generator is None:
            reasons.append("a centered child encoding has no error bound")
        beta = None if None in means or generator is None else max(Fraction(value) for value in means) + generator
    inner = None
    if beta is not None:
        value = _upward_float(2 * beta + beta * beta)
        inner = 0.0 if value == 0 else _row_frame(compose_recovery(mean, mean, value), frame, 2)
        if inner is None:
            reasons.append("the QSP bank bias exceeds the binary64 range of the row frame")
    vector, model, model_reasons = _model_bias(method, problem, common, data, kernel, facts, total, mean, cap, end,
                                               frame)
    return LCHSPairBias(
        step_rule=None, inner_relation="qsp_bank", product_formula_operator=None, mean_weight_bound=mean,
        model_vector_bound=vector, inner_bias=inner, model_bias=model, observable_bound=_observable_bound(terms),
        kernel_allowances=None if isinstance(kernel, DiscreteKernel) else
        (0.5 * float(method.approximation_tolerance),) * 2, reasons=tuple(reasons + model_reasons))


def qsp_record(method, forecast, prepared, binding, generator, expected):
    """Return the `LCHSPairQSP` of a planned bank, its prepared errors and the row choices."""
    from nwqlib.subroutines.block_encoding.joint import access_summary
    from nwqlib.subroutines.trotterization.error_budget import _upward_float
    from .native import LCHS_QSP_EPSILON_FRACTION
    policy, _, margin, cutoff = bank_options(method)
    bucket, member_x, member_y, state = binding
    plans = () if prepared is None else prepared.plan.buckets
    prepared_buckets = () if prepared is None else prepared.buckets
    notes = ["bank coefficients, recoveries, polynomial errors and phase residuals are binary64 numerical values "
             "without an arithmetic enclosure",
             "the generator term bounds the rounding of the branch weights and stored angles on the whole support, "
             "assuming libm arccos and atan2 within 2 ulp",
             "the stochastic allowance bounds the recovered mean amplitude; each member keeps its own individual "
             "error" if policy == "stochastic" else
             "the allowance bounds each recovered bucket polynomial"]

    def summary_error(child):
        return 0.0 if child is None else access_summary(child).error_bound
    return LCHSPairQSP(
        policy=policy, mean_amplitude_allowance=LCHS_QSP_EPSILON_FRACTION * float(method.approximation_tolerance),
        margin=margin,
        cutoffs=tuple(plan.cutoff for plan in plans) if policy == "stochastic" else (),
        coverage=forecast.coverage, node_bound=forecast.node_bound, active_time=forecast.active_time,
        alpha_l=forecast.alpha_l, alpha_h=forecast.alpha_h, child_error_l=summary_error(forecast.l_child),
        child_error_h=summary_error(forecast.h_child), bucket_times=tuple(plan.polynomial_time for plan in plans),
        recoveries=tuple(plan.recovery for plan in plans), counts=tuple(tuple(plan.counts) for plan in plans),
        shells=tuple(plan.shells for plan in plans),
        member_signals=tuple(tuple(member.target.signals for member in plan.members) for plan in plans),
        individual_errors=tuple(None if item.normalized_individual_error is None else
                                item.plan.recovery * item.normalized_individual_error for item in prepared_buckets),
        mean_errors=tuple(item.recovered_mean_error for item in prepared_buckets),
        score_factor=_upward_float(forecast.factor), flag_qubits=forecast.flag_qubits, bucket=FrozenArray(bucket),
        member_x=FrozenArray(member_x), member_y=FrozenArray(member_y), member_draw_state=state,
        solver_calls=0 if prepared is None else prepared.solver_calls,
        generator_error=None if generator is None else _upward_float(generator),
        expected_pair_signals=expected, qualification="; ".join(notes))


def qsp_pair_resources(plan):
    """Return the populations, phase preparation and query census of a sampled QSP Plan without building a circuit.

    ``rows``, ``bindings``, ``preparations`` and ``shots`` are as for a
    product-formula Plan. Phase preparation is reported separately from the
    rows: ``phase_targets`` (every member of every bucket, fitted once) and
    ``solver_calls``.
    In a row with a bucket, each branch runs one controlled qubitized
    sequence of d signals on its scaled query, a zero branch included: its
    query ``D(0) (x) I`` calls no child but keeps the bucket's common
    recovery. A both-zero row runs no sequence. A sequence makes ``d``
    calls of its query with two controls for a direct full-involution
    walk, or 2d singly controlled calls for the generic lift, d walk
    reflections and ``3(d + 1) + 1`` processing rotations
    (`plan_qubitized_evolution`). One scaled query
    calls the scale table once and each present child once, in its own
    direction; ``child_query_calls_total`` counts these child calls.
    ``row_signals`` and ``scaled_query_calls`` give each row's (min, max)
    and the table total over every executed shot. ``expected_pair_signals``
    is the planning value `expected_pair_signals` over the whole outer and
    member laws, None for a continuous outer law; ``maximum_pair_signals``
    is twice the largest member degree. Input
    preparations are counted as for product-formula rows, and no CX law of
    the controlled queries or depth is given.

    ``supplied_forward_calls_total`` and ``supplied_adjoint_calls_total``
    count, over every executed shot, the calls of the user's supplied
    encodings and of their adjoints (`_supplied_calls`): a supplied
    Hermitian part is called once per call of its child, in that call's
    direction, and every call of a single-A child of a general A calls A
    and A^dagger once each, singly controlled on its Hermitianization
    helper. The calls of the two children of one A are added. Native
    children, zero branches and both-zero rows call none.
    """
    from nwqlib.subroutines.block_encoding.joint import _branch_parts
    rec, payload = plan.reconstruction, plan._native.get("native_data")
    rows, record = rec.pair_rows, rec.pair_qsp
    bank = payload.qsp_bank
    signals = np.zeros(rows.clusters, dtype=np.int64)
    calls = np.zeros(rows.clusters, dtype=np.int64)
    reflections = np.zeros(rows.clusters, dtype=np.int64)
    rotations = np.zeros(rows.clusters, dtype=np.int64)
    child_calls = np.zeros(rows.clusters, dtype=np.int64)
    supplied = np.zeros((rows.clusters, 2), dtype=np.int64)
    for row, branches in enumerate(row_evolutions(payload, rows, record)):
        for access, evolution in branches:
            lift = evolution.signal_access == "generic_lift"
            signals[row] += evolution.degree
            calls[row] += (2 if lift else 1) * evolution.degree
            reflections[row] += evolution.query_census["walk_reflections"]
            rotations[row] += evolution.query_census["processing_rotations"]
            present = _branch_parts(access)[0]
            child_calls[row] += (2 if lift else 1) * evolution.degree * len(present)
            # Each scaled-query call of the sequence (its "controlled" census rows: d forward calls for the
            # direct walk, d forward and d adjoint calls for the lift) calls every present child once.
            for _, _, adjoint, count in evolution.query_census["controlled"]:
                for _, child, _, _ in present:
                    supplied[row] += count * np.asarray(_supplied_calls(child, adjoint), dtype=np.int64)
    mixed = int(np.count_nonzero(rows.kind_x.array != rows.kind_y.array))
    preparations = dict(uncontrolled=rows.clusters - mixed, controlled_pairs=mixed) if payload.prepares_inputs else None
    maximum = 2 * max((max(member) for member in record.member_signals), default=0)
    m = rows.repeats
    return dict(rows=rows.clusters, repeats=m, bindings=rows.clusters, preparations=rows.clusters,
                shots=rows.clusters * m, templates=len(plan.construction.selections),
                qubits=1 + payload.num_system_qubits + record.flag_qubits,
                clbits=1 + payload.num_system_qubits + record.flag_qubits, policy=record.policy,
                buckets=len(record.bucket_times), phase_targets=0 if bank is None else bank.plan.target_count,
                solver_calls=record.solver_calls,
                both_zero_rows=int(np.count_nonzero(record.bucket.array < 0)),
                row_signals=(int(signals.min()), int(signals.max())), signals_total=int(signals.sum()) * m,
                scaled_query_calls=(int(calls.min()), int(calls.max())), scaled_query_calls_total=int(calls.sum()) * m,
                child_query_calls_total=int(child_calls.sum()) * m,
                supplied_forward_calls_total=int(supplied[:, 0].sum()) * m,
                supplied_adjoint_calls_total=int(supplied[:, 1].sum()) * m,
                walk_reflections_total=int(reflections.sum()) * m, processing_rotations_total=int(rotations.sum()) * m,
                expected_pair_signals=record.expected_pair_signals,
                maximum_pair_signals=maximum, input_preparations=preparations, input_preparation_cx=None,
                scaled_query_cx=None)


def _supplied_calls(child, adjoint):
    """Return ``(forward, adjoint)`` calls of a user's supplied encoding made by one call of a branch child.

    ``adjoint`` is the direction of the child call. A supplied Hermitian
    part is that encoding and is called once in that direction. A single-A
    child calls its ``a_child`` as its census rows say (``query_census``
    of `joint.plan_single_a_generator`: one A and one A^dagger, singly
    controlled on the helper, which the external controls never reach), in
    either direction, because its adjoint is the same construction. A
    native child calls none.
    """
    from nwqlib.subroutines.block_encoding.core import BlockEncodingPlan
    from nwqlib.subroutines.block_encoding.joint import JointEncodingPlan
    if isinstance(child, BlockEncodingPlan):
        return 0, 0
    if isinstance(child, JointEncodingPlan):
        rows = [row for row in child.query_census["controlled_calls"] if row[0] == "a_child"]
        return (sum(count for _, _, flag, count in rows if not flag),
                sum(count for _, _, flag, count in rows if flag))
    return (0, 1) if adjoint else (1, 0)


def expected_pair_signals(laws, forecast):
    """Return ``E[d_x + d_y]`` over the outer law and the member laws, or None for a continuous outer law.

    A row in bucket b runs two independent members with mean degree
    ``E_b[d]`` each (``expected_independent_pair_signals``), and a both-zero
    row runs none. For a discrete outer law the two branches are iid with
    ``F(t) = P(s <= t)`` over the finite category and node laws, so
    ``P(bucket <= b) = F(t_b)**2`` and the both-zero mass is ``F(0)**2`` (the
    finite-law factorization of B19/§7 of the policy), which avoids
    enumerating pairs.
    """
    bank = forecast.bank
    if bank is None:
        return 0.0
    if laws.continuous_source or laws.node_law is None:
        return None
    category = laws.category_law.counts / float(1 << 52)
    node = laws.node_law.counts / float(1 << 52)
    s = branch_products(np.asarray(laws.kernel.nodes, dtype=float)[None, :], laws.elapsed[:, None],
                        forecast.alpha_l, forecast.alpha_h)
    weight = category[:, None] * node[None, :]
    zero = fsum(weight[s == 0].ravel())
    expected, below = 0.0, zero
    for plan in bank.buckets:
        cumulative = fsum(weight[s <= plan.polynomial_time].ravel())
        expected += (cumulative * cumulative - below * below) * plan.expected_independent_pair_signals
        below = cumulative
    return expected
