"""Signed pair scores of a sampled LCHS observable, their joint moments and finite-sample confidence.

The sampled route estimates the quadratic form Q = v^dagger O v and the norm
squared N = ||v||^2 of the selected LCHS vector
``v = integral a(x) |chi_x> dmu(x)``, with unit directions chi_x and
``B = integral |a| dmu`` finite. A classical proposal q covering the support
of a gives the importance weight w = a/q. For two independent draws x, y,

    v^dagger O v = E_{x,y} Re[conj(w(x)) w(y) <chi_x|O|chi_y>],

and the cross terms x != y are part of this mean (An, Liu and Lin,
arXiv:2303.01029v2, Supplement IV; Chakraborty, arXiv:2302.13555v4, Sec. 3.2).

One shot of a row prepares the balanced branch state
``(|0>|chi_x> + exp(i*theta)|1>|chi_y>)/sqrt(2)`` with
``theta = arg w(y) - arg w(x)`` and measures the branch qubit in X and the
system in the basis of one Pauli term P_l of ``O = sum_l o_l P_l``. Direct
expansion gives ``<X (x) O> = Re(exp(i*theta) <chi_x|O|chi_y>)``. With the
branch outcome s_a and the system parity s_P, both in {-1, +1}, and the
term l drawn with the probability p_l of an integer law, the per-shot scores

    Z_Q = (o_l/p_l) |w(x) w(y)| s_a s_P,     Z_N = |w(x) w(y)| s_a

have the means Q and N for any p_l that is positive on every nonzero o_l,
because sum_l p_l*(o_l/p_l)*P_l = O. Their ranges are H_O*W**2 and W**2
for a weight bound W and H_O = max_l |o_l|/p_l, which is sum_l |o_l| only
when p_l is exactly |o_l|/sum_l |o_l|. The
branch amplitudes are exactly 1/sqrt(2), so the weight magnitudes enter only
these scores. One shot gives both scores, so the reducer keeps their joint
moments.

Every score here is in the binary frame of the row table: it is the
physical score divided by g**2, where g = ``LCHSPairMoments.scale`` is the
common weight scale. A physical moment is recovered from a scaled one by
``_quantum_readout.physical_moment``, which multiplies by g**2 without
forming it in binary64.

Rows are independent clusters. A row holds every random choice that stays
fixed during its m conditionally independent shots, including the drawn
observable term. Writing V_b for the variance of the conditional mean
mu(x, y) and V_w for the mean conditional shot variance, B rows of m shots
give ``Var = V_b/B + V_w/(B*m)``. At a fixed total N = B*m the ratio to
fresh pairs (m = 1) is ``1 + (m-1)*rho`` with ``rho = V_b/(V_b + V_w)``, so
repeating a row can add almost no information. The finite-confidence rule
below is the bounded-score Hoeffding inequality for the B equal-weight
cluster means (Hoeffding, doi:10.1080/01621459.1963.10500830, Theorem 2).
It holds for every m and does not improve when m grows.
"""

from math import fsum, isfinite, log, sqrt

import numpy as np
from pydantic import model_validator

from nwqlib._quantum_readout import weighted_group_moments
from nwqlib.core.records import ContentID, FrozenArray, Nonnegative, PositiveInt, Real, Record
from nwqlib.problems.inputs import PhysicalScale


class LCHSPairMoments(Record):
    """Signed joint moments of the scaled pair scores Z_Q and Z_N over the completed rows of one row table.

    A continued run recomputes the moments from all its saved rows, and
    `add_rows` refuses a row that is already included. A raw norm estimate
    `sum_n/shots` can be negative. It is kept as it is and never clipped.

    Attributes:
        table_id: Content hash of the row table whose rows were measured.
        clusters: Rows B of the table, the planned terminal sample.
        repeats: Shots m of every row.
        rows: Sorted indices of the completed rows. A row is completed when
            all m of its shots returned.
        sum_q, sum_n: Sums of Z_Q and Z_N over all shots of the completed rows.
        sum_qq, sum_nn, sum_qn: Sums of Z_Q**2, Z_N**2 and Z_Q*Z_N over those
            shots. Z_Q*Z_N is the system parity times a constant, so it is
            the same-shot covariance information.
        cluster_q, cluster_n: Sums over completed rows of the row means of
            Z_Q and Z_N.
        cluster_qq, cluster_nn, cluster_qn: Sums over completed rows of the
            squares and the product of the row means.
        score_bound_q, score_bound_n: Bounds H_Q = H_O*W**2 and H_N = W**2 on
            |Z_Q| and |Z_N| in the scaled frame, where W bounds every scaled
            weight magnitude of the table.
        scale: Common weight scale g. A physical score is the scaled score
            times g**2.
        contribution_ids: Content hashes of the observation chunks that
            supplied the counts.
    """

    table_id: ContentID
    clusters: PositiveInt
    repeats: PositiveInt
    rows: FrozenArray
    sum_q: Real = 0.0
    sum_n: Real = 0.0
    sum_qq: Nonnegative = 0.0
    sum_nn: Nonnegative = 0.0
    sum_qn: Real = 0.0
    cluster_q: Real = 0.0
    cluster_n: Real = 0.0
    cluster_qq: Nonnegative = 0.0
    cluster_nn: Nonnegative = 0.0
    cluster_qn: Real = 0.0
    score_bound_q: Nonnegative
    score_bound_n: Nonnegative
    scale: PhysicalScale
    contribution_ids: tuple[ContentID, ...] = ()

    @model_validator(mode="after")
    def _rows(self):
        rows = self.rows.array
        if rows.dtype != np.int64 or rows.ndim != 1:
            raise ValueError("completed rows must be a one-dimensional int64 array")
        if rows.size and (rows[0] < 0 or rows[-1] >= self.clusters or np.any(np.diff(rows) <= 0)):
            raise ValueError("completed rows must be distinct sorted indices of the row table")
        return self

    @property
    def completed(self):
        """Number of completed rows."""
        return int(self.rows.array.size)

    @property
    def shots(self):
        """Number of shots of the completed rows."""
        return self.completed * self.repeats


def _rebuilt(moments, **changes):
    """Return a new record with ``changes`` and no parent link.

    Moments of the same rows are the same scientific record whatever order
    the rows were acquired or combined in, so they keep no revision chain.
    """
    values = {name: getattr(moments, name) for name in type(moments).model_fields}
    values.update(changes, parent_id=None)
    return type(moments).model_validate(values)


def empty_pair_moments(table, factor=1.0):
    """Moments of no completed row of ``table``, the start of an acquisition or a continuation.

    ``factor`` bounds every row's recipe or bank factor over its whole law
    (Gamma_num or mu_cap**2, `sampled_recipes`; max(1, max_b zeta_b**2) of a
    QSP bank, `sampled_qsp`), so the score ranges are the table's
    exact-product bounds times it.
    """
    return LCHSPairMoments(table_id=table.content_id, clusters=table.clusters, repeats=table.repeats,
        rows=FrozenArray(np.zeros(0, dtype=np.int64)), score_bound_q=table.score_bound_q * factor,
        score_bound_n=table.score_bound_n * factor, scale=table.scale)


def row_scores(bits, counts, *, branch_mask, parity_mask, magnitude, signed_range, good_mask=0):
    """Return the scaled score sums ``(q, n, qq, nn, qn)`` of one row's returned counts.

    ``bits`` and ``counts`` are the distinct observed outcomes and their
    counts. ``branch_mask`` selects the branch qubit, measured in X after
    its phase, and ``parity_mask`` the system qubits in the support of the
    row's Pauli term, measured in that term's basis. ``magnitude`` is
    |w(x) w(y)| and ``signed_range`` is o_l/p_l, both in the scaled
    frame.

    The parities come from `weighted_group_moments`. Because s_a**2 = 1,
    Z_Q*Z_N = signed_range*magnitude**2*s_P is a parity of ``parity_mask``
    alone, and Z_Q**2 and Z_N**2 are constants.

    A nonzero ``good_mask`` selects the success flags of a sampled QSP row,
    and ``magnitude`` then includes its recovery zeta_b**2. With the
    indicator I_g of every flag being zero, the shot scores are
    ``Z_N = r I_g s_a`` and ``Z_Q = h r I_g s_a s_P`` (r = magnitude, h =
    signed_range), and ``I_g**2 = I_g`` gives ``Z_Q**2 = h**2 r**2 I_g``,
    ``Z_N**2 = r**2 I_g`` and ``Z_Q Z_N = h r**2 I_g s_P`` (B16, B17 of the
    finite-bank relations in `sampled_qsp`). With the good count M_g and the integer sums
    ``S_a``, ``S_aP``, ``S_P`` of the signs over good shots, the sums are
    ``(h r S_aP, r S_a, h**2 r**2 M_g, r**2 M_g, h r**2 S_P)`` (B18). Every
    bad shot is a zero observation: nothing is renormalized, and the row
    mean still divides by all its returned shots, so a row whose shots are
    all bad completes with five zero sums.
    """
    shots = int(np.sum(counts, dtype=np.int64))
    if shots <= 0:
        raise ValueError("a row needs a positive returned shot count")
    if good_mask:
        bits = np.asarray(bits, dtype=np.uint64)
        counts = np.asarray(counts, dtype=np.int64)
        good = (bits & np.uint64(good_mask)) == 0
        bits, counts = bits[good], counts[good]
        s_a = 1 - 2 * (np.bitwise_count(bits & np.uint64(branch_mask)) & 1).astype(np.int64)
        s_p = 1 - 2 * (np.bitwise_count(bits & np.uint64(parity_mask)) & 1).astype(np.int64)
        total_good = int(counts.sum())
        sum_a, sum_ap, sum_p = (int((counts * sign).sum()) for sign in (s_a, s_a * s_p, s_p))
        square = magnitude * magnitude
        return (signed_range * magnitude * sum_ap, magnitude * sum_a, signed_range * signed_range * square * total_good,
                square * total_good, signed_range * square * sum_p)
    mean_q = weighted_group_moments(bits, counts, 1.0, [branch_mask | parity_mask], [signed_range * magnitude])[0]
    mean_n = weighted_group_moments(bits, counts, 1.0, [branch_mask], [magnitude])[0]
    mean_qn = weighted_group_moments(bits, counts, 1.0, [parity_mask], [signed_range * magnitude * magnitude])[0]
    square = magnitude * magnitude
    return (mean_q * shots, mean_n * shots, signed_range * signed_range * square * shots, square * shots,
            mean_qn * shots)


def add_rows(moments, rows, scores, contribution_ids=()):
    """Return ``moments`` with completed rows and their score sums added.

    ``rows`` are row indices not yet in ``moments`` and ``scores`` their
    `row_scores` tuples, each from exactly ``moments.repeats`` shots, since the
    estimator's sample unit is a complete row. The caller leaves out a row
    whose attempt returned fewer shots, so it adds nothing to the completed-row
    population, and a continuation does not measure it again, because its
    attempt is final.
    """
    rows = np.asarray(rows, dtype=np.int64)
    if rows.size != len(scores):
        raise ValueError("every added row needs its score sums")
    if np.intersect1d(rows, moments.rows.array).size or np.unique(rows).size != rows.size:
        raise ValueError("a completed row cannot be added twice")
    m = moments.repeats
    q, n, qq, nn, qn = (np.array([s[i] for s in scores], dtype=float) for i in range(5))
    mq, mn = q / m, n / m
    merged = np.sort(np.concatenate((moments.rows.array, rows)))
    return _rebuilt(moments, rows=FrozenArray(merged),
        sum_q=fsum((moments.sum_q, *q)), sum_n=fsum((moments.sum_n, *n)), sum_qq=fsum((moments.sum_qq, *qq)),
        sum_nn=fsum((moments.sum_nn, *nn)), sum_qn=fsum((moments.sum_qn, *qn)),
        cluster_q=fsum((moments.cluster_q, *mq)), cluster_n=fsum((moments.cluster_n, *mn)),
        cluster_qq=fsum((moments.cluster_qq, *(mq * mq))), cluster_nn=fsum((moments.cluster_nn, *(mn * mn))),
        cluster_qn=fsum((moments.cluster_qn, *(mq * mn))),
        contribution_ids=moments.contribution_ids + tuple(contribution_ids))


def pair_estimates(moments):
    """Return the scaled raw estimates ``(Q_hat, N_hat)``, the equal-weight means of the completed rows' means.

    With equal m per row this equals the mean over all shots. ``None`` for
    no completed row. A negative ``N_hat`` is a legitimate raw value of the
    signed estimator.
    """
    b = moments.completed
    if b == 0:
        return None
    return moments.cluster_q / b, moments.cluster_n / b


def hoeffding_radius(bound, failure, clusters):
    """Return ``bound*sqrt(2*log(2/failure)/clusters)``, the two-sided Hoeffding radius of a mean of bounded scores.

    Each of the ``clusters`` independent row means lies in
    ``[-bound, bound]``, a range of width 2*bound. Hoeffding's Theorem 2
    gives ``P(|mean - mu| >= eps) <= 2*exp(-clusters*eps**2/(2*bound**2))``,
    and setting the right side to ``failure`` gives the radius.
    """
    if not 0.0 < failure < 1.0:
        raise ValueError("failure probability must lie in (0, 1)")
    return bound * sqrt(2.0 * log(2.0 / failure) / clusters)


def pair_confidence(moments, *, failure, quantities, bias_q=None, bias_n=None):
    """Return the scaled estimates and covering radii of the requested quantities, or ``None`` before the terminal sample.

    The guarantee applies to the fixed terminal sample: every planned row is
    completed and every row has the same m. A partial run therefore has no
    finite-confidence radius, only its raw moments and completion. The total
    failure probability ``failure`` is split equally over ``quantities``, a
    nonempty subset of ``("q", "n")``, so a simultaneous Q/N claim uses
    failure/2 for each.

    ``bias_q`` and ``bias_n`` are deterministic bias bounds of the selected
    approximation in the scaled frame (product-formula, kernel, source,
    sampler and preparation errors), each counted once. A requested
    quantity whose bias is ``None`` has no covering radius, because an
    unknown bias is not zero.

    Returns:
        confidence (dict | None): For each requested quantity ``"q"`` or
            ``"n"``, ``(estimate, radius, failure_share)``, where radius is
            the Hoeffding radius plus the bias, or ``None`` in place of the
            radius when the bias is unknown.
    """
    if not quantities or set(quantities) - {"q", "n"} or len(set(quantities)) != len(quantities):
        raise ValueError("quantities must be a nonempty subset of ('q', 'n')")
    if moments.completed != moments.clusters:
        return None
    q_hat, n_hat = pair_estimates(moments)
    share = failure / len(quantities)
    result = {}
    for name in quantities:
        estimate, bound, bias = ((q_hat, moments.score_bound_q, bias_q) if name == "q"
                                 else (n_hat, moments.score_bound_n, bias_n))
        if bias is not None and (not isfinite(bias) or bias < 0):
            raise ValueError("deterministic bias bounds must be finite and nonnegative")
        radius = None if bias is None else hoeffding_radius(bound, share, moments.clusters) + bias
        result[name] = (estimate, radius, share)
    return result


def ratio_interval(q, radius_q, n, radius_n):
    """Return the interval of Q/N over the jointly covered rectangle, or ``None`` unless the norm interval is positive.

    With |Q - q| <= radius_q and |N - n| <= radius_n holding together, q/n
    over the rectangle is monotone in each coordinate when every n in it is
    positive, so its extremes are at the four corners. When ``n - radius_n``
    is not positive the normalized expectation is unresolved: the raw
    estimates and their radii remain, and nothing is clipped or divided.
    """
    if radius_q is None or radius_n is None or not n - radius_n > 0:
        return None
    corners = [(q + sq * radius_q) / (n + sn * radius_n) for sq in (-1.0, 1.0) for sn in (-1.0, 1.0)]
    return min(corners), max(corners)


def norm_domain_interval(n, radius_n):
    """Return the covering interval of N intersected with N >= 0, leaving the signed raw estimate unchanged."""
    if radius_n is None:
        return None
    return max(0.0, n - radius_n), max(0.0, n + radius_n)
