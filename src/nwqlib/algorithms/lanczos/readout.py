"""SDK-free signed readout over supplied histograms; no operator application.

A Lanczos count outcome stores the SELECT index in its least significant
bits. Its used row supplies a sign and the system-support mask. The signed
value is the sign times the support parity, with zero for an unused address.
Reflection readout returns +1 only for index zero. Vectorized decoding uses
the same first and second moments and the returned count population. Exact
probability reduction preserves the supplied mass without renormalization.
"""

from collections.abc import Mapping
from math import fsum
from numbers import Integral

import numpy as np

from nwqlib.operators.access import _check_bytes
from .records import MomentStatistics


def packed_lanczos_census(table):
    """Derive the Lanczos frame and ordered SELECT metadata from packed Pauli words.

    A row is identity when every word of x OR z is zero. The identity
    coefficients sum to center, and the absolute nonzero nonidentity
    coefficients sum to alpha, using fsum in table order. The selected signs
    are sign(c.real), support masks are x OR z, and K coefficients are c.real
    divided by alpha. A zero alpha selects the algebraic scalar-operator path
    and performs no division.

    A native row ``(x, z, c)`` represents ``c i**popcount(x&z) X**x Z**z``.
    Its support is ``x|z``. It is identity exactly when every support word is
    zero. For an admitted Hermitian table, coefficients are real. Exact-zero
    coefficients and identity rows are removed in source table order. For
    the remaining J rows

        center = fsum(c_j : support_j = 0),
        alpha = fsum(|c_j| : support_j != 0, c_j != 0),
        K = sum_(j<J) (c_j/alpha) P_j,

    the encoding of Kirby, Motta and Mezzacapo, arXiv:2208.00567v4, Section
    2.1, Eqs. (2)-(4), after the identity term is removed. The sign is
    ``sign(c_j.real)`` and the parity mask is ``x_j|z_j``. No extra Y phase
    belongs in that sign: the basis rotation already maps the actual
    Hermitian Pauli P_j to the measured Z parity. The identity mask also
    handles a table with more than one identity row. Keep `fsum`, as
    `_pauli_census` does, if correctly rounded center and alpha are part of
    equivalence. Replacing it with np.sum can move the frame and near-tie
    decisions. With one nonidentity term K is +-P, so its even
    moments are one. The SELECT index width is zero for J<=1 and
    ``(J-1).bit_length()`` otherwise.

    Args:
        table: Admitted PauliTerms with (M, W) uint64 x/z words and (M,)
            complex128 coefficients.

    Returns:
        ``(center, alpha, rows, support, signs, scaled)``: the frame, the
        source-table row of each used SELECT address, its (J, W) uint64
        support mask, its int8 sign and its float64 K coefficient, all in
        SELECT address order.

    Raises:
        ValueError: A coefficient has a nonzero imaginary part.
    """
    coefficients = table.coefficients
    if np.any(coefficients.imag != 0):
        raise ValueError("Lanczos needs real Pauli coefficients")
    support = table.x | table.z
    identity = ~np.any(support, axis=1)
    used = (~identity) & (coefficients != 0)
    center = fsum(map(float, coefficients.real[identity]))
    physical = coefficients.real[used]
    alpha = fsum(map(float, np.abs(physical)))
    scaled = physical / alpha if alpha else np.empty(0, dtype=float)
    return (center, alpha, np.flatnonzero(used), support[used],
            np.sign(physical).astype(np.int8), scaled)


class LanczosReadout:
    """The SELECT readout table of one bound Plan, derived from its packed Pauli table.

    The selected operator's packed Pauli table fixes the SELECT address order
    (packed_lanczos_census). The Plan stores only the scalar frame and the
    table identity. Loading derives this table again from the saved operator,
    so the classical action and the quantum decoder use a common term order.

    It also keeps the moment statistics already decoded for the bound Plan,
    keyed by the chunk identity and the setting identity. A second analysis
    of the same chunks decodes nothing.

    The constructor takes the admitted PauliTerms table and the byte limit
    max_bytes, which it checks before the census allocates. For M source
    rows and W uint64 words per mask, the census and readout constructor
    admit M*(16*W+56) additional array-data bytes plus a 65,536-byte fixed
    allowance before allocation. The data law covers simultaneous support,
    coefficient, index and sign arrays and the constructor's owning row
    copy. The input Pauli table is excluded. The fixed allowance covers
    headers and bookkeeping on the qualified stack, including empty and
    small tables. Derivation: at the largest return-expression frontier,
    support arrays contribute 8W(M+J), two masks 2M and the selected
    coefficient/index/sign arrays at most 33J bytes, so the frontier holds
    at most 8W(M+J)+2M+33J <= M(16W+35) <= M(16W+56) bytes for J <= M used
    rows. Earlier identity, mask and reduction phases fit that envelope, and
    the constructor's owning row copy of at most J*(8W+25) bytes fits after
    the full-size census scratch dies (ENGINEERING_CONSTANTS.md, "Chebyshev
    Lanczos defaults").

    Attributes:
        center: Identity shift of the frame, in the operator's units.
        alpha: Scale of the frame, in the operator's units. Zero marks a scalar operator.
        index_width: SELECT index qubits a, zero for at most one used term.
        rows: Read-only source-table row of each used SELECT address.
        signs: Read-only int8 sign(c_j) per used SELECT address.
        masks: Read-only (J, W) uint64 system-support mask x_j|z_j per used SELECT address.
        coefficients: Read-only float64 K coefficient c_j/alpha per used SELECT address.
    """

    __slots__ = ("center", "alpha", "index_width", "rows", "signs", "masks", "coefficients",
                 "_wide_masks", "_statistics")

    def __init__(self, table, *, max_bytes):
        rows_count, words = table.x.shape
        # B_census = H0 + M*(16*W+56) with H0 = 65536 (class docstring).
        _check_bytes(65536 + rows_count*(16*words+56), max_bytes, "Lanczos Pauli census")
        center, alpha, rows, masks, signs, scaled = packed_lanczos_census(table)
        # Read-only owners published as views cannot be made writeable again,
        # so the table and the statistics decoded from it stay consistent. An
        # array that is itself a view is copied first so that it owns its data.
        arrays = {}
        for name, array in (("rows", rows), ("masks", masks), ("signs", signs), ("coefficients", scaled)):
            if array.base is not None:
                array = array.copy()
            array.flags.writeable = False
            arrays[name] = array.view()
        values = dict(center=center, alpha=alpha,
                      index_width=0 if len(signs) <= 1 else (len(signs) - 1).bit_length(),
                      **arrays, _wide_masks=None, _statistics={})
        for name, value in values.items():
            object.__setattr__(self, name, value)

    def __setattr__(self, name, value):
        raise AttributeError("a LanczosReadout is read-only")

    def __delattr__(self, name):
        raise AttributeError("a LanczosReadout is read-only")

    def wide_masks(self):
        """Return the support masks joined into Python integers, for readouts wider than 64 bits."""
        if self._wide_masks is None:
            object.__setattr__(self, "_wide_masks", tuple(
                sum(word << (64 * j) for j, word in enumerate(row)) for row in self.masks.tolist()))
        return self._wide_masks

    def chunk_moments(self, setting, chunk, *, counts):
        """Return the MomentStatistics of one chunk, decoding it at most once per setting.

        The key is the chunk's content identity and the setting's identity.
        The cache lives with this table, so it belongs to one bound Plan.
        """
        key = (chunk.content_id, setting.content_id)
        if key not in self._statistics:
            self._statistics[key] = decode_histogram(self, setting, chunk.histogram(), counts=counts)
        return self._statistics[key]


def outcome(readout, kind, bits):
    """Return the signed single-shot value in {-1, 0, 1} of one measured outcome.

    This is the scalar form of signed_outcomes, for outcomes of any width
    as Python integers. The mean of these values over the walk state
    |psi_j> = (RU)^j (|G>|psi>), j = floor(k/2), of Kirby, Motta and
    Mezzacapo, arXiv 2208.00567v4, Eq. (24), p. 5, is the moment mu_k by
    their Eq. (26), p. 6:
    mu_2j = <psi_j|R|psi_j> and mu_(2j+1) = <psi_j|U|psi_j>.

    The low num_index_qubits bits hold the SELECT address and the remaining
    bits the system register. A reflection readout returns +1 when the index
    register reads all zeros after PREP^dagger and -1 otherwise, which
    measures R=2|G><G|-I (Kirby Eq. (6) and Section 3.1, step 2). A select
    readout of a used address i returns sign(c_i) times the parity of the
    system bits where P_i acts:
    f(b) = s_i [1 - 2 (popcount((b >> a) & m_i) mod 2)].
    The label-controlled basis change applied before measurement, or before
    an exact trajectory saves the marginal and undoes the change, makes that
    parity the P_i eigenvalue, so the mean is <U> for
    U=sum_i |i><i| sign(c_i) P_i (Eq. (27)). This coherent readout is
    NWQLib's variant of step 3. Padding addresses, which the ideal walk never
    populates, return 0 and stay in the denominator.

    Args:
        readout: LanczosReadout of the bound Plan.
        kind: The setting's readout, ``reflection`` or ``select``.
        bits: Measured outcome as a nonnegative integer, index bits least significant.

    Returns:
        +1 or -1 for a reflection readout. sign(c_i) times the P_i parity
        for a select readout of a used address, and 0 for a padding address.
    """
    a = readout.index_width
    index = bits & ((1 << a) - 1)
    if kind == "reflection":
        # PREP^dagger maps R = G(2|0><0|-I)G^dagger to 2|0><0|-I on the index
        # register. Its eigenvalue is +1 on the all-zero outcome and -1 on every
        # other outcome, so this value averages to mu_2j = <psi_j|R|psi_j>.
        return 1 if index == 0 else -1
    if index >= len(readout.signs):
        return 0
    # After the basis change P_i acts as a Z string on the bits in mask, whose
    # eigenvalue is -1 exactly when an odd number of those bits read one.
    return int(readout.signs[index]) * (-1 if ((bits >> a) & readout.wide_masks()[index]).bit_count() % 2 else 1)


def signed_outcomes(bits, *, index_width, histogram_width, signs, masks, reflection=False):
    """Return the int8 signed value of every uint64 outcome, the vectorized form of outcome.

    For an unsigned integer outcome b, the low a bits are exactly
    i = b mod 2**a, and a right shift by a gives the system-coordinate bits.
    For a used row, f(b) = s_i [1 - 2 (popcount((b >> a) & m_i) mod 2)],
    the scalar outcome definition. Padding rows return zero. For a reflection
    setting, f(b) = 2*1_(i=0) - 1, independent of the upper bits. With a=0
    every reflection outcome is +1 and every select outcome uses row zero.
    Width 64 uses the full uint64 domain without constructing a uint64 2**64.
    Reflection probability settings have width a, while count settings can
    carry a+n bits, so the width check uses the setting's histogram width.

    Args:
        bits: One-dimensional uint64 outcomes, index bits least significant.
        index_width: SELECT index bits a.
        histogram_width: Admitted bit width of an outcome, at most 64.
        signs: int8 sign per used SELECT address.
        masks: uint64 system-support mask per used SELECT address.
        reflection: True for a reflection setting.

    Raises:
        ValueError: The outcomes are not a one-dimensional uint64 array, the
            widths are outside 0..64, an outcome exceeds the readout width or
            the sign and mask tables do not align.
    """
    bits = np.asarray(bits)
    if bits.dtype != np.dtype("uint64") or bits.ndim != 1:
        raise ValueError("outcomes must be a one-dimensional uint64 array")
    if not 0 <= index_width <= histogram_width <= 64:
        raise ValueError("uint64 decoding requires widths within 0..64")
    if histogram_width < 64 and np.any(bits >> np.uint64(histogram_width)):
        raise ValueError("histogram outcome exceeds readout width")
    index_mask = np.uint64((1 << index_width) - 1)
    index = bits & index_mask
    if reflection:
        return np.where(index == 0, 1, -1).astype(np.int8)
    signs = np.asarray(signs, dtype=np.int8)
    masks = np.asarray(masks, dtype=np.uint64)
    if signs.shape != masks.shape or signs.ndim != 1:
        raise ValueError("sign and mask tables must align")
    result = np.zeros(bits.size, dtype=np.int8)
    used = index < len(signs)
    rows = index[used].astype(np.intp)
    system = bits[used] >> np.uint64(index_width)
    parity = np.bitwise_count(system & masks[rows]) & 1
    result[used] = signs[rows] * (1 - 2 * parity.astype(np.int8))
    return result


def readout_values(readout, kind, width, histogram):
    """Return the int8 signed values of a histogram's outcomes under one readout kind and width.

    kind is ``reflection`` or ``select`` and width the setting's admitted
    histogram width. Widths up to 64 bits decode the uint64 indices with
    signed_outcomes. Wider histograms decode their Python-integer indices
    with outcome, entry by entry, after the same range check.

    Args:
        readout: LanczosReadout of the bound Plan.
        kind: ``reflection`` or ``select``.
        width: Admitted bit width of an outcome.
        histogram: Histogram with width, indices() and index_list().
    """
    if width <= 64 and histogram.width <= 64:
        return signed_outcomes(
            histogram.indices(),
            index_width=readout.index_width,
            histogram_width=width,
            signs=readout.signs,
            masks=readout.masks[:, 0],
            reflection=kind == "reflection",
        )
    values = np.empty(histogram.entries, dtype=np.int8)
    for position, bits in enumerate(histogram.index_list()):
        if not 0 <= bits < 1 << width:
            raise ValueError("histogram outcome exceeds readout width")
        values[position] = outcome(readout, kind, bits)
    return values

def reduce_moments(values, weights, *, counts):
    """Reduce signed values and their weights to the first two moments.

    With counts, the weights are nonnegative integer frequencies, the mean is
    sum(w*x)/sum(w) and the population is the exact integer count total.
    Without counts, the weights are exact probabilities used as returned,
    with population 1, so missing probability, negative roundoff and padding
    leakage stay in the moments. x*x is one on used select rows and zero on
    padding, and one everywhere for reflection. The products are formed
    elementwise and exactly, and fsum reduces them in stored order.

    Decode the signed marginal with denominator one and zero weight for
    unused SELECT addresses. Comparison adds probability-evaluation and
    signed-sum errors to both routes' state contributions. Unknown execution
    error is not replaced by the population floor.

    Decode `m_hat=fsum(s_j*p_hat_j)` with exact signs `s_j` in `{-1,0,1}`,
    zero signs on unused SELECT addresses, and denominator one. Let
    `psi_tilde` be the computed state entering probability formation,
    `R2=||psi_tilde||_2**2`, `w` its full native width, `N=2**w`,
    `u=2**-53`, and `eta=2**-1074`. For binary64 component-square/add
    probabilities with round-to-nearest, gradual underflow, no overflow,
    truncation or renormalization, and `2*N*u<=1/2`, use
    `e_prob=max(exact_readout_roundoff(w)*u, (2*N*u)/(1-2*N*u))` and
    `U_prob=4*N*eta`; the latter bounds absolute probability-evaluation
    underflow and may be zero under an established normal-range or
    exact-underflow premise. With correctly rounded `fsum`, the absolute
    arithmetic error relative to `sum_j s_j*p_j`, where `p_j` is the exact
    bin probability of `psi_tilde`, satisfies
    `a_moment <= e_prob*R2 + u*sum_j abs(p_hat_j) + U_prob + eta
    <= [e_prob+u*(1+e_prob)]*R2 + (1+u)*U_prob + eta`. If the receipt
    supplies a state-distance allowance `delta` from a unit target,
    `R2 <= (1+delta)**2`; otherwise that substitution is unavailable.
    Evaluate bound expressions outward when publishing them as numerical
    upper bounds.

    The first inequality uses `|s_j|<=1` and a single rounded signed sum.
    The second uses `sum|p_hat_j| <= (1+e_prob)R2+U_prob`. The final `eta`
    covers the sum's rounding separately. Algebraically supplied moments
    bypass this readout and have no `a_moment` charge.

    Comparison of an exact trajectory moment with a separately prepared one:
    when both routes have justified total state allowances `Delta_traj,k`
    and `Delta_prefix,k` to the same ideal readout, and absolute scalar
    allowances `a_traj,k` and `a_prefix,k` (the `a_moment` above), the
    triangle inequality through that common ideal value gives
    `|m_hat_k,traj - m_hat_k,prefix| <= D(Delta_traj,k) + D(Delta_prefix,k)
    + a_traj,k + a_prefix,k` with `D(x) = 2*x + x**2`. It applies separately
    to every requested degree, needs no probabilistic union factor, and does
    not permit root-sum-square combination of the coherent residuals. With
    an unamplified sum `delta_exec,k` of valid local absolute execution
    errors, the trajectory's allowance in the unitary-reference case is
    `Delta_traj,k <= b_k + (1 + b_k)*delta_exec,k`, where `b_k` is the
    construction-only difference that the restoration residuals of the
    earlier readout views accumulate (``method._trajectory_program``). If a
    single bound already measures the entire computed execution against its
    complete represented circuit, the triangle inequality instead gives
    `Delta <= b + delta`, without counting the construction twice.

    For a view whose inverse is a different numerical construction of the
    stored table's adjoint, such as an inverted UCG decomposition, the
    restoration residual enters the recurrence for `b_k` in
    ``method._trajectory_program`` as its `eps_i`. For one target with
    stored table entries `U_c`, the represented decomposition `C_c` and
    the actually lowered inverse block `A_c`, let `d_c = ||C_c - U_c||_2`,
    `nu_c = ||A_c - C_c^dagger||_2` and `g_c = ||U_c^dagger U_c - I||_2`.
    Then `A_c U_c - I = (A_c - C_c^dagger) U_c + (C_c - U_c)^dagger U_c
    + (U_c^dagger U_c - I)` gives
    `r_c = ||A_c U_c - I||_2 <= (d_c + nu_c)*||U_c||_2 + g_c`, without
    assuming that either set of computed matrices is exactly unitary. With
    `r_q = max_c ||A_(q,c) U_(q,c) - I_2||_2` over the addresses of target
    `q`, orthogonality of the control projectors and the tensor-product
    expansion give `eps_B = rho_phi + prod_q (1 + r_q) - 1`, where
    `rho_phi = |exp(i*Phi) - exp(i*sum_q phi_q)|` charges the assembly of the
    inverse's phase `Phi` from the factor phases `phi_q`. The odd view's
    inverse here is the native adjoint table, whose residual is the Gram
    defect stated in ``method._trajectory_program``.

    `state_error()` is unavailable when the executed native circuit has an
    unresolved exclusion. Report the actual exclusions and operation count
    for each route. Multiplexer application and matrix-unitary construction
    or application remain outside the current state-error derivation. A
    separately bounded readout restoration residual is an additional
    construction contribution and does not clear those exclusions. A
    missing state bound remains unavailable in the derived moment
    comparison. When the receipts of both routes list an exclusion, such as
    Aer's native `multiplexer`, neither route has a state bound (`delta` is
    None) and the comparison supplies no numerical acceptance tolerance.
    Knowing `eps_B`, observing a nearly unit norm or reducing the operation
    count does not provide the missing execution error, and the `1e-12`
    probability-window floor does not replace it. A named regression at
    full native width at most three compares with `atol=2e-12, rtol=0`, an
    empirical threshold, and a wider register records the degreewise
    differences and both receipts without a derived pass or fail tolerance.

    Sources: Aer 0.17.2, `QubitVector::probability` and `probabilities`
    (https://github.com/Qiskit/qiskit-aer/blob/0.17.2/src/simulators/statevector/qubitvector.hpp#L2090-L2148),
    CPython 3.12.14, `math.fsum`
    (https://github.com/python/cpython/blob/v3.12.14/Modules/mathmodule.c#L1249-L1465),
    and the gamma propagation model in Higham, *Accuracy and Stability of
    Numerical Algorithms*, 2nd ed., DOI 10.1137/1.9780898718027. The `fsum`
    statement retains the qualification against excess precision and unsafe
    reassociation: platforms subject to the documented excess-precision
    exception need an additional correctly-rounded-sum allowance or a proved
    conservative summation owner.

    Args:
        values: int8 signed values.
        weights: int64 counts or float64 probabilities, aligned with values.
        counts: True for count weights.

    Returns:
        MomentStatistics with shots equal to the count total, or None for probabilities.
    """
    first = values * weights
    second = (values * values) * weights
    if not counts:
        population = 1
    elif weights.size and int(weights.max()) * weights.size > np.iinfo(np.int64).max:
        # Each count fits int64, but their sum may not: add Python integers.
        population = sum(weights.tolist())
    else:
        population = int(weights.sum())
    if population <= 0:
        raise ValueError("empty returned count population")
    return MomentStatistics(
        mean=fsum(first.tolist()) / population,
        second_moment=fsum(second.tolist()) / population,
        shots=population if counts else None,
    )


def decode_histogram(readout, setting, histogram, *, counts):
    """Reduce one returned histogram to the first two moments of its signed outcome.

    histogram is a Histogram (``ObservationChunk.histogram()``) or a mapping
    from outcome to weight. Mapping keys are bit strings, ``0x`` hex strings
    or nonnegative integers, and they are validated as integers before any
    uint64 conversion. With counts, each weight is a nonnegative integer
    frequency, so the mean is sum(w*x)/sum(w) over the outcome values x.
    Without counts, the weights are exact probabilities that are used as
    returned, with population 1. Missing probability, negative roundoff and
    padding leakage then stay in the moments. Weights may also be analytical
    fixture weights.

    Calling this kernel makes no acquisition/provenance claim. Runtime analysis
    separately validates actual ObservationChunks against their completed trace.

    Returns:
        MomentStatistics with the mean and second moment of the signed
        outcome and shots equal to the returned count total, or None for
        probabilities.
    """
    if isinstance(histogram, Mapping):
        histogram, weights = _mapping_histogram(histogram, setting.histogram_width, counts=counts)
    else:
        weights = histogram.weights
    values = readout_values(readout, setting.readout, setting.histogram_width, histogram)
    return reduce_moments(values, weights, counts=counts)


def _mapping_histogram(mapping, width, *, counts):
    """Validate a mapping histogram and return it as (indices, weights) arrays."""
    keys, weights = [], []
    for key, weight in mapping.items():
        if isinstance(key, str):
            value = key.replace(" ", "")
            bits = int(value, 16 if value.startswith("0x") else 2)
        elif isinstance(key, Integral) and not isinstance(key, bool):
            bits = int(key)
        else:
            raise ValueError("histogram outcomes must be bit strings or integers")
        if not 0 <= bits < 1 << width:
            raise ValueError("histogram outcome exceeds readout width")
        if counts and (isinstance(weight, bool) or not isinstance(weight, Integral) or weight < 0):
            raise ValueError("counts require nonnegative integer frequencies")
        if counts and weight > np.iinfo(np.int64).max:
            raise ValueError("a count exceeds the int64 readout limit 2**63 - 1")
        keys.append(bits)
        weights.append(weight)
    return (_Outcomes(width, keys),
            np.array(weights, dtype=np.int64 if counts else np.float64).reshape(len(weights)))


class _Outcomes:
    """Validated mapping outcomes with the Histogram index accessors readout_values reads."""

    __slots__ = ("width", "entries", "_keys")

    def __init__(self, width, keys):
        self.width, self.entries, self._keys = width, len(keys), keys

    def indices(self):
        return np.array(self._keys, dtype=np.uint64).reshape(self.entries)

    def index_list(self):
        return self._keys
